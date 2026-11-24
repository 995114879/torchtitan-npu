# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""CPU-side stop requests and replica-local safe-boundary coordination."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class LastWordsConfig:
    enable: bool = False
    coordination_backend: str = "service"
    run_id: str = ""
    """Unique launch ID shared by every replica; change it on restart."""
    timeout_seconds: int = 300
    heartbeat_timeout_seconds: int = 60
    resume_path: str = ""
    """A published candidate directory; empty selects the latest compatible candidate."""

    def validate(self) -> None:
        if self.coordination_backend != "service":
            raise ValueError("Only the service coordination backend is supported")
        if not self.enable:
            return
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("last_words.run_id must be a unique nonempty path component shared by all replicas")
        if self.timeout_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("Last-words timeouts must be positive")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_checkpoint_tree(path: Path, *, root: Path) -> None:
    """Persist copied checkpoint data and directory entries before publication."""
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint durability root must contain the checkpoint")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {path}")

    def failed_walk(error):
        raise error

    for directory, _, files in os.walk(path, topdown=False, onerror=failed_walk):
        for name in files:
            with (Path(directory) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(Path(directory))
    parent = path.parent
    while parent.is_relative_to(root):
        _sync_directory(parent)
        if parent == root:
            break
        parent = parent.parent


def atomic_json(path: Path, value: Any, *, durable: bool = False, sync_file: bool = True) -> None:
    missing = []
    parent = path.parent
    while durable and not parent.exists():
        missing.append(parent)
        parent = parent.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            if sync_file or durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            _sync_directory(path.parent)
            for directory in missing:
                _sync_directory(directory.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_fault_reports(directory: Path) -> list:
    reports = []
    try:
        for path in directory.glob("request-*.json"):
            try:
                reports.append(read_json(path))
            except (OSError, ValueError) as error:
                logger.warning("Skipping unreadable fault report %s: %s", path, error)
    except OSError as error:
        logger.warning("Could not list fault reports in %s: %s", directory, error)
    return reports


class LastWordsExitError(RuntimeError):
    """This training run ends after an emergency save or failed dump."""


class LastWordsStopRequestedError(Exception):
    """The replica must leave ordinary training at a checkpoint boundary."""


class LastWordsWorkerController:
    def __init__(
        self,
        config: LastWordsConfig,
        *,
        folder: str,
        replica_id: int,
        num_replicas: int,
        rank: int,
        world_size: int,
        on_stop: Callable[[], None] | None = None,
    ):
        config.validate()
        self.config = config
        self.replica_id = replica_id
        self.num_replicas = num_replicas
        self.rank = rank
        self.world_size = world_size
        self._on_stop = on_stop
        self.root = Path(folder) / "last_words" / "control" / config.run_id
        if "://" in folder:
            raise ValueError("Last words requires a shared POSIX filesystem checkpoint folder")
        self._lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._status: dict[str, Any] = {"phase": "initializing", "replica_id": replica_id, "rank": rank}
        self._status["world_size"] = world_size
        self._status["timeout_seconds"] = config.timeout_seconds
        self._status["heartbeat_timeout_seconds"] = config.heartbeat_timeout_seconds
        backend = os.environ.get("LAST_WORDS_COORDINATION_BACKEND", config.coordination_backend)
        if backend != "service":
            raise ValueError("Only the service coordination backend is supported")
        from .client import WorkerSessionClient

        self._service_client = WorkerSessionClient(config.run_id, replica_id, rank)
        session = self._service_client.session
        if (
            session["num_replicas"] != num_replicas
            or session["world_size"] != world_size
            or session["heartbeat_seconds"] != config.heartbeat_timeout_seconds
            or session["timeout_seconds"] != config.timeout_seconds
        ):
            self._service_client.client.close()
            raise ValueError("Training topology/timeouts differ from service registration")
        self._closed = threading.Event()
        self._requested = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._monitor, name="last-words-control", daemon=True)
        self._thread.start()

    def update_status(self, phase: str, *, state: dict[str, Any] | None = None) -> bool:
        with self._lock:
            # A heartbeat can make an initializing rank ineligible while the
            # training thread is still completing its first step.
            if self._status["phase"] == "ineligible":
                return False
            self._status["phase"] = phase
            if phase == "ineligible":
                self._status.pop("published_round", None)
                self._status.pop("validated_round", None)
            if state is not None:
                self._status["state"] = dict(state)
        logger.debug("Last words replica=%d rank=%d phase=%s state=%s", self.replica_id, self.rank, phase, state)
        return True

    def request_terminal_stop(self, reason: str | None = None) -> None:
        callback = None
        with self._lock:
            if not self._requested.is_set():
                self._requested.set()
                callback = self._on_stop
        if callback is not None:
            callback()
        if reason is None:
            return
        # This fault-only file preserves the original exception for diagnosis.
        # Coordinator never reads it to authorize or select a writer.
        path = self.root / f"request-{self.replica_id}-{self.rank}.json"
        try:
            atomic_json(
                path,
                {
                    "reporting_replica": self.replica_id,
                    "reporting_rank": self.rank,
                    "reason": reason,
                    "time": time.time(),
                },
            )
        except OSError as error:
            logger.warning("Could not record stop reason %r in %s: %s", reason, path, error)

    @property
    def requested(self) -> bool:
        if isinstance(self._error, LastWordsExitError):
            raise self._error
        if self._error is not None:
            raise LastWordsExitError("Last-words Coordinator control channel failed") from self._error
        return self._requested.is_set()

    def _monitor(self):
        try:
            while not self._closed.wait(0.5):
                snapshot = self._report_status()
                if snapshot.get("replica_finished"):
                    return
                if snapshot["stop"]:
                    self.request_terminal_stop()
                if snapshot["result"] is not None and not snapshot["result"]["success"]:
                    raise LastWordsExitError(snapshot["result"]["error"])
        except BaseException as error:
            self._error = error
            self.request_terminal_stop()
            logger.exception("Last-words control monitor failed")

    def _report_status(self, *, deadline=None):
        # Read status after acquiring the send lock: an older heartbeat must
        # never overtake the final completion/failure report.
        heartbeat_deadline = time.monotonic() + self.config.heartbeat_timeout_seconds
        deadline = min(heartbeat_deadline, deadline) if deadline is not None else heartbeat_deadline
        acquired = self._report_lock.acquire(timeout=max(0, deadline - time.monotonic()))
        try:
            # The heartbeat may already have published this rank's validation
            # and cached the immutable decision while the training thread waited.
            snapshot = self._service_client.get_cached_snapshot()
            if snapshot is not None and snapshot["result"] is not None:
                return snapshot
            if not acquired:
                raise TimeoutError("Timed out waiting to send last-words status")
            with self._lock:
                requested = self._requested.is_set()
                # Only vote after all peers have drained. Serialize the final
                # local vote with stop requests, then await the service decision.
                if (
                    self._status["phase"] == "finishing"
                    and snapshot
                    and snapshot.get("confirm_finish")
                    and not requested
                ):
                    self._status["phase"] = "finish_vote"
                status = dict(self._status)
            return self._service_client.report(status, requested, deadline=deadline)
        finally:
            if acquired:
                self._report_lock.release()

    def wait_for_replica_freeze(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if not self.update_status("frozen", state=state):
            raise LastWordsExitError("Replica contains an unsafe rank")
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            # Read in this order so a newer monitor error cannot interrupt an older freeze snapshot.
            monitor_failed = self._error is not None
            snapshot = self._service_client.get_cached_snapshot()
            report = snapshot["reports"].get(str(self.replica_id)) if snapshot else None
            statuses = report["shards"] if report else []
            if len(statuses) == self.world_size and all(
                status and status.get("phase") == "frozen" for status in statuses
            ):
                if any(status.get("state") != state for status in statuses):
                    raise LastWordsExitError("Replica ranks stopped at different completed versions")
                # A peer may already be in a collective. Observe the result after preparation.
                return None
            saved = self._save_result(snapshot)
            if saved is not None:
                return saved
            if any(status and status.get("phase") == "ineligible" for status in statuses):
                raise LastWordsExitError("Replica contains an unsafe rank")
            if monitor_failed:
                _ = self.requested
            time.sleep(0.05)
        raise LastWordsExitError("Timed out collecting complete replica shards")

    def finish_training(self) -> None:
        self.update_status("finishing")
        # All shards remain available to freeze until the service accepts the
        # replica's normal finish. The monitor reports intent and any local stop.
        while True:
            snapshot = self._service_client.get_cached_snapshot()
            if snapshot and snapshot.get("replica_finished"):
                return
            if snapshot and snapshot["stop"]:
                raise LastWordsStopRequestedError
            _ = self.requested
            time.sleep(0.05)

    def fail_training(self) -> None:
        self.request_terminal_stop()
        self.update_status("ineligible")
        self._report_status()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)
        if not self._thread.is_alive():
            self._service_client.client.close()

    def await_save_authorization(self, state, signature):
        self._service_client.set_ready(state, signature)
        while True:
            snapshot = self._service_client.get_cached_snapshot()
            result = self._save_result(snapshot)
            if result is not None:
                return False, result
            grant = snapshot["grant"] if snapshot else None
            if (
                snapshot is not None
                and grant
                and snapshot["phase"] == "SAVING"
                and grant["replica_id"] == self.replica_id
            ):
                if grant["state"] != state or grant["signature"] != signature:
                    raise LastWordsExitError("Native grant does not match frozen state")
                self._save_grant = dict(
                    instance=snapshot["instance"],
                    run_id=self.config.run_id,
                    round=grant["round"],
                    replica_id=self.replica_id,
                )
                return True, self._save_grant
            _ = self.requested
            time.sleep(0.05)

    def _save_result(self, snapshot):
        if snapshot is None or snapshot["result"] is None:
            return None
        result = snapshot["result"]
        if not result["success"]:
            raise LastWordsExitError(result["error"])
        if result["replica_id"] is None:
            raise LastWordsExitError("Expected an emergency save result")
        return dict(coordinator_instance=snapshot["instance"], run_id=snapshot["run_id"], **result)

    def wait_save_result(self):
        while True:
            result = self._save_result(self._service_client.get_cached_snapshot())
            if result is not None:
                if (
                    result["coordinator_instance"] != self._save_grant["instance"]
                    or result["run_id"] != self._save_grant["run_id"]
                    or result["replica_id"] != self._save_grant["replica_id"]
                    or result["round"] != self._save_grant["round"]
                ):
                    raise LastWordsExitError("Save result does not match the writer grant")
                return result
            _ = self.requested
            time.sleep(0.05)

    def confirm_publication(self, deadline):
        with self._lock:
            self._status["published_round"] = self._save_grant["round"]
        snapshot = self._report_status(deadline=deadline)
        if self.rank != 0:
            return
        while time.monotonic() < deadline:
            self._save_result(snapshot)
            report = snapshot["reports"].get(str(self.replica_id))
            shards = report["shards"] if report else []
            if len(shards) == self.world_size and all(
                shard and shard["phase"] == "frozen" and shard.get("published_round") == self._save_grant["round"]
                for shard in shards
            ):
                return
            _ = self.requested
            time.sleep(0.05)
            snapshot = self._service_client.get_cached_snapshot()
        raise LastWordsExitError("Timed out waiting for writer publication reports")

    def confirm_validation(self, deadline):
        with self._lock:
            self._status["validated_round"] = self._save_grant["round"]
        self._report_status(deadline=deadline)
