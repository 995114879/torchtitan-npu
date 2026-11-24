# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Publish full emergency DCP candidates with explicitly bound replica cursors."""

from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torchtitan.components.checkpoint import (
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
    CheckpointManager,
)
from torchtitan.tools import filesystem

from ..checkpoint import TorchFTCheckpointManagerEx
from .checkpoint_schema import replica_cursor_schema, state_schema
from .checkpoint_validation import (
    contained,
    has_confirmed_save_result,
    periodic_model_preflight,
    read_periodic_binding,
    recovery_preflight,
    select_cursor,
    validate_tokenizer,
)
from .rng import RNG_SCHEMA
from .worker import (
    LastWordsExitError,
    LastWordsStopRequestedError,
    atomic_json,
    read_fault_reports,
    read_json,
    sync_checkpoint_tree,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..manager import ManagerEx
    from .rng import RNGCheckpointState
    from .worker import LastWordsWorkerController

logger = logging.getLogger(__name__)
_ReadResult = TypeVar("_ReadResult")


class LastWordsCheckpointManager(TorchFTCheckpointManagerEx):
    _agreed_load_step: int | None = None
    last_words_resume_signatures: tuple[str, ...] = ()
    tokenizer_sha256: str | None = None
    ft_manager: ManagerEx | None  # pyrefly: ignore [bad-override, bad-override-mutable-attribute]
    ft_states: dict[str, Any]
    last_words: LastWordsWorkerController | None
    rng_state: RNGCheckpointState | None

    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTCheckpointManagerEx.Config):
        pass

    def __init__(self, *args, **kwargs):
        self.last_words = None
        self.last_words_signature = ""
        self.last_words_state = None
        self.last_words_cursor_current = False
        self._durability_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="last-words-durability")
        self._last_words_pending_cursor = None
        self._last_words_pending_binding = None
        self._last_words_control_pg = None
        self._last_words_cursor_anchors = {}
        self._last_words_upload_error = None
        self.rng_state = None
        super().__init__(*args, **kwargs)

    def close(self):
        try:
            super().close()
        finally:
            executor = getattr(self, "_durability_executor", None)
            if executor is not None:
                executor.shutdown(wait=False)

    def _recovery_metadata(self):
        metadata: dict[str, Any] = {"tokenizer_sha256": self.tokenizer_sha256} if self.tokenizer_sha256 else {}
        if self.rng_state is not None:
            assert self.last_words is not None
            metadata.update(rng_schema=RNG_SCHEMA, rng_shards=self.last_words.world_size)
        return metadata

    @property
    def _rng_device_type(self):
        return self.rng_state.device.type if self.rng_state is not None else None

    def _cursor_record(self, state, *, cursor=None):
        return dict(
            state=dict(state),
            signature=self.last_words_signature,
            recovery_schema=state_schema(self.ft_states if cursor is None else cursor, kind="cursor"),
            **self._recovery_metadata(),
        )

    def _restore_rng(self):
        if self.rng_state is not None:
            self.rng_state.restore()

    def _prepare_save_state(self, state_dict):
        super()._prepare_save_state(state_dict)
        if self.last_words is None or OPTIMIZER not in state_dict:
            return

        def describe_model():
            # Use the state handed to DCP before staging begins. Per-replica
            # dataloader state has its own schema and immutable cursor marker.
            excluded = set(self.states) - {MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE}
            self._periodic_model_schema = state_schema(
                {key: value for key, value in state_dict.items() if key not in excluded}, kind="model"
            )

        self._replica_read(describe_model)

    def _save(self, curr_step, last_step=False):
        if self.last_words is None:
            return super()._save(curr_step, last_step)
        writer = not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager and self.ft_manager.participating_rank() == 0
        )
        if self.last_words_stop_requested():
            raise LastWordsStopRequestedError
        if self.last_words_state is None or self.last_words_state["step"] != curr_step:
            return False
        # Drain the previous cursor on every shard before another upload or
        # a periodic full-save collective can start.
        if not self._drain_periodic_cursor():
            raise LastWordsStopRequestedError
        if self.enable_ft_dataloader_checkpoints:
            self._ft_save(curr_step)
            # Non-periodic cursors stay asynchronous; periodic writers must
            # coordinate a newly failed cursor before saving the full state.
            if writer and self._should_save(curr_step, last_step) and not self._drain_periodic_cursor():
                raise LastWordsStopRequestedError
        # Preserve TorchFT's full-checkpoint writer selection; its per-replica
        # side channel has already completed and must not be launched twice.
        saved = False
        if writer:
            saved = CheckpointManager._save(self, curr_step, last_step)
        if saved and not (last_step and self.last_save_model_only) and dist.get_rank() == 0:
            assert self.last_words_state is not None
            path = Path(self._create_checkpoint_id(curr_step)) / "last_words_binding.json"
            binding = {
                **self._recovery_metadata(),
                "recovery_schema": self._periodic_model_schema,
                "signature": self.last_words_signature,
                "run_id": self.last_words.config.run_id,
                "state": dict(self.last_words_state),
                "anchors": {
                    str(replica): str(anchor.resolve().relative_to(Path(self.folder).resolve()))
                    for replica, anchor in self._last_words_cursor_anchors.items()
                },
            }
            if self.save_future is None:
                atomic_json(path, binding, durable=True)
            else:
                self._last_words_pending_binding = (path, binding)
        return saved

    def _drain_periodic_cursor(self):
        assert self.last_words is not None
        try:
            uploaded = self._drain_last_words_upload()
        except (Exception, CheckpointException) as error:
            self.last_words.request_terminal_stop(f"Unsafe background checkpoint failure: {error}")
            self.last_words.update_status("ineligible")
            raise
        if not uploaded:
            self.last_words.request_terminal_stop("Completed cursor storage upload failed")
        return uploaded

    def finish_pending_save(self) -> None:
        assert self.last_words is not None
        try:
            self.maybe_wait_for_saving()
        except (Exception, CheckpointException):
            if self._last_words_upload_error is None or not self._last_words_upload_error[0]:
                raise
            # The service coordinates this stop with shards already waiting to finish.
            self.last_words.request_terminal_stop("Completed cursor storage upload failed")

    def last_words_stop_requested(self):
        assert self.last_words is not None
        group = self._last_words_control_pg
        assert group is not None
        failure_reason = ""
        try:
            vote = int(self.last_words.requested)
        except LastWordsExitError as error:
            # A peer may already be voting. Values above 1 elect a rank to share its error.
            vote = 2 + dist.get_rank(group)
            failure_reason = str(error)
        result = torch.tensor(vote, dtype=torch.int32)
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=group)
        decision = int(result.item())
        if decision > 1:
            reason = [failure_reason]
            dist.broadcast_object_list(reason, group=group, group_src=decision - 2)
            raise LastWordsExitError(reason[0])
        return bool(decision)

    def _read_on_shards(self, read: Callable[[], _ReadResult]) -> list[_ReadResult]:
        if not dist.is_initialized():
            return [read()]
        try:
            local = (read(), "")
        except (Exception, CheckpointException) as caught:
            local = (None, f"{type(caught).__name__}: {caught}")
        replies: list[tuple[Any, str]] = [(None, "")] * dist.get_world_size(self.pg)
        dist.all_gather_object(replies, local, group=self.pg)
        for _, failure in replies:
            if failure:
                raise LastWordsExitError(failure)
        return [answer for answer, _ in replies]

    def _replica_read(self, read: Callable[[], _ReadResult], *, every_rank=False) -> _ReadResult:
        """Resolve a filesystem decision before any shard enters the next collective."""
        if every_rank:
            answers = self._read_on_shards(read)
            if any(answer != answers[0] for answer in answers[1:]):
                raise LastWordsExitError("Replica ranks observed different checkpoint contents")
            return answers[0]
        if not dist.is_initialized():
            return read()
        result, error = None, ""
        if dist.get_rank(self.pg) == 0:
            try:
                result = read()
            except (Exception, CheckpointException) as caught:
                error = f"{type(caught).__name__}: {caught}"
        response: list[Any] = [result, error]
        dist.broadcast_object_list(response, group=self.pg, group_src=0)
        result, error = response
        if error:
            raise LastWordsExitError(error)
        return cast("_ReadResult", result)

    def _recovery_paths(self, root: Path, pattern: str) -> list[Path]:
        # A directory listing can lag behind a peer's view on a shared filesystem.
        listings = self._read_on_shards(lambda: list(root.glob(pattern)))
        return sorted({path for listing in listings for path in listing})

    def _ft_save(self, step):
        if self.last_words is None:
            return super()._ft_save(step)
        self.maybe_wait_for_saving()
        assert self.last_words_state is not None
        state = dict(self.last_words_state)
        path = Path(self._create_checkpoint_id(step, folder=self._ft_folder())) / "last_words_cursor.json"
        record = None

        def prepare_cursor():
            nonlocal record
            if path.is_file():
                existing = read_json(path)
                validate_tokenizer(existing, self.tokenizer_sha256)
                if existing["state"] != state:
                    raise LastWordsExitError("Refusing to overwrite an immutable cursor version")
                return False
            record = self._cursor_record(state)
            return True

        if not self._replica_read(prepare_cursor):
            return
        super()._ft_save(step)
        assert self.save_future is not None
        self._last_words_pending_cursor = (path, state)
        if dist.get_rank() != 0:
            return
        upload = self.save_future

        def persist_cursor():
            upload.result()
            sync_checkpoint_tree(path.parent, root=Path(self.folder))
            atomic_json(path, record, durable=True)

        self.save_future = self._durability_executor.submit(persist_cursor)

    def _wait_for_saving(self):
        future = self.save_future
        try:
            super()._wait_for_saving()
            self._last_words_pending_cursor = None
            if self._last_words_pending_binding is not None:
                path, binding = self._last_words_pending_binding
                atomic_json(path, binding, durable=True)
                self._last_words_pending_binding = None
        except (Exception, CheckpointException) as error:
            if self.last_words is not None:
                # Only an ended cursor upload with storage errors can fall back.
                # A failed full save or communication/staging error is unsafe.
                failures = error.failures.values() if isinstance(error, CheckpointException) else [(error, None)]
                safe = (
                    self._last_words_pending_cursor is not None
                    and self._last_words_pending_binding is None
                    and future is not None
                    and future.done()
                    and self.staging_future is None
                    and all(isinstance(failure[0], OSError) for failure in failures)
                )
                self._last_words_upload_error = (safe, str(error))
                self._last_words_pending_cursor = None
                self._last_words_pending_binding = None
                if future is not None and future.done():
                    self.save_future = None
            raise

    def _ft_folder(self):
        folder = super()._ft_folder()
        if self.last_words is None:
            return folder
        branch = Path(folder) / self.last_words.config.run_id
        return str(branch)

    def _expected_cursor_schemas(self):
        assert self.last_words is not None
        control = self.last_words
        local = state_schema(self.ft_states, kind="cursor")
        # TorchFT assigns a contiguous batch-rank range to each replica.
        batch_world = getattr(
            self.ft_states[DATALOADER],
            "dp_world_size",
            control.num_replicas * control.world_size,
        )
        batch_degree, remainder = divmod(batch_world, control.num_replicas)
        if batch_degree < 1 or remainder:
            raise ValueError("Invalid dataloader world size for the replica topology")
        return {
            replica: replica_cursor_schema(local, replica, batch_degree=batch_degree, num_shards=control.world_size)
            for replica in range(control.num_replicas)
        }

    def _select_cursor(self, replica, state, *, expected=None):
        assert self.last_words is not None
        return select_cursor(
            self.folder,
            replica,
            state,
            signature=self.last_words_signature,
            run_id=self.last_words.config.run_id,
            anchors=self._last_words_cursor_anchors,
            tokenizer_sha256=self.tokenizer_sha256,
            expected=expected,
            expected_device_type=self._rng_device_type,
            num_shards=(self.last_words.world_size if self.rng_state is not None else None),
        )

    def _periodic_recovery(self, step):
        assert self.last_words is not None
        expected_cursors = {}

        def check_model():
            assert self.last_words is not None
            if step:
                expected_cursors.update(self._expected_cursor_schemas())
            return periodic_model_preflight(
                self.folder,
                step,
                signature=self.last_words_signature,
                compatible_signatures=self.last_words_resume_signatures,
                tokenizer_sha256=self.tokenizer_sha256,
                num_replicas=self.last_words.num_replicas,
                num_shards=(self.last_words.world_size if self.rng_state is not None else None),
                expected_model=(state_schema(self._full_states(), kind="model") if step else None),
            )

        # A model rejection must reach every rank before any cursor collective.
        signature, options = self._replica_read(check_model, every_rank=True)
        selected = {}
        if options:
            for replica in range(self.last_words.num_replicas):
                cursor = select_cursor(
                    self.folder,
                    replica,
                    signature=signature,
                    expected=expected_cursors[replica],
                    expected_device_type=self._rng_device_type,
                    replica_read=self._replica_read,
                    list_paths=self._recovery_paths,
                    **options,
                )
                if cursor is None:
                    raise LastWordsExitError("Periodic model checkpoint has no compatible safe replica cursor")
                selected[replica] = cursor[0]
        return signature, selected

    def _any_shard_matches(self, check):
        # Even one readable identity record can rule out a reset to step zero.
        try:
            matched = bool(check())
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            matched = False
        if not dist.is_initialized():
            return matched
        vote = torch.tensor(int(matched), dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MAX, group=self.pg)
        return bool(vote.item())

    def _read_periodic_binding(self, step):
        path = contained(self.folder, self._create_checkpoint_id(step))
        return read_periodic_binding(
            path, self.last_words_signature, self.last_words_resume_signatures, self.tokenizer_sha256
        )

    def _latest_periodic_candidate(self, step, *, after_step=-1):
        steps = [step]
        if step == -1:
            paths = self._recovery_paths(Path(self.folder), "step-*")
            suffixes = (path.name.removeprefix("step-") for path in paths)
            steps = sorted({int(suffix) for suffix in suffixes if suffix.isdecimal()}, reverse=True)
        recovery_error: str | None = None
        for candidate_step in steps:
            if candidate_step <= after_step or (candidate_step == 0 and recovery_error is not None):
                break
            try:
                signature, anchors = self._periodic_recovery(candidate_step)
            except (Exception, CheckpointException) as error:
                if step != -1:
                    raise LastWordsExitError(f"Requested periodic checkpoint is not recoverable: {error}") from error
                if (
                    candidate_step > 0
                    and after_step <= 0
                    and recovery_error is None
                    and self._any_shard_matches(partial(self._read_periodic_binding, candidate_step))
                ):
                    recovery_error = (
                        f"Periodic checkpoint {self._create_checkpoint_id(candidate_step)} is not recoverable: {error}"
                    )
                logger.warning(
                    "Ignoring unpublished or incompatible periodic checkpoint %s: %s",
                    self._create_checkpoint_id(candidate_step),
                    error,
                )
                continue
            return candidate_step, signature, anchors
        if recovery_error is not None:
            raise LastWordsExitError(recovery_error)
        return None

    def _import_cursor_anchors(self, anchors):
        assert self.last_words is not None
        imported = {}
        error = ""
        for replica, source in anchors.items():
            if source.resolve().is_relative_to(Path(self.folder).resolve()):
                imported[replica] = source
                continue
            target = (
                Path(self.folder)
                / f"ft-replicat-{self.last_words.replica_id}"
                / self.last_words.config.run_id
                / "resume-anchors"
                / f"replica-{replica}"
            )
            imported[replica] = target
            if self.last_words.rank == 0:
                try:
                    shutil.copytree(source, target)
                    sync_checkpoint_tree(target, root=Path(self.folder))
                except Exception as caught:
                    error = str(caught)
        if dist.is_initialized():
            failures: list[str | None] = [None] * dist.get_world_size(self.pg)
            dist.all_gather_object(failures, error, group=self.pg)
            error = next((failure for failure in failures if failure), "")
        if error:
            raise LastWordsExitError(f"Failed importing resume cursors: {error}")
        return imported

    def _find_load_step(self, folder=""):
        # The upstream initial-load path calls this again with step=-1. A newly
        # visible directory must not change the replica's agreed selection.
        if self._agreed_load_step is not None and not folder:
            return self._agreed_load_step
        return super()._find_load_step(folder)

    def _load_periodic(self, step, *, candidate=None, recovery_error=None):
        assert self.last_words is not None
        required = {MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE, DATALOADER, "torchft"}
        excluded = required.intersection(self.exclude_from_loading)
        if excluded:
            raise ValueError(f"Last words complete recovery cannot exclude training state: {sorted(excluded)}")
        signature = self.last_words_signature
        if candidate is None:
            candidate = self._latest_periodic_candidate(step)
        # step-0 loads only initial weights; it cannot recover committed progress.
        if candidate is not None and (recovery_error is None or candidate[0] > 0):
            step, signature, anchors = candidate
        elif recovery_error is not None:
            raise LastWordsExitError(recovery_error)
        elif self._replica_read(self._find_load_step) != -1:
            raise LastWordsExitError("No published periodic checkpoint has a complete recovery chain")
        else:
            anchors = {}

        def verify_source():
            if candidate is not None:
                checkpoint_id = self._create_checkpoint_id(step)
                flags = ()
            else:
                checkpoint_id = self.initial_load_path
                if not checkpoint_id and self.initial_load_in_hf and self.sd_adapter:
                    checkpoint_id = self.sd_adapter.hf_assets_path
                flags = (self.initial_load_in_hf, self.initial_load_in_hf_quantized, self.initial_load_model_only)
            if checkpoint_id and not filesystem.isdir(checkpoint_id):
                raise LastWordsExitError(f"Checkpoint is not visible: {checkpoint_id}")
            return checkpoint_id, flags

        self._replica_read(verify_source, every_rank=True)
        imported = self._import_cursor_anchors(anchors) if anchors else {}
        original_states = self.states
        self.states = {key: value for key, value in original_states.items() if key != DATALOADER}
        self._agreed_load_step = step
        try:
            loaded = CheckpointManager._load(self, step)
        finally:
            self.states = original_states
            self._agreed_load_step = None
        if not loaded:
            return False
        if self.states[TRAIN_STATE].state_dict()["step"] > 0:
            if not anchors:
                raise LastWordsExitError("Initial full checkpoint has no bound cursor recovery chain")
            self._last_words_cursor_anchors = imported
            dcp.load(
                self.ft_states,
                checkpoint_id=imported[self.last_words.replica_id],
                process_group=self.pg,
            )
            self._restore_rng()
            self.last_words_signature = signature
        return True

    def _check_resume_candidate(self, base, manifest, requested):
        assert self.last_words is not None
        recovery_preflight(
            base,
            manifest,
            self.last_words.num_replicas,
            expected_model=state_schema(self._full_states(), kind="model"),
            expected_cursors=self._expected_cursor_schemas(),
            expected_device_type=self._rng_device_type,
            tokenizer_sha256=self.tokenizer_sha256,
        )
        return has_confirmed_save_result(base, manifest, allow_missing_legacy_receipt=requested)

    def _read_resume_manifest(self, path):
        manifest = read_json(path)
        if manifest["signature"] not in (self.last_words_signature, *self.last_words_resume_signatures):
            raise ValueError("Last-words checkpoint topology or data configuration differs")
        validate_tokenizer(manifest, self.tokenizer_sha256)
        step = manifest["state"]["step"]
        return manifest, type(step) is int and step > 0

    @torch.no_grad()
    def _load(self, step=-1):
        control = self.last_words
        if control is None:
            return super()._load(step)
        root = Path(self.folder) / "last_words"
        requested = control.config.resume_path

        self._replica_read(
            lambda: (
                step,
                requested,
                self.last_words_signature,
                self.last_words_resume_signatures,
                self.tokenizer_sha256,
                self.rng_state is not None,
                tuple(sorted(self.exclude_from_loading)),
            ),
            every_rank=True,
        )

        if requested:
            path = Path(requested)
            paths = [self._replica_read(lambda: path / "manifest.json" if path.is_dir() else path)]
        else:
            paths = self._recovery_paths(root, "*/replica-*/manifest.json")
        candidates = []
        recovery_error = None
        for manifest_path in paths:
            base = manifest_path.parent
            matching_progress = False
            try:
                manifest, matching_progress = self._replica_read(partial(self._read_resume_manifest, manifest_path))
                saved_step = manifest["state"]["step"]
                if self.rng_state is not None and manifest.get("rng_schema") != RNG_SCHEMA:
                    raise ValueError("Last-words checkpoint has no bound RNG recovery chain")
                if step != -1 and saved_step != step:
                    continue
                confirmed = self._replica_read(
                    partial(self._check_resume_candidate, base, manifest, bool(requested)),
                    every_rank=True,
                )
            except (Exception, CheckpointException) as error:
                if requested:
                    raise ValueError(f"Requested last-words candidate is not recoverable: {error}") from error
                if step == -1 and (
                    matching_progress
                    or self._any_shard_matches(lambda path=manifest_path: self._read_resume_manifest(path)[1])
                ):
                    recovery_error = f"Last-words checkpoint {base} is not recoverable: {error}"
                logger.warning(
                    "Ignoring incomplete last-words candidate %s: %s",
                    manifest_path,
                    error,
                )
                continue
            if not confirmed:
                if not requested:
                    logger.warning("Checkpoint %s requires external confirmation", base)
                    if step == -1 and matching_progress:
                        recovery_error = (
                            f"Last-words checkpoint {base} has no confirmed save. Verify coordinator-binding.json "
                            "and service-result.json against the original service records. For v1 with a missing "
                            "receipt, select the verified candidate with --fault-tolerance.last-words.resume-path."
                        )
                    continue
                raise ValueError(
                    f"Requested last-words candidate has no matching coordinator binding and receipt: {base}"
                )
            candidates.append(
                (
                    manifest["state"]["last_words_num_updates"],
                    -manifest["replica_id"],
                    manifest_path,
                    manifest,
                )
            )
        if recovery_error is not None:
            candidates = [entry for entry in candidates if entry[3]["state"]["step"] > 0]
        if not candidates:
            if requested:
                raise ValueError("No matching last-words candidate")
            return self._load_periodic(step, recovery_error=recovery_error)
        _, _, manifest_path, manifest = max(candidates, key=lambda entry: (entry[0], entry[1], str(entry[2])))
        # A later periodic checkpoint takes precedence over an older emergency save.
        if not requested and step == -1:
            periodic = self._latest_periodic_candidate(-1, after_step=manifest["state"]["step"])
            if periodic is not None:
                return self._load_periodic(-1, candidate=periodic)
        base = manifest_path.parent
        anchors = {int(replica): base / cursor["path"] for replica, cursor in manifest["cursors"].items()}
        imported = self._import_cursor_anchors(anchors)
        cursor = manifest["cursors"][str(control.replica_id)]
        states = self._full_states()
        dcp.load(states, checkpoint_id=base / "model", process_group=self.pg)
        self.states[MODEL].load_state_dict(states)
        dcp.load(self.ft_states, checkpoint_id=base / cursor["path"], process_group=self.pg)
        self._restore_rng()
        self._last_words_cursor_anchors = imported
        # Keep subsequent uploads and saves bound to the recovered cursor lineage.
        self.last_words_signature = manifest["signature"]
        logger.warning(
            "Loaded last words model=%s cursor replica=%d state=%s",
            manifest["state"],
            control.replica_id,
            cursor["state"],
        )
        return True

    def _full_states(self):
        self.states[OPTIMIZER]._refresh_cached_state_dict()
        states = {key: self.states[key] for key in (MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE)}
        return self._flattened_model_states_sd(states)

    def _drain_last_words_upload(self):
        try:
            self.maybe_wait_for_saving()
        except (Exception, CheckpointException):
            if self._last_words_upload_error is None:
                raise
        error = self._last_words_upload_error
        failed = torch.tensor(0 if error is None else (1 if error[0] else 2), dtype=torch.int32)
        if dist.is_initialized():
            # All uploader futures have ended before probing their group.
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.pg)
        if failed.item() == 2:
            raise LastWordsExitError("Background checkpoint failed outside a completed cursor storage upload")
        if failed.item() == 1:
            logger.warning(
                "Using previously published replica cursors after failed upload: %s",
                error,
            )
        return failed.item() == 0

    def _candidate_path(self, replica_id):
        assert self.last_words is not None
        return Path(self.folder) / "last_words" / self.last_words.config.run_id / f"replica-{replica_id}"

    def save_last_words(self, state):
        control = self.last_words
        assert control is not None
        saved = control.wait_for_replica_freeze(state)
        if saved is not None:
            return self._candidate_path(saved["replica_id"])
        self.last_words_state = dict(state)
        # Complete staging/upload before freezing the cursor versions.
        self.maybe_wait_for_staging()
        if self._drain_last_words_upload() and self.last_words_cursor_current:
            self._ft_save(state["step"])
            self._drain_last_words_upload()
        # Pin the validated immutable cursors before advertising readiness.
        prepared_cursors = {}
        cursor_error = ""
        if control.rank == 0:
            try:
                for replica, expected in self._expected_cursor_schemas().items():
                    cursor = self._select_cursor(replica, state, expected=expected)
                    if cursor is None:
                        raise LastWordsExitError(f"A safe cursor is missing for replica {replica}")
                    prepared_cursors[replica] = cursor
            except Exception as error:
                cursor_error = str(error)
        if dist.is_initialized():
            preparation = [cursor_error]
            dist.broadcast_object_list(preparation, group=self.pg, group_src=0)
            cursor_error = preparation[0]
        if cursor_error:
            control.update_status("ineligible")
            raise LastWordsExitError(f"Emergency cursor preparation failed: {cursor_error}")
        grant_signature = f"{self.last_words_signature}:{self.tokenizer_sha256}"
        write, selected = control.await_save_authorization(state, grant_signature)
        if not write:
            return self._candidate_path(selected["replica_id"])
        if dist.is_initialized():
            # Every shard has independently checked the same immutable grant.
            dist.barrier(group=self._last_words_control_pg)
        target = self._candidate_path(control.replica_id)
        cursors = {}
        writer_error = None
        if control.rank == 0:
            operation = f"create writer directory {target}"
            try:
                target.mkdir(parents=True, exist_ok=True)
                binding = target / "coordinator-binding.json"
                operation = f"persist writer binding {binding}"
                atomic_json(binding, selected, durable=True)
                for replica, (source, cursor) in prepared_cursors.items():
                    relative = f"cursors/replica-{replica}"
                    operation = f"prepare cursor {target / relative}"
                    shutil.copytree(contained(self.folder, source), target / relative)
                    if read_json(target / relative / "last_words_cursor.json") != cursor:
                        raise LastWordsExitError(f"Prepared cursor changed for replica {replica}")
                    cursors[str(replica)] = {"path": relative, "state": cursor["state"]}
            except Exception as error:
                writer_error = f"Failed to {operation}: {type(error).__name__}: {error}"
        if dist.is_initialized():
            writer_status = [writer_error]
            dist.broadcast_object_list(writer_status, group=self.pg, group_src=0)
            writer_error = writer_status[0]
        if writer_error is not None:
            raise LastWordsExitError(writer_error)
        # self.pg is replica-local Gloo, never a TorchFT cross-replica group.
        states = self._full_states()
        dcp.save(states, checkpoint_id=target / "model", process_group=self.pg)
        if control.rank == 0:
            sync_checkpoint_tree(target, root=Path(self.folder))
            atomic_json(
                target / "manifest.json",
                {
                    **self._recovery_metadata(),
                    "format_version": 2,
                    "recovery_schema": state_schema(states, kind="model"),
                    "run_id": control.config.run_id,
                    "replica_id": control.replica_id,
                    "signature": self.last_words_signature,
                    "state": state,
                    "cursors": cursors,
                    "fault_reports": read_fault_reports(control.root),
                },
                durable=True,
            )
            atomic_json(
                target / "publication.json",
                {"success": True, "error": ""},
                durable=True,
            )
        deadline = time.monotonic() + control.config.timeout_seconds
        publication = target / "publication.json"
        while not publication.is_file():
            _ = control.requested
            if time.monotonic() >= deadline:
                raise LastWordsExitError("Timed out waiting for checkpoint publication")
            time.sleep(0.05)
        result = read_json(publication)
        if not result["success"]:
            raise LastWordsExitError(result["error"])
        control.confirm_publication(deadline)
        if control.rank == 0:
            recovery_preflight(
                target,
                read_json(target / "manifest.json"),
                control.num_replicas,
                expected_device_type=self._rng_device_type,
            )
            control.confirm_validation(deadline)
        receipt = control.wait_save_result()
        if control.rank == 0:
            atomic_json(target / "service-result.json", receipt, durable=True)
        if dist.is_initialized():
            # Rank 0 durably writes the service receipt before joining. A writer peer
            # exiting sooner could trigger torchrun to terminate rank 0 mid-write.
            dist.barrier(group=self._last_words_control_pg)
        logger.warning("Writer rank %d completed last-words save at %s", control.rank, target)
        return target
