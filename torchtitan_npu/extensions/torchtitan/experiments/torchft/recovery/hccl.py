# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Coordinate HCCL device recovery with work and optimizer update boundaries."""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, cast

import torch
from torch.distributed import ProcessGroup

from .work import HcclWorkTracker, TrackedHcclWork

logger = logging.getLogger(__name__)


class HcclRecoveryError(RuntimeError):
    """The current HCCL attempt must end before device recovery can complete."""


class HcclRecovery:
    def __init__(
        self,
        timeout: timedelta,
        *,
        recovery_timeout: timedelta,
        warmup_timeout: timedelta | None,
        get_process_group: Callable[[], ProcessGroup | None],
        clear_process_group: Callable[[ProcessGroup], None],
    ):
        if recovery_timeout.total_seconds() <= 0:
            raise ValueError("recovery_timeout must be positive")
        if warmup_timeout is not None and warmup_timeout.total_seconds() <= 0:
            raise ValueError("warmup_timeout must be positive")
        self._timeout = timeout
        self._warmup_timeout = max(timeout, warmup_timeout or timeout)
        self._warmup_pending = False
        self.recovery_timeout = recovery_timeout
        self._get_process_group = get_process_group
        self._clear_process_group = clear_process_group
        self.work_tracker = HcclWorkTracker(self._schedule_recovery)
        self._recovery_requested = threading.Event()
        self._terminal_lock = threading.Lock()
        self._terminal_check: threading.Event | None = None
        self._stop_observed = threading.Event()
        self._device_ready = threading.Event()
        self._device_ready.set()
        self._recovery_error: BaseException | None = None
        self._errored: Exception | None = None
        self._device_index: int | None = None
        self.num_device_recoveries = 0
        self.optimizer_update_completed = False
        self.last_words_request = None
        self.last_words_stop_requested: Callable[[], bool] | None = None
        self._last_words_interrupt_work: tuple[int, int] | None = None

    @property
    def watchdog_timeout(self):
        # Leave time for Python recovery before the fatal HCCL watchdog.
        return self._warmup_timeout + self.recovery_timeout + timedelta(seconds=30)

    def errored(self):
        return self._errored

    def prepare_group(self, device_index):
        self._raise_if_recovering()
        self._device_index = device_index

    def group_created(self):
        self.work_tracker.clear()
        self._warmup_pending = True
        self._errored = None

    @property
    def _collective_timeout(self):
        return self._warmup_timeout if self._warmup_pending else self._timeout

    @contextmanager
    def track_collective(self):
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
            if self.last_words_stop_requested is not None and self.last_words_stop_requested():
                raise HcclRecoveryError("Last words stop requested; refuse another HCCL collective")
            # Register before releasing the boundary lock. A stop notification
            # racing with this launch then sees an owned token even if native
            # HCCL has not returned a Work yet.
            token = self.work_tracker.track(self._collective_timeout)
        try:
            yield
        except RuntimeError:
            # A native launch interrupted before returning Work may report a
            # backend-specific error. Preserve the existing recovery handshake
            # instead of letting cleanup touch the stopped device early.
            self._raise_if_recovering()
            raise
        finally:
            self.work_tracker.complete(token)

    def request_last_words_stop(self):
        # A local exception can race the heartbeat callback. Both observe the
        # same ownership decision before the training thread touches the device.
        with self._terminal_lock:
            if self._terminal_check is None:
                self._terminal_check = threading.Event()
                self._schedule_recovery(None, self.work_tracker.generation, self._terminal_check)
            return self._terminal_check

    def wrap_work(self, pg, work, opts):
        timeout = self._collective_timeout
        requested = getattr(opts, "timeout", None)
        if isinstance(requested, timedelta) and timedelta(0) < requested < timeout:
            timeout = requested
        return TrackedHcclWork(pg, work, timeout, tracker=self.work_tracker)

    def wait_for_completion(self, work):
        """Wait for one HCCL work item without synchronizing the compute stream."""
        tracked_work = getattr(work, "_work", None)
        if not isinstance(tracked_work, TrackedHcclWork):
            if not work.wait():
                raise RuntimeError("HCCL work did not complete")
            return
        while True:
            with self.work_tracker.boundary_lock:
                self._raise_if_recovering()
                try:
                    if tracked_work.is_completed():
                        break
                    if self.last_words_stop_requested is not None and self.last_words_stop_requested():
                        generation = self.work_tracker.generation
                        interruption = (tracked_work._token, generation)
                        if self._last_words_interrupt_work != interruption:
                            # A confirmed save request must not wait for the
                            # longer communication timer. Retain exact work
                            # ownership and the existing optimizer/restart gate.
                            self._last_words_interrupt_work = interruption
                            self._schedule_recovery(*interruption)
                except RuntimeError:
                    self._raise_if_recovering()
                    raise
            time.sleep(0.001)
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
        # WorkHCCL.wait() releases allocator-safety state. Calling it only
        # after this work's end event completed avoids waiting on unrelated
        # compute while retaining the required record-stream cleanup.
        if not work.wait():
            raise RuntimeError("HCCL work did not complete")

    @contextmanager
    def release_group(self, *, errored):
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
            if errored:
                self._errored = RuntimeError("aborted")
            pg = self._get_process_group()
            if pg is None:
                yield None
                return
            if self.work_tracker.has_active_work():
                if errored:
                    self._schedule_recovery(None, self.work_tracker.generation)
                    yield None
                    return
                raise RuntimeError("Cannot release an HCCL group with unfinished work")
            try:
                yield pg
            finally:
                self._clear_process_group(pg)
                self.work_tracker.clear()

    def _raise_if_recovering(self):
        if self._recovery_error is not None:
            raise RuntimeError("TorchFT NPU device recovery failed") from self._recovery_error
        if self._recovery_requested.is_set():
            raise HcclRecoveryError("The interrupted HCCL attempt must finish before retrying")

    def _schedule_recovery(self, token, generation, checked=None):
        # Never block the timer callback on communicator destruction.
        thread = threading.Thread(target=self._recover_device, args=(token, generation, checked), daemon=True)
        thread.start()

    def _recover_device(self, token, generation, checked=None):
        stop_reason = None
        owner = False
        try:
            with self.work_tracker.boundary_lock:
                pg = self._get_process_group()
                if generation != self.work_tracker.generation or pg is None:
                    return
                if self._recovery_requested.is_set() or not self.work_tracker.has_active_work(token):
                    return
                owner = True
                assert self._device_index is not None
                self._device_ready.clear()
                self._stop_observed.clear()
                self._recovery_requested.set()
                if checked is not None:
                    checked.set()
                self._errored = HcclRecoveryError("HCCL communication timed out during an uncommitted attempt")
                if self.last_words_request is not None:
                    stop_reason = str(self._errored)
                    # Set the stop request before device recovery or diagnostic
                    # I/O can block so the heartbeat can report it.
                    self.last_words_request()
                backend = cast("Any", pg._get_backend(torch.device("npu")))
                import torch_npu

                logger.warning("Stopping NPU %d after unfinished TorchFT HCCL communication", self._device_index)
                result = torch_npu.npu.stop_device(self._device_index)
                if result not in (None, 0):
                    raise RuntimeError(f"NPU stop_device failed: {result}")
            backend.abort_hccl_comm("reinit")
            if not self._stop_observed.wait(self.recovery_timeout.total_seconds()):
                raise TimeoutError("Training thread did not observe the interrupted NPU attempt")
            from torch.distributed.distributed_c10d import _cleanup_process_group_global_state, _pg_map

            backend.clear_workmeta_list()
            _cleanup_process_group_global_state(pg)
            self._clear_process_group(pg)
            self.work_tracker.clear()
            torch_npu.npu.restart_device(
                self._device_index, rebuild_all_resources=True, disable_tensor_unsafe_check=True
            )
            # stop_device suspends every communicator on this device. The pinned
            # restart_device restores streams/watchdogs but does not resume HCCL.
            npu_device = torch.device("npu")
            for remaining_pg in tuple(_pg_map):
                if npu_device in remaining_pg._device_types:
                    cast("Any", remaining_pg._get_backend(npu_device)).resume_hccl_comm(self._device_index)
            self.num_device_recoveries += 1
            logger.warning("Restarted NPU %d; discard the interrupted training attempt", self._device_index)
        except BaseException as error:
            self._recovery_error = error
            logger.exception("TorchFT HCCL device recovery failed")
        finally:
            if owner:
                self._device_ready.set()
            if checked is not None:
                checked.set()
        if stop_reason is not None and self.last_words_request is not None:
            self.last_words_request(stop_reason)

    def recover(self, error: Exception) -> bool:
        """Acknowledge the expected stop, then wait before accessing device state."""
        if not self._recovery_requested.is_set():
            return False
        if not isinstance(error, HcclRecoveryError) and "FORCE STOP" not in str(error).upper():
            return False
        self._wait_for_device(error)
        return True

    def prepare_for_terminal_save(self):
        """Observe the stop and wait within one budget, even if a native call stalls."""
        deadline = time.monotonic() + self.recovery_timeout.total_seconds()
        checked = self.request_last_words_stop()
        if not self._recovery_requested.is_set() and not checked.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Timed out waiting for the terminal HCCL work check")
        if self._recovery_requested.is_set():
            self._wait_for_device(timeout=max(0.0, deadline - time.monotonic()))
        elif self._recovery_error is not None:
            raise RuntimeError("TorchFT NPU device recovery failed") from self._recovery_error

    def _wait_for_device(self, error=None, *, timeout=None):
        self._stop_observed.set()
        if not self._device_ready.wait(self.recovery_timeout.total_seconds() if timeout is None else timeout):
            raise TimeoutError("Timed out waiting for TorchFT NPU device recovery") from error
        if self._recovery_error is not None:
            raise RuntimeError("TorchFT NPU device recovery failed") from self._recovery_error
        self._recovery_requested.clear()
        self._errored = None

    def begin_attempt(self):
        self.optimizer_update_completed = False

    def apply_optimizer(self, update):
        """Exclude late timeouts until the accepted local update has completed."""
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
            # Manager.should_commit has already synchronized all gradient work.
            self.work_tracker.clear()
            update()
            import torch_npu

            torch_npu.npu.current_stream().synchronize()
            self.optimizer_update_completed = True
            self._warmup_pending = False
