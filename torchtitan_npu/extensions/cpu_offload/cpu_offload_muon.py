# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap CPU staging with DistributedMuon compute without forking FlexShard."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
    _BucketWork,
    _BufferSlot,
    _CommunicationContext,
    _execute_packed_all_to_all,
    _finalize_redistributed,
    _include_compute_scratch_requirement,
    _LocalBucketPlan,
    _PipelineSlot,
    _prepare_redistributed,
    _RedistributionBucketPlan,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    DistMuon,
    _apply_muon_update,
    _normalize_param_groups,
    _prepare_muon_input,
    _storage_layout_signature,
)

from torchtitan_npu.extensions.cpu_offload.runtime import GradientClipChannel, stage_gradient

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from torch import Tensor
    from torchtitan.distributed.flex_shard.dist_muon import _ParameterComputeLayout
    from torchtitan.distributed.flex_shard.optimizer_reshard import (
        BucketConfig,
        ComputeLayout,
    )

    from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging, TransferHandle


@dataclass(slots=True)
class _LocalEvents:
    input_ready: Any
    compute_done: Any
    done: Any


@dataclass(slots=True)
class _LocalSlot:
    buffers: _BufferSlot = field(default_factory=_BufferSlot)
    events: _LocalEvents | None = None


@dataclass(slots=True)
class _CommunicationEvents:
    storage_ready: Any
    reverse_ready: Any


@dataclass(slots=True)
class _WeightPrefetchBucket:
    """One optimizer bucket's batched weight H2D lifecycle."""

    layouts: tuple[Any, ...]
    ready_event: Any | None = None


@dataclass(slots=True)
class _LocalWork[ItemT]:
    item: ItemT
    buffer: Tensor
    events: _LocalEvents


def _run_rolling_pipeline[SpecT, WorkT](
    specs: Sequence[SpecT],
    *,
    enqueue: Callable[[SpecT, int], WorkT],
    compute: Callable[[WorkT], None],
    finalize: Callable[[WorkT], None],
    release: Callable[[WorkT], None],
) -> None:
    """Keep one item prefetched and reuse each of two slots after finalization."""
    if not specs:
        return

    previous: WorkT | None = None
    current = enqueue(specs[0], 0)
    prefetched = enqueue(specs[1], 1) if len(specs) > 1 else None
    next_index = 2
    while current is not None:
        compute(current)
        if previous is not None:
            finalize(previous)
            release(previous)
            if next_index < len(specs):
                prefetched = enqueue(specs[next_index], next_index % 2)
                next_index += 1
        previous = current
        current = prefetched
        prefetched = None

    assert previous is not None
    finalize(previous)
    release(previous)


class LocalPrefetchRuntime[ItemT](_BucketedRedistributionRuntime[ItemT]):
    """Add a two-slot CPU-staging lane beside FlexShard's HCCL lane."""

    pipeline_local = True

    def __init__(
        self,
        device: torch.device,
        *,
        local_stream: torch.Stream | None = None,
    ) -> None:
        super().__init__(device)
        self._local_stream = local_stream
        # Input preparation must not queue behind local item finalization,
        # which waits for the current bucket's compute. This lane performs
        # device-side preparation only; host transfers use staging.stream.
        self._prep_stream = torch_npu.npu.Stream(device=device)
        self._local_slots: tuple[_LocalSlot, ...] = ()
        self._communication_events: dict[int, _CommunicationEvents] = {}
        self._event_context_id: int | None = None
        # The public enqueue helpers retain their upstream immediate-prefetch
        # behavior.  The bucketed ``run`` path enables deferral so that the
        # next bucket's gradient input is submitted before the current
        # bucket's weight H2D on the single staging stream.
        self._defer_parameter_prefetch_enabled = False

    @classmethod
    def from_runtime(
        cls,
        runtime: _BucketedRedistributionRuntime[ItemT],
        *,
        local_stream: torch.Stream | None = None,
    ) -> LocalPrefetchRuntime[ItemT]:
        """Reuse the upstream communication context and discard duplicate scratch."""
        replacement = cls(runtime._device, local_stream=local_stream)
        replacement._context = runtime._context
        runtime._context = None
        runtime._local_slot.buffers.clear()
        if replacement._context is not None:
            for slot in replacement._context.slots:
                slot.buffers.buffers.clear()
        return replacement

    def _ensure_events(
        self,
        context: _CommunicationContext,
        *,
        compute_stream: torch.Stream,
    ) -> None:
        if self._event_context_id == id(context):
            return

        handle = context.device_handle
        local_stream = self._local_stream or context.transfer_stream
        self._local_slots = tuple(_LocalSlot() for _ in context.slots)
        for slot in self._local_slots:
            slot.events = _LocalEvents(
                handle.Event(),
                handle.Event(),
                handle.Event(),
            )
            slot.events.input_ready.record(local_stream)
            slot.events.compute_done.record(compute_stream)
            slot.events.done.record(local_stream)

        self._communication_events = {
            id(slot): _CommunicationEvents(handle.Event(), handle.Event()) for slot in context.slots
        }
        self._event_context_id = id(context)

    @staticmethod
    def _local_items(
        plan: _LocalBucketPlan[ItemT] | _RedistributionBucketPlan[ItemT],
    ) -> Sequence[ItemT]:
        return plan.items if isinstance(plan, _LocalBucketPlan) else plan.unredistributed_items

    @staticmethod
    def _prefetch_owner(callback: Any) -> Any | None:
        owner = getattr(callback, "__self__", None)
        return owner if hasattr(owner, "_prefetch_parameters") else None

    def _defer_parameter_prefetch(self, callback: Any, items: Sequence[ItemT]) -> bool:
        """Defer weight H2D when the optimizer provides the staging queue."""
        owner = self._prefetch_owner(callback)
        if owner is None or not hasattr(owner, "_defer_parameter_prefetch"):
            return False
        owner._defer_parameter_prefetch(items)
        return True

    def _flush_deferred_parameter_prefetch(self, callback: Any) -> None:
        owner = self._prefetch_owner(callback)
        if owner is not None and hasattr(owner, "_flush_deferred_parameter_prefetch"):
            owner._flush_deferred_parameter_prefetch()

    def _should_defer_parameter_prefetch(self, callback: Any) -> bool:
        if not getattr(self, "_defer_parameter_prefetch_enabled", False):
            return False
        owner = self._prefetch_owner(callback)
        if owner is None or not hasattr(owner, "_defer_parameter_prefetch"):
            return False
        policy = getattr(owner, "_should_defer_parameter_prefetch", None)
        return policy is None or bool(policy())

    def reserve_buffers(
        self,
        plans: Sequence[_LocalBucketPlan[ItemT] | _RedistributionBucketPlan[ItemT]],
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
    ) -> None:
        if not plans:
            return

        remote_plans = tuple(
            replace(plan, unredistributed_items=()) for plan in plans if isinstance(plan, _RedistributionBucketPlan)
        )
        super().reserve_buffers(remote_plans, local_tensor_spec=local_tensor_spec)

        context = self._context
        created_context = context is None
        if context is None:
            context = _CommunicationContext.create(self._device)
            self._context = context

        handle = context.device_handle
        compute_stream = handle.current_stream(self._device)
        local_stream = self._local_stream or context.transfer_stream
        if created_context:
            for slot in context.slots:
                slot.compute_input_ready.record(context.transfer_stream)
                slot.compute_done.record(compute_stream)
                slot.done.record(local_stream)
        self._ensure_events(context, compute_stream=compute_stream)

        requirements = tuple({} for _ in self._local_slots)
        for plan in plans:
            for item_index, item in enumerate(self._local_items(plan)):
                _include_compute_scratch_requirement(
                    requirements[item_index % len(requirements)],
                    item,
                    local_tensor_spec,
                )

        for slot, required in zip(self._local_slots, requirements, strict=True):
            slot.buffers.reserve(
                required,
                device_handle=handle,
                compute_stream=compute_stream,
                transfer_stream=local_stream,
            )
            slot.buffers.record_compute_stream(local_stream)
        for slot in context.slots:
            slot.buffers.record_compute_stream(self._prep_stream)

    def _enqueue_local(
        self,
        item: ItemT,
        slot_index: int,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
        prepare: Callable[[ItemT, Tensor], None],
    ) -> _LocalWork[ItemT]:
        handle = context.device_handle
        local_stream = self._local_stream or context.transfer_stream
        slot = self._local_slots[slot_index]
        assert slot.events is not None
        with handle.stream(local_stream):
            local_stream.wait_event(slot.events.done)
            shape, dtype, device = local_tensor_spec(item)
            buffer = slot.buffers.compute_buffer(shape, dtype=dtype, device=device)
            prepare(item, buffer)
            slot.events.input_ready.record(local_stream)
            prefetch = getattr(getattr(prepare, "__self__", None), "_prefetch_parameters", None)
            if prefetch is not None:
                if self._should_defer_parameter_prefetch(prepare):
                    self._flush_deferred_parameter_prefetch(prepare)
                    self._defer_parameter_prefetch(prepare, (item,))
                else:
                    prefetch((item,), stream=local_stream)
        return _LocalWork(item, buffer, slot.events)

    @staticmethod
    def _compute_local(
        work: _LocalWork[ItemT],
        caller: torch.Stream,
        handle: Any,
        *,
        compute: Callable[[ItemT, Tensor], None],
    ) -> None:
        with handle.stream(caller):
            caller.wait_event(work.events.input_ready)
            compute(work.item, work.buffer)
            work.events.compute_done.record(caller)

    def _finalize_local(
        self,
        work: _LocalWork[ItemT],
        context: _CommunicationContext,
        *,
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        local_stream = self._local_stream or context.transfer_stream
        with handle.stream(local_stream):
            local_stream.wait_event(work.events.compute_done)
            self._flush_deferred_parameter_prefetch(finalize)
            finalize(work.item, work.buffer)
            work.events.done.record(local_stream)

    @staticmethod
    def _release_local(work: _LocalWork[ItemT], caller: torch.Stream) -> None:
        caller.wait_event(work.events.done)

    def _run_local_items(
        self,
        items: Sequence[ItemT],
        context: _CommunicationContext,
        caller: torch.Stream,
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
        prepare: Callable[[ItemT, Tensor], None],
        compute: Callable[[ItemT, Tensor], None],
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle

        def enqueue(item: ItemT, slot_index: int) -> _LocalWork[ItemT]:
            return self._enqueue_local(
                item,
                slot_index,
                context,
                local_tensor_spec=local_tensor_spec,
                prepare=prepare,
            )

        _run_rolling_pipeline(
            items,
            enqueue=enqueue,
            compute=lambda work: self._compute_local(
                work,
                caller,
                handle,
                compute=compute,
            ),
            finalize=lambda work: self._finalize_local(
                work,
                context,
                finalize=finalize,
            ),
            release=lambda work: self._release_local(work, caller),
        )
        flush = getattr(getattr(finalize, "__self__", None), "_flush_parameter_d2h", None)
        if flush is not None:
            flush()

    def _compute_without_redistribution(
        self,
        items: Sequence[ItemT],
        slot: _BufferSlot,
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
        prepare: Callable[[ItemT, Tensor], None],
        compute: Callable[[ItemT, Tensor], None],
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        del slot
        context = self._context
        if context is None:
            raise RuntimeError("local-prefetch runtime must be reserved before run")
        caller = context.device_handle.current_stream(self._device)
        self._run_local_items(
            items,
            context,
            caller,
            local_tensor_spec=local_tensor_spec,
            prepare=prepare,
            compute=compute,
            finalize=finalize,
        )

    def _compute_bucket(
        self,
        work: _BucketWork[ItemT],
        slot: _PipelineSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
        prepare: Callable[[ItemT, Tensor], None],
        compute: Callable[[ItemT, Tensor], None],
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        self._run_local_items(
            work.plan.unredistributed_items,
            context,
            caller_stream,
            local_tensor_spec=local_tensor_spec,
            prepare=prepare,
            compute=compute,
            finalize=finalize,
        )
        remote_work = _BucketWork(
            replace(work.plan, unredistributed_items=()),
            work.slot,
            work.storage_buffer,
            work.compute_fragment_buffer,
        )
        super()._compute_bucket(
            remote_work,
            slot,
            caller_stream,
            context,
            local_tensor_spec=local_tensor_spec,
            prepare=prepare,
            compute=compute,
            finalize=finalize,
        )

    def _communication_events_for(
        self,
        slot: _PipelineSlot,
    ) -> _CommunicationEvents:
        try:
            return self._communication_events[id(slot)]
        except KeyError as exc:
            raise RuntimeError("unknown FlexShard pipeline slot") from exc

    def _enqueue_storage_to_compute(
        self,
        plan: _RedistributionBucketPlan[ItemT],
        slot: _PipelineSlot,
        context: _CommunicationContext,
        *,
        prepare: Callable[[ItemT, Tensor], None],
    ) -> _BucketWork[ItemT]:
        handle = context.device_handle
        storage_stream = self._prep_stream
        communication_stream = context.transfer_stream
        events = self._communication_events_for(slot)
        with handle.stream(storage_stream):
            # The previous occupant finalized on the local stream. Keep
            # that reuse fence, without joining the current bucket's tail.
            storage_stream.wait_event(slot.done)
            storage_buffer, compute_buffer = slot.buffers.communication_buffers(plan)
            work = _BucketWork(plan, slot, storage_buffer, compute_buffer)
            prefetch_momenta = getattr(getattr(prepare, "__self__", None), "_prefetch_bucket_momenta", None)
            if prefetch_momenta is not None:
                prefetch_momenta(plan.redistributed_items, stream=storage_stream)
            _prepare_redistributed(
                plan,
                slot.buffers,
                storage_buffer,
                prepare=prepare,
            )
            events.storage_ready.record(storage_stream)
        with handle.stream(communication_stream):
            communication_stream.wait_event(events.storage_ready)
            _execute_packed_all_to_all(
                plan.storage_to_compute_schedule,
                output=compute_buffer,
                input=storage_buffer,
            )
            slot.compute_input_ready.record(communication_stream)
        # Weight copies are deliberately delayed until this bucket's input
        # H2D and all-to-all have been submitted.  Flush the previous bucket's
        # deferred weights first, then defer this bucket's weights so the next
        # bucket's gradient H2D gets priority on the single staging stream.
        prefetch = getattr(getattr(prepare, "__self__", None), "_prefetch_parameters", None)
        if prefetch is not None:
            if self._should_defer_parameter_prefetch(prepare):
                self._flush_deferred_parameter_prefetch(prepare)
                self._defer_parameter_prefetch(prepare, plan.redistributed_items)
            else:
                prefetch(plan.redistributed_items, stream=storage_stream)
        return work

    def _enqueue_compute_to_storage(
        self,
        work: _BucketWork[ItemT],
        context: _CommunicationContext,
        *,
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        storage_stream = self._local_stream or context.transfer_stream
        communication_stream = context.transfer_stream
        events = self._communication_events_for(work.slot)
        with handle.stream(communication_stream):
            communication_stream.wait_event(work.slot.compute_done)
            _execute_packed_all_to_all(
                work.plan.compute_to_storage_schedule,
                output=work.storage_buffer,
                input=work.compute_fragment_buffer,
            )
            events.reverse_ready.record(communication_stream)
        with handle.stream(storage_stream):
            storage_stream.wait_event(events.reverse_ready)
            self._flush_deferred_parameter_prefetch(finalize)
            _finalize_redistributed(
                work,
                work.slot.buffers,
                finalize=finalize,
            )
            flush = getattr(getattr(finalize, "__self__", None), "_flush_parameter_d2h", None)
            if flush is not None:
                flush()
            work.slot.done.record(storage_stream)

    def run(
        self,
        plans: Sequence[_LocalBucketPlan[ItemT] | _RedistributionBucketPlan[ItemT]],
        *,
        local_tensor_spec: Callable[[ItemT], tuple[torch.Size, torch.dtype, torch.device]],
        prepare: Callable[[ItemT, Tensor], None],
        compute: Callable[[ItemT, Tensor], None],
        finalize: Callable[[ItemT, Tensor], None],
    ) -> None:
        context = self._context
        if plans and context is None:
            raise RuntimeError("local-prefetch runtime must be reserved before run")
        if context is None:
            return

        handle = context.device_handle
        caller = handle.current_stream(self._device)
        local_stream = self._local_stream or context.transfer_stream
        local_stream.wait_stream(caller)
        self._prep_stream.wait_stream(caller)
        self._defer_parameter_prefetch_enabled = True
        try:
            super().run(
                plans,
                local_tensor_spec=local_tensor_spec,
                prepare=prepare,
                compute=compute,
                finalize=finalize,
            )
        except BaseException:
            local_stream.wait_stream(caller)
            self._prep_stream.wait_stream(caller)
            caller.wait_stream(local_stream)
            caller.wait_stream(self._prep_stream)
            raise
        finally:
            self._defer_parameter_prefetch_enabled = False


class CpuOffloadDistributedMuon(DistMuon):
    """Use upstream Muon planning while keeping canonical tensors on CPU."""

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        staging: CpuStaging,
        clip_channel: GradientClipChannel | None = None,
        offload_states: bool = True,
        **kwargs: Any,
    ) -> None:
        self._staging = staging
        self._clip_channel = clip_channel
        self._offload_states = offload_states
        self._compute_device = staging.device  # pyrefly: ignore [read-only]
        self._commit_stream = torch_npu.npu.Stream(device=self._compute_device)
        self._scratch_by_slot: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
        self._scratch_releases: dict[int, TransferHandle] = {}
        self._retired_scratch: list[torch.Tensor] = []
        # Prepared parameter/momentum work is kept by logical parameter until
        # finalize.  This lets the next bucket H2D its weights before the
        # current bucket finishes Muon, while one bucket-level foreach D2H
        # commits all completed updates together.
        self._prefetched_parameters: dict[int, tuple[torch.Tensor, _WeightPrefetchBucket]] = {}
        self._prefetched_momenta: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_parameter_prefetch: list[_WeightPrefetchBucket] = []
        self._pending_d2h: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
        super().__init__(params, **kwargs)

    def _validate_parameter_storage(self) -> torch.device:
        local_devices = set()
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    raise TypeError("CPU-offloaded Muon requires DTensor parameters")
                local_devices.add(param.to_local().device)
        if len(local_devices) != 1:
            raise ValueError("CPU-offloaded Muon requires one storage device per process")
        storage_device = local_devices.pop()
        if storage_device.type != "cpu":
            raise ValueError("CPU-offloaded Muon requires CPU parameter storage")
        return self._compute_device

    def _momentum(self, compute_layout: _ParameterComputeLayout, grad: DTensor) -> DTensor:
        # NPU-resident momentum skips the per-step H2D/D2H round trip in
        # ``_prepare_local``; CPU-canonical momentum keeps upstream placement.
        state = self.state[compute_layout.param]
        if "momentum_buffer" not in state:
            momentum = torch.zeros_like(grad, memory_format=torch.preserve_format)
            if not self._offload_states:
                momentum = momentum.to(self._compute_device)
            state["momentum_buffer"] = momentum
        return state["momentum_buffer"]

    def _validate_momentum(self, compute_layout: _ParameterComputeLayout) -> None:
        """Validate restored momentum with the storage device excepted.

        Upstream derives the expected layout signature from the parameter,
        whose local storage is the CPU under CPU offload; the NPU-resident
        plane legitimately differs in ``local.device`` only. Every other
        signature component (shape, stride, placements, dtype, contiguity)
        must still match, and the momentum must stay a DTensor.
        """
        if self._offload_states:
            super()._validate_momentum(compute_layout)
            return
        momentum = self.state.get(compute_layout.param, {}).get("momentum_buffer")
        if momentum is None:
            return
        if not isinstance(momentum, DTensor):
            raise RuntimeError(f"momentum storage layout changed for {compute_layout.fqn!r}")
        # ``_storage_layout_signature`` tuples end with
        # (..., local.dtype, local.device, local.is_contiguous()); the device
        # sits at index 6.
        expected = list(compute_layout.storage_layout_signature)
        expected[6] = self._compute_device
        if _storage_layout_signature(momentum) != tuple(expected):
            raise RuntimeError(f"momentum storage layout changed for {compute_layout.fqn!r}")

    def _initialize_plan(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
    ) -> None:
        super()._initialize_plan(compute_layouts)
        self._bucket_plans = tuple(
            replace(plan, device=self._compute_device) if isinstance(plan, _RedistributionBucketPlan) else plan
            for plan in self._bucket_plans
        )
        self._scratch_by_slot.clear()
        self._scratch_releases.clear()
        self._retired_scratch.clear()
        self._prefetched_parameters.clear()
        self._prefetched_momenta.clear()
        self._pending_parameter_prefetch.clear()
        self._pending_d2h.clear()

    def _should_defer_parameter_prefetch(self) -> bool:
        """Defer weights only when the next input has a staging H2D.

        With NPU-resident optimizer state, the clip channel supplies the
        already staged NPU gradient and momentum never leaves the device.
        There is then no next-bucket optimizer H2D for a weight copy to block,
        so delaying weights only moves useful prefetch work later.
        """
        if self._offload_states or self._clip_channel is None:
            return True
        return getattr(self._clip_channel, "_pending", None) is None

    def _defer_parameter_prefetch(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
    ) -> None:
        """Queue a bucket's weight copy after the next input batch is queued.

        Gradient and momentum H2D copies are on the critical path to the next
        redistribution all-to-all.  We therefore keep weight copies out of
        the staging stream until the following bucket has submitted its
        input copies.  The caller owns the small pending queue and flushes
        the previous bucket immediately before enqueuing the next one.
        """
        self._pending_parameter_prefetch.append(_WeightPrefetchBucket(tuple(compute_layouts)))

    def _flush_deferred_parameter_prefetch(self) -> None:
        if not self._pending_parameter_prefetch:
            return
        bucket = self._pending_parameter_prefetch.pop(0)
        self._prefetch_parameters(bucket.layouts, stream=self._staging.stream, bucket=bucket)

    def _local_tensor_spec(
        self,
        compute_layout: _ParameterComputeLayout,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        shape, dtype, _ = super()._local_tensor_spec(compute_layout)
        return shape, dtype, self._compute_device

    def _scratch_like(
        self,
        tensor: torch.Tensor,
        *,
        owner: torch.Tensor,
        kind: str,
    ) -> torch.Tensor:
        # The owner is the stable logical parameter, not a storage base:
        # different CPU parameters can be slices of the same pinned slab.
        key = (id(owner), kind, tensor.dtype)
        scratch = self._scratch_by_slot.get(key)
        if scratch is None or scratch.numel() < tensor.numel():
            if scratch is not None:
                self._retired_scratch.append(scratch)
            scratch = torch.empty(
                tensor.numel(),
                dtype=tensor.dtype,
                device=self._compute_device,
            )
            self._scratch_by_slot[key] = scratch
        return scratch[: tensor.numel()].view(tensor.shape)

    def _wait_for_scratch(self, scratch: torch.Tensor, stream: torch.Stream) -> None:
        release = self._scratch_releases.pop(
            scratch.untyped_storage().data_ptr(),
            None,
        )
        if release is not None:
            release.wait_on(stream)

    def _release_scratch(
        self,
        scratch: torch.Tensor,
        handle: TransferHandle,
    ) -> None:
        self._scratch_releases[scratch.untyped_storage().data_ptr()] = handle

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyrefly: ignore [bad-override]
        self._staging.stream.wait_stream(torch_npu.npu.current_stream(self._compute_device))
        result = super().step(closure)
        self._flush_parameter_d2h()
        self._staging.wait()
        self._scratch_releases.clear()
        self._retired_scratch.clear()
        self._prefetched_parameters.clear()
        self._prefetched_momenta.clear()
        return result

    def load_state_dict(self, state_dict):
        """Relocate restored momentum to the compute device when NPU-resident.

        PyTorch's ``Optimizer.load_state_dict`` casts state tensors to the
        parameter device, which is CPU storage under CPU offload; without
        this correction the first post-restore ``_prepare_local`` would mix
        a CPU momentum with the NPU gradient buffer.
        """
        super().load_state_dict(state_dict)
        if self._offload_states:
            return
        for state in self.state.values():
            momentum = state.get("momentum_buffer")
            if momentum is None:
                continue
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.device != self._compute_device:
                state["momentum_buffer"] = momentum.to(self._compute_device)

    def _prepare_local(
        self,
        compute_layout: _ParameterComputeLayout,
        out: torch.Tensor,
    ) -> None:
        _, momentum, momentum_state, group = self._local_gradient_and_momentum(compute_layout)
        stream = torch_npu.npu.current_stream(self._compute_device)
        parameter_gradient = compute_layout.param.grad
        assert parameter_gradient is not None
        canonical_gradient = (
            parameter_gradient.to_local() if isinstance(parameter_gradient, DTensor) else parameter_gradient
        )
        stage_gradient(
            compute_layout.param,
            canonical_gradient,
            out,
            staging=self._staging,
            channel=self._clip_channel,
            stream=stream,
        )
        if self._offload_states:
            if id(compute_layout) not in self._prefetched_momenta:
                self._prefetch_bucket_momenta((compute_layout,), stream=stream)
            momentum_work, _ = self._prefetched_momenta[id(compute_layout)]
            _prepare_muon_input(
                out,
                momentum_work,
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                out=out,
            )
            self._prefetched_momenta[id(compute_layout)] = (momentum_work, momentum)
        else:
            # NPU-resident momentum: blend in place on the compute device.
            _prepare_muon_input(
                out,
                momentum,
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                out=out,
            )
        torch.autograd.graph.increment_version(momentum_state)

    def _prefetch_bucket_momenta(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
        *,
        stream: torch.Stream,
    ) -> None:
        """Batch the next bucket's CPU momentum before preparing HCCL inputs."""
        if not self._offload_states:
            return
        groups: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        h2d_stream = self._staging.stream
        with torch_npu.npu.stream(h2d_stream):
            for layout in compute_layouts:
                _, momentum, _, _ = self._local_gradient_and_momentum(layout)
                work = self._scratch_like(momentum, owner=layout.param, kind="momentum-prefetch")
                self._wait_for_scratch(work, h2d_stream)
                sources, destinations = groups.setdefault(momentum.dtype, ([], []))
                sources.append(momentum)
                destinations.append(work)
                work.record_stream(stream)
                self._prefetched_momenta[id(layout)] = (work, momentum)
            for sources, destinations in groups.values():
                ready = self._staging.submit_h2d_batch(sources, destinations, stream=h2d_stream)
                ready.wait_on(stream)

    def _prefetch_parameters(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
        *,
        stream: torch.Stream,
        bucket: _WeightPrefetchBucket | None = None,
    ) -> None:
        """Stage one weight bucket and publish one device-side ready event."""
        layouts = tuple(layout for layout in compute_layouts if id(layout) not in self._prefetched_parameters)
        if not layouts:
            return
        bucket = bucket or _WeightPrefetchBucket(layouts)
        bucket.layouts = layouts
        groups: dict[torch.dtype, list[tuple[_ParameterComputeLayout, torch.Tensor, torch.Tensor]]] = {}
        # Never queue H2D behind a consumer's compute_done/reverse_ready.
        # All weights and momenta share this one transfer lane.
        consumer_stream = stream
        transfer_stream = self._staging.stream
        with torch_npu.npu.stream(transfer_stream):
            for compute_layout in layouts:
                local_param = compute_layout.param.to_local()
                parameter_work = self._scratch_like(
                    local_param,
                    owner=compute_layout.param,
                    kind="parameter-prefetch",
                )
                self._wait_for_scratch(parameter_work, transfer_stream)
                groups.setdefault(local_param.dtype, []).append((compute_layout, local_param, parameter_work))
            for group in groups.values():
                ready = self._staging.submit_h2d_batch(
                    [source for _layout, source, _work in group],
                    [work for _layout, _source, work in group],
                    stream=transfer_stream,
                )
                for compute_layout, _source, work in group:
                    work.record_stream(consumer_stream)
                    self._prefetched_parameters[id(compute_layout)] = (work, bucket)
            bucket.ready_event = torch_npu.npu.Event()
            bucket.ready_event.record(transfer_stream)

    @staticmethod
    def _wait_for_weight_bucket(bucket: _WeightPrefetchBucket, stream: torch.Stream) -> None:
        if bucket.ready_event is not None:
            stream.wait_event(bucket.ready_event)

    def _apply_update(
        self,
        compute_layout: _ParameterComputeLayout,
        direction: torch.Tensor,
    ) -> None:
        # Current TorchTitan stores the canonical parameter in
        # ``compute_layout.param`` and no longer exposes the old
        # ``local_storage_view`` field. The runtime stages weights separately
        # from prepare; direct callback users retain a synchronous-in-stream
        # fallback without making communication wait for those weights.
        local_param = compute_layout.param.to_local()
        assert local_param is not None
        stream = torch_npu.npu.current_stream(self._compute_device)
        if id(compute_layout) not in self._prefetched_parameters:
            # Keep a safe fallback for callers that invoke finalize without
            # the normal runtime prepare path.
            self._prefetch_parameters((compute_layout,), stream=stream)
        parameter_work, weight_bucket = self._prefetched_parameters.pop(id(compute_layout))
        self._wait_for_weight_bucket(weight_bucket, stream)
        parameter_work.record_stream(stream)
        group = self._group(compute_layout)
        _apply_muon_update(
            parameter_work,
            direction,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            adjust_lr_fn=group["adjust_lr_fn"],
            compute_matrix_shape=compute_layout.global_compute_shape,
        )
        self._pending_d2h.append((parameter_work, local_param, stream))
        momentum_pair = self._prefetched_momenta.pop(id(compute_layout), None)
        if momentum_pair is not None:
            self._pending_d2h.append((momentum_pair[0], momentum_pair[1], stream))
        torch.autograd.graph.increment_version(compute_layout.param)

    def _flush_parameter_d2h(self) -> None:
        """Commit all finalized parameters/momenta with bucket-level foreach."""
        if not self._pending_d2h:
            return
        groups: dict[
            tuple[torch.dtype, Any],
            tuple[Any, list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for source, destination, producer in self._pending_d2h:
            group = groups.setdefault((source.dtype, producer), (producer, [], []))
            group[1].append(source)
            group[2].append(destination)
        for (_source_dtype, _producer_id), (producer, sources, destinations) in groups.items():
            handle = self._staging.submit_d2h_batch(
                sources,
                destinations,
                producer_stream=producer,
                stream=self._commit_stream,
            )
            for source in sources:
                self._release_scratch(source, handle)
        self._pending_d2h.clear()


def build_cpu_offload_distributed_muon(
    params: Iterable[dict[str, Any]],
    *,
    staging: CpuStaging,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
    offload_states: bool = True,
    enable_hsdp_replica_dedup: bool = False,
    **kwargs: Any,
) -> CpuOffloadDistributedMuon:
    """Build the CPU-storage variant through upstream Muon configuration."""
    if enable_hsdp_replica_dedup:
        raise ValueError("HSDP replica dedup is not supported with CPU-offload Muon")
    optimizer = CpuOffloadDistributedMuon(
        _normalize_param_groups(params),
        staging=staging,
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
        offload_states=offload_states,
        **kwargs,
    )
    upstream_runtime = optimizer._redistribution_runtime
    optimizer._redistribution_runtime = LocalPrefetchRuntime.from_runtime(
        upstream_runtime,
        local_stream=torch_npu.npu.Stream(device=staging.device),
    )
    optimizer._redistribution_runtime.reserve_buffers(
        optimizer._bucket_plans,
        local_tensor_spec=optimizer._local_tensor_spec,
    )
    return optimizer
