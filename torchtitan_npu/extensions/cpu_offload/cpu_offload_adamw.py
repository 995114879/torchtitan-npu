# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AdamW with CPU canonical tensors and NPU functional updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from torch import Tensor

_BATCH_BYTES = 64 * 1024 * 1024


from torchtitan_npu.extensions.cpu_offload.runtime import (
    GradientClipChannel,
    local_tensor,
    require_cpu_tensor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging


@dataclass(slots=True)
class _AdamWWork:
    parameter: Tensor
    gradient: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor


@dataclass(slots=True)
class _AdamWBatchWork:
    entries: tuple[_AdamWEntry, ...]
    works: tuple[_AdamWWork, ...]
    h2d: tuple[Any, ...] = ()
    buffer_keys: tuple[tuple[int, int, int, torch.dtype], ...] = ()


@dataclass(slots=True)
class _AdamWEntry:
    parameter: Tensor
    group: dict[str, Any]
    state: dict[str, Tensor]
    canonical: tuple[Tensor, Tensor, Tensor, Tensor]


class CpuOffloadAdamW(torch.optim.AdamW):
    """Keep standard AdamW state on CPU and execute each update on NPU."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        staging: CpuStaging,
        clip_channel: GradientClipChannel | None = None,
        offload_states: bool = True,
    ) -> None:
        unsupported = {
            "foreach": bool(foreach),
            "fused": bool(fused),
            "amsgrad": amsgrad,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        if option := next((name for name, enabled in unsupported.items() if enabled), None):
            raise ValueError(f"CPU-offloaded AdamW does not support {option}")

        self._staging = staging
        self._clip_channel = clip_channel
        self._offload_states = offload_states
        self._compute_device = staging.device  # pyrefly: ignore [read-only]
        self._work_buffers: dict[tuple[int, int, torch.dtype], Tensor] = {}
        self._batch_work_buffers: dict[tuple[int, int, int, torch.dtype], Tensor] = {}
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            maximize=maximize,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        for group in self.param_groups:
            if option := next((name for name in unsupported if group.get(name)), None):
                raise ValueError(f"CPU-offloaded AdamW does not support {option}")

    def materialize_state(self) -> None:
        """Create the standard AdamW state directly on canonical CPU storage.

        Materialization is eager by design: the CPU-offload pipeline stages
        state to the NPU from the first step on, and checkpointing material-
        izes optimizer state the same way (``init_optim_state``), so the
        canonical storage must exist before the first ``step()`` instead of
        being created lazily on the NPU.
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                require_cpu_tensor(parameter, what="AdamW parameters")
                state = self.state[parameter]
                if state:
                    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise RuntimeError("unexpected AdamW state keys")
                    if not self._offload_states:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if state[key].device.type != self._compute_device.type:
                                state[key] = state[key].to(self._compute_device)
                    continue
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = self._momentum_like(parameter)
                state["exp_avg_sq"] = self._momentum_like(parameter)

    def _momentum_like(self, parameter: Tensor) -> Tensor:
        """Allocate a moment mirroring the parameter; NPU-resident when the
        optimizer state does not join the CPU-offload plane."""
        moment = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if not self._offload_states:
            moment = moment.to(self._compute_device)
        return moment

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.materialize_state()
        self._reserve_work_buffers()
        compute_stream = torch_npu.npu.current_stream(self._compute_device)
        transfer_stream = self._staging.stream
        entries: list[_AdamWEntry] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                local_parameter = local_tensor(parameter)
                local_gradient = local_tensor(parameter.grad)
                state = self.state[parameter]
                local_exp_avg = local_tensor(state["exp_avg"])
                local_exp_avg_sq = local_tensor(state["exp_avg_sq"])
                canonical = (
                    local_parameter,
                    local_gradient,
                    local_exp_avg,
                    local_exp_avg_sq,
                )
                if local_parameter.device.type != "cpu" or local_gradient.device.type != "cpu":
                    raise ValueError("CPU-offloaded AdamW requires CPU parameter and gradient storage")
                expected_state_device = "cpu" if self._offload_states else self._compute_device.type
                if (
                    local_exp_avg.device.type != expected_state_device
                    or local_exp_avg_sq.device.type != expected_state_device
                ):
                    raise ValueError(
                        "CPU-offloaded AdamW optimizer state must be "
                        f"{expected_state_device}-resident, got exp_avg on "
                        f"{local_exp_avg.device} and exp_avg_sq on {local_exp_avg_sq.device}"
                    )
                if any(tensor.is_complex() for tensor in canonical):
                    raise ValueError("CPU-offloaded AdamW does not support complex tensors")

                entries.append(_AdamWEntry(parameter, group, state, canonical))

        if entries:
            # Keep each batch bounded so CPU offload does not turn into a
            # full-optimizer NPU allocation. Dtype grouping still happens
            # inside each batch.
            for batch_entries in self._iter_entry_batches(tuple(entries)):
                current = self._enqueue_batch(0, batch_entries, compute_stream)
                for handle in current.h2d:
                    handle.wait_on(compute_stream)
                with torch_npu.npu.stream(compute_stream):
                    for entry, work in zip(current.entries, current.works, strict=True):
                        if self._clip_channel is not None:
                            self._clip_channel.apply_pending(work.gradient)
                        self._update_one(entry, work)
                        for tensor in (work.parameter, work.gradient, work.exp_avg, work.exp_avg_sq):
                            tensor.record_stream(compute_stream)
                        torch.autograd.graph.increment_version(entry.parameter)
                self._submit_batch_d2h(current, compute_stream, transfer_stream)
        self._staging.wait()
        return loss

    def _iter_entry_batches(self, entries: tuple[_AdamWEntry, ...]) -> Iterable[tuple[_AdamWEntry, ...]]:
        batch: list[_AdamWEntry] = []
        batch_bytes = 0
        for entry in entries:
            entry_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in entry.canonical
                if self._offload_states or tensor is entry.canonical[0] or tensor is entry.canonical[1]
            )
            if batch and batch_bytes + entry_bytes > _BATCH_BYTES:
                yield tuple(batch)
                batch = []
                batch_bytes = 0
            batch.append(entry)
            batch_bytes += entry_bytes
        if batch:
            yield tuple(batch)

    @staticmethod
    def _copy_groups(sources: list[Tensor], destinations: list[Tensor]) -> list[tuple[list[Tensor], list[Tensor]]]:
        groups: dict[
            tuple[torch.dtype, torch.dtype, torch.device, torch.device],
            tuple[list[Tensor], list[Tensor]],
        ] = {}
        for source, destination in zip(sources, destinations, strict=True):
            key = (source.dtype, destination.dtype, source.device, destination.device)
            group = groups.setdefault(key, ([], []))
            group[0].append(source)
            group[1].append(destination)
        return list(groups.values())

    def _enqueue_batch(
        self,
        slot: int,
        entries: tuple[_AdamWEntry, ...],
        compute_stream: Any,
    ) -> _AdamWBatchWork:
        works: list[_AdamWWork] = []
        h2d_sources: list[Tensor] = []
        h2d_destinations: list[Tensor] = []
        buffer_keys: set[tuple[int, int, int, torch.dtype]] = set()
        for entry in entries:
            parameter, gradient, exp_avg, exp_avg_sq = entry.canonical
            cached_gradient = (
                self._clip_channel.take_gradient(entry.parameter, gradient) if self._clip_channel is not None else None
            )
            if cached_gradient is not None:
                working_gradient = cached_gradient.view_as(gradient)
                if working_gradient.device.type == "npu":
                    working_gradient.record_stream(compute_stream)
            else:
                working_gradient = self._batch_work_buffer(slot, 1, gradient)
                buffer_keys.add((slot, 1, id(gradient), gradient.dtype))
                h2d_sources.append(gradient)
                h2d_destinations.append(working_gradient)
            sources = [parameter]
            destinations = [self._batch_work_buffer(slot, 0, parameter)]
            buffer_keys.add((slot, 0, id(parameter), parameter.dtype))
            if self._offload_states:
                sources.extend((exp_avg, exp_avg_sq))
                destinations.extend(
                    (
                        self._batch_work_buffer(slot, 2, exp_avg),
                        self._batch_work_buffer(slot, 3, exp_avg_sq),
                    )
                )
                buffer_keys.add((slot, 2, id(exp_avg), exp_avg.dtype))
                buffer_keys.add((slot, 3, id(exp_avg_sq), exp_avg_sq.dtype))
            h2d_sources.extend(sources)
            h2d_destinations.extend(destinations)
            works.append(
                _AdamWWork(
                    destinations[0],
                    working_gradient,
                    destinations[1] if self._offload_states else exp_avg,
                    destinations[2] if self._offload_states else exp_avg_sq,
                )
            )
        handles = tuple(
            self._staging.submit_h2d_batch(group_sources, group_destinations, stream=self._staging.stream)
            for group_sources, group_destinations in self._copy_groups(h2d_sources, h2d_destinations)
        )
        return _AdamWBatchWork(entries, tuple(works), handles, tuple(buffer_keys))

    @staticmethod
    def _update_one(entry: _AdamWEntry, work: _AdamWWork) -> None:
        gradient = -work.gradient if entry.group["maximize"] else work.gradient
        lr = entry.group["lr"]
        beta1, beta2 = entry.group["betas"]
        weight_decay = entry.group["weight_decay"]
        eps = entry.group["eps"]
        entry.state["step"] += 1
        step = int(entry.state["step"].item())
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        if weight_decay != 0:
            work.parameter.mul_(1 - lr * weight_decay)
        work.exp_avg.lerp_(gradient, 1 - beta1)
        work.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        denom = (work.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        work.parameter.addcdiv_(work.exp_avg, denom, value=-step_size)

    def _submit_batch_d2h(
        self,
        batch: _AdamWBatchWork,
        compute_stream: Any,
        transfer_stream: Any,
    ) -> None:
        sources: list[Tensor] = []
        destinations: list[Tensor] = []
        for entry, work in zip(batch.entries, batch.works, strict=True):
            sources.append(work.parameter)
            destinations.append(entry.canonical[0])
            if self._offload_states:
                sources.extend((work.exp_avg, work.exp_avg_sq))
                destinations.extend((entry.canonical[2], entry.canonical[3]))
        for group_sources, group_destinations in self._copy_groups(sources, destinations):
            self._staging.submit_d2h_batch(
                group_sources,
                group_destinations,
                producer_stream=compute_stream,
                stream=transfer_stream,
            )
        # Staging records the transfer stream on the working tensors, so
        # dropping the cache cannot recycle storage before D2H completes.
        for key in batch.buffer_keys:
            self._batch_work_buffers.pop(key, None)

    def _reserve_work_buffers(self) -> None:
        """Keep allocation lazy; each batch owns only its active slot."""
        self._batch_work_buffers.clear()

    def _batch_work_buffer(self, slot: int, kind: int, tensor: Tensor) -> Tensor:
        key = (slot, kind, id(tensor), tensor.dtype)
        buffer = self._batch_work_buffers.get(key)
        if buffer is None or buffer.numel() < tensor.numel():
            buffer = torch.empty(tensor.numel(), dtype=tensor.dtype, device=self._compute_device)
            self._batch_work_buffers[key] = buffer
        # Keep the legacy per-slot view populated for diagnostics and older
        # callers that inspect the optimizer's staged planes.  The batched
        # path itself is keyed by tensor id and drops entries after D2H, so
        # this compatibility map retains only the latest buffer per plane.
        self._work_buffers[(slot, kind, tensor.dtype)] = buffer
        return buffer[: tensor.numel()].view(tensor.shape)
