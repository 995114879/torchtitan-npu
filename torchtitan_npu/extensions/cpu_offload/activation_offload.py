# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layer-boundary activation offload for the eager FullAC path.

The scheduler owns only logical activation tokens. NovaSwap owns the CPU
buffer, transfer streams, events and storage release. Candidate tensors are
captured while the checkpoint wrapper saves its input; registration and D2H
are submitted only after the layer forward returns.
"""

from __future__ import annotations

import logging
import weakref
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils._pytree import tree_flatten

from torchtitan_npu.extensions.novaswap import swap_api

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ActivationToken:
    tensor: Tensor | None
    restored: weakref.ReferenceType[Tensor] | None
    name: str
    group: _ActivationGroup
    state: str = "candidate"
    duplicate: bool = False
    prefetched: bool = False


@dataclass(slots=True)
class _ActivationGroup:
    index: int
    tokens: list[_ActivationToken] = field(default_factory=list)
    selected: _ActivationToken | None = None
    selected_index: int = -1


class ActivationBoundaryOffload:
    """Coordinate one activation-offload instance per trainer."""

    def __init__(self, *, min_tensor_bytes: int = 1 << 20) -> None:
        if min_tensor_bytes < 0:
            raise ValueError("min_tensor_bytes must be non-negative")
        self.min_tensor_bytes = min_tensor_bytes
        self._generation = 0
        self._group_counter = 0
        self._active_group: _ActivationGroup | None = None
        self._groups: list[_ActivationGroup] = []
        self._live_names: set[str] = set()
        self._closed = False
        self._running = False
        self._stats: Counter[str] = Counter()
        self._skips: Counter[str] = Counter()
        self._patched_forwards: list[tuple[CheckpointWrapper, Any]] = []

    @contextmanager
    def microbatch(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("activation offload scheduler is closed")
        if self._running:
            raise RuntimeError("activation offload microbatches cannot be nested")
        self._running = True
        self._generation += 1
        self._groups.clear()
        self._active_group = None
        try:
            yield
        finally:
            try:
                self._cleanup()
            finally:
                self._active_group = None
                self._groups.clear()
                self._running = False
                if self._generation == 1:
                    log = logger.info if self._stats["d2h"] else logger.warning
                    log(
                        "Activation offload first microbatch: D2H=%d, H2D=%d, prefetch=%d, "
                        "offloaded=%.2f MiB, skipped=%s",
                        self._stats["d2h"],
                        self._stats["h2d"],
                        self._stats["prefetch"],
                        self._stats["bytes"] / (1 << 20),
                        dict(self._skips),
                    )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._cleanup()
        finally:
            for module, original_forward in reversed(self._patched_forwards):
                module.forward = original_forward
                if getattr(module, "_activation_boundary_offload", False):
                    delattr(module, "_activation_boundary_offload")
            self._patched_forwards.clear()
            self._closed = True

    def install(self, model_parts: list[torch.nn.Module]) -> int:
        """Patch existing FullAC wrappers without changing module hierarchy."""
        installed = 0
        for model in model_parts:
            for module in model.modules():
                if not isinstance(module, CheckpointWrapper):
                    continue
                if getattr(module, "_activation_boundary_offload", False):
                    continue
                original_forward = module.forward

                def wrapped_forward(
                    wrapped_self: CheckpointWrapper,
                    *args: Any,
                    _original_forward=original_forward,
                    **kwargs: Any,
                ) -> Any:
                    if not self._running:
                        return _original_forward(*args, **kwargs)
                    group = self._begin_group()
                    try:
                        with saved_tensors_hooks(self.pack, self.unpack):
                            output = _original_forward(*args, **kwargs)
                    except Exception:
                        self._discard_group(group)
                        raise
                    self._finish_group(group, output)
                    return output

                module.forward = MethodType(wrapped_forward, module)
                module._activation_boundary_offload = True  # type: ignore[attr-defined]
                self._patched_forwards.append((module, original_forward))
                installed += 1
        if installed == 0:
            raise ValueError("activation offload requires at least one FullAC CheckpointWrapper")
        logger.info("Installed activation boundary offload on %d FullAC layers", installed)
        return installed

    def _begin_group(self) -> _ActivationGroup:
        if self._active_group is not None:
            raise RuntimeError("nested activation checkpoint forward is unsupported")
        group = _ActivationGroup(self._group_counter)
        self._group_counter += 1
        self._active_group = group
        return group

    def _discard_group(self, group: _ActivationGroup) -> None:
        if self._active_group is group:
            self._active_group = None
        for token in group.tokens:
            self._remove_token(token)

    @staticmethod
    def _storage_key(tensor: Tensor) -> tuple[str, int, int]:
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.size()

    @staticmethod
    def _has_compact_activation_storage(value: Tensor) -> bool:
        # FSDP's RegisterPostBackwardFunction returns identity views of layer
        # inputs. These span the complete storage and are safe to swap together
        # with their base; rejecting every view disables offload under FSDP.
        try:
            base = value
            while base._base is not None:
                base = base._base
            if isinstance(value, torch.nn.Parameter) or isinstance(base, torch.nn.Parameter):
                return False
            if not value.is_contiguous() or value.storage_offset() != 0:
                return False
            if torch._debug_has_internal_overlap(value) != 0:
                return False
            return value.untyped_storage().size() == value.numel() * value.element_size()
        except Exception:
            # DTensor and unallocated/retired storages may reject metadata
            # queries. Treat them as ineligible instead of breaking training.
            return False

    def _candidate(self, value: Any) -> bool:
        reason = None
        if not isinstance(value, Tensor):
            reason = "not_tensor"
        elif value.device.type != "npu":
            reason = "not_npu"
        elif not value.requires_grad:
            reason = "no_grad"
        elif value.numel() == 0 or value.numel() * value.element_size() < self.min_tensor_bytes:
            reason = "too_small"
        elif not self._has_compact_activation_storage(value):
            reason = "unsafe_storage_or_parameter"
        if reason is not None:
            if self._generation == 1:
                self._skips[reason] += 1
            return False
        return True

    def pack(self, value: Any) -> Any:
        group = self._active_group
        if group is None or not self._candidate(value):
            return value

        tensor = value.detach()
        key = self._storage_key(tensor)
        for token in group.tokens:
            if token.tensor is not None and self._storage_key(token.tensor) == key:
                token.duplicate = True
                return tensor

        name = f"activation.{id(self)}.{self._generation}.{group.index}.{len(group.tokens)}"
        token = _ActivationToken(tensor=tensor, restored=None, name=name, group=group)
        group.tokens.append(token)
        return token

    def unpack(self, value: Any) -> Any:
        if not isinstance(value, _ActivationToken):
            return value
        token = value
        if token.state == "consumed":
            restored = token.restored() if token.restored is not None else None
            if restored is None:
                raise RuntimeError("activation offload does not support reusing a consumed backward graph")
            return restored
        if token.state in {"candidate", "resident"}:
            tensor = token.tensor
            if tensor is None:
                restored = token.restored() if token.restored is not None else None
                if restored is None:
                    raise RuntimeError(f"Missing resident activation {token.name}")
                return restored
            token.restored = weakref.ref(tensor)
            token.tensor = None
            token.state = "consumed"
            return tensor
        if token.state != "offloaded":
            return token.tensor

        # Put the demand transfer first on NovaSwap's FIFO prefetch stream.
        # Waiting for the current layer must not wait behind the next layer.
        if not token.prefetched:
            swap_api.execute(token.name, "H2D")
            if self._generation == 1:
                self._stats["h2d"] += 1
        self._prefetch_previous(token.group)
        swap_api.execute(token.name, "WAIT_DEVICE")
        payload = swap_api.pop_tensor_by_name(token.name, device_waited=True)
        if not isinstance(payload, Tensor):
            raise RuntimeError(f"Missing restored activation {token.name}")
        token.state = "consumed"
        self._live_names.discard(token.name)
        # The checkpoint recomputation now owns the restored tensor. Keeping a
        # strong reference in the scheduler retains every reloaded layer until
        # the end of backward instead of freeing it after layer consumption.
        token.restored = weakref.ref(payload)
        token.tensor = None
        return payload

    def _prefetch_previous(self, group: _ActivationGroup) -> None:
        if group.selected_index <= 0:
            return
        previous = self._groups[group.selected_index - 1].selected
        if previous is None or previous.state != "offloaded" or previous.prefetched:
            return
        swap_api.execute(previous.name, "H2D")
        previous.prefetched = True
        if self._generation == 1:
            self._stats["h2d"] += 1
            self._stats["prefetch"] += 1

    @staticmethod
    def _output_aliases(output: Any, candidate: Tensor) -> bool:
        try:
            candidate_key = (candidate.device, candidate.untyped_storage().data_ptr())
            values, _ = tree_flatten(output)
        except Exception:
            # Unknown output containers are retained on device to avoid an
            # unsafe D2H that could invalidate an autograd-visible alias.
            return True
        for value in values:
            if isinstance(value, Tensor):
                try:
                    value_key = (value.device, value.untyped_storage().data_ptr())
                except Exception:
                    return True
                if value_key == candidate_key:
                    return True
            elif not isinstance(value, (type(None), bool, int, float, complex, str, bytes)):
                # torch pytree treats an unregistered custom container as a
                # leaf. Keep the candidate resident rather than assuming the
                # container cannot hold an alias.
                return True
        return False

    def _finish_group(self, group: _ActivationGroup, output: Any) -> None:
        if self._active_group is not group:
            raise RuntimeError("activation checkpoint group is no longer active")
        self._active_group = None
        candidates = []
        for token in group.tokens:
            if not token.duplicate and token.tensor is not None:
                candidates.append((token, token.tensor))
        if candidates:
            selected, tensor = max(candidates, key=lambda item: item[1].numel())
            if not self._output_aliases(output, tensor):
                try:
                    swap_api.register_tensor(tensor, selected.name)
                except ValueError:
                    # SwapRegistry uses ValueError for an unsupported candidate;
                    # retain the tensor on device and continue the layer.
                    for token in group.tokens:
                        self._remove_token(token)
                    return
                registered = True
                self._live_names.add(selected.name)
                try:
                    swap_api.execute(selected.name, "D2H")
                except Exception:
                    if registered:
                        try:
                            swap_api.remove_tensor(selected.name)
                        except Exception as cleanup_error:
                            self._live_names.add(selected.name)
                            raise RuntimeError(
                                f"Failed to roll back activation registration {selected.name}"
                            ) from cleanup_error
                        self._live_names.discard(selected.name)
                    raise
                else:
                    if self._generation == 1:
                        self._stats["d2h"] += 1
                        self._stats["bytes"] += tensor.numel() * tensor.element_size()
                    selected.state = "offloaded"
                    group.selected = selected
                    group.selected_index = len(self._groups)
                    self._groups.append(group)
                    for token in group.tokens:
                        if token is not selected:
                            self._remove_token(token)
                    return
        for token in group.tokens:
            self._remove_token(token)

    def _remove_token(self, token: _ActivationToken) -> None:
        if token.state == "candidate":
            token.state = "resident"

    def _cleanup(self) -> None:
        cleanup_error: Exception | None = None
        for name in tuple(self._live_names):
            try:
                swap_api.remove_tensor(name)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
            else:
                self._live_names.discard(name)
        if cleanup_error is not None:
            raise RuntimeError("Failed to clean one or more activation swaps") from cleanup_error
