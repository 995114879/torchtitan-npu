# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AscendC MoonEP transport primitives; expert computation stays in PyTorch.

The external ``ops`` backend owns the single-slot symmetric heap. Each prefetch
returns owned weight snapshots so later layers can reuse the communication slot
while autograd and activation checkpointing retain earlier weights.
"""

from typing import Any

import torch
from torch._library.opaque_object import (
    CustomClassBase,  # pyrefly: ignore [missing-module-attribute]
    register_opaque_type,
)


class MoonEPContextHandle(CustomClassBase):
    """Reference to a context whose lifetime is managed by the integration."""

    def __init__(self, context: Any):
        self.context = context

    def __eq__(self, other):
        return isinstance(other, MoonEPContextHandle) and self.context is other.context

    def __hash__(self):
        return id(self.context)


register_opaque_type(MoonEPContextHandle, typ="reference")


@torch.library.custom_op("torchtitan_npu::moonep_plan", mutates_args=(), device_types="npu")
def plan(
    counts: torch.Tensor,
    ids: torch.Tensor,
    num_ranks: int,
    context: MoonEPContextHandle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return owned virtual routing, cumulative boundaries and replica table."""
    from ops import moonep_plan  # pyrefly: ignore [missing-import]

    _, cu, etc, _, virtual_ids = moonep_plan.moonep_plan(
        context.context, counts.to(torch.int32).contiguous(), ids.to(torch.int32).contiguous()
    )
    return virtual_ids.reshape_as(ids).clone(), cu.clone(), etc.clone()


@plan.register_fake
def _plan_fake(counts, ids, num_ranks, context):
    num_local_experts = counts.numel() // num_ranks
    return (
        ids.new_empty(ids.shape, dtype=torch.int32),
        counts.new_empty((2 * num_local_experts,), dtype=torch.int32),
        counts.new_empty((num_ranks, num_local_experts), dtype=torch.int32),
    )


@torch.library.custom_op("torchtitan_npu::moonep_prefetch_weight", mutates_args=(), device_types="npu")
def prefetch_weight(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    experts_to_copy: torch.Tensor,
    context: MoonEPContextHandle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three owned BF16 weight snapshots, including the replica rows.

    Copy inside the custom-op boundary so compiled execution and SAC see owned
    outputs. On the current single stream, the next plan's rank handshake
    follows these copies and precedes the next prefetch that reuses the slot.
    """
    from ops import prefetch_weight  # pyrefly: ignore [missing-import]

    weights = (gate, up, down)
    num_local_experts = gate.shape[0]
    width = gate[0].numel()
    views = tuple(
        prefetch_weight.create_weight_view(
            context.context, num_local_experts, width, group=index, groups=3, slot=0, slots=1
        )
        for index in range(3)
    )
    prefetch_weight.prefetch_weight(
        context.context,
        experts_to_copy,
        tuple(w.reshape(num_local_experts, width) for w in weights),
        views,
        num_local_experts,
    )
    # Keep clone: later layers overwrite the shared slot before backward.
    # Autograd and SAC must retain this layer's weights in independent storage.
    snapshots = tuple(
        view.reshape(2 * num_local_experts, *w.shape[1:]).clone() for view, w in zip(views, weights, strict=True)
    )
    return snapshots[0], snapshots[1], snapshots[2]


@prefetch_weight.register_fake
def _prefetch_fake(gate, up, down, experts_to_copy, context):
    return tuple(w.new_empty((2 * w.shape[0], *w.shape[1:])) for w in (gate, up, down))


@torch.library.custom_op("torchtitan_npu::moonep_reduce_grad", mutates_args=(), device_types="npu")
def reduce_grad(
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    down_grad: torch.Tensor,
    experts_to_copy: torch.Tensor,
    context: MoonEPContextHandle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return home gradients after adding the contributions of replica rows."""
    from ops import reduce_grad  # pyrefly: ignore [missing-import]

    grads = (gate_grad, up_grad, down_grad)
    num_local_experts = experts_to_copy.shape[1]
    width = gate_grad.numel() // (2 * num_local_experts)
    home_grads = []
    for index, grad in enumerate(grads):
        rows = grad.reshape(2 * num_local_experts, width)
        slot = reduce_grad.create_grad_slot_view(
            context.context, num_local_experts, width, group=index, groups=3, grad_dtype=grad.dtype
        )
        slot.copy_(rows[num_local_experts:])
        home_grads.append(rows[:num_local_experts].clone())
    # This backend forwards `barrier` to finish_op's async_finish parameter.
    reduce_grad.reduce_grad(context.context, experts_to_copy, tuple(home_grads), num_local_experts, barrier=False)
    results = tuple(
        g.reshape(num_local_experts, *original.shape[1:]) for g, original in zip(home_grads, grads, strict=True)
    )
    return results[0], results[1], results[2]


@reduce_grad.register_fake
def _reduce_fake(gate_grad, up_grad, down_grad, experts_to_copy, context):
    return tuple(g.new_empty((experts_to_copy.shape[1], *g.shape[1:])) for g in (gate_grad, up_grad, down_grad))


def _prefetch_setup_context(ctx, inputs, output):
    gate, up, down, experts_to_copy, context = inputs
    ctx.save_for_backward(experts_to_copy)
    ctx.context = context
    ctx.shapes = tuple(w.shape for w in (gate, up, down))
    ctx.device = gate.device
    ctx.dtype = gate.dtype
    ctx.set_materialize_grads(False)


def _prefetch_backward(ctx, gate_grad, up_grad, down_grad):
    (experts_to_copy,) = ctx.saved_tensors
    # All ranks enter the same three-group collective, including empty experts
    # and groups with no local contribution. Do not skip a remote contribution
    # based on the presence of a local grad output.
    grads = tuple(
        torch.zeros((2 * shape[0], *shape[1:]), device=ctx.device, dtype=ctx.dtype)
        if grad is None
        else grad.contiguous()
        for grad, shape in zip((gate_grad, up_grad, down_grad), ctx.shapes, strict=True)
    )
    home_grads = torch.ops.torchtitan_npu.moonep_reduce_grad(*grads, experts_to_copy, ctx.context)
    return (*home_grads, None, None)


torch.library.register_autograd(
    "torchtitan_npu::moonep_prefetch_weight", _prefetch_backward, setup_context=_prefetch_setup_context
)
