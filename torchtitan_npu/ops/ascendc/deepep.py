# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE dispatch/combine adapter for ``cann_ops_transformer.ElasticBuffer``.

The adapter deliberately implements only the communication primitives consumed by
TorchTitan's compact ``DeepEPTokenDispatcher`` training path. The dispatcher creates
and owns the ``ElasticBuffer`` instance; this module only transports that reference
through the custom-op boundary. Engram storage and the DeepEP expand/cudagraph
inference layout are outside its scope.
"""

__all__ = ["DispatchHandle", "ElasticBufferHandle"]

from typing import Any, cast

import torch
import torch_npu  # noqa: F401 - registers the Ascend device type
from cann_ops_transformer import ElasticBuffer, EPHandle
from torch._library.opaque_object import (
    CustomClassBase,  # pyrefly: ignore [missing-module-attribute]
    register_opaque_type,
)


class ElasticBufferHandle(CustomClassBase):
    """Opaque reference to an ``ElasticBuffer`` owned by the dispatcher."""

    def __init__(self, value: ElasticBuffer):
        self.value: ElasticBuffer = value

    def __eq__(self, other):
        return isinstance(other, ElasticBufferHandle) and self.value is other.value

    def __hash__(self):
        return id(self.value)


register_opaque_type(ElasticBufferHandle, typ="reference")


class DispatchHandle(CustomClassBase):
    """Opaque wrapper carrying a CANN ``EPHandle`` across custom-op boundaries."""

    def __init__(self, value: EPHandle | None = None):
        self.value: Any = value

    def __eq__(self, other):
        return isinstance(other, DispatchHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "DispatchHandle()", {"DispatchHandle": DispatchHandle}


register_opaque_type(DispatchHandle, typ="reference")


@torch.library.custom_op("deepep::dispatch", mutates_args=(), device_types="npu")
def _dispatch_op_impl(
    x: list[torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    buffer: ElasticBufferHandle,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, DispatchHandle]:
    """Dispatch ``[x]`` or ``[qdata, scale]`` with one shared routing handle.

    Tensor lists give both payload formats the same custom-op schema and
    ``deepep.dispatch.default`` selective-AC save entry. Only ElasticBuffer's
    Python boundary receives a tuple for the quantized payload.

    The MXFP8 caller must use no_grad here: its external autograd Function
    connects the received payload to the original high-precision activation
    and router scores. Encoded uint8 scales have no payload gradient.
    """
    payload_tensors = x
    if len(payload_tensors) not in (1, 2):
        raise ValueError("DeepEP dispatch expects [x] or [qdata, scale]")
    payload = (
        payload_tensors[0].contiguous()
        if len(payload_tensors) == 1
        else (payload_tensors[0].contiguous(), payload_tensors[1].contiguous())
    )
    topk_idx = topk_idx.contiguous()
    topk_weights = topk_weights.contiguous()
    elastic_buffer = buffer.value

    recv_payload, _recv_topk_idx, recv_scores, handle = elastic_buffer.dispatch(
        payload,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        # In non-cached dispatch, there is no handle with a previously known receive count.
        # ``do_cpu_sync=True`` resets a host-pinned counter to -1; after the NPU
        # finishes computing the number of expert-routed token rows received by
        # local rank (the sum across its local experts), its dispatch kernel writes
        # that value to the counter. ``dispatch`` then CPU-spins until it observes
        # a non-negative value before allocating the exact receive tensors and routed metadata.
        do_cpu_sync=True,
    )
    if len(payload_tensors) == 2:
        if not isinstance(recv_payload, tuple) or len(recv_payload) != 2:
            raise RuntimeError(
                "ElasticBuffer must support tuple (qdata, scale) dispatch; rebuild the CANN ops and wheel"
            )
        received_payload = [cast("torch.Tensor", recv_payload[0]), cast("torch.Tensor", recv_payload[1])]
    else:
        received_payload = [cast("torch.Tensor", recv_payload)]
    return (
        received_payload,
        cast("torch.Tensor", recv_scores),
        handle.num_recv_tokens_per_expert,
        DispatchHandle(value=handle),
    )


@_dispatch_op_impl.register_fake
def _dispatch_fake(
    x: list[torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    buffer: ElasticBufferHandle,
):
    payload_tensors = x
    if len(payload_tensors) not in (1, 2):
        raise ValueError("DeepEP dispatch expects [x] or [qdata, scale]")
    # Every payload has the same data-dependent receive count, so a
    # static upper bound would fail inductor's assert_size_stride. This dynamic
    # symbol represents the actual number of rows produced by dispatch.
    num_recv = torch.library.get_ctx().new_dynamic_size()
    return (
        [tensor.new_empty((num_recv, tensor.shape[1])) for tensor in payload_tensors],
        payload_tensors[0].new_empty((num_recv,), dtype=torch.float32),
        payload_tensors[0].new_empty((num_local_experts,), dtype=torch.int64),
        DispatchHandle(),
    )


@torch.library.custom_op("deepep::dispatch_quantized_cached", mutates_args=(), device_types="npu")
def _dispatch_quantized_cached_op_impl(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse routing inside combine's first-order VJP; no payload autograd."""
    recv_payload, _, _, _ = buffer.value.dispatch(
        (qdata.contiguous(), scale.contiguous()),
        handle=handle.value,
        # Both payloads follow the original route, with no new count synchronization.
        do_cpu_sync=False,
    )
    if not isinstance(recv_payload, tuple):
        raise RuntimeError("ElasticBuffer cached dispatch must return (qdata, scale)")
    return recv_payload


@_dispatch_quantized_cached_op_impl.register_fake
def _dispatch_quantized_cached_fake(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
):
    # Use the saved dispatch score tensor's row count, as the BF16 path does.
    return (
        qdata.new_empty((num_permuted_tokens, qdata.shape[1])),
        scale.new_empty((num_permuted_tokens, scale.shape[1])),
    )


@torch.library.custom_op("deepep::combine", mutates_args=(), device_types="npu")
def _combine_op_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_permuted_tokens_src: torch.Tensor,
    buffer: ElasticBufferHandle,
) -> torch.Tensor:
    """Pure-reduction combine; routing scores are applied outside this op.

    ``num_permuted_tokens_src`` has one entry per routed row. It is saved
    for backward to recover the row count from a dispatch tensor input,
    avoiding an unbacked size from a recomputed expert intermediate.
    The combine kernel does not read its values.
    """
    x = x.contiguous()
    elastic_buffer = buffer.value
    combined, _ = elastic_buffer.combine(x, handle.value, topk_weights=None)
    return combined


@_combine_op_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_permuted_tokens_src: torch.Tensor,
    buffer: ElasticBufferHandle,
):
    # CANN combine permutes the routed tokens back to the original token order,
    # so its output has num_tokens rows, NOT the combine input's num_recv rows.
    # Let torch.compile keep the true (dynamic) routed count.
    return x.new_empty((num_tokens, x.shape[1]))


@torch.library.custom_op("deepep::dispatch_backward", mutates_args=(), device_types="npu")
def _dispatch_backward_op_impl(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    has_grad_scores: bool,
    topk_idx: torch.Tensor,
    handle: DispatchHandle,
    buffer: ElasticBufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``dispatch`` implemented by CANN's cached ``combine``."""
    grad_hidden = grad_hidden.contiguous()
    grad_scores = grad_scores.contiguous()
    elastic_buffer = buffer.value
    grad_x, grad_topk_weights = elastic_buffer.combine(
        grad_hidden,
        handle.value,
        topk_weights=(grad_scores.reshape(-1).float().contiguous() if has_grad_scores else None),
    )
    if not has_grad_scores:
        # The custom-op schema always returns a Tensor for this slot; the
        # autograd wrapper discards it when the forward score output is unused.
        grad_topk_weights = grad_scores.new_empty(topk_idx.shape)
    return grad_x, cast("torch.Tensor", grad_topk_weights)


@_dispatch_backward_op_impl.register_fake
def _dispatch_backward_fake(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    has_grad_scores: bool,
    topk_idx: torch.Tensor,
    handle: DispatchHandle,
    buffer: ElasticBufferHandle,
):
    return (
        grad_hidden.new_empty((topk_idx.shape[0], grad_hidden.shape[1])),
        grad_hidden.new_empty(topk_idx.shape, dtype=torch.float32),
    )


@torch.library.custom_op("deepep::combine_backward", mutates_args=(), device_types="npu")
def _combine_backward_op_impl(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
) -> torch.Tensor:
    """Backward of ``combine`` implemented by CANN's cached ``dispatch``."""
    grad_combined = grad_combined.contiguous()
    elastic_buffer = buffer.value
    grad_x, _recv_topk_idx, _recv_scores, _new_handle = elastic_buffer.dispatch(
        grad_combined,
        handle=handle.value,
        # Cached dispatches reuse the count stored in the handle
        # and explicitly disable this synchronization.
        do_cpu_sync=False,
    )
    return cast("torch.Tensor", grad_x)


@_combine_backward_op_impl.register_fake
def _combine_backward_fake(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
):
    return grad_combined.new_empty((num_permuted_tokens, grad_combined.shape[1]))


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, _topk_weights, _num_experts, _num_local_experts, buffer = inputs
    if len(x) != 1:
        raise RuntimeError("MXFP8 DeepEP payloads require torch.no_grad() and the MXFP8 dispatcher's logical backward")
    (
        _recv_x,
        _recv_scores,
        _num_recv_tokens_per_expert,
        handle,
    ) = output
    ctx.save_for_backward(topk_idx)
    ctx.handle = handle
    ctx.buffer = buffer
    ctx.input_dtype = x[0].dtype
    ctx.hidden = x[0].shape[1]


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_scores,
    _grad_num_recv_per_expert,
    _grad_handle,
):
    """Route data and routing-score gradients through the CANN combine path."""
    if grad_recv_x is None and grad_recv_scores is None:
        return None, None, None, None, None, None

    has_grad_x = grad_recv_x is not None
    has_grad_scores = grad_recv_scores is not None
    if grad_recv_x is None:
        grad_recv_x = torch.zeros(
            (grad_recv_scores.numel(), ctx.hidden),
            dtype=ctx.input_dtype,
            device=grad_recv_scores.device,
        )
    if grad_recv_scores is None:
        grad_recv_scores = grad_recv_x.new_empty((0,), dtype=torch.float32)
    (topk_idx,) = ctx.saved_tensors
    grad_x, grad_topk_weights = torch.ops.deepep.dispatch_backward(
        grad_recv_x,
        grad_recv_scores,
        has_grad_scores,
        topk_idx,
        ctx.handle,
        ctx.buffer,
    )
    return (
        grad_x.to(ctx.input_dtype) if has_grad_x else None,
        None,
        grad_topk_weights if has_grad_scores else None,
        None,
        None,
        None,
    )


def _combine_setup_context(ctx, inputs, output) -> None:
    _, handle, _, num_permuted_tokens_src, buffer = inputs
    ctx.handle = handle
    ctx.buffer = buffer
    ctx.save_for_backward(num_permuted_tokens_src)


def _combine_backward(ctx, grad_combined):
    """Scatter combined gradients through the cached CANN dispatch path."""
    (num_permuted_tokens_src,) = ctx.saved_tensors
    grad_x = torch.ops.deepep.combine_backward(
        grad_combined,
        ctx.handle,
        num_permuted_tokens_src.shape[0],
        ctx.buffer,
    )
    return grad_x, None, None, None, None


def _dispatch_list_backward(ctx, grad_recv_x, grad_recv_scores, grad_counts, grad_handle):
    # Tensor[] gradients mirror the input list. Keep the tensor-based VJP
    # helper shared with TorchAO's logical MXFP8 activation/score backward.
    # ``grad_recv_x`` is None when the whole payload list output is unused, so
    # guard the index and let ``_dispatch_backward`` synthesize/route the rest.
    grad_x = None if grad_recv_x is None else grad_recv_x[0]
    grad_inputs = _dispatch_backward(ctx, grad_x, grad_recv_scores, grad_counts, grad_handle)
    return [grad_inputs[0]], *grad_inputs[1:]


torch.library.register_autograd("deepep::dispatch", _dispatch_list_backward, setup_context=_dispatch_setup_context)
torch.library.register_autograd("deepep::combine", _combine_backward, setup_context=_combine_setup_context)
