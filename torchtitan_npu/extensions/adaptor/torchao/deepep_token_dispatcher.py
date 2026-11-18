# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 DeepEP dispatcher and autograd adaptation for torchtitan-npu.

``torchao_npu`` owns the pure quantization helpers (payload layout, wire tensor
and the score-boundary dequantizer). This adaptor binds them to the
``torchtitan_npu`` DeepEP dispatcher, ops and override implementations, so the
standalone ``torchao_npu`` distribution keeps no reverse dependency on
``torchtitan_npu``.
"""

from __future__ import annotations

__all__ = ["get_mxfp8_deepep_dispatcher_cls"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchao_npu.ops.mxfp8_deepep_payload import (
    _DequantizeGradient,
    _quantize_payload,
    _wire_tensor,
)
from torchao_npu.quantized_tensors.mx_tensor import MXTensor, _mxfp8_dequantize_tensor

if TYPE_CHECKING:
    from torchao_npu.ops.mxfp8_all_to_all import MXFP8QuantParams


class _MXFP8DeepEPDispatch(torch.autograd.Function):
    """Communicate FP8 activations forward and high-precision input gradients backward."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        activation,
        recv_qdata,
        recv_scale,
        recv_scores,
        counts,
        topk_idx,
        topk_scores,
        handle,
        buffer,
        quant_params,
    ):
        # Match the existing DeepEP backward context to reuse its score-gradient routing.
        ctx.save_for_backward(topk_idx)
        ctx.handle, ctx.buffer = handle, buffer
        ctx.input_dtype, ctx.hidden = activation.dtype, activation.shape[1]
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(counts)
        routed = _wire_tensor(
            recv_qdata,
            recv_scale,
            activation.dtype,
            quant_params,
            activation.requires_grad,
        )
        return routed, recv_scores, counts

    @staticmethod
    def backward(ctx, grad_routed, grad_scores, grad_counts):  # pyrefly: ignore [bad-override]
        # Import only on the CANN execution path; configuration conversion needs no ElasticBuffer instance.
        from torchtitan_npu.ops.ascendc.deepep import _dispatch_backward

        if isinstance(grad_routed, MXTensor):
            grad_routed = _mxfp8_dequantize_tensor(grad_routed)
        # The existing BF16 combine also restores grad(topk_scores); quant_params has no gradient.
        grad_inputs = _dispatch_backward(ctx, grad_routed, grad_scores, grad_counts, None)
        grad_x, _, grad_topk_weights, _, _, _ = grad_inputs
        # qdata/scale/received-score/counts are communication outputs; their VJP is
        # consumed here, while the input router score receives the routed gradient.
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            grad_topk_weights,
            None,
            None,
            None,
        )


class _MXFP8DeepEPCombine(torch.autograd.Function):
    """Keep forward reduction in BF16/FP16; use a cached FP8 dispatch for its VJP."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx, activation, num_permuted_tokens_src, handle, num_tokens, buffer, quant_params
    ):
        if isinstance(activation, MXTensor):
            activation = _mxfp8_dequantize_tensor(activation)
        ctx.handle, ctx.buffer, ctx.quant_params = handle, buffer, quant_params
        # Only the dispatch-produced row count is needed in compiled backward.
        # Score multiplication and its saved tensors belong to the caller's graph.
        ctx.save_for_backward(num_permuted_tokens_src)
        return torch.ops.deepep.combine(activation, handle, num_tokens, num_permuted_tokens_src, buffer)

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        (num_permuted_tokens_src,) = ctx.saved_tensors
        qdata, scale = _quantize_payload(grad_output, ctx.quant_params)
        recv_qdata, recv_scale = torch.ops.deepep.dispatch_quantized_cached(
            qdata, scale, ctx.handle, num_permuted_tokens_src.shape[0], ctx.buffer
        )
        grad_input = _wire_tensor(
            recv_qdata, recv_scale, grad_output.dtype, ctx.quant_params, grad_output.requires_grad
        )
        # The row-count source is metadata, not a differentiable combine operand.
        return grad_input, None, None, None, None, None


_dispatcher_cache: dict[type, type] = {}


def get_mxfp8_deepep_dispatcher_cls(parent_cls: type) -> type:
    """Retain the selected CANN dispatcher's buffer, process group and metadata lifecycle."""
    from torchtitan_npu.override.common.token_dispatcher import AscDeepEPTokenDispatcher
    from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

    if parent_cls in _dispatcher_cache.values():
        return parent_cls
    if parent_cls is DeepEPTokenDispatcher:
        parent_cls = AscDeepEPTokenDispatcher
    if not issubclass(parent_cls, AscDeepEPTokenDispatcher):
        raise TypeError("MXFP8 DeepEP requires an AscDeepEPTokenDispatcher subclass")
    if parent_cls in _dispatcher_cache:
        return _dispatcher_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]

    class MXFP8DeepEPTokenDispatcher(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[valid-type, misc]
            mxfp8_quant_params: MXFP8QuantParams

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            if config.mxfp8_quant_params[0] != 32:
                raise ValueError("MXFP8 DeepEP requires block_size=32")
            self.mxfp8_quant_params = config.mxfp8_quant_params

        def _dispatch_activation_exchange(self, x_td, topk_expert_ids_tk, topk_scores_tk):
            # Transport produces detached payloads; the Function below owns the
            # logical activation/score VJP. Keep the original x and scores for apply.
            # This also avoids requiring autograd formulas on raw FP8/uint8 ops.
            with torch.no_grad():
                qdata, scale = _quantize_payload(x_td, self.mxfp8_quant_params)
                recv_payload, recv_scores, counts, handle = torch.ops.deepep.dispatch(
                    [qdata, scale],
                    topk_expert_ids_tk,
                    topk_scores_tk,
                    self.num_experts,
                    self.num_experts // self._deepep_world_size,
                    self._deepep_buffer_handle,
                )
                recv_qdata, recv_scale = recv_payload
            routed, recv_scores, counts = _MXFP8DeepEPDispatch.apply(
                x_td,
                recv_qdata,
                recv_scale,
                recv_scores,
                counts,
                topk_expert_ids_tk,
                topk_scores_tk,
                handle,
                self._deepep_buffer_handle,  # pyrefly: ignore [missing-attribute]
                self.mxfp8_quant_params,
            )
            return routed, recv_scores, counts, handle

        def _combine_activation_exchange(self, routed_output_rd, metadata, num_tokens):
            # The parent combine already applied scores using ordinary autograd.
            # MulBackward needs a plain tensor; experts with pre-W2 absorption
            # can consume the MXTensor gradient directly and reuse its FP8 data.
            if not self.absorb_router_scores:  # pyrefly: ignore [missing-attribute]
                routed_output_rd = _DequantizeGradient.apply(routed_output_rd)
            return _MXFP8DeepEPCombine.apply(
                routed_output_rd,
                metadata.recv_scores,
                metadata.handle,
                num_tokens,
                self._deepep_buffer_handle,  # pyrefly: ignore [missing-attribute]
                self.mxfp8_quant_params,
            )

    MXFP8DeepEPTokenDispatcher.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8DeepEPTokenDispatcher.__qualname__ = f"MXFP8{parent_cls.__qualname__}"
    _dispatcher_cache[parent_cls] = MXFP8DeepEPTokenDispatcher
    return MXFP8DeepEPTokenDispatcher
