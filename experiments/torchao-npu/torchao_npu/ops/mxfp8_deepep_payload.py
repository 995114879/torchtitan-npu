# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 DeepEP payload helpers.

Transport-agnostic-to-torchtitan primitives for the CANN ElasticBuffer DeepEP
path: the payload layout adapter, the wire-tensor builder and the score-boundary
dequantizer. Binding them to ``torchtitan_npu``'s DeepEP dispatcher, ops and
override lives in
``torchtitan_npu.extensions.adaptor.torchao.deepep_token_dispatcher`` so this
distribution keeps no reverse dependency on ``torchtitan_npu``.
"""

from __future__ import annotations

__all__ = ["_DequantizeGradient", "_quantize_payload", "_wire_tensor"]

import torch

from torchao_npu.ops.mxfp8_all_to_all import (
    MXFP8QuantParams,
    _mxfp8_quantize_or_empty,
    _mxfp8_wire_tensor,
)
from torchao_npu.quantized_tensors.mx_tensor import MXTensor, _mxfp8_dequantize_tensor


def _quantize_payload(activation: torch.Tensor, quant_params: MXFP8QuantParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Standard's quantizer, then adapt its scale layout to CANN's ABI."""
    if activation.ndim != 2 or activation.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("MXFP8 DeepEP expects a 2D BF16/FP16/FP32 activation")
    # CANN infers the number of MX scales from hidden / 32; there is no block-size argument.
    if quant_params[0] != 32:
        raise ValueError("MXFP8 DeepEP requires block_size=32 for ElasticBuffer's MX scale layout")
    if isinstance(activation, MXTensor):
        activation = _mxfp8_dequantize_tensor(activation)
    qdata, scale = _mxfp8_quantize_or_empty(activation, *quant_params)
    # A byte view preserves E8M0 bits. CANN's C++ adapter tags uint8 scales as ACL_FLOAT8_E8M0.
    # Explicit dimensions also work for zero rows; reshape(0, -1) would be ambiguous.
    return qdata, scale.view(torch.uint8).reshape(scale.shape[0], scale.shape[1] * 2)


def _wire_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    quant_params: MXFP8QuantParams,
    requires_grad: bool,
) -> MXTensor:
    # ElasticBuffer returns expert-major rows for BOTH payloads, so no local reroute is needed.
    if qdata.ndim != 2 or scale.ndim != 2:
        raise ValueError("DeepEP MXFP8 payloads must be rank-2 qdata and scale tensors")
    if qdata.shape[0] != scale.shape[0]:
        raise ValueError(
            f"DeepEP MXFP8 payload row mismatch: qdata has {qdata.shape[0]} rows, scale has {scale.shape[0]} rows"
        )
    if scale.shape[1] % 2 != 0:
        raise ValueError(f"DeepEP MXFP8 packed scale width must be even, got {scale.shape[1]}")
    scale = scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)
    return _mxfp8_wire_tensor(qdata, scale, dtype, quant_params, requires_grad)  # pyrefly: ignore [not-callable]


class _DequantizeGradient(torch.autograd.Function):
    """Let ordinary autograd consume a gradient returned as an MXTensor."""

    @staticmethod
    def forward(ctx, tensor):  # pyrefly: ignore [bad-override]
        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        # This boundary changes representation only; MulBackward owns both
        # the expert-output gradient and the routing-score gradient.
        if isinstance(grad_output, MXTensor):
            return _mxfp8_dequantize_tensor(grad_output)
        return grad_output
