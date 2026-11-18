# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for :class:`torchao_npu.quantized_tensors.block_mx_tensor.BlockMXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor


def test_from_hp_matches_block_mx_quantize():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = BlockMXTensor.from_hp(x, config)
    qdata, scale1, scale2 = block_mx_quantize(x, config)

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"
    assert y.pack_axis is None, f"FP8 needs no pack axis, got {y.pack_axis}"

    assert y.qdata.shape == qdata.shape, f"qdata shape {tuple(y.qdata.shape)} != the primitive's {tuple(qdata.shape)}"
    assert torch.equal(y.qdata.view(torch.uint8), qdata.view(torch.uint8)), "qdata values differ from the primitive's"

    assert y.scale1.shape == scale1.shape, (
        f"scale1 shape {tuple(y.scale1.shape)} != the primitive's {tuple(scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), scale1.view(torch.uint8)), (
        "scale1 values differ from the primitive's"
    )

    assert y.scale2.shape == scale2.shape, (
        f"scale2 shape {tuple(y.scale2.shape)} != the primitive's {tuple(scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), scale2.view(torch.uint8)), (
        "scale2 values differ from the primitive's"
    )


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)

    y = BlockMXTensor.from_hp(x, config)

    assert y.requires_grad, "from_hp dropped the input's requires_grad"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_uses_the_last_dim_scale(elem_dtype):
    """For a freshly quantized tensor the innermost-contiguous quant axis is the last dim.

    Its scale, ``scale1``, is the one whose dequantization the result must match; using the
    other scale would give different values, so this pins the choice.
    """
    config = BlockMXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = BlockMXTensor.from_hp(tensor, config)

    dequantized = x.dequantize()
    expected = mx_dequantize(
        x.qdata,
        x.scale1,
        x.qdata.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == expected.stride(), f"stride mismatch: {dequantized.stride()} vs {expected.stride()}"
    assert torch.equal(dequantized, expected), "the result is not the last-dim scale's"


def test_dequantize_after_a_swap_dequantizes_the_same_axis():
    """A swap exchanges the two scales, and with them the trailing dim each one covers.

    ``t()`` makes the unit-stride dim second-to-last, so ``dequantize`` has to fall through to
    the second scale -- which now covers the last dim, the axis ``scale1`` covered before the
    swap. The values are therefore the same quantization, transposed.
    """
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = BlockMXTensor.from_hp(tensor, config)

    y = x.t()
    dequantized = y.dequantize()
    expected = mx_dequantize(
        x.qdata,
        x.scale1,
        x.qdata.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == y.shape, f"shape mismatch: {dequantized.shape} vs {y.shape}"
    assert dequantized.stride() == expected.t().stride(), (
        f"stride mismatch: {dequantized.stride()} vs {expected.t().stride()}"
    )
    assert torch.equal(dequantized, expected.t()), "the swapped tensor dequantized the wrong scale"
