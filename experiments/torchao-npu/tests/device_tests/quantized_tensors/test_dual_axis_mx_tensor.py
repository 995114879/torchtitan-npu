# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for :class:`torchao_npu.quantized_tensors.dual_axis_mx_tensor.DualAxisMXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize_dual_axis
from torchao_npu.quantized_tensors.dual_axis_mx_tensor import DualAxisMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_from_hp_matches_mx_quantize_dual_axis(elem_dtype):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = DualAxisMXTensor.from_hp(x, config)
    qdata1, scale1, qdata2, scale2 = mx_quantize_dual_axis(x, config)

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"

    # the op packs both qdata along the input's last dim; FP8 has no packing at all
    expected_pack_axis = y.ndim - 1 if elem_dtype is torch.float4_e2m1fn_x2 else None
    assert y.pack_axis == expected_pack_axis, f"pack axis {y.pack_axis}, expected {expected_pack_axis}"

    assert y.qdata1.shape == qdata1.shape, (
        f"qdata1 shape {tuple(y.qdata1.shape)} != the primitive's {tuple(qdata1.shape)}"
    )
    assert torch.equal(y.qdata1.view(torch.uint8), qdata1.view(torch.uint8)), (
        "qdata1 values differ from the primitive's"
    )

    assert y.scale1.shape == scale1.shape, (
        f"scale1 shape {tuple(y.scale1.shape)} != the primitive's {tuple(scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), scale1.view(torch.uint8)), (
        "scale1 values differ from the primitive's"
    )

    assert y.qdata2.shape == qdata2.shape, (
        f"qdata2 shape {tuple(y.qdata2.shape)} != the primitive's {tuple(qdata2.shape)}"
    )
    assert torch.equal(y.qdata2.view(torch.uint8), qdata2.view(torch.uint8)), (
        "qdata2 values differ from the primitive's"
    )

    assert y.scale2.shape == scale2.shape, (
        f"scale2 shape {tuple(y.scale2.shape)} != the primitive's {tuple(scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), scale2.view(torch.uint8)), (
        "scale2 values differ from the primitive's"
    )


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)

    y = DualAxisMXTensor.from_hp(x, config)

    assert y.requires_grad, "from_hp dropped the input's requires_grad"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_to_mx_tensor_equals_single_axis_from_hp(elem_dtype):
    """The pair stored for one quant axis must match a single-axis quantization."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = DualAxisMXTensor.from_hp(x, config).to_mx_tensor(-2)
    y_ref = MXTensor.from_hp(x, config, axis=-2)

    assert y.pack_axis == y_ref.pack_axis, (
        f"pack axis {y.pack_axis} != the single-axis quantization's {y_ref.pack_axis}"
    )

    assert y.qdata.shape == y_ref.qdata.shape, (
        f"qdata shape {tuple(y.qdata.shape)} != the single-axis quantization's {tuple(y_ref.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), y_ref.qdata.view(torch.uint8)), (
        "qdata values differ from the single-axis quantization's"
    )

    assert y.scale.shape == y_ref.scale.shape, (
        f"scale shape {tuple(y.scale.shape)} != the single-axis quantization's {tuple(y_ref.scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), y_ref.scale.view(torch.uint8)), (
        "scale values differ from the single-axis quantization's"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_uses_the_last_dim_pair(elem_dtype):
    """For a freshly quantized tensor the innermost-contiguous quant axis is the last dim.

    The pair that covers it, ``qdata1``/``scale1``, is the one whose dequantization the result
    must match; using the other pair would give different values, so this pins the choice.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = DualAxisMXTensor.from_hp(tensor, config)

    dequantized = x.dequantize()
    expected = mx_dequantize(
        x.qdata1,
        x.scale1,
        x.qdata1.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == expected.stride(), f"stride mismatch: {dequantized.stride()} vs {expected.stride()}"
    assert torch.equal(dequantized, expected), "the result is not the last-dim pair's"


def test_dequantize_after_a_swap_dequantizes_the_same_axis():
    """A swap moves the two pairs, and with them the trailing dim each one covers.

    ``t()`` makes the unit-stride dim second-to-last, so ``dequantize`` has to fall through to
    the second pair -- which now covers the last dim, the axis the first pair covered before the
    swap. The values are therefore the same quantization, transposed.
    """
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = DualAxisMXTensor.from_hp(tensor, config)

    y = x.t()
    dequantized = y.dequantize()
    expected = mx_dequantize(
        x.qdata1,
        x.scale1,
        x.qdata1.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == y.shape, f"shape mismatch: {dequantized.shape} vs {y.shape}"
    assert dequantized.stride() == expected.t().stride(), (
        f"stride mismatch: {dequantized.stride()} vs {expected.t().stride()}"
    )
    assert torch.equal(dequantized, expected.t()), "the swapped tensor dequantized the wrong pair"
