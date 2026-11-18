# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for :class:`torchao_npu.quantized_tensors.mx_tensor.MXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.ops.mx_ops import to_mx_then_bmm, to_mx_then_grouped_mm, to_mx_then_mm
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_fake_quantize
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


def test_slice_leading_dim_slices_qdata_and_scale():
    config = MXQuantizeConfig()
    x = MXTensor.from_hp(torch.randn(4, 64, device="npu", dtype=torch.bfloat16), config, axis=-1)

    y = x[1:3]

    assert torch.equal(y.qdata.view(torch.uint8), x.qdata[1:3].view(torch.uint8))
    assert torch.equal(y.scale, x.scale[1:3])


def test_slice_keyword_dim_rejects_quant_axis():
    config = MXQuantizeConfig()
    x = MXTensor.from_hp(torch.randn(4, 64, device="npu", dtype=torch.bfloat16), config, axis=-1)

    with pytest.raises(ValueError, match="quantization blocks"):
        torch.ops.aten.slice.Tensor(x, dim=1, start=0, end=32)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_mm_matches_to_mx_then_mm(elem_dtype):
    """A pre-quantized weight must give what the training path computes.

    ``to_mx_then_mm`` quantizes both operands itself; the MXTensor path reuses
    the weight's stored quantization. Matching them is the train/inference
    consistency this class exists for.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.dtype is torch.bfloat16
    assert torch.equal(y, y_ref)


def test_bmm_matches_to_mx_then_bmm():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_bmm(a, w.transpose(-2, -1), config, config)

    assert torch.equal(y, y_ref)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_out_dtype_is_honored(elem_dtype, out_dtype):
    """``torch.mm(..., out_dtype=...)`` takes the ``aten.mm.dtype`` overload into the kernel."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1), out_dtype=out_dtype)
    y_default = torch.mm(a, weight.transpose(-2, -1))

    assert y.dtype is out_dtype, f"out_dtype {out_dtype} not honored: got {y.dtype}"
    if out_dtype is y_default.dtype:
        assert torch.equal(y, y_default), "asking for the dtype the product already has changed it"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_bmm_out_dtype_is_honored(elem_dtype):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_addmm_out_dtype_is_honored(elem_dtype):
    """``out_dtype`` is the fourth positional argument of the ``aten.addmm.dtype`` overload."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize(("beta", "alpha"), [(2, 1), (1, 3), (2, 3)])
def test_addmm_scales_the_bias_and_the_product(beta, alpha):
    """``beta``/``alpha`` scale the kernel's output: the fused epilogue has no scaling terms."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), beta=beta, alpha=alpha)
    y_ref = alpha * to_mx_then_mm(a, w.transpose(-2, -1), config, config) + beta * bias

    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "scaled addmm differs from the scaled training path"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_grouped_mm_matches_to_mx_then_grouped_mm(elem_dtype):
    """A pre-quantized expert weight must give what the training path computes."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    m, k, n, experts = 64, 128, 64, 4
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(experts, n, k, device="npu", dtype=torch.bfloat16)  # stored [E, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    offs = torch.tensor([8, 32, 48, 64], device="npu", dtype=torch.int32)  # uneven groups

    y = torch._grouped_mm(a, weight.transpose(-2, -1), offs=offs)
    y_ref = to_mx_then_grouped_mm(a, w.transpose(-2, -1), offs, config, config)

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert torch.equal(y, y_ref), "grouped matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_matches_to_mx_then_mm(elem_dtype, shape):
    """``F.linear`` takes the stored ``[N, K]`` weight; that must give the training path's result.

    A 3D activation goes through the path ``F.linear`` itself flattens, so the shape it restores
    is checked along with the values. Every trailing dim is a multiple of 32, as the training
    path's dual-axis quantization of either operand requires.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.nn.functional.linear(a, weight)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "linear differs from the training path's"


@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_with_bias_matches_to_mx_then_mm(shape):
    """With a bias, ``F.linear`` routes through ``addmm``, which fuses the bias into the kernel."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(n, device="npu", dtype=torch.float32)

    y = torch.nn.functional.linear(a, weight, bias)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config) + bias

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "linear with a bias differs from the training path's plus the bias"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.linear`` handler itself, which ``F.linear`` never reaches.

    ``aten::linear`` is a composite op, decomposed into ``t``/``addmm`` before dispatch, so the
    handler runs only when something dispatches ``linear`` itself -- a backend registering it,
    as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way. The
    activation flattens to a multiple of 32 rows for the training path's sake.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.linear.default, (MXTensor,), (a, weight, None), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the linear handler differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_matches_to_mx_then_mm(elem_dtype, shape):
    """``torch.matmul`` against a stored ``[N, K]`` weight must give the training path's result.

    A 3D activation is folded into the rows by the composite ``matmul`` before its ``mm``, so the
    shape it restores is checked along with the values.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.matmul(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.matmul`` handler itself, which ``torch.matmul`` reaches only through its leaves.

    ``aten::matmul`` is a composite op, decomposed into ``mm``/``bmm``/``expand`` before dispatch,
    so the handler runs only when something dispatches ``matmul`` itself -- a backend registering
    it, as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.matmul.default, (MXTensor,), (a, weight.transpose(-2, -1)), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the matmul handler differs from the training path's"


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)

    x = MXTensor.from_hp(tensor, config, axis=-1)

    assert x.requires_grad, "from_hp dropped the input's requires_grad"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_matches_mx_fake_quantize(elem_dtype):
    """``dequantize`` undoes ``from_hp``'s quantization, reproducing ``mx_fake_quantize``.

    ``mx_fake_quantize`` quantizes and dequantizes the permuted tensor in one op, so for a
    tensor both permute the same way they see the same ``qdata``/``scale`` and must agree bit
    for bit.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    dequantized = x.dequantize()
    fake_quantized = mx_fake_quantize(tensor, -1, config)

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == fake_quantized.stride(), (
        f"stride mismatch: {dequantized.stride()} vs {fake_quantized.stride()}"
    )
    assert torch.equal(dequantized, fake_quantized), "dequantized values differ from mx_fake_quantize"


def test_dequantize_honors_output_dtype():
    """Every MX value fits in bfloat16, so both output dtypes agree bit for bit."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    as_bfloat16 = x.dequantize()
    as_float32 = x.dequantize(torch.float32)

    assert as_bfloat16.dtype is torch.bfloat16, f"dtype {as_bfloat16.dtype} != torch.bfloat16"
    assert as_float32.dtype is torch.float32, f"dtype {as_float32.dtype} != torch.float32"
    assert as_float32.stride() == as_bfloat16.stride(), (
        f"stride mismatch: {as_float32.stride()} vs {as_bfloat16.stride()}"
    )
    assert torch.equal(as_float32, as_bfloat16.to(torch.float32)), "the two output dtypes disagree"


def test_dequantize_rejects_a_strided_quant_axis():
    """A pass-through quantization keeps the quant axis strided, which cannot be dequantized."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(4, 256, 128, device="npu", dtype=torch.bfloat16)

    x = MXTensor.from_hp(tensor, config, axis=1)  # dense, non-trailing axis: no permutation
    assert x.qdata.stride(x.quant_axis) != 1, "the pass-through route should keep the quant axis strided"

    with pytest.raises(AssertionError, match="unit stride"):
        x.dequantize()


def test_to_block_mx_matches_block_mx_quantize_of_the_same_tensor():
    """``from_hp`` + ``to_block_mx`` must reproduce the block-MX quantization of the same tensor.

    The two paths derive the kernel layout differently (``mx_quantize``'s stride rule vs
    ``_to_block_quant_layout``), so equal components pin the permutation and its inverse.
    """
    fp4_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    block_config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=fp4_config)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    converted = MXTensor.from_hp(tensor, fp4_config, axis=-1).to_block_mx(block_config)
    reference = BlockMXTensor.from_hp(tensor, block_config, axis=-1)

    assert converted.pack_axis is None, f"an FP8 result gained pack_axis {converted.pack_axis}"
    for name in ("qdata", "scale1", "scale2"):
        actual = getattr(converted, name)
        expected = getattr(reference, name)
        assert actual.shape == expected.shape, f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"{name} values differ from the directly quantized block MX tensor"
        )
