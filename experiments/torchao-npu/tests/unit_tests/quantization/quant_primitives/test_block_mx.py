# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import mxfp4_to_block_mx


def _stub_fused_op(monkeypatch, seen):
    def fake(fp4, mxscale, *, dst_type=None, x_type=None):
        seen.append((fp4, mxscale))
        kernel_shape = (*fp4.shape[:-1], fp4.shape[-1] * 2)
        return (
            torch.zeros(kernel_shape, dtype=torch.float8_e4m3fn),
            torch.zeros(*kernel_shape[:-1], 1, 2, dtype=torch.float8_e8m0fnu),
            torch.zeros(*kernel_shape[:-2], 1, kernel_shape[-1], 2, dtype=torch.float8_e8m0fnu),
        )

    # ``raising=False``: the op is registered by the CANN package on NPU machines
    # but absent from a bare CPU environment running this test in isolation.
    monkeypatch.setattr(torch.ops.cann_ops_nn, "mx_to_block_mx_quant", fake, raising=False)


@pytest.mark.parametrize(
    "shape, quant_axis, kernel_perm",
    [
        ((64, 128), -1, [0, 1]),
        ((64, 128), 0, [1, 0]),
        ((2, 64, 128), -2, [0, 2, 1]),
    ],
    ids=["2d-last-dim", "2d-first-dim", "3d-second-last"],
)
def test_mxfp4_to_block_mx_permutes_the_logical_payload(monkeypatch, shape, quant_axis, kernel_perm):
    """The logical layout is permuted to the kernel layout before the fused op and restored after."""
    fq_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=fq_config)
    qdata = torch.randint(0, 256, shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    seen = []
    _stub_fused_op(monkeypatch, seen)

    q, _, _ = mxfp4_to_block_mx(qdata, scale, fq_config, config, quant_axis=quant_axis)

    kernel_fp4, kernel_scale = seen[0]
    expected_fp4 = qdata.permute(kernel_perm)
    assert kernel_fp4.data_ptr() == expected_fp4.data_ptr(), "the kernel must see the same storage"
    assert kernel_fp4.stride() == expected_fp4.stride(), (
        f"kernel-layout strides {kernel_fp4.stride()} != the expected {expected_fp4.stride()}"
    )
    expected_scale = scale.permute([*kernel_perm, scale.ndim - 1])
    assert kernel_scale.stride() == expected_scale.stride(), (
        f"kernel-layout scale strides {kernel_scale.stride()} != the expected {expected_scale.stride()}"
    )

    logical_shape = list(qdata.shape)
    logical_shape[quant_axis % qdata.ndim] *= 2
    assert q.shape == torch.Size(logical_shape), f"q shape {tuple(q.shape)} != the logical {logical_shape}"


def test_mxfp4_to_block_mx_forwards_a_kernel_layout_payload(monkeypatch):
    """``quant_axis=None`` forwards the components unchanged: they are already in the kernel layout."""
    fq_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=fq_config)
    qdata = torch.randint(0, 256, (64, 64), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    seen = []
    _stub_fused_op(monkeypatch, seen)

    q, _, _ = mxfp4_to_block_mx(qdata, scale, fq_config, config)

    kernel_fp4, kernel_scale = seen[0]
    assert kernel_fp4.data_ptr() == qdata.data_ptr(), "the kernel-layout payload must be forwarded"
    assert kernel_fp4.stride() == qdata.stride(), f"forwarded strides {kernel_fp4.stride()} != {qdata.stride()}"
    assert kernel_scale.stride() == scale.stride(), (
        f"forwarded scale strides {kernel_scale.stride()} != {scale.stride()}"
    )
    assert q.shape == torch.Size((64, 128)), f"q shape {tuple(q.shape)} != the logical (64, 128)"
