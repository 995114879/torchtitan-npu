# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16


@pytest.mark.parametrize(
    ("quant_mode", "group_size"),
    [("mxfp4_bf16", 16), ("mxfp8_bf16", 32)],
    ids=["main-kv-fp4", "swa-fp8"],
)
def test_kv_fake_quantize_real_epilog_round_trip_and_backward(quant_mode, group_size):
    x = torch.tensor([0.5, 1.0, 2.0], device="npu", dtype=torch.bfloat16)[:, None].repeat(1, 512)
    x.requires_grad_()

    result = fake_quantize_mx_bf16(x, quant_group_size=group_size, quant_mode=quant_mode)

    # FP8 round_scale=False uses amax / 448 stored as BF16; FP4 powers of two
    # are exact. Account for BF16 rounding of FP8's stored group scale.
    expected = (
        x.detach() if quant_mode == "mxfp4_bf16" else (x.detach().float() / 448).bfloat16().float().mul(448).bfloat16()
    )
    assert result.dtype == x.dtype
    assert torch.equal(result, expected)
    upstream = torch.linspace(-1, 1, x.numel(), device=x.device, dtype=x.dtype).reshape_as(x)
    result.backward(upstream)
    assert torch.equal(x.grad, upstream)
