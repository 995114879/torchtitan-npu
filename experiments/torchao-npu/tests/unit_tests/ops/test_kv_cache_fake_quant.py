# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("shape", [(2, 3, 32), (6, 1, 32)], ids=["compressor", "smla"])
def test_kv_fake_quantize_decodes_packed_cache_and_preserves_ste(monkeypatch, shape, cpu_custom_ops):
    from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16

    x = torch.linspace(-1, 1, 192, dtype=torch.bfloat16).reshape(shape).requires_grad_()
    seen = []

    def packed_fp4_encoder(cache, rows, slots, *, quant_group_size, quant_mode, round_scale):
        seen.append(rows.detach().clone())
        assert cache.dtype == torch.uint8
        assert cache.shape == (6, 32)
        assert torch.equal(slots, torch.arange(6, dtype=torch.int32))
        assert (quant_group_size, quant_mode, round_scale) == (16, "mxfp4_bf16", False)
        # E2M1 low-first pair [0.5, 1.0], with two distinct BF16 group scales.
        cache[:, :16] = 0x21
        cache[:, 16:20] = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16).view(torch.uint8)

    monkeypatch.setattr(
        torch.ops.custom,
        "kv_compress_epilog_v2",
        SimpleNamespace(default=packed_fp4_encoder),
        raising=False,
    )

    result = fake_quantize_mx_bf16(x)

    expected = torch.tensor([0.25, 0.5] * 8 + [1.0, 2.0] * 8, dtype=x.dtype).repeat(6).reshape(shape)
    assert len(seen) == 1
    assert torch.equal(seen[0], x.detach().reshape(6, 32))
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.equal(result, expected)
    grad = torch.linspace(-0.5, 0.5, x.numel(), dtype=x.dtype).reshape_as(x)
    result.backward(grad)
    assert torch.equal(x.grad, grad)
