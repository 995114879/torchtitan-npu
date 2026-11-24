# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.models.common.rope import ComplexRoPE

import torchtitan_npu.override.common.rope as fused
from torchtitan_npu.override.common.rope import AscComplexRoPE, AscHalfRotation


def test_fused_half_adapter_preserves_fp32_tables_and_batches(monkeypatch):
    import torchtitan_npu.override.common.rope as fused

    calls = []

    def kernel(x, cos, sin, *, rotary_mode):
        calls.append((x, cos, sin, rotary_mode))
        return x + 2

    monkeypatch.setattr(fused.torch_npu, "npu_rotary_mul", kernel)
    x = torch.ones(2, 3, 1, 4, dtype=torch.bfloat16)
    cos = torch.linspace(0.1, 0.9, 2 * 3 * 2).reshape(2, 3, 1, 2)
    sin = cos.flip(1)
    actual = AscHalfRotation.Config().build()(x, cos, sin, inverse=True)
    value, sent_cos, sent_sin, rotary_mode = calls[0]
    assert value.dtype == sent_cos.dtype == sent_sin.dtype == torch.float32
    assert sent_cos.shape[0] == 1
    expected_cos = torch.cat((cos, cos), -1)
    expected_sin = -torch.cat((sin, sin), -1)
    # Distinct per-image tables fold batch into the shared axis.
    expected_cos = expected_cos.flatten(0, 1).unsqueeze(0)
    expected_sin = expected_sin.flatten(0, 1).unsqueeze(0)
    torch.testing.assert_close(sent_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sent_sin, expected_sin, rtol=0, atol=0)
    assert rotary_mode == "half"
    torch.testing.assert_close(actual, torch.full_like(x, 3), rtol=0, atol=0)


def test_fused_complex_adapter_preserves_distinct_row_positions_and_gradients(monkeypatch):
    calls = []

    def kernel(x, cos, sin, *, rotary_mode):
        calls.append((x, cos, sin, rotary_mode))
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    monkeypatch.setattr(fused.torch_npu, "npu_rotary_mul", kernel)
    config = ComplexRoPE.Config(dim=8, max_seq_len=16)
    reference = ComplexRoPE(config)
    module = AscComplexRoPE(AscComplexRoPE.Config(dim=8, max_seq_len=16))
    positions = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 0, 0, 1]])
    generator = torch.Generator().manual_seed(1042)
    query = torch.randn(2, 5, 2, 8, generator=generator)
    key = torch.randn(2, 5, 2, 8, generator=generator)
    query_fused, query_ref = (value.detach().clone().requires_grad_(True) for value in (query, query))
    key_fused, key_ref = (value.detach().clone().requires_grad_(True) for value in (key, key))

    actual_q, actual_k = module(query_fused, key_fused, positions=positions)
    expected_q, expected_k = reference(query_ref, key_ref, positions=positions)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-6, atol=1e-6)
    grad_q = torch.randn(actual_q.shape, generator=generator)
    grad_k = torch.randn(actual_k.shape, generator=generator)
    actual_grads = torch.autograd.grad((actual_q, actual_k), (query_fused, key_fused), (grad_q, grad_k))
    expected_grads = torch.autograd.grad((expected_q, expected_k), (query_ref, key_ref), (grad_q, grad_k))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(calls) == 2
    for value, cos, sin, rotary_mode in calls:
        assert value.shape[0] == cos.shape[0] == sin.shape[0] == 1
        assert value.shape[1] == cos.shape[1] == sin.shape[1] == positions.numel()
        assert rotary_mode == "interleave"
