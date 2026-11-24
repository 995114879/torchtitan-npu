# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-NPU complex RoPE checks with independent packed-row positions.

Requires one visible NPU and the CANN ``npu_rotary_mul`` operator. BF16 query/key
use shape [B=2, S=7, H=2, D=16]. The second position row resets independently;
the reference executes the workaround implementation separately for each row.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from torchtitan_npu.override.common.rope import AscComplexRoPE, WorkaroundComplexRoPE

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("inverse", [False, True])
def test_asc_complex_matches_row_wise_reference_forward_and_backward(npu_device, inverse):
    module = AscComplexRoPE(AscComplexRoPE.Config(dim=16, max_seq_len=16)).to(npu_device)
    reference = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=16, max_seq_len=16)).to(npu_device)
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 2, 0, 1]],
        device=npu_device,
    )
    generator = torch.Generator().manual_seed(42)
    inputs = [torch.randn(2, 7, 2, 16, generator=generator) for _ in range(4)]
    query, key, grad_q, grad_k = [tensor.to(device=npu_device, dtype=torch.bfloat16) for tensor in inputs]
    query.requires_grad_()
    key.requires_grad_()

    actual_q, actual_k = module(query, key, positions=positions, inverse=inverse)
    actual_grads = torch.autograd.grad((actual_q, actual_k), (query, key), (grad_q, grad_k))

    expected_q, expected_k, expected_grad_q, expected_grad_k = [], [], [], []
    for row in range(2):
        row_q = query.narrow(0, row, 1).detach().clone().requires_grad_(True)
        row_k = key.narrow(0, row, 1).detach().clone().requires_grad_(True)
        out_q, out_k = reference(row_q, row_k, positions=positions.narrow(0, row, 1), inverse=inverse)
        row_grads = torch.autograd.grad(
            (out_q, out_k),
            (row_q, row_k),
            (grad_q.narrow(0, row, 1), grad_k.narrow(0, row, 1)),
        )
        expected_q.append(out_q)
        expected_k.append(out_k)
        expected_grad_q.append(row_grads[0])
        expected_grad_k.append(row_grads[1])

    for actual, expected in zip(
        (actual_q, actual_k, *actual_grads),
        (torch.cat(expected_q), torch.cat(expected_k), torch.cat(expected_grad_q), torch.cat(expected_grad_k)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
