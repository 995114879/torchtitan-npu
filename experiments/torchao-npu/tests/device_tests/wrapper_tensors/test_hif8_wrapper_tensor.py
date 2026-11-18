# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the HiF8 parameter wrapper and its operation dispatch."""

import pytest
import torch
import torch.nn.functional as F
from torchao.float8.float8_utils import compute_error
from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig
from torchao_npu.wrapper_tensors import HiF8TrainingWeightWrapperTensor

_SQNR_THRESHOLD = 12.0


def _config() -> HiF8QuantizeConfig:
    return HiF8QuantizeConfig()


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda activation, weight, bias: torch.mm(activation, weight.T), id="mm"),
        pytest.param(lambda activation, weight, bias: torch.matmul(activation, weight.T), id="matmul"),
        pytest.param(lambda activation, weight, bias: F.linear(activation, weight, bias), id="linear"),
        pytest.param(lambda activation, weight, bias: torch.addmm(bias, activation, weight.T), id="addmm"),
        pytest.param(
            lambda activation, weight, bias: torch.addmm(bias, activation, weight.T, beta=2.0, alpha=0.5),
            id="addmm-scaled",
        ),
    ],
)
def test_mm_dispatch_forward_backward_matches_dense_reference(operation):
    config = _config()
    activation_data = torch.randn(32, 1024, device="npu", dtype=torch.bfloat16)
    weight_data = torch.randn(2048, 1024, device="npu", dtype=torch.bfloat16)
    bias_data = torch.randn(2048, device="npu", dtype=torch.bfloat16)

    activation = torch.nn.Parameter(activation_data.clone())
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(
            weight_data.clone(),
            weight_config=config,
            activation_config=config,
        )
    )
    bias = torch.nn.Parameter(bias_data.clone())
    actual = operation(activation, weight, bias)
    actual.sum().backward()

    activation_ref = torch.nn.Parameter(activation_data.clone())
    weight_ref = torch.nn.Parameter(weight_data.clone())
    bias_ref = torch.nn.Parameter(bias_data.clone())
    expected = operation(activation_ref, weight_ref, bias_ref)
    expected.sum().backward()

    assert actual.shape == expected.shape
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD


def test_bmm_dispatch_forward_backward_matches_dense_reference():
    config = _config()
    activation_data = torch.randn(8, 32, 1024, device="npu", dtype=torch.bfloat16)
    weight_data = torch.randn(8, 1024, 2048, device="npu", dtype=torch.bfloat16)

    activation = torch.nn.Parameter(activation_data.clone())
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(
            weight_data.clone(),
            weight_config=config,
            activation_config=config,
        )
    )
    actual = torch.bmm(activation, weight)
    actual.sum().backward()

    activation_ref = torch.nn.Parameter(activation_data.clone())
    weight_ref = torch.nn.Parameter(weight_data.clone())
    expected = torch.bmm(activation_ref, weight_ref)
    expected.sum().backward()

    assert actual.shape == expected.shape
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD


def test_grouped_mm_dispatch_matches_dense_reference():
    config = _config()
    num_tokens, num_experts, k, n = 16, 4, 1024, 2048
    activation = torch.randn(num_tokens, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight_data = torch.randn(num_experts, n, k, device="npu", dtype=torch.bfloat16)
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(weight_data, weight_config=config, activation_config=config)
    )
    offsets = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device="npu")

    actual = torch._grouped_mm(activation, weight.transpose(-2, -1), offs=offsets)
    actual.sum().backward()

    activation_ref = activation.detach().clone().requires_grad_()
    weight_ref = weight_data.detach().clone().requires_grad_()
    expected = torch._grouped_mm(activation_ref, weight_ref.transpose(-2, -1), offs=offsets)
    # Materialize the grad: ``sum()`` yields a zero-stride tensor that the NPU
    # grouped-mm backward rejects.
    expected.backward(torch.ones_like(expected))

    assert actual.shape == (num_tokens, n)
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD
