# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the HiF8 parameter wrapper and its operation dispatch."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase
from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig
from torchao_npu.wrapper_tensors import HiF8TrainingWeightWrapperTensor


def _config() -> HiF8QuantizeConfig:
    return HiF8QuantizeConfig()


def _wrapper(shape, device="cpu", *, dtype=torch.float32):
    config = _config()
    return HiF8TrainingWeightWrapperTensor(
        torch.randn(*shape, device=device, dtype=dtype),
        weight_config=config,
        activation_config=config,
    )


def test_init_rejects_missing_or_wrong_config():
    data = torch.randn(64, 128)
    config = _config()

    with pytest.raises(ValueError, match=r"^`weight_config` is required"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=None, activation_config=config)
    with pytest.raises(ValueError, match=r"^`activation_config` is required"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=config, activation_config=None)

    class DummyConfig(FakeQuantizeConfigBase):
        pass

    with pytest.raises(ValueError, match=r"^Only `HiF8QuantizeConfig` is supported"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=DummyConfig(), activation_config=config)
    with pytest.raises(ValueError, match=r"^Only `HiF8QuantizeConfig` is supported"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=config, activation_config=DummyConfig())


def test_init_rejects_non_pertensor_quant_mode():
    """HiF8QuantizeConfig also serves Lightning Indexer Q/K quantization,
    where quant_mode may be non-"pertensor" -- this wrapper's per-expert
    scale broadcast and backward-reuse-via-transpose scheme only hold for a
    single scalar scale, so a non-"pertensor" config must be rejected here
    rather than silently mishandled.
    """
    data = torch.randn(64, 128)
    pertoken_config = HiF8QuantizeConfig(quant_mode="pertoken")
    pertensor_config = _config()

    with pytest.raises(ValueError, match=r"weight_config.quant_mode.*must be 'pertensor'"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=pertoken_config, activation_config=pertoken_config)
    with pytest.raises(ValueError, match=r"activation_config.quant_mode.*must be 'pertensor'"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=pertensor_config, activation_config=pertoken_config)


@pytest.mark.parametrize("operation", ["select", "transpose", "detach", "clone"])
def test_metadata_operations_preserve_wrapper(operation):
    weight = _wrapper((4, 64, 128))

    with torch._C.DisableTorchFunctionSubclass():
        if operation == "select":
            actual = weight[0]
            expected = weight._data[0]
        elif operation == "transpose":
            actual = weight.transpose(0, 1)
            expected = weight._data.transpose(0, 1)
        elif operation == "detach":
            actual = weight.detach()
            expected = weight._data.detach()
        else:
            actual = weight.clone()
            expected = weight._data.clone()

    assert isinstance(actual, HiF8TrainingWeightWrapperTensor)
    assert actual.weight_config is weight.weight_config
    assert actual.activation_config is weight.activation_config
    torch.testing.assert_close(actual._data, expected)


@pytest.mark.parametrize("operation", [torch.mm, F.linear])
def test_dispatch_rejects_wrapped_activation(operation):
    weight = _wrapper((64, 128))

    with pytest.raises(AssertionError, match=r"^A should not be a HiF8TrainingWeightWrapperTensor$"):
        operation(weight, weight)


@pytest.mark.parametrize("operation", [torch.addmm, F.linear])
def test_dispatch_rejects_unwrapped_weight(operation):
    activation = torch.randn(16, 64)
    weight = torch.randn(64, 128)
    wrapped_bias = _wrapper((128,))

    with pytest.raises(AssertionError, match=r"^B should be a HiF8TrainingWeightWrapperTensor$"):
        if operation is torch.addmm:
            operation(wrapped_bias, activation, weight)
        else:
            operation(activation, weight, wrapped_bias)


def test_init_stores_data_and_configs():
    weight_config = _config()
    activation_config = _config()
    data = torch.randn(64, 128, device="cpu")

    wrapper = HiF8TrainingWeightWrapperTensor(
        data,
        weight_config=weight_config,
        activation_config=activation_config,
    )

    assert wrapper._data is data
    assert wrapper._data.data_ptr() == data.data_ptr()
    assert torch.equal(wrapper._data, data)
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is activation_config


def test_non_quantized_operation_preserves_plain_tensor_result():
    plain = torch.randn(64, 128, device="cpu")
    wrapped = _wrapper((64, 128), "cpu")

    actual = torch.add(plain, wrapped)
    expected = torch.add(plain, wrapped._data)

    assert isinstance(actual, torch.Tensor)
    assert not isinstance(actual, HiF8TrainingWeightWrapperTensor)
    torch.testing.assert_close(actual, expected)


def test_deepcopy_preserves_wrapper_contract():
    wrapper = _wrapper((64, 128), "cpu", dtype=torch.bfloat16)
    copied = copy.deepcopy(wrapper)

    assert isinstance(copied, HiF8TrainingWeightWrapperTensor)
    assert copied._data is not wrapper._data
    assert copied.weight_config == wrapper.weight_config
    assert copied.activation_config == wrapper.activation_config
    torch.testing.assert_close(copied._data, wrapper._data)
