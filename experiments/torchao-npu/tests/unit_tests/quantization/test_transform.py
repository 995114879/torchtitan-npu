# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from _support.reference_moe import MoE, MoEArgs
from _support.testing_utils import _expert_weight_filter
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.filters import (
    _is_expert,
    _is_parameter,
    _is_parameter_with_wrapped_data,
)
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import (
    _replace_params_with_custom_fn_if_matches_filter,
    unwrap_param,
)
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
    Float8TrainingWeightWrapperTensor,
)


def _create_cpu_moe_model():
    """Keep the real parameter hierarchy small; prepare never runs the MoE forward."""
    args = MoEArgs(num_experts=4, num_shared_experts=2, use_grouped_mm=True, load_balance_coeff=None)
    model = MoE(args, dim=32, hidden_dim=64)
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param)
    return model


def test_replace_params_filters_and_replaces():
    """_replace_params_with_custom_fn_if_matches_filter: filter selects params, replacement is applied."""

    recorded = []

    def my_filter(param, fqn):
        return "weight" in fqn  # match only weight, not bias

    def my_replacement(module, param_fqn, param, extra_args):
        recorded.append((param_fqn, extra_args))
        return torch.nn.Parameter(param.data.clone() * 2)

    model = torch.nn.Linear(2, 2)  # has "weight" and "bias"
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()
    _replace_params_with_custom_fn_if_matches_filter(model, my_replacement, my_filter, extra_args=(42,))

    # Only weight matched, bias skipped
    assert len(recorded) == 1, f"Expected 1 match, got {len(recorded)}"
    assert recorded[0][0] == "weight"
    assert recorded[0][1] == (42,)

    # Weight was replaced with 2x original value, bias was unchanged
    assert torch.equal(model.weight.data, original_weight * 2)
    assert torch.equal(model.bias.data, original_bias)


def test_replace_params_default_filter():
    """Default filter (None) uses _is_parameter, which wraps all nn.Parameters."""

    called = []

    def replacement(module, param_fqn, param, extra_args):
        called.append(param_fqn)
        return param  # no change

    model = torch.nn.Linear(2, 2)
    _replace_params_with_custom_fn_if_matches_filter(model, replacement, None)

    assert len(called) == 2, f"Expected weight + bias, got {len(called)}"
    assert any("weight" in f for f in called)
    assert any("bias" in f for f in called)


def test_replace_params_recursive():
    """_replace_params_with_custom_fn_if_matches_filter recurses into submodules."""

    called = []

    def my_filter(param, fqn):
        return True

    def my_replacement(module, param_fqn, param, extra_args):
        called.append(param_fqn)
        return param

    inner = torch.nn.Linear(2, 2)
    outer = torch.nn.Sequential(inner)
    _replace_params_with_custom_fn_if_matches_filter(outer, my_replacement, my_filter)

    # inner.weight, inner.bias -- cur_fqn should include the outer prefix
    assert any("0.weight" in f for f in called)
    assert any("0.bias" in f for f in called)


def test_convert_with_unsupported_wrapper_raises_before_the_quant_kernel():
    """Wrappers without a to_inference_weight (e.g. Float8) are rejected before any NPU quant kernel runs."""
    cfg = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    wrapped = Float8TrainingWeightWrapperTensor(
        torch.randn(4, 32, dtype=torch.bfloat16),
        weight_config=cfg,
    )
    param = torch.nn.Parameter(wrapped, requires_grad=False)

    with pytest.raises(NotImplementedError, match="to_inference_weight"):
        unwrap_param(torch.nn.Linear(32, 4), "weight", param)


def test_is_expert_filter():
    """_is_expert returns True for module FQNs ending with 'experts' or 'shared_experts'."""

    class DummyModule(torch.nn.Module):
        pass

    assert _is_expert(DummyModule(), "model.layers.0.experts")
    assert _is_expert(DummyModule(), "shared_experts")
    assert not _is_expert(DummyModule(), "model.layers.0.router")
    assert not _is_expert(DummyModule(), "model.layers.0.attention")


def test_is_parameter_filter():
    """_is_parameter returns True for all nn.Parameter instances."""

    param = torch.nn.Parameter(torch.randn(4, 4))
    assert _is_parameter(param, "any.fqn") is True
    assert _is_parameter(torch.randn(4, 4), "any.fqn") is False
    assert _is_parameter(torch.nn.Module(), "any.fqn") is False
    assert _is_parameter(None, "any.fqn") is False


def test_convert_unwraps_wrapper_without_weight_config():
    """A wrapper without a weight_config unwraps to the raw high-precision tensor on convert."""
    orig = torch.randn(4, 32, dtype=torch.bfloat16)
    wrapped = Float8TrainingWeightWrapperTensor(orig)
    assert wrapped.weight_config is None
    param = torch.nn.Parameter(wrapped, requires_grad=True)

    result = unwrap_param(torch.nn.Linear(32, 4), "weight", param)

    assert not isinstance(result.data, BaseTrainingWeightWrapperTensor)
    assert result.requires_grad
    assert torch.equal(result.data, orig)


@pytest.mark.parametrize(
    "weight_config, act_config, wrapper_cls",
    [
        (Float8FakeQuantizeConfig(), None, Float8TrainingWeightWrapperTensor),
        (BlockMXQuantizeConfig(), MXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor),
    ],
)
def test_prepare_wraps_expert_weights(weight_config, act_config, wrapper_cls):
    """Prepare wraps expert weights with the configured tensor subclass."""
    # use_grouped_mm only affects the forward computation path -- no forward run here.
    moe_model = _create_cpu_moe_model()
    orig_values = {name: param.data.clone() for name, param in moe_model.named_parameters()}

    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped_count = 0
    for name, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped_count += 1
            assert torch.equal(param.data.to_tensor(), orig_values[name]), (
                f"Values of {name} should match after being wrapped in the prepare step."
            )
        else:
            assert torch.equal(param.data, orig_values[name]), (
                f"values of {name} should not be changed after the prepare step."
            )
    assert wrapped_count == 3, f"Expected 3 wrapped params, got {wrapped_count}"


@pytest.mark.parametrize(
    "weight_config, act_config, wrapper_cls",
    [
        (Float8FakeQuantizeConfig(), None, Float8TrainingWeightWrapperTensor),
        (BlockMXQuantizeConfig(), MXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor),
    ],
)
def test_prepare_skips_non_expert_params(weight_config, act_config, wrapper_cls):
    """params_filter_fn excluding 2D params skips router.gate.weight."""
    # use_grouped_mm only affects the forward computation path -- no forward run here.
    moe_model = _create_cpu_moe_model()
    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped = 0
    for name, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped += 1
            assert param.ndim == 3, f"Wrapped param {name} should be 3D, got {param.ndim}D"
    assert wrapped == 3, f"All 3D expert params should be wrapped, got {wrapped}"
    # base class as wildcard -- any wrapper type on non-expert params is a bug
    assert not isinstance(moe_model.router.gate.weight.data, BaseTrainingWeightWrapperTensor), (
        "router.gate.weight should not be wrapped"
    )


@pytest.mark.parametrize(
    "base_config, wrapper_cls",
    [
        (
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            Float8TrainingWeightWrapperTensor,
        ),
    ],
)
def test_config_prepare_with_base_config(base_config, wrapper_cls):
    """Model can be prepared using base_config instead of explicit weight_config."""

    # use_grouped_mm only affects the forward computation path -- no forward run here.
    moe_model = _create_cpu_moe_model()
    qat_config = ParamSwapConfig(
        base_config=base_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    wrapped = sum(1 for _, p in moe_model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"


@pytest.mark.parametrize(
    "weight_config, act_config, wrapper_cls",
    [
        (Float8FakeQuantizeConfig(), None, Float8TrainingWeightWrapperTensor),
        (BlockMXQuantizeConfig(), MXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor),
    ],
)
def test_is_expert_integration(weight_config, act_config, wrapper_cls):
    """_is_expert as filter_fn: only expert submodules are transformed, router skipped."""

    # use_grouped_mm only affects the forward computation path -- no forward run here.
    moe_model = _create_cpu_moe_model()
    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=_is_expert)

    wrapped = sum(1 for _, p in moe_model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"
    # base class as wildcard -- any wrapper type on non-expert params is a bug
    assert not isinstance(moe_model.router.gate.weight.data, BaseTrainingWeightWrapperTensor), (
        "router.gate.weight should not be wrapped"
    )


@pytest.mark.parametrize(
    "weight_config, act_config, wrapper_cls",
    [
        (Float8FakeQuantizeConfig(), None, Float8TrainingWeightWrapperTensor),
        (BlockMXQuantizeConfig(), MXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor),
    ],
)
def test_is_parameter_integration(weight_config, act_config, wrapper_cls):
    """Default filter (_is_parameter) wraps all parameters including 2D gate."""

    # use_grouped_mm only affects the forward computation path -- no forward run here.
    moe_model = _create_cpu_moe_model()
    qat_config = ParamSwapConfig(weight_config=weight_config, activation_config=act_config, step="prepare")
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped_count = 0
    for _, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped_count += 1
    assert wrapped_count == 7, f"Expected 7 wrapped params, got {wrapped_count}"


@pytest.mark.parametrize(
    "weight_config, wrapper_cls",
    [(Float8FakeQuantizeConfig(), Float8TrainingWeightWrapperTensor)],
)
def test_is_parameter_with_wrapped_data_filter(weight_config, wrapper_cls):
    """_is_parameter_with_wrapped_data returns True only for wrapped nn.Parameters."""

    w = torch.randn(64, 128, device="cpu")
    wrapped = wrapper_cls(w, weight_config=weight_config)
    wrapped_param = torch.nn.Parameter(wrapped)
    plain_param = torch.nn.Parameter(w)

    assert _is_parameter_with_wrapped_data(wrapped_param, "any.fqn") is True
    assert _is_parameter_with_wrapped_data(plain_param, "any.fqn") is False
    assert _is_parameter_with_wrapped_data(wrapped, "any.fqn") is False
    assert _is_parameter_with_wrapped_data(None, "any.fqn") is False
