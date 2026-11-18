# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy

import pytest
import torch
from _support.reference_moe import MoE
from _support.testing_utils import (
    _expert_weight_filter,
    create_moe_model,
    target_devices,
)
from torchao.quantization.quant_api import quantize_
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.transform import (
    unwrap_param,
)
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
    MXTrainingWeightWrapperTensor,
)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "weight_config, act_config, wrapper_cls",
    [
        (MXQuantizeConfig(), MXQuantizeConfig(), MXTrainingWeightWrapperTensor),
        (BlockMXQuantizeConfig(), MXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor),
    ],
)
def test_convert_produces_inference_weight(device, weight_config, act_config, wrapper_cls):
    """Convert converts the wrapped parameters to inference weights (``MXTensor``).

    The training-time weight_config is stored on the wrapper, so the bare
    convert step (no config) quantizes via ``to_inference_weight``.
    """

    # use_grouped_mm only affects the forward computation path, which is
    # never triggered here -- only prepare/convert lifecycle is tested.
    # bf16 master: the block-MX quant kernel rejects fp32 inputs.
    moe_model = create_moe_model(device, use_grouped_mm=True, dtype=torch.bfloat16)
    model = copy.deepcopy(moe_model)

    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped_fqns = [name for name, p in model.named_parameters() if isinstance(p.data, wrapper_cls)]
    assert len(wrapped_fqns) == 3, f"Only {len(wrapped_fqns)} nn.Parameters are wrapped, 3 expected."

    for (name, param), (orig_name, orig_param) in zip(
        model.named_parameters(), moe_model.named_parameters(), strict=True
    ):
        assert name == orig_name, f"Parameter order changed: {name} vs {orig_name}"
        assert torch.equal(param, orig_param), f"Values of {name} should match after prepare"

    qat_config = ParamSwapConfig(step="convert")
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    # base class as wildcard -- no wrapper of any type should survive convert
    wrapped = sum(1 for _, p in model.named_parameters() if isinstance(p.data, BaseTrainingWeightWrapperTensor))
    assert wrapped == 0, f"{wrapped} parameters should not be wrapped after convert"

    params = dict(model.named_parameters())
    for name in wrapped_fqns:
        assert isinstance(params[name].data, MXTensor), (
            f"{name}: expected MXTensor after convert, got {type(params[name].data).__name__}"
        )


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "weight_config, wrapper_cls, expected_elem_dtype",
    [
        pytest.param(MXQuantizeConfig(), MXTrainingWeightWrapperTensor, torch.float8_e4m3fn, id="mx"),
        pytest.param(BlockMXQuantizeConfig(), BlockMXTrainingWeightWrapperTensor, torch.float8_e4m3fn, id="block_fp8"),
        pytest.param(
            BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            BlockMXTrainingWeightWrapperTensor,
            torch.float4_e2m1fn_x2,
            id="block_fp8_mxfp4",
        ),
    ],
)
def test_convert_with_weight_config_produces_mx_tensor(device, weight_config, wrapper_cls, expected_elem_dtype):
    """Convert of a wrapper with a weight_config quantizes via its to_inference_weight.

    Covers every to_inference_weight shape: MX (FP8), block-MX FP8 (converted
    via ``BlockMXTensor.to_mx_tensor``), and mxfp4-QAT (FP4 qdata).
    """
    wrapped = wrapper_cls(
        torch.randn(256, 128, dtype=torch.bfloat16, device=device),
        weight_config=weight_config,
        activation_config=MXQuantizeConfig(),
    )
    param = torch.nn.Parameter(wrapped, requires_grad=False)

    result = unwrap_param(torch.nn.Linear(128, 256), "weight", param)

    assert isinstance(result.data, MXTensor), f"got {type(result.data).__name__}"
    assert not result.requires_grad
    assert result.data.quant_axis == result.data.ndim - 1
    assert result.data.quant_config.elem_dtype is expected_elem_dtype, (
        f"expected {expected_elem_dtype} qdata, got {result.data.quant_config.elem_dtype}"
    )
