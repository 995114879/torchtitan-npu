# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy

import pytest
import torch
from _support.testing_utils import target_devices
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
from torchao.quantization.qat.fake_quantize_config import (
    Float8FakeQuantizeConfig,
)
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
    Float8TrainingWeightWrapperTensor,
    MXTrainingWeightWrapperTensor,
)
from torchao_npu.wrapper_tensors.base_wrapper_tensor import _pad_uneven_shard_dim0

_BASE_WRAPPER_CASES = [
    (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (
        BaseTrainingWeightWrapperTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(),
    ),
]


_NON_BASE_WRAPPER_CASES = [
    (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (
        Float8TrainingWeightWrapperTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(),
    ),
    (
        MXTrainingWeightWrapperTensor,
        MXQuantizeConfig(),
        MXQuantizeConfig(),
    ),
    (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    (
        BlockMXTrainingWeightWrapperTensor,
        BlockMXQuantizeConfig(),
        MXQuantizeConfig(),
    ),
]


_ALL_WRAPPER_CASES = _BASE_WRAPPER_CASES + _NON_BASE_WRAPPER_CASES


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_deepcopy(wrapper_cls, weight_config, act_config, device):
    """deepcopy creates an independent wrapper with independent _data."""
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    wrapper_copy = copy.deepcopy(wrapper)

    assert wrapper_copy is not wrapper
    assert isinstance(wrapper, wrapper_cls)
    assert isinstance(wrapper_copy, wrapper_cls)

    assert wrapper_copy._data is not wrapper._data
    assert torch.equal(wrapper_copy._data, wrapper._data)

    assert wrapper_copy.weight_config == wrapper.weight_config
    assert wrapper_copy.weight_config is not wrapper.weight_config

    assert wrapper_copy.activation_config == wrapper.activation_config
    if act_config is not None:
        assert wrapper_copy.activation_config is not wrapper.activation_config

    activation = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
    out = torch.mm(activation, wrapper.T)
    out_copy = torch.mm(activation, wrapper_copy.T)
    assert torch.equal(out, out_copy), "The cloned tensor should yield identical results."
    if wrapper_cls is BaseTrainingWeightWrapperTensor:
        # Base class passes through without fake quantization.
        ref = torch.mm(activation, wrapper._data.T)
        assert torch.equal(out, ref), "Base class __torch_function__ should pass through without fake quantization."


@pytest.mark.parametrize("device", target_devices)
def test_pad_uneven_shard_dim0_packs_fp4_via_uint8(device):
    """Packed FP4 (which has no dim-0 copy kernel) is padded through a uint8 byte view."""
    from torchao_npu.quantized_tensors.mx_tensor import MXTensor

    fp4 = MXTensor.from_hp(
        torch.randn(1, 64, 128, device=device, dtype=torch.bfloat16),
        quant_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        axis=-1,
        act_quant_config=MXQuantizeConfig(),
    ).qdata

    (padded,) = _pad_uneven_shard_dim0((fp4,), torch.Size([3, *fp4.shape[1:]]), 2)

    assert padded.shape == (2, *fp4.shape[1:])
    assert padded.dtype is torch.float4_e2m1fn_x2
    assert torch.equal(padded[:1].view(torch.uint8), fp4.view(torch.uint8))
    assert torch.count_nonzero(padded[1:].view(torch.uint8)) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_spec",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_fsdp_post_all_gather_existing_out_same_dtype_dtensor(wrapper_spec, dtype, device, mock_distributed_env):
    """out is DTensor with wrapped local_tensor -- configs restored on local_tensor."""
    wrapper_cls, weight_config, act_config = wrapper_spec

    w = torch.empty(2, 32, 64, device="meta")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    mesh = DeviceMesh(device, torch.arange(1))
    out_data = torch.randn(4, 64, 128, dtype=dtype, device=device)
    out_local = wrapper_cls(out_data, weight_config=weight_config, activation_config=act_config)
    out = DTensor.from_local(out_local, mesh, [Shard(0)])
    local_tensor = out._local_tensor
    local_tensor.activation_config = None
    local_tensor.weight_config = None

    data = local_tensor._data
    result = wrapper.fsdp_post_all_gather((data,), None, dtype, out=out)

    assert result is None
    assert local_tensor._data is data
    assert local_tensor.weight_config is weight_config
    assert local_tensor.activation_config is act_config

    # If data has the same dtype but different storage, the assertion should fire
    storage_mismatch = data.clone()
    with pytest.raises(AssertionError):
        wrapper.fsdp_post_all_gather((storage_mismatch,), None, dtype, out=out)


@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_spec",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_fsdp_post_all_gather_existing_out_cross_dtype_dtensor(
    wrapper_spec, in_dtype, out_dtype, device, mock_distributed_env
):
    """out is DTensor with wrapped local_tensor, cross-dtype: configs restored, copy_ applied."""
    wrapper_cls, weight_config, act_config = wrapper_spec

    w = torch.empty(2, 32, 64, device="meta")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    mesh = DeviceMesh(device, torch.arange(1))
    out_data = torch.randn(4, 64, 128, dtype=out_dtype, device=device)
    out_local = wrapper_cls(out_data, weight_config=weight_config, activation_config=act_config)
    out = DTensor.from_local(out_local, mesh, [Shard(0)])
    local_tensor = out._local_tensor
    local_tensor.activation_config = None
    local_tensor.weight_config = None

    data = torch.randn(4, 64, 128, dtype=in_dtype, device=device)
    out_local_tensor_data_before = local_tensor._data
    result = wrapper.fsdp_post_all_gather((data,), None, out_dtype, out=out)

    assert result is None
    assert local_tensor._data is out_local_tensor_data_before  # in-place copy_: same object
    assert local_tensor.weight_config is weight_config
    assert local_tensor.activation_config is act_config
    assert torch.equal(local_tensor._data, data.to(out_dtype))

    # If param_dtype doesn't match out_data.dtype, the assertion should fire
    bad_param_dtype = torch.float64
    expected_msg = (
        f"^`out`\\(dtype={out_dtype}\\) does not match the mixed precision policy param_dtype {bad_param_dtype}$"
    )
    with pytest.raises(AssertionError, match=expected_msg):
        wrapper.fsdp_post_all_gather((data,), None, bad_param_dtype, out=out)
