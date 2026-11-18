# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import re

import pytest
import torch
from _support.testing_utils import _SizeOnlyMesh
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.qat.fake_quantize_config import (
    FakeQuantizeConfigBase,
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


_DEFAULT_WRAPPER_CASES = [
    (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
]


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_init_keeps_the_supplied_data(wrapper_cls, weight_config, act_config):
    """``_data`` is the supplied tensor; detaching is the caller's job or the ``detach`` op."""
    w = torch.randn(64, 128, requires_grad=True)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    assert wrapper.requires_grad is True
    assert wrapper._data is w
    assert wrapper._data.requires_grad is True


def test_nn_parameter_wrapper_keeps_grad_on_wrapper_only():
    """Autograd tracks the wrapper; ``_data`` stays a detached storage tensor."""
    wrapper = BaseTrainingWeightWrapperTensor(torch.randn(8, 16))

    param = nn.Parameter(wrapper)

    assert isinstance(param, BaseTrainingWeightWrapperTensor)
    assert param.requires_grad is True
    assert param._data.requires_grad is False

    param.requires_grad_(False)

    assert param.requires_grad is False
    assert param._data.requires_grad is False


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_deepcopy_preserves_requires_grad(wrapper_cls, weight_config, act_config):
    """``__deepcopy__`` restores the wrapper's flag without giving ``_data`` one."""
    wrapper = wrapper_cls(torch.randn(16, 32), activation_config=act_config, weight_config=weight_config)
    wrapper.requires_grad_(True)

    wrapper_copy = copy.deepcopy(wrapper)

    assert wrapper_copy.requires_grad is True
    assert wrapper_copy._data.requires_grad is False
    assert wrapper_copy is not wrapper
    assert isinstance(wrapper_copy, wrapper_cls)
    assert wrapper_copy._data is not wrapper._data
    assert torch.equal(wrapper_copy._data, wrapper._data)
    assert wrapper_copy.weight_config == wrapper.weight_config
    assert wrapper_copy.weight_config is not wrapper.weight_config
    assert wrapper_copy.activation_config == wrapper.activation_config
    if act_config is not None:
        assert wrapper_copy.activation_config is not wrapper.activation_config


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _BASE_WRAPPER_CASES)
def test_base_to_inference_weight_is_not_implemented(wrapper_cls, weight_config, act_config):
    """The base class serves no quantized inference weight: subclasses have to implement it."""
    wrapper = wrapper_cls(torch.randn(2, 4), weight_config=weight_config, activation_config=act_config)

    with pytest.raises(NotImplementedError, match="to_inference_weight"):
        wrapper.to_inference_weight()


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _DEFAULT_WRAPPER_CASES)
def test_wrapper_untyped_storage_delegates_to_data(wrapper_cls, weight_config, act_config):
    """untyped_storage delegates to _data, exposing the real (valid) data storage."""
    w = torch.randn(64, 128)
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)

    # The wrapper holds no storage of its own; it must return _data's storage.
    assert wrapper.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    # That storage must be a valid, non-null address (not an "invalid python storage").
    assert wrapper.untyped_storage().data_ptr() != 0


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _DEFAULT_WRAPPER_CASES)
def test_wrapper_data_ptr_delegates_to_data(wrapper_cls, weight_config, act_config):
    """data_ptr delegates to _data, reporting the real data address instead of 0."""
    w = torch.randn(64, 128)
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)

    assert wrapper.data_ptr() == w.data_ptr()
    # Regression: a storage-less wrapper used to silently return 0x0 here.
    assert wrapper.data_ptr() != 0
    # For a zero-offset contiguous tensor, data_ptr and the storage pointer agree.
    assert wrapper.data_ptr() == wrapper.untyped_storage().data_ptr()


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _DEFAULT_WRAPPER_CASES)
def test_meta_weights(wrapper_cls, weight_config, act_config):
    """Wrapper can be constructed on meta device."""
    with torch.device("meta"):
        wrapper = wrapper_cls(torch.randn(64, 128), weight_config=weight_config, activation_config=act_config)
    assert wrapper._data.is_meta


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
    ids=["fp32-fp32", "fp32-bf16", "bf16-fp32", "bf16-bf16"],
)
def test_fsdp_pre_all_gather(wrapper_cls, weight_config, act_config, dtypes, mock_distributed_env):
    """The base fsdp_pre_all_gather casts _data to mp_policy.param_dtype and returns it."""
    tensor_dtype, param_dtype = dtypes
    w = torch.randn(64, 128, dtype=tensor_dtype)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype)
    mesh = DeviceMesh("cpu", torch.arange(1))

    all_gather_inputs, metadata = wrapper.fsdp_pre_all_gather(mesh, w.shape, w.stride(), None, mp_policy)
    (data,) = all_gather_inputs
    assert metadata == ()
    assert data.dtype == param_dtype
    assert data.shape == w.shape
    assert torch.equal(data, w.to(param_dtype))


@pytest.mark.parametrize(
    "local_shape, outer_size, world_size",
    [
        # Uneven: the padded rank (1 of 3 rows over 2 ranks) pads to the largest chunk.
        (torch.Size([1, 8, 16]), torch.Size([3, 8, 16]), 2),
        # Even sharding: the hook returns the cast input untouched.
        (torch.Size([2, 8, 16]), torch.Size([4, 8, 16]), 2),
    ],
)
def test_fsdp_pre_all_gather_pads_uneven_shard(local_shape, outer_size, world_size):
    """The hook pads dim 0 to the largest chunk for uneven shards; even shards pass through."""
    w = torch.randn(local_shape)
    wrapper = BaseTrainingWeightWrapperTensor(w)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

    all_gather_inputs, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(world_size), outer_size, w.stride(), None, mp_policy
    )

    (data,) = all_gather_inputs
    assert metadata == ()
    assert data.dtype == torch.bfloat16
    local_dim0 = local_shape[0]
    padded_dim0 = -(-outer_size[0] // world_size)
    assert data.shape == (padded_dim0, *local_shape[1:])
    assert torch.equal(data[:local_dim0], w.to(torch.bfloat16))
    if local_dim0 < padded_dim0:
        assert torch.count_nonzero(data[local_dim0:]) == 0


@pytest.mark.parametrize(
    "local_shape, outer_size, world_size",
    [
        # Uneven: the padded rank (1 of 3 rows over 2 ranks) pads to the largest chunk.
        (torch.Size([1, 8, 16]), torch.Size([3, 8, 16]), 2),
        # Uneven: rank 0 already holds the largest chunk, left untouched.
        (torch.Size([2, 8, 16]), torch.Size([3, 8, 16]), 2),
        # Uneven: an empty shard (2 rows over 4 ranks) pads to one zero row.
        (torch.Size([0, 8, 16]), torch.Size([2, 8, 16]), 4),
        # Even sharding or a single rank: no padding.
        (torch.Size([2, 8, 16]), torch.Size([4, 8, 16]), 2),
        (torch.Size([3, 8, 16]), torch.Size([3, 8, 16]), 1),
    ],
)
def test_pad_uneven_shard_dim0(local_shape, outer_size, world_size):
    """Uneven dim-0 FSDP shards are zero-padded to the largest chunk; others pass through unchanged."""
    t = torch.randn(local_shape)
    (padded,) = _pad_uneven_shard_dim0((t,), outer_size, world_size)
    local_dim0 = local_shape[0]
    padded_dim0 = -(-outer_size[0] // world_size)
    assert padded.shape == (padded_dim0, *local_shape[1:])
    if local_dim0 < padded_dim0:
        assert torch.equal(padded[:local_dim0], t)
        assert torch.count_nonzero(padded[local_dim0:]) == 0
    else:
        assert padded is t


def test_pad_uneven_shard_dim0_pads_all_inputs():
    """A multi-tensor payload (e.g. data + block scales) is padded on dim 0 together."""
    inputs = (torch.randn(1, 8, 16), torch.randn(1, 8, 1), torch.randn(1, 1, 16))

    padded = _pad_uneven_shard_dim0(inputs, torch.Size([3, 8, 16]), 2)

    assert [p.shape for p in padded] == [(2, 8, 16), (2, 8, 1), (2, 1, 16)]
    assert torch.equal(padded[1][:1], inputs[1])
    assert torch.count_nonzero(padded[2][1]) == 0


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fsdp_post_all_gather_existing_out_same_dtype(wrapper_cls, weight_config, act_config, dtype):
    """out is bare wrapper, same dtype: configs restored, storage pointer verified."""
    w = torch.empty(2, 32, 64, device="meta")  # different shape/device -- must not be used
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out = wrapper_cls(
        torch.randn(4, 64, 128, dtype=dtype),
        weight_config=weight_config,
        activation_config=act_config,
    )
    out.activation_config = None
    out.weight_config = None

    data = out._data
    result = wrapper.fsdp_post_all_gather((data,), None, dtype, out=out)
    assert result is None
    assert out._data is data  # storage-sharing: same pointer reused
    assert out.weight_config is weight_config
    assert out.activation_config is act_config

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
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_fsdp_post_all_gather_existing_out_cross_dtype(wrapper_cls, weight_config, act_config, in_dtype, out_dtype):
    """out is bare wrapper, different dtype: configs restored, out_data.copy_(data)."""
    w = torch.empty(2, 32, 64, device="meta")  # different shape/device -- must not be used
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out = wrapper_cls(
        torch.randn(4, 64, 128, dtype=out_dtype),
        weight_config=weight_config,
        activation_config=act_config,
    )
    out.activation_config = None
    out.weight_config = None

    data = torch.randn(4, 64, 128, dtype=in_dtype)
    out_data_before = out._data
    result = wrapper.fsdp_post_all_gather((data,), None, out_dtype, out=out)

    assert result is None
    assert out._data is out_data_before  # in-place copy_: same object
    assert out.weight_config is weight_config
    assert out.activation_config is act_config
    assert torch.equal(out._data, data.to(out_dtype))

    # If param_dtype doesn't match out_data.dtype, the assertion should fire
    bad_param_dtype = torch.float64
    expected_msg = (
        f"^`out`\\(dtype={out_dtype}\\) does not match the mixed precision policy param_dtype {bad_param_dtype}$"
    )
    with pytest.raises(AssertionError, match=expected_msg):
        wrapper.fsdp_post_all_gather((data,), None, bad_param_dtype, out=out)


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_fsdp_post_all_gather_existing_out_wrong_type(wrapper_cls, weight_config, act_config):
    """out with wrong type raises RuntimeError."""
    w = torch.randn(4, 64, 128)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out_type = re.escape(str(type(torch.randn(1))))
    expected_msg = (
        f"^expected out to be {re.escape(wrapper_cls.__name__)} or "
        f"DTensor with local_tensor={re.escape(wrapper_cls.__name__)}, "
        f"but got {out_type}$"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        wrapper.fsdp_post_all_gather((w,), None, torch.float32, out=torch.randn(4, 64, 128))


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        BaseTrainingWeightWrapperTensor,
        Float8TrainingWeightWrapperTensor,
    ],
)
def test_wrapper_init_accepts_none_weight_config(wrapper_cls):
    """Both wrapper classes accept weight_config=None."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, weight_config=None)
    assert wrapper.weight_config is None
    assert torch.equal(wrapper._data, w)


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_init_stores_attrs(wrapper_cls, weight_config, act_config):
    """__init__ stores _data, weight_config, and activation_config."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    assert wrapper._data is w
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is act_config


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config, expected_match",
    [
        (
            Float8TrainingWeightWrapperTensor,
            None,
            None,
            (
                r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` "
                r"in Float8TrainingWeightWrapperTensor\.$"
            ),
        ),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow(dim=-2)),
            None,
            r"^Only the row-wise granularity is supported\.$",
        ),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor()),
            None,
            r"^Only the row-wise granularity is supported\.$",
        ),
        (
            MXTrainingWeightWrapperTensor,
            None,
            MXQuantizeConfig(),
            r"^Only `MXQuantizeConfig` is supported for `weight_config` in MXTrainingWeightWrapperTensor\.$",
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            None,
            MXQuantizeConfig(),
            r"^Only `BlockMXQuantizeConfig` is supported for `weight_config` in BlockMXTrainingWeightWrapperTensor\.$",
        ),
    ],
)
def test_wrapper_init_rejects_invalid_weight_config(wrapper_cls, weight_config, act_config, expected_match):
    """Wrapper subclass rejects non-matching weight_config type or granularity."""
    if weight_config is None:

        class DummyConfig(FakeQuantizeConfigBase):
            pass

        weight_config = DummyConfig()

    w = torch.randn(64, 128, device="cpu")
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, weight_config=weight_config, activation_config=act_config)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config, expected_match",
    [
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            None,
            (
                r"^Only `Float8FakeQuantizeConfig` is supported for `activation_config` "
                r"in Float8TrainingWeightWrapperTensor\.$"
            ),
        ),
        (
            Float8TrainingWeightWrapperTensor,
            None,
            None,
            (
                r"^Only `Float8FakeQuantizeConfig` is supported for `activation_config` "
                r"in Float8TrainingWeightWrapperTensor\.$"
            ),
        ),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor()),
            r"^Only the row-wise granularity is supported for `activation_config`\.$",
        ),
        (
            Float8TrainingWeightWrapperTensor,
            None,
            Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor()),
            r"^Only the row-wise granularity is supported for `activation_config`\.$",
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            None,
            r"^Only `MXQuantizeConfig` is supported for `activation_config` in MXTrainingWeightWrapperTensor\.$",
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            None,
            r"^Only `MXQuantizeConfig` is supported for `activation_config` in BlockMXTrainingWeightWrapperTensor\.$",
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            None,
            None,
            r"^`weight_config` is required for BlockMXTrainingWeightWrapperTensor\.$",
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            None,
            BlockMXQuantizeConfig(),
            r"^`weight_config` is required for BlockMXTrainingWeightWrapperTensor\.$",
        ),
    ],
)
def test_wrapper_init_rejects_invalid_activation_config(wrapper_cls, weight_config, act_config, expected_match):
    """Wrapper subclass rejects non-matching activation_config type or granularity."""
    if act_config is None:

        class DummyConfig(FakeQuantizeConfigBase):
            pass

        act_config = DummyConfig()

    w = torch.randn(64, 128, device="cpu")
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, activation_config=act_config, weight_config=weight_config)


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _DEFAULT_WRAPPER_CASES)
def test_wrapper_to_tensor(wrapper_cls, weight_config, act_config):
    """to_tensor returns the underlying raw tensor."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)
    result = wrapper.to_tensor()
    assert isinstance(result, torch.Tensor)
    assert result is wrapper._data
    assert result is w


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _DEFAULT_WRAPPER_CASES)
def test_wrapper_repr(wrapper_cls, weight_config, act_config):
    """__repr__ includes class name, data, configs."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)
    expected = f"{wrapper_cls.__name__}(data={w}, activation_config={act_config}, weight_config={weight_config})"
    assert repr(wrapper) == expected


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_tensor_flatten(wrapper_cls, weight_config, act_config):
    """__tensor_flatten__ returns _data tensor and metadata dict."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    tensor_names, metadata = wrapper.__tensor_flatten__()
    assert tensor_names == ["_data"]
    assert metadata["weight_config"] is weight_config
    assert metadata["activation_config"] is act_config


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_wrapper_tensor_unflatten(wrapper_cls, weight_config, act_config):
    """__tensor_unflatten__ reconstructs the wrapper from flattened parts."""
    w = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    # simulate PyTorch: record size/stride during flatten, pass back during unflatten
    tensor_names, metadata = wrapper.__tensor_flatten__()
    outer_size = wrapper._data.size()
    outer_stride = wrapper._data.stride()
    tensor_data_dict = {name: getattr(wrapper, name) for name in tensor_names}

    reconstructed = wrapper_cls.__tensor_unflatten__(tensor_data_dict, metadata, outer_size, outer_stride)
    assert isinstance(reconstructed, wrapper_cls)
    assert torch.equal(reconstructed._data, w)
    assert reconstructed.weight_config is weight_config
    assert reconstructed.activation_config is act_config


@pytest.mark.parametrize("on_meta", [False, True])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", _ALL_WRAPPER_CASES)
def test_fsdp_post_all_gather_first_step(wrapper_cls, weight_config, act_config, on_meta):
    """fsdp_post_all_gather with out=None aliases the gathered payload and preserves the configs."""
    w_device = "meta" if on_meta else "cpu"
    w = torch.empty(64, 128, device=w_device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    gathered = torch.empty(4, 64, 128, device="cpu")

    result, inner_tensors = wrapper.fsdp_post_all_gather((gathered,), None, torch.float32, out=None)
    assert isinstance(result, wrapper_cls)
    assert result._data is gathered
    assert result._data.untyped_storage().data_ptr() == gathered.untyped_storage().data_ptr()
    assert isinstance(inner_tensors, tuple)
    assert len(inner_tensors) == 1
    assert inner_tensors[0] is gathered
    assert result.weight_config is weight_config
    assert result.activation_config is act_config
