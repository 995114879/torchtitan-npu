# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from _support.testing_utils import _SizeOnlyMesh
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor


def test_block_mx_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(),
        activation_config=MXQuantizeConfig(),
    )
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[BlockMXQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, BlockMXTrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.data.activation_config is config.activation_config
    assert result.requires_grad


@pytest.mark.parametrize(
    "shape",
    [(32, 128), (64, 32), (64, 96)],
    ids=["short_rows", "short_cols", "cols_not_block_aligned"],
)
def test_fp8_prequantize_rejects_misaligned_shards(shape):
    """A shard whose trailing dims are not 64-aligned would be padded mid-gather."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(shape),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(RuntimeError, match="Cannot quantize sharded weight before FSDP"):
        wrapper.fsdp_pre_all_gather(
            _SizeOnlyMesh(2),
            wrapper.shape,
            wrapper.stride(),
            None,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )


def test_prequantize_size_one_mesh_returns_an_empty_payload():
    """A size-1 mesh runs no collective; the empty payload keeps FSDP's buffer at zero bytes."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(1),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert metadata == ()
    assert len(payload) == 1
    assert payload[0].numel() == 0, "the payload must be empty so FSDP allocates no gather buffer"
    assert payload[0].dtype is torch.bfloat16


@pytest.mark.parametrize("master_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_disabled_prequantize_size_one_mesh_skips_the_gather_buffer(master_dtype):
    """A size-1 mesh skips the buffer even without ``fsdp_prequantize``.

    The high-precision weight is rebuilt without casting to ``param_dtype``: the matmul
    quantizes it anyway, so a cast would only add a full-size copy.
    """
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=master_dtype),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(1),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert metadata == ()
    assert len(payload) == 1
    assert payload[0].numel() == 0, "the payload must be empty so FSDP allocates no gather buffer"
    assert payload[0].dtype is torch.bfloat16

    output, inner = wrapper.fsdp_post_all_gather(payload, metadata, torch.bfloat16)

    assert isinstance(output, BlockMXTrainingWeightWrapperTensor)
    assert not isinstance(output._data, (BlockMXTensor, MXTensor)), "no quantization is expected"
    assert output._data.shape == wrapper.shape
    assert output._data.dtype is master_dtype, "the high-precision weight keeps the master dtype at size 1"
    assert output._data.data_ptr() == wrapper._data.data_ptr(), "the sharded weight backs the unsharded one"
    assert inner is payload


def test_prequantize_disabled_falls_back_to_hp_payload():
    """With the flag off, the base hook gathers the shard cast to ``param_dtype``."""
    weight = torch.randn(32, 32, dtype=torch.float32)
    wrapper = BlockMXTrainingWeightWrapperTensor(
        weight,
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert metadata == ()
    assert len(payload) == 1
    (data,) = payload
    assert data.dtype is torch.bfloat16
    assert data.shape == weight.shape
    assert torch.equal(data, weight.to(torch.bfloat16))


@pytest.mark.parametrize(
    "local_shape, outer_size, world_size",
    [
        # Uneven: the padded rank (1 of 3 rows over 2 ranks) pads to the largest chunk.
        (torch.Size([1, 8, 16]), torch.Size([3, 8, 16]), 2),
        # Uneven: rank 0 already holds the largest chunk, left untouched.
        (torch.Size([2, 8, 16]), torch.Size([3, 8, 16]), 2),
    ],
)
def test_disabled_prequantize_falls_back_to_padded_hp_payload(local_shape, outer_size, world_size):
    """The high-precision fallback pads an uneven dim-0 shard like the base hook."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(local_shape),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(world_size),
        outer_size,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    (data,) = payload
    assert metadata == ()
    local_dim0 = local_shape[0]
    padded_dim0 = -(-outer_size[0] // world_size)
    assert data.shape == (padded_dim0, *local_shape[1:])
    assert torch.equal(data[:local_dim0], wrapper._data.to(torch.bfloat16))
    if local_dim0 < padded_dim0:
        assert torch.count_nonzero(data[local_dim0:]) == 0


def test_disabled_prequantize_multi_rank_post_rebuilds_from_the_payload():
    """With the flag off, the gathered HP payload is rebound into the wrapper, not copied."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )
    gathered = torch.randn(32, 32, dtype=torch.bfloat16)
    payload = (gathered,)

    output, inner = wrapper.fsdp_post_all_gather(payload, (), torch.bfloat16)

    assert isinstance(output, BlockMXTrainingWeightWrapperTensor)
    assert not isinstance(output._data, (BlockMXTensor, MXTensor)), "no quantization is expected"
    assert output._data.dtype is torch.bfloat16
    assert output._data.data_ptr() == gathered.data_ptr(), "the unsharded weight should alias the gathered payload"
    assert inner is payload

    out = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    assert wrapper.fsdp_post_all_gather(payload, (), torch.bfloat16, out=out) is None
    assert out._data.data_ptr() == gathered.data_ptr(), "the step-1+ path should rebind ``_data`` to the payload"
    assert out.weight_config is wrapper.weight_config
    assert out.activation_config is wrapper.activation_config


@pytest.mark.parametrize("num_components", [0, 4], ids=["none", "too_many"])
def test_fsdp_post_all_gather_rejects_unsupported_payload_counts(num_components):
    """Only 1 (HP), 2 (MXFP4), and 3 (FP8) component payloads exist."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(RuntimeError, match=r"`len\(all_gather_outputs\)` in \(1, 2, 3\)"):
        wrapper.fsdp_post_all_gather(tuple(torch.empty(0) for _ in range(num_components)), {}, torch.bfloat16)


def test_fsdp_post_all_gather_single_payload_requires_hp_dtype():
    """The 1-component fallback is only valid for a high-precision gathered shard."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(AssertionError, match="expected to be in high precision"):
        wrapper.fsdp_post_all_gather((torch.empty(64, 128, dtype=torch.float8_e4m3fn),), (), torch.bfloat16)


@pytest.mark.parametrize("experts", [3, 4], ids=["uneven", "even"])
def test_hp_post_all_gather_reused_out_preserves_logical_experts(experts):
    # FSDP retains the padded payload on step 0, then hands post-gather a
    # logical-size wrapper on subsequent steps. Padding must stay out of _data.
    logical = torch.arange(experts * 8, dtype=torch.float32).reshape(experts, 2, 4)
    padded = torch.cat((logical, torch.zeros(4 - experts, 2, 4)))
    source = BlockMXTrainingWeightWrapperTensor(
        logical[:2],
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    initial, buffers = source.fsdp_post_all_gather((padded,), (), torch.float32)

    assert initial.shape == (4, 2, 4)
    assert buffers[0] is padded
    out = torch.as_strided(initial, logical.shape, logical.stride())
    for step in (1, 2):
        gathered = padded + step
        source.fsdp_post_all_gather((gathered,), (), torch.float32, out=out)
        assert out.shape == out._data.shape == logical.shape
        assert torch.equal(out._data, logical + step)
        assert out._data.untyped_storage().data_ptr() == gathered.untyped_storage().data_ptr()
