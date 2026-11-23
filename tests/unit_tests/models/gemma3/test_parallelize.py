# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims

from torchtitan_npu.models.gemma3 import config_registry


@pytest.fixture(autouse=True)
def preserve_dynamo_cache_policy(monkeypatch):
    # Upstream AC disables a process-global LRU cache without a getter. Isolate
    # this PP/Flex tuning workaround; real checkpoint wrappers and math still run.
    monkeypatch.setattr("torchtitan.distributed.activation_checkpoint._disable_dynamo_lru_cache", lambda: None)


def _parallelize(recipe, model, dims):
    return recipe.model_spec.parallelize_fn(
        model,
        parallel_dims=dims,
        training=recipe.training,
        parallelism=recipe.parallelism,
        compile_config=recipe.compile,
        ac_config=recipe.activation_checkpoint,
        dump_folder="unused",
    )


def test_fsdp_recipe_passes_parallel_config_to_native_consumer(recipe, sft_paths):
    parsed = ConfigManager().parse_args(["--module", "torchtitan_npu.models.gemma3", "--config", "sft_gemma3_4b_fsdp2"])
    # A non-default policy detects accidental hardcoding in the Gemma3 adapter.
    parsed.model_spec = replace(parsed.model_spec, model=recipe.model_spec.model)
    parsed.parallelism = replace(parsed.parallelism, fsdp_reshard_after_forward="never")
    model = parsed.model_spec.model.build()
    shared_weight = model.tok_embeddings.weight
    dims = ParallelDims.from_config(parsed.parallelism, world_size=2)
    assert (dims.dp_shard, dims.world_size, dims.spmd_backend) == (2, 2, "partial_dtensor")
    mesh = Mock(ndim=1, mesh_dim_names=("dp_shard",))
    mesh.size.return_value = 2
    with (
        patch.object(dims, "get_mesh", return_value=mesh) as get_mesh,
        patch("torchtitan_npu.models.gemma3.parallelize.apply_fsdp_to_decoder") as fsdp,
    ):
        result = _parallelize(parsed, model, dims)

    get_mesh.assert_called_once_with("fsdp")
    fsdp.assert_called_once_with(
        model,
        mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        reshard_after_forward_policy="never",
    )
    assert result is model
    assert model.tok_embeddings.weight is shared_weight
    assert model.lm_head.weight is model.tok_embeddings.weight
    assert all(p.dtype is torch.float32 for p in model.parameters())


def test_full_ac_compile_then_fsdp_consumer_preserves_cpu_outputs_and_gradients(recipe, model, sft_paths):
    parsed = ConfigManager().parse_args(["--module", "torchtitan_npu.models.gemma3", "--config", "sft_gemma3_4b_fsdp2"])
    parsed.model_spec = replace(parsed.model_spec, model=recipe.model_spec.model)
    parsed.compile = replace(parsed.compile, enable=True, components=["model"], backend="aot_eager")
    reference = deepcopy(model)
    dims = ParallelDims.from_config(parsed.parallelism, world_size=2)
    mesh = Mock(ndim=1, mesh_dim_names=("dp_shard",))
    mesh.size.return_value = 2
    compiled_layers = []

    def compile_after_full_ac(candidate, **kwargs):
        assert candidate is model
        assert all(hasattr(layer, "_checkpoint_wrapped_module") for layer in candidate.layers.values())
        # CPU Inductor cannot lower Gemma3's FlexAttention. The CPU UT checks
        # the compile consumer contract; real compilation is covered on NPU.
        compiled_layers.extend(candidate.layers.values())

    def observe_fsdp_after_compile(target, mesh, **kwargs):
        # Only device sharding and compilation are simulated; the native
        # consumer receives the model after the FullAC/compile hook boundary.
        assert target is model
        assert compiled_layers == list(model.layers.values())

    tokens = torch.tensor([[2, 5, 7, 11, 13]])
    with (
        torch._dynamo.config.patch(
            capture_scalar_outputs=torch._dynamo.config.capture_scalar_outputs,
            skip_fwd_side_effects_in_bwd_under_checkpoint=(
                torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint
            ),
        ),
        patch.object(dims, "get_mesh", return_value=mesh),
        patch("torchtitan_npu.models.gemma3.parallelize.apply_compile", side_effect=compile_after_full_ac) as compiler,
        patch(
            "torchtitan_npu.models.gemma3.parallelize.apply_fsdp_to_decoder",
            side_effect=observe_fsdp_after_compile,
        ) as fsdp,
    ):
        result = _parallelize(parsed, model, dims)
        actual = result(tokens)
        expected = reference(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()

    compiler.assert_called_once_with(model, compile_config=parsed.compile, parallel_dims=dims)
    fsdp.assert_called_once_with(
        model,
        mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        reshard_after_forward_policy=parsed.parallelism.fsdp_reshard_after_forward,
    )
    assert model.lm_head.weight is model.tok_embeddings.weight
    for actual_parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        assert actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-5)


def test_fsdp_recipe_rejects_unsupported_replication_before_mesh_creation(recipe, model):
    config = config_registry.debug_gemma3_fsdp2()
    config.model_spec = replace(config.model_spec, model=recipe.model_spec.model)
    dims = ParallelDims(dp_replicate=2, dp_shard=1, tp=1, pp=1, cp=1, ep=1, world_size=2)
    with patch.object(dims, "get_mesh") as get_mesh, pytest.raises(ValueError, match="world_size=dp_shard"):
        _parallelize(config, model, dims)
    get_mesh.assert_not_called()
