# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE file.

"""Small real-operator TorchFT profile for complete-state recovery validation."""

from torchtitan_npu.models.deepseek_v4.config_registry import (
    _torchft_flash_config,
    deepseek_v4_debugmodel,
)


def deepseek_v4_small_torchft():
    base = deepseek_v4_debugmodel(num_mtp_layers=1)
    # Both debug and Flash use _make_v4_config/DeepSeekV4Model. Name this scaled
    # Flash variant explicitly, as the existing four-layer Flash recipe does.
    base.model_spec.flavor = "deepseek_v4_flash_small_4layers_4experts"
    config = _torchft_flash_config(base, expert_parallel_degree=2, data_parallel_shard_degree=2, global_batch_size=8)
    config.training.seq_len = 256
    config.debug.seed = 1234
    assert describe_model(config) == {
        "dim": 256,
        "vocab_size": 129280,
        "n_layers": 4,
        "mtp_layers": 1,
        "num_experts": [4, 4, 4, 4],
        "top_k": [3, 3, 3, 3],
        "compress_ratios": [0, 0, 4, 128],
        "expert_parallel_degree": 2,
        "data_parallel_shard_degree": 2,
        "global_batch_size": 8,
        "seq_len": 256,
    }
    return config


def describe_model(config):
    model = config.model_spec.model
    return {
        "dim": model.dim,
        "vocab_size": model.vocab_size,
        "n_layers": model.n_layers,
        "mtp_layers": len(model.mtp_layers),
        "num_experts": [layer.moe.num_experts for layer in model.layers],
        "top_k": [layer.moe.router.top_k for layer in model.layers],
        "compress_ratios": list(model.compress_ratios),
        "expert_parallel_degree": config.parallelism.expert_parallel_degree,
        "data_parallel_shard_degree": config.parallelism.data_parallel_shard_degree,
        "global_batch_size": config.training.global_batch_size,
        "seq_len": config.training.seq_len,
    }
