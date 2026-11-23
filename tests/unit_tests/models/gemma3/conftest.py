# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType

from torchtitan_npu.models.gemma3.config_registry import debug_gemma3_single_node

from .fixtures import recipe_model_config

# Keep the numerical oracle's HF fields independent from the local Config
# fixture.  The duplicated values are intentional: deriving the oracle from
# recipe_text_config() could hide a mismatch in the Config factory.
_HF_ORACLE_FIELDS = {
    "model_type": "gemma3_text",
    "vocab_size": 2008,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "query_pre_attn_scalar": 16,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 4,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "max_position_embeddings": 256,
    "tie_word_embeddings": True,
    "bos_token_id": 2002,
    "eos_token_id": 2003,
    "pad_token_id": 2001,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "hidden_activation": "gelu_pytorch_tanh",
}


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        # Seed CPU only; torch.manual_seed also mutates the NPU/CUDA RNG state.
        torch.random.default_generator.manual_seed(20261014)
        yield


@pytest.fixture(autouse=True)
def force_cpu_checkpoint_device(monkeypatch):
    # CPU UTs must not let torch_npu's global NPU default affect checkpoint recompute.
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")


@pytest.fixture
def recipe():
    config = debug_gemma3_single_node()
    model_config = recipe_model_config()
    return replace(config, model_spec=replace(config.model_spec, model=model_config))


@pytest.fixture
def model(recipe):
    decoder = recipe.model_spec.model.build().float()
    decoder.init_states(buffer_device=torch.device("cpu"))
    return decoder


@pytest.fixture
def hf_reference():
    # Imported lazily because this independent numerical oracle is only needed
    # by parity and checkpoint tests.
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(**deepcopy(_HF_ORACLE_FIELDS))
    config._attn_implementation = "eager"
    reference = Gemma3ForCausalLM(config).float()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.1, 0.1)
    return reference


@pytest.fixture
def sft_paths(tmp_path, monkeypatch):
    assets = tmp_path / ".gemma3_gsm8k" / "models" / "gemma-3-4b-pt"
    assets.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (assets / "config.json").write_text("{}", encoding="utf-8")
    (assets / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    return assets
