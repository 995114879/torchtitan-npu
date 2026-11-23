# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer
from transformers import Gemma3TextConfig as HFTextConfig

from torchtitan_npu.models.gemma3 import (
    Gemma3Model,
    _make_gemma3_config,
    config_registry,
    gemma3_4b_config,
    gemma3_4b_text_config,
    model_registry,
    parse_gemma3_text_config,
)
from .fixtures import _write_tokenizer


def test_four_billion_recipe_matches_pinned_hf_text_configuration():
    # Gemma3 4B PT/IT share the same text decoder configuration. Expand the
    # sparse text_config with the independent HF implementation.
    text = {
        "model_type": "gemma3_text",
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 34,
        "sliding_window": 1024,
        "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    }
    reference = HFTextConfig(**text)
    parsed = parse_gemma3_text_config(
        {"text_config": text, "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27}}
    )
    recipe = gemma3_4b_text_config()
    model_config = gemma3_4b_config()

    assert "vision_config" not in parsed.to_dict()
    assert parsed.to_dict() == recipe.to_dict()
    assert (model_config.dim, model_config.vocab_size, len(model_config.layers)) == (
        reference.hidden_size,
        reference.vocab_size,
        reference.num_hidden_layers,
    )
    assert model_config.enable_weight_tying == reference.tie_word_embeddings
    assert model_config.norm.eps == reference.rms_norm_eps
    for index, layer in enumerate(model_config.layers):
        attention = layer.attention
        assert attention.layer_type == reference.layer_types[index]
        assert (attention.qkv_linear.wq.in_features, attention.qkv_linear.wq.out_features) == (
            reference.hidden_size,
            reference.num_attention_heads * reference.head_dim,
        )
        assert attention.qkv_linear.wkv.out_features == reference.num_key_value_heads * reference.head_dim
        assert attention.qkv_linear.wq.bias == reference.attention_bias
        assert attention.qk_norm.eps == reference.rms_norm_eps
        assert attention.query_pre_attn_scalar == reference.query_pre_attn_scalar
        assert layer.feed_forward.w1.out_features == reference.intermediate_size
        assert layer.feed_forward.activation.approximate == "tanh"
    rope_fields = {"rope_scaling", "rope_theta", "rope_local_base_freq", "param_init", "sharding_config"}
    for name, actual in recipe.to_dict().items():
        if name not in rope_fields:
            assert actual == getattr(reference, name), name
    # HF 4.x exposes separate RoPE fields; 5.x groups them by attention type.
    # Both representations must describe the same production configuration.
    rope_parameters = getattr(reference, "rope_parameters", None)
    if rope_parameters is None:
        assert reference.rope_scaling == recipe.rope_scaling
        assert reference.rope_theta == recipe.rope_theta
        assert reference.rope_local_base_freq == recipe.rope_local_base_freq
    else:
        assert rope_parameters["full_attention"] == {
            "rope_type": "linear",
            "factor": 8.0,
            "rope_theta": recipe.rope_theta,
        }
        assert rope_parameters["sliding_attention"] == {
            "rope_type": "default",
            "rope_theta": recipe.rope_local_base_freq,
        }
    assert recipe.rope_scaling == {"factor": 8.0, "rope_type": "linear"}
    assert json.loads(json.dumps(parsed.to_dict())) == parsed.to_dict()


@pytest.mark.parametrize(
    ("flavor", "dim", "vocab_size", "num_layers"),
    [("debug", 64, 128, 4), ("4B", 2560, 262208, 34)],
    ids=["debug", "four-billion"],
)
def test_registry_returns_independent_root_configs_without_parameters(flavor, dim, vocab_size, num_layers):
    # Guard creation itself, including parameters a factory might discard.
    with patch.object(
        torch.nn.Parameter, "__new__", side_effect=AssertionError("Config factories must not allocate parameters")
    ):
        first = model_registry(flavor).model
        second = model_registry(flavor).model

    assert isinstance(first, Gemma3Model.Config)
    assert isinstance(second, Gemma3Model.Config)
    assert first.dim == dim
    assert first.vocab_size == vocab_size
    assert len(first.layers) == num_layers
    nodes = {fqn: config for fqn, config, _, _ in first.traverse(Module.Config, recurse=True)}
    assert {"tok_embeddings", "norm", "lm_head"} <= nodes.keys()
    layer_paths = (
        "attention",
        "attention.qk_norm",
        "attention.rope",
        "feed_forward",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    )
    for index in range(num_layers):
        assert f"layers.{index}" in nodes
        assert {f"layers.{index}.{path}" for path in layer_paths} <= nodes.keys()

    nodes["layers.0.attention.qk_norm"].eps = 2e-6
    assert first.layers[0].attention.qk_norm.eps == 2e-6
    assert first.layers[1].attention.qk_norm.eps == 1e-6
    assert second.layers[0].attention.qk_norm.eps == 1e-6


@pytest.mark.parametrize("invalid", ["missing-width", "nondivisible-gqa"], ids=["missing-width", "nondivisible-gqa"])
def test_external_text_config_rejects_missing_width_and_invalid_gqa(invalid):
    text = gemma3_4b_text_config().to_dict()
    if invalid == "missing-width":
        text.pop("hidden_size")
        match = "hidden_size"
    else:
        text["num_attention_heads"] = 3
        match = "divisible"
    with pytest.raises(ValueError, match=match):
        parse_gemma3_text_config({"text_config": text})


def test_parser_preserves_optional_padding_id():
    text = gemma3_4b_text_config().to_dict()
    text["pad_token_id"] = None

    parsed = parse_gemma3_text_config({"text_config": text})

    assert parsed.pad_token_id is None


def test_factory_rejects_linear_rope_without_factor():
    text = gemma3_4b_text_config()
    text.rope_scaling = {"rope_type": "linear"}

    with pytest.raises(ValueError, match="requires factor"):
        _make_gemma3_config(text)


def test_full_module_config_manager_selects_sft_loader_and_explicit_assets(sft_paths):
    default = config_registry.sft_gemma3_4b_fsdp2()
    assert Path(default.hf_assets_path).resolve() == sft_paths
    assert default.checkpoint.initial_load_path is None
    assets = sft_paths.parent / "cli-assets"
    assets.mkdir()
    parsed = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.gemma3",
            "--config",
            "sft_gemma3_4b_fsdp2",
            "--hf-assets-path",
            str(assets),
        ]
    )
    assert parsed.model_spec.name == "gemma3"
    assert isinstance(parsed.model_spec.model, Gemma3Model.Config)
    assert isinstance(parsed.tokenizer, HuggingFaceTokenizer.Config)
    assert parsed.loss.global_vocab_size == parsed.model_spec.model.vocab_size
    assert isinstance(parsed.dataloader, ChatDataLoader.Config)
    assert Path(parsed.hf_assets_path).resolve() == assets
    assert parsed.checkpoint.initial_load_path is None
    assert parsed.dataloader.dataset_path == "openai/gsm8k"
    assert parsed.dataloader.load_dataset_kwargs == {"name": "main", "split": "train"}
    assert parsed.parallelism.data_parallel_shard_degree == 2
    assert parsed.parallelism.spmd_backend == "partial_dtensor"


def test_offline_sft_batch_preserves_packing_and_assistant_only_loss(tmp_path, sft_paths, recipe, model):
    _write_tokenizer(sft_paths)
    data_path = tmp_path / "samples.jsonl"
    samples = [{"question": "one", "answer": "two"}, {"question": "three", "answer": "four five"}]
    data_path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")
    parsed = ConfigManager().parse_args(["--module", "torchtitan_npu.models.gemma3", "--config", "sft_gemma3_4b_fsdp2"])
    parsed.model_spec = replace(parsed.model_spec, model=recipe.model_spec.model)
    parsed.loss = replace(parsed.loss, global_vocab_size=recipe.model_spec.model.vocab_size)
    parsed.training = replace(parsed.training, seq_len=40, local_batch_size=1)
    parsed.dataloader = replace(
        parsed.dataloader,
        dataset_path="json",
        load_dataset_kwargs={"data_files": str(data_path), "split": "train", "cache_dir": str(tmp_path / "cache")},
        infinite=False,
    )
    tokenizer = parsed.tokenizer.build(tokenizer_path=parsed.hf_assets_path)
    loader = parsed.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=parsed.training.seq_len,
        local_batch_size=parsed.training.local_batch_size,
    )
    input_dict, labels = next(iter(loader))

    # Hand-written IDs from the shared byte-level vocabulary and Gemma3 chat template.
    # Allow either shuffle order; both complete documents and the tail are checked.
    documents = [
        [2002, 2004, 2006, 111, 110, 101, 2005, 2004, 2007, 116, 119, 111, 2005],
        [2002, 2004, 2006, 116, 104, 114, 101, 101, 2005, 2004, 2007, 102, 111, 117, 114, 102, 105, 118, 101, 2005],
    ]
    targets = [
        [-100] * 8 + [116, 119, 111, 2005, 2003],
        [-100] * 10 + [102, 111, 117, 114, 102, 105, 118, 101, 2005, 2003],
    ]
    if input_dict["input"][0, 3].item() == 116:
        documents.reverse()
        targets.reverse()
    expected_tokens = torch.tensor([documents[0] + documents[1] + [2003] * 7])
    expected_labels = torch.tensor([targets[0] + targets[1] + [-100] * 7])
    expected_positions = torch.tensor(
        [list(range(len(documents[0]))) + list(range(len(documents[1]))) + list(range(7))]
    )
    assert set(input_dict) == {"input", "positions"}
    torch.testing.assert_close(input_dict["input"], expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(input_dict["positions"], expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)

    # Supply only the state needed by the real Trainer preprocessing method;
    # this CPU test does not initialize distributed training or shard parameters.
    trainer_state = SimpleNamespace(
        model_parts=[model],
        config=parsed,
        device=torch.device("cpu"),
        parallel_dims=ParallelDims.from_config(recipe.parallelism, world_size=1),
        ntokens_seen=0,
    )
    with patch.object(model, "preprocess_inputs", wraps=model.preprocess_inputs) as preprocess:
        tokens, forwarded_labels, kwargs = Trainer.post_dataloading_process(trainer_state, input_dict, labels)
    preprocess.assert_called_once_with(
        input_dict,
        labels,
        parallel_dims=trainer_state.parallel_dims,
        device=trainer_state.device,
        parallelism=parsed.parallelism,
    )
    assert tokens is input_dict["input"]
    assert forwarded_labels is labels
    assert set(kwargs) == {"positions", "attention_masks"}
    assert kwargs["positions"] is input_dict["positions"]

    logits = model(tokens, **kwargs)
    with torch.no_grad():
        separate = torch.cat([model(torch.tensor([document])) for document in documents], dim=1)
    torch.testing.assert_close(logits[:, :33], separate, rtol=1e-5, atol=1e-5)
    valid = expected_labels != -100
    loss, _ = parsed.loss.build(compile_config=parsed.compile)(logits, forwarded_labels, valid.sum())
    expected_loss = F.cross_entropy(logits[valid], expected_labels[valid])
    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
    logits.retain_grad()
    loss.backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0
    assert torch.count_nonzero(logits.grad[valid]) > 0
    assert torch.count_nonzero(model.tok_embeddings.weight.grad) > 0


def test_production_sft_preserves_explicit_initial_load_path(sft_paths, tmp_path):
    cli_assets = tmp_path / "cli-assets"
    cli_assets.mkdir()
    initial_path = tmp_path / "separate-hf-weights"
    args = [
        "--module",
        "torchtitan_npu.models.gemma3",
        "--config",
        "sft_gemma3_4b_fsdp2",
        "--hf-assets-path",
        str(cli_assets),
        "--checkpoint.initial-load-path",
        str(initial_path),
    ]
    parsed = ConfigManager().parse_args(args)

    assert Path(parsed.hf_assets_path).resolve() == cli_assets
    assert parsed.checkpoint.initial_load_path == str(initial_path)
