# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent Gemma3 checkpoint and small-config fixtures."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

from torchtitan_npu.models.gemma3 import _make_gemma3_config, debug_gemma3_text_config

_SHARED_TOKENIZER_DIR = Path(__file__).parents[3] / "assets" / "deepseek_v3"

_TEST_CHAT_TEMPLATE = """{%- for message in messages -%}
{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\\n' + (message['content'] | trim) + '<end_of_turn>\\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<start_of_turn>model\\n' }}
{%- endif -%}"""


def _write_tokenizer(root: Path) -> None:
    """Write the shared byte-level tokenizer used by CPU SFT fixtures."""
    if not _SHARED_TOKENIZER_DIR.is_dir():
        raise FileNotFoundError("Gemma3 fixture tokenizer assets are missing")
    shutil.copy2(_SHARED_TOKENIZER_DIR / "tokenizer.json", root / "tokenizer.json")
    tokenizer_config = json.loads((_SHARED_TOKENIZER_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    added_tokens = tokenizer_config.setdefault("added_tokens_decoder", {})
    for token_id, content in {
        2004: "<start_of_turn>",
        2005: "<end_of_turn>",
        2006: "user",
        2007: "model",
    }.items():
        added_tokens[str(token_id)] = {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    tokenizer_config["chat_template"] = _TEST_CHAT_TEMPLATE
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")


def recipe_text_config():
    return replace(
        debug_gemma3_text_config(),
        vocab_size=2008,
        bos_token_id=2002,
        eos_token_id=2003,
        pad_token_id=2001,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        query_pre_attn_scalar=16,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=256,
    )


def recipe_model_config(**overrides):
    return _make_gemma3_config(replace(recipe_text_config(), **overrides))


_ATTENTION_SUFFIXES = ("qkv_linear.wq", "qkv_linear.wk", "qkv_linear.wv", "wo", "q_norm", "k_norm")
_MLP_PROJECTION_NAMES = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))
_LAYER_NORM_SUFFIXES = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)


def _hf_suffix(native_suffix):
    """Translate a native suffix using Gemma3's naming conventions."""
    components = native_suffix.split(".")
    if components[0] == "attention":
        names = {
            "qkv_linear.wq": "q_proj",
            "qkv_linear.wk": "k_proj",
            "qkv_linear.wv": "v_proj",
            "wo": "o_proj",
            "q_norm": "q_norm",
            "k_norm": "k_norm",
        }
        return f"self_attn.{names['.'.join(components[1:-1])]}.{components[-1]}"
    if components[0] == "feed_forward":
        projection = dict(_MLP_PROJECTION_NAMES)[components[1]]
        return ".".join(("mlp", projection, *components[2:]))
    return native_suffix


def hf_parameter_names(num_layers, *, prefix="model"):
    names = [
        ("tok_embeddings.weight", f"{prefix}.embed_tokens.weight"),
        ("norm.weight", f"{prefix}.norm.weight"),
        (
            "lm_head.weight",
            "lm_head.weight" if prefix == "model" else "language_model.lm_head.weight",
        ),
    ]
    native_suffixes = [*(f"attention.{suffix}.weight" for suffix in _ATTENTION_SUFFIXES)]
    native_suffixes.extend(f"feed_forward.{native}.weight" for native, _ in _MLP_PROJECTION_NAMES)
    native_suffixes.extend(f"{suffix}.weight" for suffix in _LAYER_NORM_SUFFIXES)
    for layer_index in range(num_layers):
        native_prefix = f"layers.{layer_index}."
        hf_prefix = f"{prefix}.layers.{layer_index}."
        names.extend(
            (native_prefix + native_suffix, hf_prefix + _hf_suffix(native_suffix)) for native_suffix in native_suffixes
        )
    return dict(names)
