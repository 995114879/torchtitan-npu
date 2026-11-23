# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 text model configuration, HF parsing, and model registration."""

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torchtitan.models.common.attention import FlexAttention, QKVLinear
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.protocols.model_spec import ModelSpec

from .attention import Gemma3Attention, Gemma3FlexAttention, Gemma3RotaryEmbedding
from .model import (
    Gemma3DecoderLayer,
    Gemma3MLP,
    Gemma3Model,
    Gemma3RMSNorm,
    Gemma3ScaledEmbedding,
)
from .parallelize import parallelize_gemma3
from .state_dict_adapter import Gemma3StateDictAdapter

# These values are the effective defaults of the pinned Gemma3TextConfig
# reference (Transformers 4.57.6), not values copied from vision_config.
_REFERENCE_DEFAULTS: dict[str, Any] = {
    "vocab_size": 262208,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "query_pre_attn_scalar": 256,
    "max_position_embeddings": 131072,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "hidden_activation": "gelu_pytorch_tanh",
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
}

_REQUIRED_TEXT_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
)


def _default_layer_types(num_hidden_layers: int) -> list[str]:
    """Expand the reference's six-layer local/global attention pattern."""

    return ["full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(num_hidden_layers)]


@dataclass(kw_only=True, slots=True)
class Gemma3TextConfig:
    """HF-shaped text parameters consumed by the decoder configuration factory."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    query_pre_attn_scalar: int | float
    layer_types: list[str]
    sliding_window: int
    rope_local_base_freq: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    attn_logit_softcapping: float | None
    final_logit_softcapping: float | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or None")
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _effective_layer_types(text: Mapping[str, Any], num_hidden_layers: int) -> list[str]:
    raw = text.get("layer_types")
    layer_types = (
        _default_layer_types(num_hidden_layers)
        if raw is None
        else list(raw)
        if isinstance(raw, (list, tuple))
        else None
    )
    if layer_types is None:
        raise ValueError("layer_types must be a list")
    if len(layer_types) != num_hidden_layers:
        raise ValueError(f"layer_types length must equal num_hidden_layers ({len(layer_types)} != {num_hidden_layers})")
    allowed = {"sliding_attention", "full_attention"}
    if any(item not in allowed for item in layer_types):
        raise ValueError("layer_types entries must be sliding_attention or full_attention")
    return layer_types


def parse_gemma3_text_config(composite_config: Mapping[str, Any]) -> Gemma3TextConfig:
    """Extract and validate the text decoder from a composite config."""

    composite = _require_mapping(composite_config, "composite_config")
    text = _require_mapping(composite.get("text_config"), "text_config")
    missing = [name for name in _REQUIRED_TEXT_FIELDS if name not in text]
    if missing:
        raise ValueError("text_config missing required field(s): " + ", ".join(missing))
    if text["model_type"] != "gemma3_text":
        raise ValueError(f"text_config.model_type must be 'gemma3_text', got {text['model_type']!r}")

    hidden_size = _positive_int(text["hidden_size"], "hidden_size")
    intermediate_size = _positive_int(text["intermediate_size"], "intermediate_size")
    num_hidden_layers = _positive_int(text["num_hidden_layers"], "num_hidden_layers")
    values = {**_REFERENCE_DEFAULTS, **text}
    num_attention_heads = _positive_int(values["num_attention_heads"], "num_attention_heads")
    num_key_value_heads = _positive_int(values["num_key_value_heads"], "num_key_value_heads")
    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads "
            f"({num_attention_heads} % {num_key_value_heads} != 0)"
        )
    head_dim = _positive_int(values["head_dim"], "head_dim")
    query_pre_attn_scalar = values["query_pre_attn_scalar"]
    if not isinstance(query_pre_attn_scalar, (int, float)) or query_pre_attn_scalar <= 0:
        raise ValueError("query_pre_attn_scalar must be positive")
    sliding_window = _positive_int(values.get("sliding_window", 1024), "sliding_window")
    max_position_embeddings = _positive_int(values["max_position_embeddings"], "max_position_embeddings")
    rope_scaling = values.get("rope_scaling")
    if rope_scaling is not None:
        rope_scaling = dict(_require_mapping(rope_scaling, "rope_scaling"))
        if "factor" in rope_scaling:
            rope_scaling["factor"] = float(rope_scaling["factor"])
    return Gemma3TextConfig(
        model_type="gemma3_text",
        vocab_size=_positive_int(values["vocab_size"], "vocab_size"),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        query_pre_attn_scalar=query_pre_attn_scalar,
        layer_types=_effective_layer_types(text, num_hidden_layers),
        sliding_window=sliding_window,
        rope_local_base_freq=float(values["rope_local_base_freq"]),
        rope_theta=float(values["rope_theta"]),
        rope_scaling=rope_scaling,
        attn_logit_softcapping=values.get("attn_logit_softcapping"),
        final_logit_softcapping=values.get("final_logit_softcapping"),
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=_boolean(values["tie_word_embeddings"], "tie_word_embeddings"),
        bos_token_id=int(values["bos_token_id"]),
        eos_token_id=int(values["eos_token_id"]),
        pad_token_id=_optional_int(values["pad_token_id"], "pad_token_id"),
        rms_norm_eps=float(values["rms_norm_eps"]),
        attention_bias=_boolean(values["attention_bias"], "attention_bias"),
        hidden_activation=str(values["hidden_activation"]),
    )


def debug_gemma3_text_config() -> Gemma3TextConfig:
    """Return the small text-only input configuration."""

    return Gemma3TextConfig(
        model_type="gemma3_text",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4,
        rope_local_base_freq=10000.0,
        rope_theta=1000000.0,
        rope_scaling={"factor": 8.0, "rope_type": "linear"},
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )


def _make_gemma3_attention_config(config: Gemma3TextConfig, layer_idx: int) -> Gemma3Attention.Config:
    if layer_idx < 0 or layer_idx >= config.num_hidden_layers:
        raise ValueError(f"layer_idx out of range: {layer_idx}")
    rope_scaling = 1.0
    if config.rope_scaling is not None:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        if rope_type == "linear":
            if "factor" not in config.rope_scaling:
                raise ValueError("linear Gemma3 rope_scaling requires factor")
            rope_scaling = float(config.rope_scaling["factor"])
        elif rope_type != "default":
            raise ValueError(f"unsupported Gemma3 rope_type: {rope_type!r}")
    is_sliding = config.layer_types[layer_idx] == "sliding_attention"
    inner_attention = (
        FlexAttention.Config()
        if config.attn_logit_softcapping is None
        else Gemma3FlexAttention.Config(softcap=config.attn_logit_softcapping)
    )
    return Gemma3Attention.Config(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        qkv_linear=QKVLinear.Config(
            head_dim=config.head_dim,
            wq=Linear.Config(
                in_features=config.hidden_size,
                out_features=config.num_attention_heads * config.head_dim,
                bias=config.attention_bias,
            ),
            wkv=Linear.Config(
                in_features=config.hidden_size,
                out_features=config.num_key_value_heads * config.head_dim,
                bias=config.attention_bias,
            ),
        ),
        wo=Linear.Config(
            in_features=config.num_attention_heads * config.head_dim,
            out_features=config.hidden_size,
            bias=config.attention_bias,
        ),
        qk_norm=Gemma3RMSNorm.Config(normalized_shape=config.head_dim, eps=config.rms_norm_eps),
        rope=Gemma3RotaryEmbedding.Config(
            dim=config.head_dim,
            theta=config.rope_local_base_freq if is_sliding else config.rope_theta,
            scaling_factor=1.0 if is_sliding else rope_scaling,
            max_seq_len=config.max_position_embeddings,
        ),
        inner_attention=inner_attention,
        query_pre_attn_scalar=config.query_pre_attn_scalar,
        layer_type=config.layer_types[layer_idx],
        sliding_window=config.sliding_window,
    )


def _make_gemma3_mlp_config(config: Gemma3TextConfig) -> Gemma3MLP.Config:
    if config.hidden_activation != "gelu_pytorch_tanh":
        raise ValueError(
            f"Gemma3 only supports hidden_activation='gelu_pytorch_tanh'; got {config.hidden_activation!r}"
        )
    linear_init: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "weight": partial(torch.nn.init.kaiming_uniform_, a=math.sqrt(5))
    }
    ffn_config = make_ffn_config(
        dim=config.hidden_size,
        hidden_dim=config.intermediate_size,
        w1_param_init=linear_init,
        w2w3_param_init=linear_init,
    )
    # The upstream helper retains the supplied dictionaries by reference.
    for projection in (ffn_config.w1, ffn_config.w2, ffn_config.w3):
        projection.param_init = dict(linear_init)
    return Gemma3MLP.Config(
        w1=ffn_config.w1,
        w2=ffn_config.w2,
        w3=ffn_config.w3,
        param_init=ffn_config.param_init,
        sharding_config=ffn_config.sharding_config,
        activation=GELU.Config(approximate="tanh"),
    )


def _make_gemma3_layer_config(config: Gemma3TextConfig, layer_idx: int) -> Gemma3DecoderLayer.Config:
    return Gemma3DecoderLayer.Config(
        attention=_make_gemma3_attention_config(config, layer_idx),
        feed_forward=_make_gemma3_mlp_config(config),
        input_layernorm=Gemma3RMSNorm.Config(normalized_shape=config.hidden_size, eps=config.rms_norm_eps),
        post_attention_layernorm=Gemma3RMSNorm.Config(normalized_shape=config.hidden_size, eps=config.rms_norm_eps),
        pre_feedforward_layernorm=Gemma3RMSNorm.Config(normalized_shape=config.hidden_size, eps=config.rms_norm_eps),
        post_feedforward_layernorm=Gemma3RMSNorm.Config(normalized_shape=config.hidden_size, eps=config.rms_norm_eps),
    )


def _make_gemma3_config(config: Gemma3TextConfig) -> Gemma3Model.Config:
    return Gemma3Model.Config(
        dim=config.hidden_size,
        vocab_size=config.vocab_size,
        tok_embeddings=Gemma3ScaledEmbedding.Config(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        ),
        lm_head=Linear.Config(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            param_init={"weight": skip_param_init} if config.tie_word_embeddings else None,
        ),
        norm=Gemma3RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
        ),
        layers=[_make_gemma3_layer_config(config, layer_idx) for layer_idx in range(config.num_hidden_layers)],
        enable_weight_tying=config.tie_word_embeddings,
        max_position_embeddings=config.max_position_embeddings,
        final_logit_softcapping=config.final_logit_softcapping,
    )


def debug_gemma3_config() -> Gemma3Model.Config:
    """Return a small decoder configuration tree with local and global attention."""

    return _make_gemma3_config(debug_gemma3_text_config())


def gemma3_4b_text_config() -> Gemma3TextConfig:
    """Return the pinned 4B text-only input configuration."""

    return parse_gemma3_text_config(
        {
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_hidden_layers": 34,
                "sliding_window": 1024,
                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            },
        }
    )


def gemma3_4b_config() -> Gemma3Model.Config:
    """Return the pinned 4B decoder configuration tree."""

    return _make_gemma3_config(gemma3_4b_text_config())


_FLAVORS = {
    "debug": debug_gemma3_config,
    "4B": gemma3_4b_config,
}


def model_registry(flavor: str) -> ModelSpec:
    try:
        model_config = _FLAVORS[flavor]()
    except KeyError as error:
        supported = ", ".join(sorted(_FLAVORS))
        raise ValueError(f"Unknown Gemma3 flavor {flavor!r}; expected one of: {supported}") from error

    return ModelSpec(
        name="gemma3",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_gemma3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Gemma3StateDictAdapter,
    )
