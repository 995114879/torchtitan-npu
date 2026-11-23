# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace state-dict mapping for the Gemma3 text decoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder

_NATIVE_TO_HF_SUFFIX = {
    "attention.qkv_linear.wq.weight": "self_attn.q_proj.weight",
    "attention.qkv_linear.wk.weight": "self_attn.k_proj.weight",
    "attention.qkv_linear.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention.qkv_linear.wq.bias": "self_attn.q_proj.bias",
    "attention.qkv_linear.wk.bias": "self_attn.k_proj.bias",
    "attention.qkv_linear.wv.bias": "self_attn.v_proj.bias",
    "attention.wo.bias": "self_attn.o_proj.bias",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
}
_HF_TO_NATIVE_SUFFIX = {value: key for key, value in _NATIVE_TO_HF_SUFFIX.items()}
_NATIVE_ROOT_SUFFIX = {
    "tok_embeddings.weight": "embed_tokens.weight",
    "norm.weight": "norm.weight",
}
_HF_ROOT_TO_NATIVE = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "model.lm_head.weight": "lm_head.weight",
}


def _checkpoint_model_prefix(path: str | Path) -> str:
    """Return the text-model prefix found in an actual HF checkpoint."""
    root = Path(path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        keys = list(index["weight_map"])
    else:
        keys = []
        for shard_path in sorted(root.glob("*.safetensors")):
            with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                keys.extend(checkpoint.keys())

    if any(key.startswith("language_model.model.") for key in keys):
        return "language_model.model"
    if any(key.startswith("model.") or key == "lm_head.weight" for key in keys):
        return "model"
    raise ValueError(f"Gemma3 HF checkpoint has no recognized text-model keys: {root}")


def _rename_hf_keys(state_dict: dict[str, Any], source_prefix: str, target_prefix: str) -> dict[str, Any]:
    """Rename model and language-model head keys between HF layouts."""
    renamed: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key == "lm_head.weight":
            target_key = "language_model.lm_head.weight" if target_prefix == "language_model.model" else key
        elif key == "language_model.lm_head.weight":
            target_key = "lm_head.weight" if target_prefix == "model" else key
        else:
            target_key = key.replace(source_prefix + ".", target_prefix + ".", 1)
        renamed[target_key] = value
    return renamed


def _expected_native_shapes(config: Decoder.Config) -> dict[str, tuple[int, ...]]:
    """Read parameter shapes from the same child configs used to build the model."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name, child, _, _ in config.traverse(Module.Config, recurse=True):
        # QKVLinear and GQAttention each build two modules from one child config.
        if name.endswith(".qkv_linear.wkv"):
            prefix = name.removesuffix(".wkv")
            names = (f"{prefix}.wk", f"{prefix}.wv")
        elif name.endswith(".qk_norm"):
            prefix = name.removesuffix(".qk_norm")
            names = (f"{prefix}.q_norm", f"{prefix}.k_norm")
        else:
            names = (name,)
        for module_name in names:
            if isinstance(child, Embedding.Config):
                shapes[f"{module_name}.weight"] = (child.num_embeddings, child.embedding_dim)
            elif isinstance(child, Linear.Config):
                shapes[f"{module_name}.weight"] = (child.out_features, child.in_features)
                if child.bias:
                    shapes[f"{module_name}.bias"] = (child.out_features,)
            elif isinstance(child, RMSNorm.Config) and child.elementwise_affine:
                shapes[f"{module_name}.weight"] = (child.normalized_shape,)
    return shapes


def _native_to_hf_key(native_key: str, *, model_prefix: str = "model") -> str:
    root_suffix = _NATIVE_ROOT_SUFFIX.get(native_key)
    if root_suffix is not None:
        return f"{model_prefix}.{root_suffix}"
    if native_key == "lm_head.weight":
        return "lm_head.weight" if model_prefix == "model" else f"{model_prefix.removesuffix('.model')}.lm_head.weight"
    if not native_key.startswith("layers."):
        raise ValueError(f"unknown Gemma3 native parameter key: {native_key}")
    parts = native_key.split(".", 2)
    if len(parts) != 3 or parts[2] not in _NATIVE_TO_HF_SUFFIX:
        raise ValueError(f"unknown Gemma3 native parameter key: {native_key}")
    return f"{model_prefix}.layers.{parts[1]}.{_NATIVE_TO_HF_SUFFIX[parts[2]]}"


def _canonical_hf_key(hf_key: str) -> str | None:
    """Normalize standalone-text and composite text prefixes to ``model.*``."""

    if hf_key == "lm_head.weight":
        return "model.lm_head.weight"
    if hf_key.startswith("language_model.model."):
        return "model." + hf_key.removeprefix("language_model.model.")
    if hf_key.startswith("language_model.lm_head."):
        return "model.lm_head." + hf_key.removeprefix("language_model.lm_head.")
    if hf_key.startswith("model."):
        return hf_key
    return None


def _hf_to_native_key(canonical_hf_key: str) -> str:
    native_root = _HF_ROOT_TO_NATIVE.get(canonical_hf_key)
    if native_root is not None:
        return native_root
    if not canonical_hf_key.startswith("model.layers."):
        raise ValueError(f"unknown Gemma3 HF text parameter key: {canonical_hf_key}")
    parts = canonical_hf_key.split(".", 3)
    if len(parts) != 4 or parts[3] not in _HF_TO_NATIVE_SUFFIX:
        raise ValueError(f"unknown Gemma3 HF text parameter key: {canonical_hf_key}")
    return f"layers.{parts[2]}.{_HF_TO_NATIVE_SUFFIX[parts[3]]}"


class Gemma3StateDictAdapter(StateDictAdapter):
    """Map Gemma3 text parameters while explicitly filtering composite weights."""

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self._tied = bool(model_config.enable_weight_tying)
        self._model_prefix = "model"
        self._pending_hf_state_dict: dict[str, Any] | None = None
        if self.fqn_to_index_mapping is not None:
            if any(key.startswith("language_model.model.") for key in self.fqn_to_index_mapping):
                self._model_prefix = "language_model.model"
            filtered_mapping: dict[str, int] = {}
            for key, index in self.fqn_to_index_mapping.items():
                if _canonical_hf_key(key) is not None:
                    filtered_mapping[key] = index
            self.fqn_to_index_mapping = filtered_mapping
        elif hf_assets_path:
            for shard_path in sorted(Path(hf_assets_path).glob("*.safetensors")):
                with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                    checkpoint_keys = checkpoint.keys()
                    if any(key.startswith("language_model.model.") for key in checkpoint_keys):
                        self._model_prefix = "language_model.model"
                        break

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader:
        if from_quantized:
            raise ValueError("Gemma3 text adapter does not support quantized checkpoints")
        if self._pending_hf_state_dict is not None:
            # TorchTitan 0.3.0 creates the load dictionary before supplying the path.
            actual_prefix = _checkpoint_model_prefix(path)
            source_prefix = self._model_prefix
            if actual_prefix != source_prefix:
                renamed = _rename_hf_keys(self._pending_hf_state_dict, source_prefix, actual_prefix)
                self._pending_hf_state_dict.clear()
                self._pending_hf_state_dict.update(renamed)
            self._pending_hf_state_dict = None
        root = Path(path)
        if (root / "model.safetensors.index.json").is_file() or list(root.glob("*.safetensors")):
            return HuggingFaceStorageReader(path)
        raise ValueError("Gemma3 HF loading requires safetensors; load DCP through CheckpointManager instead")

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = dict(state_dict)
        expected = _expected_native_shapes(cast("Decoder.Config", self.model_config))
        unknown = sorted(set(state_dict) - set(expected))
        if unknown:
            raise ValueError("unknown native key(s): " + ", ".join(unknown))
        missing = set(expected) - set(state_dict)
        if self._tied:
            missing.discard("lm_head.weight")
        if missing:
            raise ValueError("missing native key(s): " + ", ".join(sorted(missing)))

        result: dict[str, Any] = {}
        for native_key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state-dict value for {native_key} must be a Tensor")
            expected_shape = expected[native_key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"shape mismatch for {native_key}: expected {expected_shape}, got {tuple(value.shape)}"
                )
            if native_key == "lm_head.weight" and self._tied:
                if (
                    "tok_embeddings.weight" in state_dict
                    and value is not state_dict["tok_embeddings.weight"]
                    and not torch.equal(value, state_dict["tok_embeddings.weight"])
                ):
                    raise ValueError("tied lm_head.weight conflicts with tok_embeddings.weight")
                if self.fqn_to_index_mapping is not None:
                    hf_key = _native_to_hf_key(native_key, model_prefix=self._model_prefix)
                    self.fqn_to_index_mapping.pop(hf_key, None)
                    canonical_hf_key = _canonical_hf_key(hf_key)
                    if canonical_hf_key is not None:
                        self.fqn_to_index_mapping.pop(canonical_hf_key, None)
                continue
            result[_native_to_hf_key(native_key, model_prefix=self._model_prefix)] = value
        self._pending_hf_state_dict = result
        return result

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        expected = _expected_native_shapes(cast("Decoder.Config", self.model_config))
        canonical: dict[str, Any] = {}
        sources: dict[str, str] = {}
        unknown: list[str] = []

        for source_key, value in hf_state_dict.items():
            if source_key.startswith(("vision_tower.", "multi_modal_projector.")):
                continue
            canonical_key = _canonical_hf_key(source_key)
            if canonical_key is None:
                unknown.append(source_key)
                continue
            try:
                native_key = _hf_to_native_key(canonical_key)
            except ValueError:
                unknown.append(source_key)
                continue
            if canonical_key in canonical:
                previous = sources[canonical_key]
                raise ValueError(f"duplicate HF mapping for {canonical_key}: {previous} and {source_key}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state-dict value for {source_key} must be a Tensor")
            expected_shape = expected[native_key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"shape mismatch for {source_key}: expected {expected_shape}, got {tuple(value.shape)}"
                )
            canonical[canonical_key] = value
            sources[canonical_key] = source_key

        if unknown:
            raise ValueError("unknown HF key(s): " + ", ".join(sorted(unknown)))

        required = set(expected)
        if self._tied:
            required.discard("lm_head.weight")
        missing = []
        for native_key in sorted(required):
            canonical_key = _canonical_hf_key(_native_to_hf_key(native_key))
            if canonical_key not in canonical:
                missing.append(native_key)
        if missing:
            # Use the input's names even when no HF asset path was supplied.
            source_prefixes: set[str] = set()
            for key in sources.values():
                if key.startswith("language_model."):
                    source_prefixes.add("language_model.model")
                else:
                    source_prefixes.add("model")
            model_prefixes = sorted(source_prefixes or {"model", "language_model.model"})
            descriptions: list[str] = []
            for native_key in missing:
                hf_keys = [_native_to_hf_key(native_key, model_prefix=prefix) for prefix in model_prefixes]
                hf_name = hf_keys[0] if len(hf_keys) == 1 else "one of [" + " | ".join(hf_keys) + "]"
                descriptions.append(f"{hf_name} (native: {native_key})")
            raise ValueError("missing HF text key(s): " + ", ".join(descriptions))

        result = {_hf_to_native_key(key): value for key, value in canonical.items()}
        if self._tied:
            embedding = result["tok_embeddings.weight"]
            if "lm_head.weight" in result and not torch.equal(embedding, result["lm_head.weight"]):
                raise ValueError("tied embedding and lm_head weights differ")
            result["lm_head.weight"] = embedding
        return result
