# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative name rules for loading, quantization, and frozen-weight recovery.

These rules identify source tensors: quantization targets/exclusions, vision and
MTP names, and DCP training-state entries. The written artifact -- its names,
dtypes, description entries, and loadability checks -- lives in
``vllm_export_contract``.

Quantization filters match adapter-output tensor names, not model-config paths.
"""

import re
from dataclasses import dataclass, field

# Quantization targets.

# Select DeepSeek attention and shared-expert Linear weights by logical-name suffix.
DEEPSEEK_V4_DENSE_WEIGHT_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
)
# V4.1 adds the Engram context-gate value projection. The model applies it with
# ``F.linear`` over the fetched memory, and the serving checkpoint stores it
# MXFP8 like every other Linear; the elementwise gate parameters stay dense.
DEEPSEEK_V41_SERVING_ONLY_DENSE_WEIGHT_SUFFIXES = (".engram.wkv.weight",)
DEEPSEEK_V41_DENSE_WEIGHT_SUFFIXES = (
    *DEEPSEEK_V4_DENSE_WEIGHT_SUFFIXES,
    *DEEPSEEK_V41_SERVING_ONLY_DENSE_WEIGHT_SUFFIXES,
)
# Select Qwen3.5 attention and dense MLP Linear weights by HF-name suffix.
QWEN35_DENSE_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
# Select DeepSeek routed-expert w1/w2/w3 weights; fused w13 is split before matching.
ROUTED_WEIGHT_PATTERN = r".*\.ffn\.experts\.\d+\.w[123]\.weight"


@dataclass(frozen=True)
class QuantizationExportRule:
    """HF-name filters needed to quantize one exported model family."""

    dense_weight_suffixes: tuple[str, ...]
    dense_filter_key: str
    routed_weight_pattern: str | None = ROUTED_WEIGHT_PATTERN
    routed_filter_key: str | None = "routed_expert"
    # Training-namespace templates that ``to_hf`` fuses into one HF tensor.
    # Training quantizes the split projections separately, so replay applies
    # their shared config to the fused export weight.
    fused_hf_fqns: dict[str, str] = field(default_factory=dict)
    # Dense suffixes the serving format quantizes although training leaves the
    # parameter unwrapped (it is a bare ``nn.Parameter``, not a Linear). Replay
    # reuses the dense config so the artifact matches a recipe-based export.
    serving_only_dense_weight_suffixes: tuple[str, ...] = ()


# ``to_hf`` concatenates the split GatedDeltaNet projections into one HF
# ``in_proj_qkv`` tensor; replay maps each split FQN back to that fused name.
_QWEN35_FUSED_HF_FQNS = {
    f"layers.{{}}.attn.in_proj_{projection}.weight": ("model.language_model.layers.{}.linear_attn.in_proj_qkv.weight")
    for projection in ("q", "k", "v")
}


# Keep model-specific naming in one registry.  Adding a model should require a
# new entry here and a recipe/filter-key declaration in the adaptor config, not
# another conditional branch in the exporter.
QUANTIZATION_EXPORT_RULES: dict[str, QuantizationExportRule] = {
    "deepseek_v4": QuantizationExportRule(DEEPSEEK_V4_DENSE_WEIGHT_SUFFIXES, dense_filter_key="dense"),
    "deepseek_v4_1": QuantizationExportRule(
        DEEPSEEK_V41_DENSE_WEIGHT_SUFFIXES,
        dense_filter_key="dense",
        serving_only_dense_weight_suffixes=DEEPSEEK_V41_SERVING_ONLY_DENSE_WEIGHT_SUFFIXES,
    ),
    "qwen3_5": QuantizationExportRule(
        QWEN35_DENSE_WEIGHT_SUFFIXES,
        dense_filter_key="dense_model",
        routed_weight_pattern=None,
        routed_filter_key=None,
        fused_hf_fqns=_QWEN35_FUSED_HF_FQNS,
    ),
}


def get_quantization_export_rule(model_name: str) -> QuantizationExportRule:
    """Return the HF-name quantization rule for a canonical model name."""
    rule = QUANTIZATION_EXPORT_RULES.get(model_name)
    if rule is None:
        raise ValueError(
            f"unsupported quantized model {model_name!r}; expected one of {tuple(QUANTIZATION_EXPORT_RULES)}"
        )
    return rule


# Quantization exclusions and frozen-weight recovery.

# Vision tower and aligner names, in local and HF spellings. The marker vectors are
# listed separately: a checkpoint carrying only markers is not a complete tower.
VISION_TOWER_PREFIXES = (
    "model.visual.",
    "model.visual_merger.",
    "visual.",
    "vision.",
    "vision_encoder.",
    "aligner.",
)
# V4.1 release checkpoints spell the frozen marker vectors as bare top-level names.
VISION_MARKER_NAMES = ("image_start", "image_end", "image_newline")
# Local namespace of the same marker vectors.
IMAGE_MARKER_PREFIX = "image_marker_embeddings."
# Skip vision quantization and select original HF vision weights for restoration;
# also allow extra vision entries during DCP validation. This is the widest set:
# the tower check deliberately drops the marker entries again (``has_vision_tower``).
VISION_PREFIXES = (*VISION_TOWER_PREFIXES, IMAGE_MARKER_PREFIX, *VISION_MARKER_NAMES)
# Vision-owned routing bias that lives in the language layers' HF namespace. A
# text-only flavor builds no such parameter, but vLLM Ascend builds one per MoE
# layer whenever the served model has a vision tower.
VISION_BIAS_RE = re.compile(r"^layers\.\d+\.ffn\.gate\.bias_vl$")
# Skip MTP quantization even when its weights match the target suffixes.
MTP_PREFIXES = ("mtp.", "mtp_layers.", "mtp.layers.", "model.mtp.")


# DCP load validation.

# Allow these frozen vision weights to be absent from the training checkpoint.
LOCAL_VISION_PREFIXES = ("vision_encoder.", "image_marker_embeddings.")


# Expert layout.

# Extract layer, expert, and projection IDs for DeepSeek expert validation and w13 splitting.
EXPERT_FQN_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w(13|[123])\.weight$")
