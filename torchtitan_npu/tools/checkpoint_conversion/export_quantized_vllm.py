# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export DCP or Qwen3.5 HF weights as quantized safetensors for vLLM Ascend.

Quantized weights are written as ordinary ``weight``/``weight_scale`` tensors and
the applied per-parameter format is recorded in ``quant_model_description.json``.
For a quantized-training DCP checkpoint, wrapper configs are replayed by default;
``--recipe`` explicitly rebuilds the complete export plan. A checkpoint with no
embedded configs fails instead of falling back to a recipe. Replay also applies
the checkpoint's dense config to serving-format dense weights that training
leaves unwrapped, and maps split projections back onto the fused HF weight
``to_hf`` emits.
"""

import argparse
import copy
import importlib
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

if __package__:
    from . import dcp_quant_configs as _dcp_quant_configs
    from .vllm_export_contract import (
        ENGRAM_MX_BLOCK_SIZE,
        ENGRAM_TABLE_RE,
        NATIVE_DTYPES_BY_TAG,
        block_count,
        build_quant_model_description,
        effective_weight_config,
        engram_scale_name,
        has_vision_tower,
        quant_description_type,
        read_safetensors_header,
        scale_name,
        sync_vision_config,
        validate_export,
        vision_prefixes_from_hf_assets,
        write_quant_model_description,
    )
    from .vllm_export_specs import (
        EXPERT_FQN_RE,
        LOCAL_VISION_PREFIXES,
        MTP_PREFIXES,
        VISION_BIAS_RE,
        VISION_PREFIXES,
        get_quantization_export_rule,
    )
else:
    # Script mode: this file's directory is on sys.path, and the sibling modules
    # import each other by bare name.
    import dcp_quant_configs as _dcp_quant_configs  # pyrefly: ignore [missing-import]
    from vllm_export_contract import (  # pyrefly: ignore [missing-import]
        ENGRAM_MX_BLOCK_SIZE,
        ENGRAM_TABLE_RE,
        NATIVE_DTYPES_BY_TAG,
        block_count,
        build_quant_model_description,
        effective_weight_config,
        engram_scale_name,
        has_vision_tower,
        quant_description_type,
        read_safetensors_header,
        scale_name,
        sync_vision_config,
        validate_export,
        vision_prefixes_from_hf_assets,
        write_quant_model_description,
    )
    from vllm_export_specs import (  # pyrefly: ignore [missing-import]
        EXPERT_FQN_RE,
        LOCAL_VISION_PREFIXES,
        MTP_PREFIXES,
        VISION_BIAS_RE,
        VISION_PREFIXES,
        get_quantization_export_rule,
    )

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent


def _bootstrap_sys_path() -> None:
    for path in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import torchao_npu.serialization  # noqa: F401
    except ImportError:
        fallback = _REPO_ROOT / "experiments" / "torchao-npu"
        if fallback.is_dir() and str(fallback) not in sys.path:
            sys.path.insert(0, str(fallback))


_bootstrap_sys_path()


def _model_type(model_name: str):
    """Map a canonical model name to the quantized-training model type."""
    from torchtitan_npu.extensions.adaptor.torchao.configs import _NAME_TO_TYPE

    model_type = _NAME_TO_TYPE.get(model_name)
    if model_type is None:
        raise ValueError(f"unsupported quantized model {model_name!r}; expected one of {tuple(_NAME_TO_TYPE)}")
    return model_type


def build_recipe_plan(recipe: str, *, model_name: str, enable_mxfp4_qat: bool, dst_type_max: float = 0.0):
    """Use the model's training recipe factory with checkpoint tensor-name filters."""
    from torchao_npu.quantization.filters import match_fqn_regex, match_fqn_suffix
    from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig

    from torchtitan_npu.extensions.adaptor.torchao.configs import get_model_config
    from torchtitan_npu.extensions.adaptor.torchao.recipes import RecipeContext

    model_type = _model_type(model_name)
    matrix = get_model_config(model_type).recipe_matrix
    if recipe not in matrix:
        raise ValueError(f"recipe for model {model_name!r} must be one of {tuple(matrix)}, got {recipe!r}")
    export_rule = get_quantization_export_rule(model_name)
    dense_filter = match_fqn_suffix(*export_rule.dense_weight_suffixes)
    routed_filter = (
        match_fqn_regex(export_rule.routed_weight_pattern) if export_rule.routed_weight_pattern is not None else None
    )
    filters = {export_rule.dense_filter_key: dense_filter}
    if routed_filter is not None and export_rule.routed_filter_key is not None:
        filters[export_rule.routed_filter_key] = routed_filter
    ctx = RecipeContext(
        model_type=model_type,
        model_compile_enabled=False,
        fsdp_prequantize_dtype="fp8",
        enable_mxfp4_qat=enable_mxfp4_qat,
        dst_type_max=dst_type_max,
    )
    specs = matrix[recipe](filters, ctx)
    for spec in specs:
        weight_config = spec.swap_config.weight_config
        if not (type(weight_config) is MXQuantizeConfig or isinstance(weight_config, BlockMXQuantizeConfig)):
            raise ValueError(f"recipe {recipe!r}: export does not support {type(weight_config).__name__}")
        quant_description_type(weight_config)
    return [(spec.swap_config, spec.filter) for spec in specs]


def _same_quantization_config(left, right) -> bool:
    """Compare kernel options too: torchao dataclass equality omits some fields."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _same_quantization_config(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
            # These communication options are unused by ``to_inference_weight``.
            if item.name not in ("fsdp_prequantize", "fsdp_prequantize_dtype")
        )
    return left == right


def _reject_mismatched_fused_configs(extracted: dict[str, tuple[Any, Any]], export_rule) -> None:
    """Every projection fused into one HF weight must be present and carry the same quantization config."""
    required: dict[str, set[str]] = {}
    for abstract, fused_hf_fqn in export_rule.fused_hf_fqns.items():
        required.setdefault(fused_hf_fqn, set()).add(abstract)
    groups: dict[tuple[str, str], list[str]] = {}
    for train_fqn in extracted:
        abstract, layer_num = _dcp_quant_configs.abstract_train_fqn(train_fqn)
        fused_hf_fqn = export_rule.fused_hf_fqns.get(abstract)
        if fused_hf_fqn is None:
            continue
        groups.setdefault((fused_hf_fqn, layer_num or ""), []).append(train_fqn)
    for (fused_hf_fqn, layer), train_fqns in groups.items():
        # A fused weight needs one config for the whole tensor, so a missing
        # split would silently quantize that slice with its sibling's config.
        present = {_dcp_quant_configs.abstract_train_fqn(fqn)[0] for fqn in train_fqns}
        missing = required[fused_hf_fqn] - present
        if missing:
            missing_names = ", ".join(sorted(template.format(layer) if layer else template for template in missing))
            raise ValueError(
                f"training FQNs {', '.join(repr(fqn) for fqn in train_fqns)} fuse into one HF weight "
                f"({fused_hf_fqn!r}) but the checkpoint embeds no quantization config for {missing_names}; "
                "the fused export weight cannot be quantized consistently"
            )
        weight_config, activation_config = extracted[train_fqns[0]]
        for train_fqn in train_fqns[1:]:
            other_weight, other_activation = extracted[train_fqn]
            if not (
                _same_quantization_config(weight_config, other_weight)
                and _same_quantization_config(activation_config, other_activation)
            ):
                raise ValueError(
                    f"training FQNs {train_fqns[0]!r} and {train_fqn!r} fuse into one HF weight "
                    f"({fused_hf_fqn!r}) but carry different quantization configs; "
                    "the fused export weight cannot replay both"
                )


def _apply_serving_only_dense_configs(
    fqn_to_swap: dict[str, Any], hf_state_dict: dict[str, torch.Tensor], export_rule
) -> None:
    """Quantize serving-format dense weights training never wraps with the dense config.

    Some weights are applied through ``F.linear`` over a bare ``nn.Parameter``, so
    quantized training leaves them in full precision while the serving checkpoint
    stores them quantized. Replaying the dense config keeps the artifact identical
    to a recipe-based export.
    """
    missing = [
        fqn
        for fqn in hf_state_dict
        if fqn not in fqn_to_swap
        and not fqn.startswith(MTP_PREFIXES)
        and fqn.endswith(export_rule.serving_only_dense_weight_suffixes)
    ]
    if not missing:
        return
    dense_swaps = [(fqn, swap) for fqn, swap in fqn_to_swap.items() if fqn.endswith(export_rule.dense_weight_suffixes)]
    if not dense_swaps:
        raise ValueError(
            "DCP replay carries no dense quantization config, so the serving-only weight(s) "
            f"({', '.join(missing[:4])}) cannot be quantized like a recipe export"
        )
    dense_fqn, dense_swap = dense_swaps[0]
    for fqn, swap in dense_swaps[1:]:
        if not (
            _same_quantization_config(dense_swap.weight_config, swap.weight_config)
            and _same_quantization_config(dense_swap.activation_config, swap.activation_config)
        ):
            raise ValueError(
                f"DCP replay carries different dense quantization configs ({dense_fqn!r} vs {fqn!r}); "
                "cannot choose one for the serving-only weight(s)"
            )
    for fqn in missing:
        fqn_to_swap[fqn] = dense_swap
    logger.info(
        "DCP replay has no training config for %d serving-only weight(s); applying the dense replay config (first: %s)",
        len(missing),
        ", ".join(missing[:4]),
    )


def _quantizable_hf_fqns(
    hf_state_dict: dict[str, torch.Tensor], export_rule, vision_prefixes: tuple[str, ...]
) -> set[str]:
    """HF weights the export rule would quantize if the plan were complete."""
    expected: set[str] = set()
    for fqn, value in hf_state_dict.items():
        if not value.is_floating_point() or fqn.startswith(vision_prefixes) or fqn.startswith(MTP_PREFIXES):
            continue
        if fqn.endswith(export_rule.dense_weight_suffixes) or (
            export_rule.routed_weight_pattern is not None and re.fullmatch(export_rule.routed_weight_pattern, fqn)
        ):
            expected.add(fqn)
    return expected


def _build_dcp_plan(
    hf_state_dict: dict[str, torch.Tensor],
    extracted: dict[str, tuple[Any, Any]],
    sd_adapter,
    model_name: str,
    *,
    vision_prefixes: tuple[str, ...] = VISION_PREFIXES,
) -> list:
    """Build exact HF-name filters from the configs embedded in a DCP checkpoint."""
    from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig

    export_rule = get_quantization_export_rule(model_name)
    _reject_mismatched_fused_configs(extracted, export_rule)
    fqn_to_swap = _dcp_quant_configs.plan_from_dcp_configs(
        hf_state_dict, extracted, sd_adapter, fused_hf_fqns=export_rule.fused_hf_fqns
    )
    _apply_serving_only_dense_configs(fqn_to_swap, hf_state_dict, export_rule)
    uncovered = sorted(_quantizable_hf_fqns(hf_state_dict, export_rule, vision_prefixes) - set(fqn_to_swap))
    if uncovered:
        raise ValueError(
            f"DCP replay has no quantization config for {len(uncovered)} quantizable weight(s) "
            f"(first: {', '.join(uncovered[:4])}); the exported checkpoint would silently differ from training"
        )
    grouped: list[tuple[Any, set[str]]] = []
    for fqn, swap_config in sorted(fqn_to_swap.items()):
        for existing, targets in grouped:
            if _same_quantization_config(
                existing.weight_config, swap_config.weight_config
            ) and _same_quantization_config(existing.activation_config, swap_config.activation_config):
                targets.add(fqn)
                break
        else:
            grouped.append((swap_config, {fqn}))

    plan = []
    for swap_config, targets in grouped:
        weight_config = swap_config.weight_config
        if not (type(weight_config) is MXQuantizeConfig or isinstance(weight_config, BlockMXQuantizeConfig)):
            raise ValueError(
                f"DCP embedded quantization config {type(weight_config).__name__} is not supported by the vLLM exporter"
            )
        quant_description_type(weight_config)
        target_fqns = frozenset(targets)

        def exact_filter(_param, fqn, target_fqns=target_fqns):
            return fqn in target_fqns

        plan.append((swap_config, exact_filter))
    return plan


def _reject_recipe_flags_when_replaying(
    *, recipe: str | None, extracted: dict[str, tuple[Any, Any]], enable_mxfp4_qat: bool, dst_type_max: float
) -> None:
    """Prevent global recipe flags from being mixed into embedded per-FQN configs."""
    if recipe is None and extracted and (enable_mxfp4_qat or dst_type_max):
        raise ValueError(
            "DCP wrapper quantization configs are being replayed, but recipe flags were also provided "
            f"(--enable_mxfp4_qat={enable_mxfp4_qat} --dst_type_max={dst_type_max}). "
            "Pass --recipe to rebuild the complete export plan, or remove these flags to replay the DCP configs."
        )


def _exportable_recipes(model_name: str) -> tuple[str, ...]:
    """Registered recipes whose plan the exporter can build (HiF8 is rejected)."""
    from torchtitan_npu.extensions.adaptor.torchao.configs import get_model_config

    exportable = []
    for recipe in get_model_config(_model_type(model_name)).recipe_matrix:
        try:
            build_recipe_plan(recipe, model_name=model_name, enable_mxfp4_qat=False, dst_type_max=0.0)
        except ValueError:
            continue
        exportable.append(recipe)
    return tuple(exportable)


def _resolve_quantization_plan(
    *,
    recipe: str | None,
    input_format: str,
    model_name: str,
    hf_state_dict: dict[str, torch.Tensor],
    extracted: dict[str, tuple[Any, Any]],
    sd_adapter,
    enable_mxfp4_qat: bool,
    dst_type_max: float,
    input_dir: Path,
    vision_prefixes: tuple[str, ...] = VISION_PREFIXES,
) -> tuple[list, str]:
    """Resolve the export plan and return ``(plan, description)``."""
    _reject_recipe_flags_when_replaying(
        recipe=recipe,
        extracted=extracted,
        enable_mxfp4_qat=enable_mxfp4_qat,
        dst_type_max=dst_type_max,
    )
    if recipe is not None:
        plan = build_recipe_plan(
            recipe,
            model_name=model_name,
            enable_mxfp4_qat=enable_mxfp4_qat,
            dst_type_max=dst_type_max,
        )
        return plan, f"recipe={recipe}"

    if not extracted:
        recipes = _exportable_recipes(model_name)
        choices = ", ".join(repr(recipe) for recipe in recipes) or "none"
        if input_format == "hf":
            raise ValueError(
                "HF input has no embedded quantization configs; pass --recipe with one of "
                f"{choices} to choose the export quantization plan"
            )
        raise ValueError(
            f"{input_dir} carries no DCP-embedded quantization configs; pass --recipe with one of "
            f"{choices} to build the plan from a recipe"
        )
    plan = _build_dcp_plan(hf_state_dict, extracted, sd_adapter, model_name, vision_prefixes=vision_prefixes)
    return plan, "DCP replay"


def _plan_summary(plan) -> str:
    """One-line plan description for cross-checking the training recipe."""
    entries: list[str] = []
    for idx, (swap, _) in enumerate(plan):
        outer = swap.weight_config
        effective = effective_weight_config(outer)
        qat = getattr(outer, "mxfp4_fake_quantize_config", None) is not None
        entries.append(
            f"{idx}: {type(outer).__name__} elem={effective.elem_dtype} block={effective.block_size} "
            f"mxfp4_qat={qat} dst_type_max={getattr(effective, 'dst_type_max', 0.0)}"
        )
    return "; ".join(entries)


def _export_scale(
    scale: torch.Tensor,
    *,
    keep_scale_tail_dim: bool,
    quant_dim_size: int,
    block_size: int,
) -> torch.Tensor:
    """Fold the packed trailing scale dimension and trim the kernel's block padding.

    The NPU kernel emits ``ceil(ceil(K / block) / 2)`` packed pairs, i.e. it pads
    the block count to an even number. vLLM Ascend allocates exactly
    ``ceil(K / block)`` scale entries per row, so the padded tail must be dropped.
    """
    if keep_scale_tail_dim:
        return scale
    if scale.ndim < 2 or scale.shape[-1] != 2:
        raise ValueError(
            f"cannot fold quantization scale: expected trailing pack dimension 2, got shape {tuple(scale.shape)}"
        )
    folded = scale.flatten(-2)
    expected = block_count(quant_dim_size, block_size)
    if folded.shape[-1] < expected:
        raise ValueError(
            f"quantization scale has {folded.shape[-1]} blocks, fewer than the expected "
            f"{expected} for quant dim {quant_dim_size} at block size {block_size}"
        )
    return folded[..., :expected]


def _export_qdata(qdata: torch.Tensor) -> torch.Tensor:
    """Plain-format payload: FP4 packed as uint8, FP8 kept in its own dtype.

    safetensors cannot represent the sub-byte FP4 dtype, so the packed payload is
    stored as raw ``uint8``. FP8 qdata has a native safetensors dtype and must keep
    it: vLLM Ascend's W8A8 parameter is ``float8_e4m3fn`` (only the scale is
    ``uint8``), and a uint8 payload would be numerically cast by the weight loader
    instead of bit-reinterpreted.
    """
    if qdata.dtype is torch.float4_e2m1fn_x2:
        return qdata.view(torch.uint8)
    return qdata


# Engram master tables are quantized in row chunks so the per-chunk temporaries
# stay bounded; the exported table layout lives in ``vllm_export_contract``.
_ENGRAM_QUANT_CHUNK_ROWS = 32768


def quantize_engram_tables(
    hf_state_dict: dict[str, torch.Tensor],
    *,
    quantize_rows: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    chunk_rows: int = _ENGRAM_QUANT_CHUNK_ROWS,
) -> set[str]:
    """Replace V4.1 Engram master tables with the MXFP8 layout vLLM Ascend loads.

    The adapter has already sliced each table to its logical rows. Rows are
    quantized in chunks so the per-chunk temporaries stay bounded; the FP32
    master is released once its table is replaced.
    """
    table_fqns = sorted(fqn for fqn in hf_state_dict if ENGRAM_TABLE_RE.match(fqn))
    if not table_fqns:
        return set()
    if quantize_rows is None:
        # Reuse the training-side quantizer so the exported table is byte-identical
        # to the MXFP8 lookup storage the same master weight produces during training.
        from torchtitan_npu.override.deepseek_v4_1.engram.mxfp8 import quantize_mxfp8_rows

        quantize_rows = quantize_mxfp8_rows
    if chunk_rows <= 0:
        raise ValueError(f"Engram quantization chunk_rows must be positive, got {chunk_rows}")

    for fqn in table_fqns:
        weight = hf_state_dict[fqn]
        if weight.ndim != 2:
            raise ValueError(f"{fqn}: Engram table must be 2D, got shape {tuple(weight.shape)}")
        rows, width = weight.shape
        codes = torch.empty((rows, width), dtype=torch.float8_e4m3fn)
        scales = torch.empty((rows, block_count(width, ENGRAM_MX_BLOCK_SIZE)), dtype=torch.float8_e8m0fnu)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            codes[start:end], scales[start:end] = quantize_rows(weight[start:end])
        hf_state_dict[fqn] = codes
        hf_state_dict[engram_scale_name(fqn)] = scales
        logger.info("Quantized Engram table %s (%d logical rows) to MXFP8", fqn, rows)
    return set(table_fqns)


def _quantize_to_plain(
    weight: torch.Tensor,
    swap_config,
    device: str,
    *,
    keep_scale_tail_dim: bool = False,
):
    """Quantize one weight on device; return CPU ``(qdata, weight_scale)`` tensors."""
    from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
    from torchao_npu.wrapper_tensors import BlockMXTrainingWeightWrapperTensor, MXTrainingWeightWrapperTensor

    if weight.ndim != 2:
        raise ValueError(f"quantized weight must be 2D, got shape {tuple(weight.shape)}")
    # npu_dynamic_block_mx_quant rejects fp32 inputs. Anything else that is not a
    # floating master weight is a pre-quantized checkpoint (fp8/int8 + ``.scale``),
    # which this exporter does not dequantize. Cast fp32 on CPU so the bf16 copy,
    # not the fp32 master, is what moves to the NPU.
    if weight.dtype is torch.float32:
        weight = weight.to(torch.bfloat16)
    elif weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"cannot quantize a {weight.dtype} weight; pre-quantized checkpoints "
            "(e.g. the fp8/fp4 DeepSeek HF release) must be dequantized first"
        )
    on_device = weight.to(device)
    weight_config = swap_config.weight_config
    # ``to_inference_weight`` quantizes with the nested FP4 config when MXFP4
    # QAT is enabled, so the written scale block size follows that config.
    scale_config = effective_weight_config(weight_config)
    if isinstance(weight_config, BlockMXQuantizeConfig):
        wrapper = BlockMXTrainingWeightWrapperTensor(
            on_device, weight_config=weight_config, activation_config=swap_config.activation_config
        )
    elif type(weight_config) is MXQuantizeConfig:
        wrapper = MXTrainingWeightWrapperTensor(
            on_device, weight_config=weight_config, activation_config=swap_config.activation_config
        )
    else:
        raise ValueError(
            f"weight_config must be `MXQuantizeConfig` or `BlockMXQuantizeConfig`, got {type(weight_config).__name__}."
        )
    mx = wrapper.to_inference_weight()
    # The NPU->CPU copy is bit-exact; storage must land on CPU for safetensors.
    qdata, scale = mx.qdata.cpu(), mx.scale.cpu()
    # vLLM Ascend reinterprets the E8M0 scale bytes as float8_e8m0fnu; the FP4
    # payload is packed bytes, while FP8 keeps its native dtype.
    scale = _export_scale(
        scale,
        keep_scale_tail_dim=keep_scale_tail_dim,
        quant_dim_size=weight.shape[-1],
        block_size=scale_config.block_size,
    )
    return _export_qdata(qdata), scale.view(torch.uint8)


def _split_fused_experts(hf_state_dict: dict[str, torch.Tensor]) -> None:
    """Split fused ``w13`` expert weights into ``w1``/``w3`` in place."""
    for fqn, value in list(hf_state_dict.items()):
        match = EXPERT_FQN_RE.match(fqn)
        if match is None or match[3] != "13":
            continue
        if value.ndim != 2 or value.shape[0] % 2:
            raise ValueError(f"fused expert {fqn!r} must be 2D with an even row count")
        stem = fqn.removesuffix(".w13.weight")
        targets = [f"{stem}.w{projection}.weight" for projection in (1, 3)]
        if any(target in hf_state_dict for target in targets):
            raise ValueError(f"cannot mix fused {fqn!r} with separate w1/w3")
        del hf_state_dict[fqn]
        hf_state_dict.update(zip(targets, value.chunk(2, dim=0), strict=True))


def quantize_state_dict(
    hf_state_dict,
    plan,
    *,
    device: str,
    keep_scale_tail_dim: bool = False,
    vision_prefixes: tuple[str, ...] = VISION_PREFIXES,
    input_sizes: dict[str, int] | None = None,
):
    """Quantize the recipe-matched weights; returns ``(state_dict, fqn -> plan index)``.

    Non-matched tensors pass through unchanged (still on CPU, master dtype). The
    input mapping is consumed and each quantized entry is replaced in ``out`` by its
    payload as soon as that payload exists, so a master weight is released once it
    has been quantized. The peak is still every master plus the current payload,
    because ``out`` starts as a shallow copy of the input mapping.

    ``input_sizes``, when given, is filled with each quantized weight's raw quant dim
    (``fqn -> K``) so the post-write check can compare the scale against the real K
    instead of re-deriving it from a packed FP4 payload.

    MTP weights are never quantized (the description records them as ``FLOAT``), so a
    plan entry that matches one is reported instead of being dropped silently.
    """
    _split_fused_experts(hf_state_dict)
    out = dict(hf_state_dict)
    quantized: dict[str, int] = {}
    unquantized_mtp: list[str] = []
    for fqn, value in hf_state_dict.items():
        # Record a quantization rule that this skip drops, then skip MTP/vision weights.
        if (
            fqn.startswith(MTP_PREFIXES)
            and value.is_floating_point()
            and any(filter_fn(nn.Parameter(value), fqn) for _, filter_fn in plan)
        ):
            unquantized_mtp.append(fqn)
        if fqn.startswith(vision_prefixes) or fqn.startswith(MTP_PREFIXES):
            continue
        # Non-floating-point tensors (e.g. the int64 ``tid2eid`` hash-router
        # table) are never recipe-matched and ``nn.Parameter`` rejects them;
        # pass them through untouched.
        if not value.is_floating_point():
            continue
        for idx, (swap_config, filter_fn) in enumerate(plan):
            if filter_fn(nn.Parameter(value), fqn):
                quantized[fqn] = idx
                break
    _group_routed_experts(quantized)
    if input_sizes is not None:
        input_sizes.update({fqn: int(hf_state_dict[fqn].shape[-1]) for fqn in quantized})
    if unquantized_mtp:
        logger.warning(
            "%d MTP weight(s) match a quantization rule but are exported unquantized "
            "(first: %s); MTP layers are not part of the quantized export",
            len(unquantized_mtp),
            unquantized_mtp[0],
        )
    fp32_fqns = [fqn for fqn in quantized if hf_state_dict[fqn].dtype is torch.float32]
    if fp32_fqns:
        logger.warning(
            "%d quantized weight(s) are fp32 and will be quantized from a bfloat16 copy "
            "(first: %s); this differs numerically from quantizing the fp32 values",
            len(fp32_fqns),
            fp32_fqns[0],
        )
    for fqn, idx in quantized.items():
        out[fqn], out[scale_name(fqn)] = _quantize_to_plain(
            hf_state_dict.pop(fqn),
            plan[idx][0],
            device,
            keep_scale_tail_dim=keep_scale_tail_dim,
        )
    return out, quantized


def _group_routed_experts(quantized: dict[str, int]) -> None:
    # ``_split_fused_experts`` runs before matching, so only w1/w2/w3 reach here.
    groups: dict[tuple[str, str], dict[int, str]] = {}
    for fqn in sorted(quantized):
        if match := EXPERT_FQN_RE.match(fqn):
            groups.setdefault((match[1], match[3]), {})[int(match[2])] = fqn
    for (layer, projection), experts in groups.items():
        missing = sorted(set(range(max(experts) + 1)) - experts.keys())
        if missing:
            raise ValueError(f"layer {layer}: incomplete per-expert quantization for w{projection}, missing {missing}")
        if len({quantized[fqn] for fqn in experts.values()}) != 1:
            raise ValueError(f"layer {layer}: experts in w{projection} use different quantization plan entries")
        if projection in ("1", "3"):
            partner = groups.get((layer, "3" if projection == "1" else "1"), {})
            if experts.keys() != partner.keys():
                raise ValueError(f"layer {layer}: w1 and w3 must match for every expert")
            if any(quantized[experts[e]] != quantized[partner[e]] for e in experts):
                raise ValueError(f"layer {layer}: w1 and w3 use different quantization plan entries")


# Assets that are never copied: weights and packaging meta files no load path reads.
# The ModelSlim index name is skipped as well: vLLM Ascend probes it before
# ``model.safetensors.index.json``, and its shards are not copied either.
_SKIPPED_ASSETS = {
    "config_recipes.json",
    "configuration.json",
    "model.safetensors.index.json",
    "quant_model_weights.safetensors.index.json",
}


def _copy_hf_assets(assets_path: Path, output_dir: Path) -> None:
    """Copy the non-weight assets (config.json, tokenizers, ...) next to the shards.

    The copied ``config.json`` is kept verbatim except for the vision switch
    that ``sync_vision_config`` aligns with the exported tensor set; vLLM
    Ascend otherwise serves the release assets as-is.
    """
    if not assets_path.is_dir():
        logger.warning("--hf_assets_path %s is not a directory; skipping asset copy", assets_path)
        return
    copied = []
    for f in sorted(assets_path.iterdir()):
        if not f.is_file() or f.suffix == ".safetensors" or f.name in _SKIPPED_ASSETS:
            continue
        shutil.copy2(f, output_dir / f.name)
        copied.append(f.name)
    logger.info("Copied %d HF asset file(s): %s", len(copied), ", ".join(copied) if copied else "(none)")


def _align_config_with_model(config_dir: Path, model_name: str, model_config: Any) -> None:
    """Rewrite the copied ``config.json`` structural fields to the exported model's truth.

    Shrunken flavors would otherwise ship the release config's layer/expert/dim
    numbers. Only the flat DeepSeek configs are derived: Qwen3.5 keeps the copied
    assets because vLLM reads its fields from a nested ``text_config``.
    """
    if model_name not in ("deepseek_v4", "deepseek_v4_1"):
        return
    if __package__:
        from .export_quantized_hf import apply_model_derived_config
    else:
        if str(_SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPT_DIR))
        from export_quantized_hf import apply_model_derived_config  # pyrefly: ignore [missing-import]

    apply_model_derived_config(config_dir, model_config)


def _reject_stale_weight_artifacts(output_dir: Path) -> None:
    """Refuse a directory that still holds weights from an earlier export.

    ``save_hf_safetensors`` writes ``model.safetensors.index.json`` only for a
    multi-shard export and never removes files a previous export left behind.
    A reused directory can therefore keep a stale index (this run is
    single-shard) or a stale single-shard ``model.safetensors`` (this run is
    multi-shard); which weights a loader picks would then depend on its probe
    order. Failing here keeps every export self-contained.
    """
    if not output_dir.is_dir():
        return
    stale = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json"))
    )
    if stale:
        raise ValueError(
            f"output_dir already contains weight/index artifact(s) from a previous export: {stale}; "
            "remove them or export into an empty directory"
        )


def _discard_failed_export(output_dir: Path) -> None:
    # Targeted removal only, never rmtree: the stale-artifact check already ran,
    # so every match belongs to this run, including a failed save's partial files.
    artifacts = [
        *sorted(output_dir.glob("model*.safetensors")),
        output_dir / "model.safetensors.index.json",
        output_dir / "quant_model_description.json",
    ]
    removed = []
    for path in artifacts:
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    logger.error(
        "export failed; removed %s from %s; directory is not a complete checkpoint",
        ", ".join(removed) or "nothing",
        output_dir,
    )


def _save_export_shards(hf_state_dict, output_dir: Path, *, max_shard_size: int | str) -> list[str]:
    """Save the shards, removing partial files when the write fails or is interrupted."""
    from torchao_npu.serialization.export import save_hf_safetensors

    try:
        return save_hf_safetensors(hf_state_dict, output_dir, max_shard_size=max_shard_size)
    except BaseException:
        _discard_failed_export(output_dir)
        raise


def _ensure_engram_index(output_dir: Path, shard_names: list[str], engram_fqns: set[str]) -> None:
    """Write the index vLLM Ascend needs to stream Engram tables.

    ``save_hf_safetensors`` writes ``model.safetensors.index.json`` only when it
    shards the export, but the Engram loader refuses a checkpoint without one.
    A single-shard export therefore gets one here.
    """
    if not engram_fqns or len(shard_names) > 1:
        return
    index_path = output_dir / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name in shard_names:
        for tensor_name, record in read_safetensors_header(output_dir / shard_name).items():
            weight_map[tensor_name] = shard_name
            total_size += record["data_offsets"][1] - record["data_offsets"][0]
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    logger.info("Wrote model.safetensors.index.json for %d Engram table(s)", len(engram_fqns))


# Short names that must resolve to the NPU package. The upstream qwen3_5 module
# imports FLA unconditionally; only the NPU module installs the compatibility
# shim before importing it.
_SHORT_MODEL_MODULES = {
    "deepseek_v4": "torchtitan_npu.models.deepseek_v4",
    "deepseek_v4_1": "torchtitan_npu.models.deepseek_v4_1",
    "qwen3_5": "torchtitan_npu.models.qwen3_5",
}


def _model_module_name(model_name: str) -> str:
    if "." in model_name:
        return model_name
    return _SHORT_MODEL_MODULES.get(model_name, f"torchtitan.models.{model_name}")


def _get_model_spec(model_name: str, model_flavor: str):
    """Resolve the model configuration without materializing its weights."""
    model_module = importlib.import_module(_model_module_name(model_name))
    return model_module.model_registry(model_flavor)


# ``HostEngramTable._save_to_state_dict`` exposes CPU row shards through the
# CheckpointableTensor protocol; the freshly allocated container must carry the
# same annotations, or ``dcp.load`` loses the global/local row mapping.
_CHECKPOINT_VIEW_ATTRS = (
    "global_shape",
    "global_offsets",
    "local_offsets",
    "local_sizes",
    "_engram_valid_rows",
)


def _carry_checkpoint_view(source: torch.Tensor, target: torch.Tensor) -> None:
    """Copy the optional CheckpointableTensor shard annotations onto ``target``."""
    for name in _CHECKPOINT_VIEW_ATTRS:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _allocate_load_container(
    meta_state_dict: dict[str, torch.Tensor],
    metadata: Metadata,
    *,
    fallback: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Allocate the CPU tensors ``dcp.load`` fills, sized from the meta model state.

    The model is built on the meta device, so its state dict carries the FQN set,
    shapes and logical dtypes but no storage: the loader's float32 storage never
    exists. Each tensor takes the checkpoint's own dtype, because ``dcp.load``
    copies the stored values in place — a float32 container would silently upcast
    the bf16 master weights, and the NPU block-MX quant kernel only accepts
    fp16/bf16 inputs.
    """
    container: dict[str, torch.Tensor] = {}
    for fqn, value in meta_state_dict.items():
        if not value.is_floating_point():
            # Integer/bool buffers keep the model dtype; ``dcp.load`` must not
            # reinterpret their storage.
            dtype = value.dtype
        else:
            entry = metadata.state_dict_metadata.get(fqn)
            dtype = entry.properties.dtype if isinstance(entry, TensorStorageMetadata) else fallback
        tensor = torch.empty(tuple(value.shape), dtype=dtype, device="cpu")
        _carry_checkpoint_view(value, tensor)
        container[fqn] = tensor
    return container


def _load_dcp_state_dict(model, input_dir: Path, master_dtype: str, read_threads: int):
    """Load the DCP checkpoint into a CPU container allocated from its metadata."""

    from torchtitan.config import TORCH_DTYPE_MAP

    from torchtitan_npu.tools.checkpoint_conversion.convert_to_hf import ParallelFileSystemReader

    _dcp_quant_configs.register_wrapper_safe_globals()
    meta_state_dict = model.state_dict()
    reader = ParallelFileSystemReader(str(input_dir), thread_count=read_threads)
    metadata = reader.read_metadata()
    checkpoint_fqns = set(metadata.state_dict_metadata)
    model_fqns = set(meta_state_dict)

    missing_from_checkpoint = model_fqns - checkpoint_fqns
    missing_nonvision = sorted(fqn for fqn in missing_from_checkpoint if not fqn.startswith(LOCAL_VISION_PREFIXES))
    if missing_nonvision:
        raise RuntimeError(
            "DCP checkpoint is missing model parameter(s) outside the frozen vision module: "
            + ", ".join(missing_nonvision[:8])
        )
    wanted = {fqn: value for fqn, value in meta_state_dict.items() if fqn not in missing_from_checkpoint}

    unexpected_checkpoint = checkpoint_fqns - model_fqns
    unexpected_model = sorted(
        fqn
        for fqn in unexpected_checkpoint
        if not fqn.startswith(VISION_PREFIXES) and not fqn.startswith(_dcp_quant_configs.DCP_TRAINING_STATE_PREFIXES)
    )
    if unexpected_model:
        raise RuntimeError(
            "DCP contains parameter(s) absent from the selected model: " + ", ".join(unexpected_model[:8])
        )

    state_dict = _allocate_load_container(wanted, metadata, fallback=TORCH_DTYPE_MAP[master_dtype])

    class _ModelOnlyReader(ParallelFileSystemReader):
        """Serve the already-validated metadata instead of re-reading it from disk."""

        def __init__(self, path, *, thread_count, metadata):
            super().__init__(path, thread_count=thread_count)
            self._metadata = metadata

        def read_metadata(self, *args, **kwargs):
            return self._metadata

    allowed_fqns = set(state_dict)
    model_metadata = copy.copy(metadata)
    kept_metadata = {fqn: value for fqn, value in metadata.state_dict_metadata.items() if fqn in allowed_fqns}
    model_metadata.state_dict_metadata = kept_metadata
    reader = _ModelOnlyReader(str(input_dir), thread_count=read_threads, metadata=model_metadata)
    logger.info(
        "Loading %d model tensor(s); skipping %d frozen/training-state entries",
        len(state_dict),
        len(checkpoint_fqns - set(state_dict)),
    )
    dcp.load(state_dict, storage_reader=reader)
    return state_dict


def _load_hf_state_dict(
    input_dir: Path,
    *,
    prefixes: tuple[str, ...] | None = None,
    extra_filter: Callable[[str], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Read single-file or indexed HF tensors onto CPU, optionally filtering by prefix.

    ``extra_filter`` admits names outside ``prefixes`` (e.g. the vision-owned
    ``layers.N.ffn.gate.bias_vl``, which lives in the text namespace).
    """
    from safetensors import safe_open

    def wanted(name: str) -> bool:
        if prefixes is None or name.startswith(prefixes):
            return True
        return extra_filter is not None and extra_filter(name)

    shard_to_names = None
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid HF safetensors index: {index_path}")
        selected = {name for name in weight_map if wanted(name)}
        shard_to_names = {}
        for name in selected:
            shard_to_names.setdefault(weight_map[name], set()).add(name)
        shard_names = sorted(shard_to_names)
    else:
        candidates = sorted(input_dir.glob("*.safetensors"))
        if not candidates:
            if prefixes is not None:
                return {}
            raise FileNotFoundError(
                f"no model.safetensors.index.json or *.safetensors found under HF input {input_dir}"
            )
        if len(candidates) > 1:
            raise ValueError(f"multiple safetensors files without model.safetensors.index.json under {input_dir}")
        shard_names = [candidates[0].name]

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = input_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"HF safetensors shard listed by the index does not exist: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            available = set(shard.keys())
            if shard_to_names is not None:
                selected_names = shard_to_names[shard_name]
                missing = selected_names - available
                if missing:
                    raise ValueError(f"HF shard {shard_path} is missing indexed tensors: {sorted(missing)[:8]}")
            else:
                selected_names = available
                if prefixes is not None or extra_filter is not None:
                    selected_names = {name for name in selected_names if wanted(name)}
            duplicate = set(state_dict).intersection(selected_names)
            if duplicate:
                raise ValueError(f"duplicate tensor names across HF shards: {sorted(duplicate)[:4]}")
            state_dict.update((name, shard.get_tensor(name)) for name in sorted(selected_names))
    logger.info("Loaded HF safetensors checkpoint %s (%d tensors)", input_dir, len(state_dict))
    return state_dict


def _restore_frozen_vision_weights(
    hf_state_dict: dict[str, torch.Tensor], hf_assets_path: Path, vision_prefixes: tuple[str, ...] = VISION_PREFIXES
) -> dict[str, torch.Tensor]:
    """Fill missing vision weights from HF assets, preserving checkpoint values.

    The fill set covers the tower/aligner/markers and the vision-owned per-MoE
    ``ffn.gate.bias_vl``; the latter is needed when a text-only flavor is
    combined with a vision-capable serving config.
    """
    vision = _load_hf_state_dict(
        hf_assets_path,
        prefixes=vision_prefixes,
        extra_filter=lambda name: VISION_BIAS_RE.match(name) is not None,
    )
    vision = {name: weight for name, weight in vision.items() if name not in hf_state_dict}
    if not vision:
        logger.debug("No missing vision tensors to restore from %s", hf_assets_path)
        return hf_state_dict
    hf_state_dict.update(vision)
    logger.info("Restored %d missing vision tensor(s) from %s", len(vision), hf_assets_path)
    return hf_state_dict


def _load_hf_dtype_map(input_dir: Path) -> dict[str, torch.dtype]:
    """Read native HF floating dtypes from safetensors headers without loading data."""
    index_path = input_dir / "model.safetensors.index.json"
    shard_to_names: dict[str, set[str]] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid HF safetensors index: {index_path}")
        shard_to_names = {}
        for name, shard in weight_map.items():
            if isinstance(name, str) and isinstance(shard, str):
                shard_to_names.setdefault(shard, set()).add(name)
        shard_names = sorted(shard_to_names)
    else:
        shard_names = sorted(input_dir.glob("*.safetensors"))
        if len(shard_names) > 1:
            raise ValueError(f"multiple safetensors files without model.safetensors.index.json under {input_dir}")
        shard_names = [path.name for path in shard_names]

    dtypes: dict[str, torch.dtype] = {}
    for shard_name in shard_names:
        header = read_safetensors_header(input_dir / shard_name)
        selected = shard_to_names[shard_name] if shard_to_names is not None else header.keys()
        for name in selected:
            record = header.get(name)
            if record is None:
                continue
            tag = record.get("dtype")
            if not isinstance(tag, str):
                continue
            dtype = NATIVE_DTYPES_BY_TAG.get(tag)
            if dtype is not None:
                dtypes[name] = dtype
    return dtypes


def _restore_native_hf_dtypes(
    hf_state_dict: dict[str, torch.Tensor], hf_assets_path: Path, *, excluded_fqns=()
) -> None:
    """Restore native HF dtypes for weights left unquantized by the export recipe."""
    try:
        native_dtypes = _load_hf_dtype_map(hf_assets_path)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        logger.warning("cannot read native HF dtypes from %s; retaining DCP dtypes: %s", hf_assets_path, exc)
        return
    restored = 0
    excluded_fqns = set(excluded_fqns)
    plain_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for fqn, value in list(hf_state_dict.items()):
        if fqn in excluded_fqns:
            continue
        target = native_dtypes.get(fqn)
        if target is None or value.dtype == target:
            continue
        if value.dtype in integer_dtypes and target in integer_dtypes:
            # Only widen: ``.to()`` wraps silently on overflow and the checkpoint's
            # integer normalization (e.g. ``tid2eid`` -> int64) must not be reverted.
            if target.itemsize <= value.itemsize or value.dtype.is_signed != target.is_signed:
                continue
        elif value.dtype not in plain_dtypes or target not in plain_dtypes:
            logger.warning(
                "cannot restore native HF dtype for %s: checkpoint=%s, native=%s; keeping checkpoint dtype",
                fqn,
                value.dtype,
                target,
            )
            continue
        logger.debug("Native HF dtype: %s -> %s (%s)", value.dtype, target, fqn)
        hf_state_dict[fqn] = value.detach().to(target)
        restored += 1
    if restored:
        logger.info("Restored native HF dtype for %d unquantized tensor(s)", restored)


def _resolve_input_format(input_dir: Path, input_format: str) -> str:
    if input_format not in ("auto", "dcp", "hf"):
        raise ValueError(f"input_format must be 'auto', 'dcp', or 'hf', got {input_format!r}")
    if input_format != "auto":
        return input_format
    if (input_dir / "model.safetensors.index.json").is_file() or any(input_dir.glob("*.safetensors")):
        return "hf"
    return "dcp"


@torch.inference_mode()
def export_quantized_vllm(args: argparse.Namespace) -> None:
    """Load weights, apply the recipe, and save the HF/vLLM checkpoint."""
    input_dir = args.input_dir
    output_dir = args.output_dir
    _reject_stale_weight_artifacts(output_dir)
    hf_assets_path = args.hf_assets_path
    model_name = args.model_name
    recipe = args.recipe
    keep_scale_tail_dim = getattr(args, "keep_scale_tail_dim", False)
    input_format = _resolve_input_format(input_dir, getattr(args, "input_format", "auto"))
    vision_prefixes = vision_prefixes_from_hf_assets(hf_assets_path)

    # Initial preparation: resolve the model and validate the input mode.  The
    # model spec is needed to build a DCP container, but the quantization plan
    # does not depend on loading the checkpoint and is created later.
    model_spec = _get_model_spec(model_name, args.model_flavor)
    if input_format == "hf" and model_spec.name != "qwen3_5":
        raise ValueError(
            "--input_format hf currently supports only Qwen3.5; the DeepSeek HF releases are "
            "already quantized (fp8/fp4 weights + .scale) and must be exported from a DCP checkpoint"
        )
    # Weight loading: DCP is converted to HF names; HF input is already in HF names.
    needs_vision = bool(getattr(args, "needs_vision", False))
    sd_adapter = None
    extracted: dict[str, tuple[Any, Any]] = {}
    if input_format == "hf":
        hf_state_dict = _load_hf_state_dict(input_dir)
    else:
        from torchtitan.components.checkpoint import ModelWrapper

        if model_spec.state_dict_adapter is None:
            raise ValueError("DCP input requires a model state_dict_adapter")
        # Extraction only needs input_dir, so it runs before the CPU model build:
        # a plain checkpoint then fails before the model is materialized.
        if recipe is None:
            extracted = _dcp_quant_configs.extract_wrapper_configs(input_dir)
            # ``_resolve_quantization_plan`` repeats this check to keep its own
            # contract for direct callers.
            _reject_recipe_flags_when_replaying(
                recipe=recipe,
                extracted=extracted,
                enable_mxfp4_qat=args.enable_mxfp4_qat,
                dst_type_max=args.dst_type_max,
            )
        model_config = model_spec.model
        with torch.device("meta"):
            model = ModelWrapper(model_config.build())
        sd_adapter = model_spec.state_dict_adapter(model_config, str(hf_assets_path))
        # Text-only flavors share the assets directory with multimodal ones; restoring
        # the vision tower unconditionally would add tensors the model does not have.
        # --needs_vision opts a text-only flavor into that fill explicitly.
        needs_vision = needs_vision or any(name.startswith(LOCAL_VISION_PREFIXES) for name in model.state_dict())
        state_dict = _load_dcp_state_dict(model, input_dir, args.master_dtype, args.read_threads)
        logger.info("Loaded DCP checkpoint %s (%d tensors)", input_dir, len(state_dict))
        hf_state_dict = sd_adapter.to_hf(state_dict)
        if needs_vision:
            hf_state_dict = _restore_frozen_vision_weights(hf_state_dict, hf_assets_path, vision_prefixes)
        del model, state_dict
    logger.info("to_hf: %d tensors", len(hf_state_dict))

    # Quantization configuration: an explicit recipe overrides DCP configs;
    # otherwise replay the per-FQN configs embedded by quantized training and
    # fail when the checkpoint carries none.
    plan, plan_desc = _resolve_quantization_plan(
        recipe=recipe,
        input_format=input_format,
        model_name=model_spec.name,
        hf_state_dict=hf_state_dict,
        extracted=extracted,
        sd_adapter=sd_adapter,
        enable_mxfp4_qat=args.enable_mxfp4_qat,
        dst_type_max=args.dst_type_max,
        input_dir=input_dir,
        vision_prefixes=vision_prefixes,
    )
    logger.info("%s plan: %s", plan_desc, _plan_summary(plan))

    # Quantization and representation preparation.
    # to_hf emits the hash-router ``tid2eid`` lookup table as float32 (the
    # upstream DSv4 checkpoint format); transformers' in-memory buffer is
    # int64. The values are small exact integers, so the cast back is lossless.
    for fqn, value in hf_state_dict.items():
        if "tid2eid" in fqn and value.dtype != torch.int64:
            logger.debug("HF dtype conversion: %s -> torch.int64 (%s)", value.dtype, fqn)
            hf_state_dict[fqn] = value.to(torch.int64)

    master_input_sizes: dict[str, int] = {}
    hf_state_dict, quantized_fqns = quantize_state_dict(
        hf_state_dict,
        plan,
        device=args.quant_device,
        keep_scale_tail_dim=keep_scale_tail_dim,
        vision_prefixes=vision_prefixes,
        input_sizes=master_input_sizes,
    )
    logger.info("Quantized %d/%d tensors on %s", len(quantized_fqns), len(hf_state_dict), args.quant_device)
    if not quantized_fqns:
        raise ValueError(
            f"{plan_desc} matched no weights for {model_spec.name}; refusing to write a "
            "quantization description for an unquantized checkpoint"
        )
    logger.info("  first matches: %s", ", ".join(sorted(quantized_fqns)[:8]))
    if input_format == "dcp":
        # These tensors already have an export-specific representation.  The
        # native dtype pass must not reinterpret their payloads using the
        # original HF header dtype.
        quantized_scale_fqns = {scale_name(fqn) for fqn in quantized_fqns}
        engram_table_fqns = {fqn for fqn in hf_state_dict if ENGRAM_TABLE_RE.match(fqn)}
        tid2eid_fqns = {fqn for fqn in hf_state_dict if "tid2eid" in fqn}
        dtype_restore_exclusions = set(quantized_fqns)
        dtype_restore_exclusions.update(quantized_scale_fqns)
        dtype_restore_exclusions.update(engram_table_fqns)
        dtype_restore_exclusions.update(tid2eid_fqns)
        _restore_native_hf_dtypes(
            hf_state_dict,
            hf_assets_path,
            excluded_fqns=dtype_restore_exclusions,
        )
    engram_fqns = quantize_engram_tables(hf_state_dict)

    # Preserve unquantized dtypes, including frozen vision weights.
    shard_names = _save_export_shards(hf_state_dict, output_dir, max_shard_size=args.max_shard_size)
    logger.info("Saved %d shard(s) to %s: %s", len(shard_names), output_dir, ", ".join(shard_names))

    try:
        _copy_hf_assets(hf_assets_path, output_dir)
        _align_config_with_model(output_dir, model_spec.name, model_spec.model)
        sync_vision_config(
            output_dir,
            model_name=model_spec.name,
            export_has_vision=has_vision_tower(hf_state_dict, vision_prefixes),
        )
        _ensure_engram_index(output_dir, shard_names, engram_fqns)
        description = build_quant_model_description(hf_state_dict, quantized_fqns, plan, engram_fqns=engram_fqns)
        write_quant_model_description(output_dir, description)
        validate_export(
            output_dir,
            shard_names,
            hf_state_dict,
            quantized_fqns,
            plan,
            description,
            model_name=model_spec.name,
            keep_scale_tail_dim=keep_scale_tail_dim,
            needs_vision=needs_vision,
            vision_prefixes=vision_prefixes,
            input_sizes=master_input_sizes,
        )
    except Exception:
        _discard_failed_export(output_dir)
        raise
    logger.info("Export complete")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a DCP or HF checkpoint as quantized safetensors for vLLM Ascend."
    )
    parser.add_argument("input_dir", type=Path, help="Input DCP directory or HF safetensors checkpoint directory.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the HF checkpoint; it must not contain weights or indexes from an earlier export.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        required=True,
        help=(
            "HF assets directory (config.json, tokenizers, and, for DCP exports whose checkpoint omits "
            "the frozen vision tower, the release model.safetensors files); non-weight files are copied "
            "to output_dir."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "One of 'deepseek_v4', 'deepseek_v4_1', 'qwen3_5', or a dotted path to a module whose "
            "model registry still reports one of those names."
        ),
    )
    parser.add_argument("--model_flavor", type=str, required=True)
    parser.add_argument(
        "--input_format",
        choices=("auto", "dcp", "hf"),
        default="auto",
        help="Input checkpoint format. 'hf' currently supports Qwen3.5 safetensors; 'auto' detects it.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "When omitted, read the per-parameter quantization configs embedded in the DCP checkpoint; "
            "the export fails if none are found. Set a registered recipe to override the DCP configs "
            "and rebuild the whole plan."
        ),
    )
    parser.add_argument(
        "--enable_mxfp4_qat",
        action="store_true",
        help=(
            "With --recipe: serve routed-expert weights as MXFP4 for mix/all_block_fp8; "
            "Qwen3.5 W4A8_MXFP enables MXFP4 for dense weights intrinsically."
        ),
    )
    parser.add_argument("--dst_type_max", type=float, default=0.0, help="With --recipe: FP4 target dtype maximum.")
    parser.add_argument(
        "--needs_vision",
        action="store_true",
        help=(
            "Export vision weights even when the selected model flavor has no vision module; "
            "missing tower/aligner/markers and per-MoE layers.N.ffn.gate.bias_vl are filled from "
            "--hf_assets_path. HF inputs cannot be filled, so there the flag only enables the "
            "vision completeness checks."
        ),
    )
    parser.add_argument(
        "--keep_scale_tail_dim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the trailing packed scale dimension of 2 in weight_scale tensors. "
            "vLLM Ascend expects cdiv(K, block_size) scale entries, so the exported checkpoint "
            "is only for inspection, not serving (disabled by default)."
        ),
    )
    parser.add_argument(
        "--master_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Fallback dtype for DCP tensors whose storage dtype is not recorded; each tensor otherwise "
            "keeps the checkpoint dtype (production master is bf16)."
        ),
    )
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="e.g. 5368709120 or '5GB'.")
    parser.add_argument("--quant_device", type=str, default="npu:0", help="NPU used for the quantization kernels.")
    parser.add_argument(
        "--read_threads",
        type=int,
        default=min(32, os.cpu_count() or 16),
        help="Threads used to read DCP shard files in parallel.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    export_quantized_vllm(args)


if __name__ == "__main__":
    main()
