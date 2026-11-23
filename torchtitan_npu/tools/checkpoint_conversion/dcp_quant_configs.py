# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read the quantization configs a training run embedded in its DCP checkpoint.

Training pickles a wrapped parameter as the whole wrapper object into the
``.distcp`` stream at the offset ``.metadata`` records, so reading it there
recovers the training-time configs that ``dcp.load`` discards. A cheap byte
probe on the stream head skips plain-tensor streams; CPU-only, side-effect
free. ``.metadata`` itself is read with plain ``pickle.load`` -- exactly what
``dcp.load`` (and ``FileSystemReader.read_metadata``) does on the same file,
which this tool performs right after extraction -- so the trust boundary is
the checkpoint directory itself, not this reader.
"""

from __future__ import annotations

import io
import logging
import pickle
import re
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Wrapper streams carry the pickled class path in their first GLOBAL opcodes.
_WRAPPER_STREAM_MARKER = b"torchao_npu.wrapper_tensors"
_PROBE_BYTES = 8192

# Checkpoint buckets that hold training state instead of model parameters. A
# quantized run also stores wrapper-typed tensors under these prefixes (EMA keeps
# a detached clone per parameter), so extraction must skip them by default.
DCP_TRAINING_STATE_PREFIXES = (
    "optimizer.",
    "lr_scheduler.",
    "dataloader.",
    "train_state.",
    "rng_state.",
    "ema_optimizer.",
)


def register_wrapper_safe_globals() -> None:
    """Allow the wrapper classes for ``weights_only`` unpickling (needed by any ``dcp.load``).

    HiF8's module registers no safe globals; add it here so a HiF8 checkpoint
    fails at ``to_inference_weight`` (owner-defined) instead of an opaque ``UnpicklingError``.
    """
    from torchao_npu.wrapper_tensors import (  # noqa: F401  (import = registration)
        BlockMXTrainingWeightWrapperTensor,
        Float8TrainingWeightWrapperTensor,
        HiF8TrainingWeightWrapperTensor,
        MXTrainingWeightWrapperTensor,
    )

    torch.serialization.add_safe_globals([HiF8TrainingWeightWrapperTensor])


def extract_wrapper_configs(input_dir: Path) -> dict[str, tuple[Any, Any]]:
    """Map each FQN saved as a quantization wrapper (training namespace) to ``(weight_config, activation_config)``.

    ``DCP_TRAINING_STATE_PREFIXES`` names the checkpoint buckets that hold training
    state instead of model parameters (optimizer, EMA, ...). EMA stores a
    detached clone of every parameter, which stays wrapper-typed and therefore
    matches the stream probe; those entries are not model quantization configs.

    Plain streams are absent; an FQN's first recorded shard decides it (all
    shards serialize the same object).
    """
    metadata_path = input_dir / ".metadata"
    if not metadata_path.is_file():
        raise ValueError(f"{input_dir} is not a DCP checkpoint directory (missing .metadata)")
    register_wrapper_safe_globals()
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    configs: dict[str, tuple[Any, Any]] = {}
    decided: set[str] = set()  # first shard decides an FQN; see docstring
    shards_checked = 0
    for index, storage in metadata.storage_data.items():
        fqn = getattr(index, "fqn", None)
        if fqn is None or fqn in decided:
            continue
        decided.add(fqn)
        if fqn.startswith(DCP_TRAINING_STATE_PREFIXES):
            continue
        shard_path = input_dir / storage.relative_path
        if not shard_path.is_file():
            raise ValueError(f"DCP shard {shard_path} referenced by .metadata is missing")
        with shard_path.open("rb") as f:
            # ``.distcp`` files concatenate torch.save archives: reads must
            # stay inside this item's ``length`` or the zip reader loads the wrong object.
            f.seek(storage.offset)
            if _WRAPPER_STREAM_MARKER not in f.read(min(_PROBE_BYTES, storage.length)):
                continue
            f.seek(storage.offset)
            stream = io.BytesIO(f.read(storage.length))
            obj = torch.load(stream, map_location="cpu", weights_only=True)
        shards_checked += 1
        weight_config = getattr(obj, "weight_config", None)
        if weight_config is None:
            logger.warning(
                "Stream for %s looks like a quantization wrapper but carries no weight_config; skipping it",
                fqn,
            )
            continue
        configs[fqn] = (weight_config, getattr(obj, "activation_config", None))
    if shards_checked and not configs:
        logger.warning(
            "%d wrapper stream(s) found but none yielded a quantization config; replay mode will be unavailable",
            shards_checked,
        )
    logger.info(
        "DCP config extraction: %d wrapper stream(s) read, %d FQN(s) with configs",
        shards_checked,
        len(configs),
    )
    return configs


def abstract_train_fqn(train_fqn: str) -> tuple[str, str | None]:
    """Training FQN template with the layer index replaced by ``{}``, plus its layer number."""
    layer_match = re.search(r"\d+", train_fqn)
    abstract = re.sub(r"(\d+)", "{}", train_fqn, count=1)
    if train_fqn.startswith("mtp_layers."):
        abstract = abstract.replace("mtp_layers.{}.", "layers.{}.", 1)
    return abstract, layer_match.group(0) if layer_match else None


def _train_fqn_to_hf_fqns(
    train_fqn: str,
    to_hf_map: dict[str, str],
    hf_state_dict,
    fused_hf_fqns: dict[str, str] | None = None,
) -> list[str]:
    """Return HF names covered by one training-namespace wrapper FQN."""
    is_mtp = train_fqn.startswith("mtp_layers.")
    abstract, layer_num = abstract_train_fqn(train_fqn)

    if train_fqn in to_hf_map:
        hf_abstract = to_hf_map[train_fqn]
    elif abstract in to_hf_map:
        hf_abstract = to_hf_map[abstract]
        if is_mtp:
            hf_abstract = hf_abstract.replace("layers.{}.", "mtp.{}.", 1)
    elif fused_hf_fqns is not None:
        hf_abstract = fused_hf_fqns.get(abstract)
        if hf_abstract is not None and is_mtp:
            hf_abstract = hf_abstract.replace("layers.{}.", "mtp.{}.", 1)
    else:
        hf_abstract = None
    if hf_abstract is None:
        raise ValueError(
            f"the checkpoint embeds a quantization config for {train_fqn!r}, but the state_dict adapter has no "
            "HF name for it (see from_hf_map); --model_flavor may not match the checkpoint, or the adapter "
            "needs a new entry to export it."
        )
    if "{}" not in hf_abstract:
        return [hf_abstract] if hf_abstract in hf_state_dict else []
    if layer_num is None:
        return []
    if hf_abstract.count("{}") == 1:
        hf_fqn = hf_abstract.format(layer_num)
        return [hf_fqn] if hf_fqn in hf_state_dict else []

    # The second placeholder is the per-expert index.
    pre, _, rest = hf_abstract.partition("{}")
    mid, _, post = rest.partition("{}")
    per_expert = re.compile("^" + re.escape(pre + layer_num + mid) + r"\d+" + re.escape(post) + "$")
    return sorted(fqn for fqn in hf_state_dict if per_expert.match(fqn))


def plan_from_dcp_configs(hf_state_dict, extracted, sd_adapter, *, fused_hf_fqns=None) -> dict[str, Any]:
    """Map DCP wrapper configs onto the exported HF tensor names."""
    from torchao_npu.configs import ParamSwapConfig

    to_hf_map = {value: key for key, value in sd_adapter.from_hf_map.items() if value is not None}
    plan: dict[str, Any] = {}
    for train_fqn, (weight_config, activation_config) in extracted.items():
        hf_fqns = _train_fqn_to_hf_fqns(train_fqn, to_hf_map, hf_state_dict, fused_hf_fqns)
        if not hf_fqns:
            raise ValueError(
                f"the checkpoint embeds a quantization config for {train_fqn!r}, but its HF name mapping matches "
                "no exported tensor; --model_flavor may not match the checkpoint."
            )
        swap_config = ParamSwapConfig(weight_config=weight_config, activation_config=activation_config)
        for hf_fqn in hf_fqns:
            plan[hf_fqn] = swap_config
    return plan
