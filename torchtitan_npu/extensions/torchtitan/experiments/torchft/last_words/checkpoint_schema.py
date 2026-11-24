# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Check DCP metadata against the application's required recovery state."""

import hashlib
import json
from dataclasses import fields, is_dataclass
from functools import partial
from math import prod
from pathlib import Path
from typing import Any


def tokenizer_asset_digest(folder: str) -> str:
    """Hash the assets selected by TorchTitan's HuggingFaceTokenizer, independent of their location."""
    root = Path(folder)
    names = ["tokenizer.json"]
    if not (root / "tokenizer.json").is_file():
        names = ["vocab.json" if (root / "vocab.json").is_file() else "vocab.txt"]
        if (root / "merges.txt").is_file():
            names.append("merges.txt")
    if (root / "tokenizer_config.json").is_file():
        names.append("tokenizer_config.json")
        if (root / "chat_template.jinja").is_file():
            names.append("chat_template.jinja")
    digests = {}
    for name in names:
        with (root / name).open("rb") as asset:
            digests[name] = hashlib.file_digest(asset, "sha256").hexdigest()
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def _config_contract(config: Any, *, legacy_compressor: bool = False) -> dict:
    def normalize(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            result = {field.name: normalize(getattr(value, field.name)) for field in fields(value)}
            if (
                legacy_compressor
                and type(value).__module__ == "torchtitan_npu.models.deepseek_v4.compressor"
                and type(value).__qualname__ == "Compressor.Config"
            ):
                # Before the CP refactor this parameterless child was present even at CP=1.
                result["carry_exchange"] = {
                    "param_init": None,
                    "sharding_config": None,
                    "ratio": result["compress_ratio"],
                }
            return result
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, partial):
            return {
                "function": normalize(value.func),
                "args": normalize(value.args),
                "keywords": normalize(value.keywords),
            }
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        return str(value)

    parallelism = normalize(config.parallelism)
    extensions = getattr(config.parallelism, "extensions", None)
    if extensions is not None and parallelism["extensions"] == {
        "disable_expert_replication": extensions.disable_expert_replication,
        "context_parallel_intra_degree": -1,
    }:
        # Preserve the pre-namespace checkpoint signature for the original flat CP exchange.
        parallelism["disable_expert_replication"] = parallelism.pop("extensions")["disable_expert_replication"]
    training = config.training
    return {
        "model": normalize(config.model_spec.model),
        "flavor": config.model_spec.flavor,
        "parallelism": parallelism,
        "dataloader": normalize(config.dataloader),
        "tokenizer": normalize(config.tokenizer),
        "optimizer": normalize(config.optimizer),
        "lr_scheduler": normalize(config.lr_scheduler),
        "training": {
            key: normalize(getattr(training, key))
            for key in ("local_batch_size", "global_batch_size", "seq_len", "steps")
        },
        "group_size": config.fault_tolerance.group_size,
        "override": normalize(config.override),
        "compile": normalize(config.compile),
    }


def checkpoint_config_signatures(config: Any) -> tuple[str, ...]:
    def digest(contract):
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    previous = _config_contract(config)
    parallel = config.parallelism
    # Bind placement and execution semantics to stable names for supported FSDP+EP jobs.
    topology = {
        "replicas": parallel.data_parallel_replicate_degree,
        "shards": parallel.data_parallel_shard_degree,
        "experts": parallel.expert_parallel_degree,
        "tensor": parallel.tensor_parallel_degree,
        "context": parallel.context_parallel_degree,
        "pipeline": parallel.pipeline_parallel_degree,
        "replicate_experts": not parallel.extensions.disable_expert_replication,
        "backend": parallel.spmd_backend,
        "reshard": parallel.fsdp_reshard_after_forward,
        "pipeline_microbatch": parallel.pipeline_parallel_microbatch_size,
    }
    current = dict(previous, version=2, parallelism=topology)
    return digest(current), digest(previous), digest(_config_contract(config, legacy_compressor=True))


def state_schema(states, *, kind):
    """Describe live DCP state without saving or loading tensor values."""
    from torch.distributed.checkpoint import DefaultSavePlanner
    from torch.distributed.checkpoint.stateful import Stateful

    values = {key: value.state_dict() if isinstance(value, Stateful) else value for key, value in states.items()}
    planner = DefaultSavePlanner()
    planner.set_up_planner(values, is_coordinator=True)
    fields = {}
    for item in planner.create_local_plan().items:
        field = {"kind": "object"}
        if item.tensor_data is not None:
            field = {
                "kind": "tensor",
                "shape": list(item.tensor_data.size),
                "dtype": str(item.tensor_data.properties.dtype),
            }
        previous = fields.setdefault(item.index.fqn, field)
        if previous != field:
            raise ValueError("Inconsistent local tensor recovery schema")
    groups: dict[str, list[str]] = {}
    reserved = {"optimizer", "lr_scheduler", "train_state", "ema", "torchft"}
    if kind == "cursor":
        reserved = {"dataloader", "last_words_rng"}
    elif kind != "model":
        raise ValueError("Unknown recovery state schema kind")
    for name in fields:
        prefix = name.split(".", 1)[0]
        group = prefix if prefix in reserved else "model"
        groups.setdefault(group, []).append(name)
    return {"version": 1, "kind": kind, "fields": fields, "groups": groups}


def replica_cursor_schema(local, replica, *, batch_degree, num_shards):
    """Expand the live rank's DCP contract to every shard of a logical replica."""
    fields = {}
    groups = {}
    for group, names in local["groups"].items():
        expanded = []
        for name in names:
            prefix, _, rest = name.partition(".")
            slot, dot, suffix = rest.partition(".")
            # ParallelAwareDataloader uses global batch ranks; RNG uses local shard ranks.
            if group == "dataloader" and slot.startswith("dp_rank_") and slot[8:].isdecimal():
                slots = [f"dp_rank_{replica * batch_degree + i}" for i in range(batch_degree)]
            elif group == "last_words_rng" and slot.startswith("rank_") and slot[5:].isdecimal():
                slots = [f"rank_{i}" for i in range(num_shards)]
            else:
                expanded.append(name)
                fields[name] = local["fields"][name]
                continue
            for target in slots:
                key = f"{prefix}.{target}{dot}{suffix}"
                if key not in fields:
                    expanded.append(key)
                fields[key] = local["fields"][name]
        groups[group] = expanded
    return {**local, "fields": fields, "groups": groups}


def _validate_description(schema, *, kind):
    if not isinstance(schema, dict) or schema.get("version") != 1 or schema.get("kind") != kind:
        raise ValueError("Checkpoint has no compatible required-state schema")
    fields, groups = schema.get("fields"), schema.get("groups")
    if not isinstance(fields, dict) or not isinstance(groups, dict):
        raise ValueError("Invalid required-state schema layout")
    required = {"model", "optimizer", "lr_scheduler", "train_state"}
    if kind == "cursor":
        required = {"dataloader", "last_words_rng"}
    if not required <= groups.keys():
        raise ValueError(f"Checkpoint schema lacks required state: {sorted(required - groups.keys())}")
    covered = []
    for group, names in groups.items():
        if not isinstance(group, str) or not isinstance(names, list) or not names:
            raise ValueError("Checkpoint schema has an empty state group")
        for name in names:
            if not isinstance(name, str) or name not in fields:
                raise ValueError("Checkpoint state group references a missing field")
            if group != "model" and name != group and not name.startswith(group + "."):
                raise ValueError("Checkpoint state group and field name disagree")
        covered.extend(names)
    if len(covered) != len(set(covered)) or set(covered) != fields.keys():
        raise ValueError("Checkpoint state groups do not partition required fields")
    for name, field in fields.items():
        if not isinstance(field, dict) or field.get("kind") not in {"tensor", "object"}:
            raise ValueError(f"Invalid recovery field schema: {name}")
        if field["kind"] == "tensor" and (
            not isinstance(field.get("shape"), list)
            or any(type(size) is not int or size < 0 for size in field["shape"])
            or not isinstance(field.get("dtype"), str)
        ):
            raise ValueError(f"Invalid tensor recovery schema: {name}")
    if kind == "model" and not any(fields[name]["kind"] == "tensor" for name in groups["model"]):
        raise ValueError("Checkpoint model schema has no tensor state")
    return fields


def validate_state_schema(path, schema, *, kind, expected=None):
    """Check published fields, then the independently derived live schema."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    required = dict(_validate_description(schema, kind=kind))
    if expected is not None:
        # The candidate's own declaration cannot relax the live application's
        # model, optimizer, EMA, counters or current logical-replica cursor.
        for name, field in _validate_description(expected, kind=kind).items():
            if name in required and required[name] != field:
                raise ValueError(f"Published and live recovery schemas differ: {name}")
            required[name] = field
    metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    for name, field in required.items():
        stored = metadata.get(name)
        if field["kind"] == "tensor":
            if (
                not isinstance(stored, TensorStorageMetadata)
                or list(stored.size) != field["shape"]
                or str(stored.properties.dtype) != field["dtype"]
            ):
                raise ValueError(f"Checkpoint has no matching required tensor: {name}")
        elif not isinstance(stored, BytesStorageMetadata):
            raise ValueError(f"Checkpoint has no required serialized state: {name}")


def validate_metadata_coverage(metadata):
    """Reject metadata with missing storage or incomplete/overlapping chunks."""
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    storage = {(item.fqn, tuple(item.offset) if item.offset is not None else None) for item in metadata.storage_data}
    for name, entry in metadata.state_dict_metadata.items():
        if isinstance(entry, BytesStorageMetadata):
            if (name, None) not in storage:
                raise ValueError(f"Checkpoint has no storage for state: {name}")
            continue
        if not isinstance(entry, TensorStorageMetadata):
            raise ValueError(f"Unsupported checkpoint metadata: {name}")
        volume, chunks = 0, []
        for chunk in entry.chunks:
            offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
            if (
                len(offsets) != len(entry.size)
                or len(sizes) != len(entry.size)
                or any(
                    offset < 0 or size < 0 or offset + size > full
                    for offset, size, full in zip(offsets, sizes, entry.size, strict=True)
                )
            ):
                raise ValueError(f"Checkpoint has invalid tensor chunks: {name}")
            current = prod(sizes)
            if current:
                if (name, offsets) not in storage:
                    raise ValueError(f"Checkpoint tensor chunk has no storage: {name}")
                for old_offsets, old_sizes in chunks:
                    if all(
                        a < b + y and b < a + x
                        for a, x, b, y in zip(offsets, sizes, old_offsets, old_sizes, strict=True)
                    ):
                        raise ValueError(f"Checkpoint tensor chunks overlap: {name}")
                chunks.append((offsets, sizes))
            volume += current
        if volume != prod(entry.size):
            raise ValueError(f"Checkpoint tensor chunks are incomplete: {name}")
