# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Read-only filesystem checks shared by recovery and checkpoint retention."""

import json
import logging
from functools import partial
from pathlib import Path

from .checkpoint_schema import validate_metadata_coverage, validate_state_schema

logger = logging.getLogger(__name__)


def contained(root, path):
    root, path = Path(root).resolve(), Path(path)
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Refusing checkpoint path outside its root: {path}")
    return resolved


def validate_storage_ranges(path):
    import torch.distributed.checkpoint as dcp

    path = Path(path).resolve()
    metadata = dcp.FileSystemReader(path).read_metadata()
    if metadata.storage_data is None:
        raise ValueError("Checkpoint has no filesystem storage map")
    validate_metadata_coverage(metadata)
    for entry in metadata.storage_data.values():
        shard = contained(path, path / entry.relative_path)
        if entry.offset < 0 or entry.length < 0 or shard.stat().st_size < entry.offset + entry.length:
            raise ValueError(f"Checkpoint shard is truncated: {shard}")


def load_train_progress(path):
    import torch.distributed.checkpoint as dcp

    validate_storage_ranges(path)
    states = {"train_state": {"step": 0, "last_words_num_updates": 0}}
    dcp.load(states, checkpoint_id=path, no_dist=True)
    return states["train_state"]


def load_torchft_progress(path):
    import torch.distributed.checkpoint as dcp

    validate_storage_ranges(path)
    states = {"torchft": {"step": 0, "batches_committed": 0}}
    dcp.load(states, checkpoint_id=path, no_dist=True)
    return states["torchft"]


def validate_rng_cursor(path, marker, num_shards, *, expected_device_type):
    import torch.distributed.checkpoint as dcp

    if (
        type(num_shards) is not int
        or num_shards < 1
        or marker.get("rng_schema") != 1
        or marker.get("rng_shards") != num_shards
    ):
        raise ValueError("Cursor RNG schema or shard count differs")
    keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    # Reject incomplete layouts before DCP planning wraps the missing-key error.
    for rank in range(num_shards):
        for field in ("schema", "device_type", "python", "cpu", "numpy"):
            prefix = f"last_words_rng.rank_{rank}.{field}"
            if not any(key == prefix or key.startswith(prefix + ".") for key in keys):
                raise ValueError(f"Cursor has no complete RNG stream for shard {rank}")
    headers = {"last_words_rng": {f"rank_{rank}": {"schema": 0, "device_type": ""} for rank in range(num_shards)}}
    dcp.load(headers, checkpoint_id=path, no_dist=True)
    for rank in range(num_shards):
        header = headers["last_words_rng"][f"rank_{rank}"]
        if (
            type(header["schema"]) is not int
            or header["schema"] != 1
            or not isinstance(header["device_type"], str)
            or not header["device_type"]
            or (expected_device_type is not None and header["device_type"] != expected_device_type)
        ):
            raise ValueError(f"Cursor RNG schema or device type differs for shard {rank}")
        if header["device_type"] != "cpu":
            name = f"last_words_rng.rank_{rank}.accelerator"
            metadata = keys.get(name)
            import torch
            from torch.distributed.checkpoint.metadata import TensorStorageMetadata

            if (
                not isinstance(metadata, TensorStorageMetadata)
                or metadata.properties.dtype != torch.uint8
                or len(metadata.size) != 1
                or metadata.size[0] <= 0
            ):
                raise ValueError(f"Cursor has no valid accelerator RNG stream for shard {rank}")


def has_confirmed_save_result(path, manifest, *, allow_missing_legacy_receipt=False):
    """Only an explicitly selected v1 checkpoint may omit the receipt."""
    version = manifest.get("format_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"Unsupported emergency checkpoint format version: {version!r}")
    if _read_json(path / "publication.json").get("success") is not True:
        raise ValueError("Emergency checkpoint publication failed")
    receipt = path / "service-result.json"
    try:
        result = _read_json(receipt)
    except FileNotFoundError:
        if version == 1:
            return allow_missing_legacy_receipt
        result = None
    except (ValueError, OSError) as error:
        logger.warning("Unreadable service receipt %s: %s", receipt, error)
        result = None
    try:
        binding = _read_json(path / "coordinator-binding.json")
    except (ValueError, OSError) as error:
        logger.warning("Unreadable coordinator binding in %s: %s", path, error)
        return False
    return (
        isinstance(binding, dict)
        and binding.get("run_id") == manifest["run_id"]
        and type(binding.get("replica_id")) is int
        and binding["replica_id"] == manifest["replica_id"]
        and type(binding.get("round")) is int
        and binding["round"] == 1
        and isinstance(binding.get("instance"), str)
        and bool(binding["instance"])
        and isinstance(result, dict)
        and result.get("success") is True
        and result.get("run_id") == binding["run_id"]
        and type(result.get("replica_id")) is int
        and result["replica_id"] == binding["replica_id"]
        and type(result.get("round")) is int
        and result["round"] == binding["round"]
        and result.get("coordinator_instance") == binding["instance"]
    )


def recovery_preflight(
    path,
    manifest,
    num_replicas,
    *,
    expected_model=None,
    expected_cursors=None,
    expected_device_type=None,
    tokenizer_sha256=None,
):
    path = Path(path).resolve()
    validate_state_schema(path / "model", manifest.get("recovery_schema"), kind="model", expected=expected_model)
    expected = {key: manifest["state"][key] for key in ("step", "last_words_num_updates")}
    if load_train_progress(path / "model") != expected:
        raise ValueError("Model progress differs from manifest")
    if "torchft" in manifest:
        progress = manifest["torchft"]
        if (
            not isinstance(progress, dict)
            or set(progress) != {"step", "batches_committed"}
            or any(type(value) is not int or value < 0 for value in progress.values())
            or type(manifest.get("event_id")) is not int
            or manifest["event_id"] <= 0
            or type(manifest.get("membership_epoch")) is not int
            or manifest["membership_epoch"] < 0
        ):
            raise ValueError("Invalid elastic checkpoint progress/identity")
        if load_torchft_progress(path / "model") != progress:
            raise ValueError("TorchFT progress differs from manifest")
    if set(manifest["cursors"]) != {str(replica) for replica in range(num_replicas)}:
        raise ValueError("Candidate does not contain every replica cursor")
    for slot, entry in manifest["cursors"].items():
        if entry["path"] != f"cursors/replica-{slot}":
            raise ValueError(f"Cursor path does not belong to replica {slot}")
        source = contained(path, path / entry["path"])
        validate_storage_ranges(source)
        marker = json.loads((source / "last_words_cursor.json").read_text())
        validate_tokenizer(marker, tokenizer_sha256 or manifest.get("tokenizer_sha256"))
        validate_state_schema(
            source,
            marker.get("recovery_schema"),
            kind="cursor",
            expected=(expected_cursors or {}).get(int(slot)),
        )
        if "rng_schema" in manifest:
            if (
                manifest["rng_schema"] != 1
                or type(manifest.get("rng_shards")) is not int
                or manifest["rng_shards"] <= 0
            ):
                raise ValueError("Invalid checkpoint RNG recovery schema")
            validate_rng_cursor(source, marker, manifest["rng_shards"], expected_device_type=expected_device_type)
        if (
            marker["signature"] != manifest["signature"]
            or marker["state"] != entry["state"]
            or any(entry["state"][key] > value for key, value in expected.items())
        ):
            raise ValueError("Candidate has no compatible safe replica cursor")


def _read_json(path):
    return json.loads(Path(path).read_text())


def validate_tokenizer(record, expected_digest):
    # Legacy checkpoints predate asset binding. New records carry the active job's digest even after a legacy resume.
    if expected_digest is not None and record.get("tokenizer_sha256", expected_digest) != expected_digest:
        raise ValueError("Checkpoint tokenizer assets differ")


def select_cursor(
    root,
    replica,
    state,
    *,
    signature,
    run_id,
    anchors,
    num_shards,
    expected_device_type,
    expected=None,
    tokenizer_sha256=None,
    replica_read=None,
    list_paths=Path.glob,
):
    """Rank compatible markers cheaply, then validate DCP in recovery order."""
    from torch.distributed.checkpoint.api import CheckpointException

    branch = Path(root) / f"ft-replicat-{replica}" / run_id

    # A step commits at most one update. The directory step bounds the best
    # possible (updates, step, path) before we open the marker. Imported
    # anchors have no directory step and use the requested boundary instead.
    paths: dict[Path, int | None] = {}
    for path in list_paths(branch, "step-*"):
        suffix = path.name.removeprefix("step-")
        if suffix.isdecimal() and int(suffix) <= state["step"]:
            paths[path] = int(suffix)
    if replica in anchors:
        paths.setdefault(anchors[replica], None)

    def upper_bound(path: Path) -> tuple[int, int, str]:
        directory_step = paths[path]
        step = state["step"] if directory_step is None else directory_step
        return min(step, state["last_words_num_updates"]), step, str(path)

    best_key: tuple[int, int, str] | None = None
    best: tuple[Path, dict] | None = None

    def check_cursor(source):
        contained(root, source)
        record = _read_json(source / "last_words_cursor.json")
        if record["signature"] != signature:
            return None
        validate_tokenizer(record, tokenizer_sha256)
        progress = record["state"]
        step, updates = progress["step"], progress["last_words_num_updates"]
        if type(step) is not int or type(updates) is not int or not 0 <= updates <= step:
            raise ValueError("Invalid cursor progress")
        if paths[source] is not None and step != paths[source]:
            raise ValueError("Cursor step differs from its directory")
        if step > state["step"] or updates > state["last_words_num_updates"]:
            return None
        key = (updates, step, str(source))
        if best_key is not None and key <= best_key:
            return None
        validate_storage_ranges(source)
        validate_state_schema(source, record.get("recovery_schema"), kind="cursor", expected=expected)
        if num_shards is not None:
            validate_rng_cursor(source, record, num_shards, expected_device_type=expected_device_type)
        return key, record

    for source in sorted(paths, key=upper_bound, reverse=True):
        if best_key is not None and upper_bound(source) <= best_key:
            break
        try:
            read = partial(check_cursor, source)
            candidate = replica_read(read, every_rank=True) if replica_read else read()
            if candidate is not None:
                best_key, record = candidate
                best = source, record
        except (Exception, CheckpointException) as error:
            logger.warning("Ignoring incomplete cursor %s: %s", source, error)
    return best


def read_periodic_binding(path, signature, compatible_signatures, tokenizer_sha256):
    binding = _read_json(path / "last_words_binding.json")
    if binding["signature"] not in (signature, *compatible_signatures):
        raise ValueError("Periodic checkpoint configuration differs")
    validate_tokenizer(binding, tokenizer_sha256)
    return binding


def periodic_model_preflight(
    root,
    step,
    *,
    signature,
    num_replicas,
    num_shards=None,
    expected_model=None,
    compatible_signatures=(),
    tokenizer_sha256=None,
):
    """Validate the periodic model and return its bound cursor search parameters."""
    root = Path(root)
    path = contained(root, root / f"step-{step}")
    if not (path / ".metadata").is_file():
        raise ValueError("Periodic checkpoint metadata is missing")
    if step == 0:
        validate_storage_ranges(path)
        return signature, {}
    binding = read_periodic_binding(path, signature, compatible_signatures, tokenizer_sha256)
    validate_state_schema(path, binding.get("recovery_schema"), kind="model", expected=expected_model)
    if (
        binding.get("rng_schema") != 1
        or type(binding.get("rng_shards")) is not int
        or binding["rng_shards"] <= 0
        or (num_shards is not None and binding["rng_shards"] != num_shards)
    ):
        raise ValueError("Periodic checkpoint has no matching RNG recovery chain")
    num_shards = binding["rng_shards"]
    state = load_train_progress(path)
    if state["step"] != step or any(binding["state"][key] != value for key, value in state.items()):
        raise ValueError("Periodic checkpoint binding progress differs")
    anchors = {}
    slots = {str(replica) for replica in range(num_replicas)}
    for slot, anchor in binding["anchors"].items():
        source = contained(root, root / anchor)
        if slot not in slots:
            raise ValueError("Periodic checkpoint refers to an unknown replica cursor")
        # Imported and emergency anchors identify their slot in the leaf name.
        # A previous periodic resume can instead bind a run's original step directory.
        parts = source.relative_to(root.resolve()).parts
        if len(parts) == 3 and parts[0] == f"ft-replicat-{slot}":
            matches = parts[2].startswith("step-") and parts[2][5:].isdecimal()
        else:
            matches = source.parent.name in {"cursors", "resume-anchors"} and source.name == f"replica-{slot}"
        if not matches:
            raise ValueError(f"Periodic checkpoint cursor path does not belong to replica {slot}")
        anchors[int(slot)] = source
    return binding["signature"], {
        "state": state,
        "run_id": binding["run_id"],
        "anchors": anchors,
        "num_shards": num_shards,
        "tokenizer_sha256": tokenizer_sha256 or binding.get("tokenizer_sha256"),
    }
