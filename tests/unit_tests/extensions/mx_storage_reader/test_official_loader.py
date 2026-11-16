# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composite official-release loading: cross-shard scales and multi-shard tensors."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint._hf_utils import CUSTOM_METADATA_KEY, SAVED_OFFSETS_KEY
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list

from torchtitan_npu.extensions.mx_storage_reader.official_hf_storage import OfficialMxHuggingFaceStorageReader


def _write_package(path: Path, shards: dict[str, dict], offsets: dict[str, dict[str, list[int]]]) -> None:
    """Write shards plus index; ``offsets`` must cover every key of a shard or the raw metadata reader fails."""
    weight_map: dict[str, str] = {}
    for name, tensors in shards.items():
        metadata = None
        if name in offsets:
            metadata = {
                CUSTOM_METADATA_KEY: json.dumps({k: {SAVED_OFFSETS_KEY: v} for k, v in offsets[name].items()})
            }
        save_file(tensors, str(path / name), metadata=metadata)
        weight_map.update(dict.fromkeys(tensors, name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


class _RegionPlanner(DefaultLoadPlanner):
    """Windowed loads: odd offsets and sizes exercise the aligned-crop read paths."""

    def __init__(self, regions: dict[str, tuple[tuple[int, int], tuple[int, int]]]):
        super().__init__()
        self._regions = regions

    def lookup_tensor(self, index):
        return self.state_dict[index.fqn]

    def create_local_plan(self):
        requests = []
        for fqn, (offsets, sizes) in self._regions.items():
            region = ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes))
            requests.extend(create_read_items_for_chunk_list(fqn, self.metadata.state_dict_metadata[fqn], [region]))
        return LoadPlan(requests)


# --- fixed reference weights -------------------------------------------------
# MX attention pair: qdata = 2.0, per-block scales 2^-1, 2^0, 2^+1.
WQ_QDATA = torch.full((4, 96), 2.0).to(torch.float8_e4m3fn)
WQ_SCALE = torch.tensor([[126, 127, 128]] * 4, dtype=torch.uint8)
WQ_EXPECTED = torch.tensor([1.0, 2.0, 4.0]).repeat_interleave(32).repeat(4, 1)

# Engram embed table (row-MX, row_block=1): rows 0-15/16-31/32-47 scaled 1x/2x/4x.
EMBED_QDATA = torch.ones(48, 32).to(torch.float8_e4m3fn)
EMBED_SCALE = torch.tensor([1.0] * 16 + [2.0] * 16 + [4.0] * 16).unsqueeze(1).to(torch.float8_e8m0fnu)
EMBED_EXPECTED = EMBED_QDATA.float() * EMBED_SCALE.float().repeat_interleave(32, 1)

# Engram gate wkv (block-MX, row_block=32, col_block=32).
WKV_QDATA = torch.full((64, 64), 2.0).to(torch.float8_e4m3fn)
WKV_SCALE = torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu)
WKV_EXPECTED = torch.tensor([[2.0, 4.0], [8.0, 16.0]]).repeat_interleave(32, 0).repeat_interleave(32, 1)

# Official attention pair: E4M3 data with a 32x32-tiled E8M0 scale, grid (3, 2).
BLK_QDATA = torch.full((96, 64), 2.0).to(torch.float8_e4m3fn)
BLK_SCALE = torch.tensor([[0.5, 2.0], [1.0, 0.5], [0.25, 1.0]]).to(torch.float8_e8m0fnu)
BLK_EXPECTED = 2.0 * BLK_SCALE.float().repeat_interleave(32, 0).repeat_interleave(32, 1)

# Official routed-expert pair: MXFP4 bytes tagged I8 by safetensors, rowwise scale.
FP4_QDATA = torch.full((64, 32), -30, dtype=torch.int8)  # byte 0xE2 -> (+1, -4)
FP4_SCALE = torch.tensor([0.5, 1.0]).repeat(64, 1).to(torch.float8_e8m0fnu)
FP4_EXPECTED = (torch.tensor([1.0, -4.0]).repeat(32) * torch.tensor([0.5] * 32 + [1.0] * 32)).expand(64, 64)


@pytest.fixture
def official_dir(tmp_path):
    """One official-style package exercising cross-shard sidecars and multi-shard tensors at once."""
    shards = {
        "model-00001-of-00005.safetensors": {
            "layers.0.attention.wq_b.weight": WQ_QDATA,
            "norm.weight": torch.tensor([1.0, 2.0]),
        },
        "model-00002-of-00005.safetensors": {
            "layers.0.attention.wq_b.scale": WQ_SCALE,
            "mtp.layers.0.attn.w.weight": WQ_QDATA[:, :32].contiguous(),
        },
        "model-00003-of-00005.safetensors": {
            "layers.1.engram.embed.weight": EMBED_QDATA[:32],
            "layers.1.engram.wkv.weight": WKV_QDATA,
            "mtp.layers.0.attn.w.scale": torch.full((4, 1), 127, dtype=torch.uint8),
        },
        "model-00004-of-00005.safetensors": {
            "layers.1.engram.embed.weight": EMBED_QDATA[32:],
            "layers.1.engram.embed.scale": EMBED_SCALE,
        },
        "model-00005-of-00005.safetensors": {"layers.1.engram.wkv.scale": WKV_SCALE},
    }
    offsets = {
        "model-00003-of-00005.safetensors": {
            "layers.1.engram.embed.weight": [0, 0],
            "layers.1.engram.wkv.weight": [0, 0],
            "mtp.layers.0.attn.w.scale": [0, 0],
        },
        "model-00004-of-00005.safetensors": {
            "layers.1.engram.embed.weight": [32, 0],
            "layers.1.engram.embed.scale": [0, 0],
        },
    }
    _write_package(tmp_path, shards, offsets)
    return tmp_path


def _official_reader(path):
    return OfficialMxHuggingFaceStorageReader(str(path), table_shapes={"layers.1.engram.embed.weight": (48, 32)})


def test_official_metadata_merges_all_three_tensor_kinds(official_dir):
    reader = _official_reader(official_dir)
    metadata = reader.read_metadata()

    # MX pairs become logical fp32; scale entries vanish, including the untargeted MTP pair.
    assert metadata.state_dict_metadata["layers.0.attention.wq_b.weight"].properties.dtype == torch.float32
    assert metadata.state_dict_metadata["layers.0.attention.wq_b.weight"].size == (4, 96)
    assert "layers.0.attention.wq_b.scale" not in metadata.state_dict_metadata
    assert set(reader._mx_tensors) == {"layers.0.attention.wq_b.weight", "mtp.layers.0.attn.w.weight"}
    # Engram weights keep their stored dtype; sidecars were consumed by the Engram reader.
    assert metadata.state_dict_metadata["layers.1.engram.embed.weight"].properties.dtype == torch.float8_e4m3fn
    assert metadata.state_dict_metadata["layers.1.engram.embed.weight"].size == (48, 32)
    assert "layers.1.engram.embed.scale" not in metadata.state_dict_metadata
    assert "layers.1.engram.wkv.scale" not in metadata.state_dict_metadata
    # Plain tensors survive untouched.
    assert metadata.state_dict_metadata["norm.weight"].properties.dtype == torch.float32


def test_official_dcp_load_full_package(official_dir, cpu_npu_boundary):
    target = {
        "layers.0.attention.wq_b.weight": torch.empty(4, 96),
        "layers.1.engram.embed.weight": torch.empty(48, 32),
        "layers.1.engram.wkv.weight": torch.empty(64, 64),
        "norm.weight": torch.empty(2),
    }
    dcp.load(target, storage_reader=_official_reader(official_dir))

    torch.testing.assert_close(target["layers.0.attention.wq_b.weight"], WQ_EXPECTED, rtol=0, atol=0)
    torch.testing.assert_close(target["layers.1.engram.embed.weight"], EMBED_EXPECTED, rtol=0, atol=0)
    torch.testing.assert_close(target["layers.1.engram.wkv.weight"], WKV_EXPECTED, rtol=0, atol=0)
    torch.testing.assert_close(target["norm.weight"], torch.tensor([1.0, 2.0]))


def test_official_dcp_load_windowed_across_shards(official_dir, cpu_npu_boundary):
    # Embed window straddles the row-chunk boundary at 32; wq window hits the crop path.
    planner = _RegionPlanner(
        {
            "layers.1.engram.embed.weight": ((17, 5), (20, 19)),
            "layers.0.attention.wq_b.weight": ((1, 7), (2, 80)),
        }
    )
    target = {
        "layers.1.engram.embed.weight": torch.empty(20, 19),
        "layers.0.attention.wq_b.weight": torch.empty(2, 80),
    }
    dcp.load(target, storage_reader=_official_reader(official_dir), planner=planner)

    torch.testing.assert_close(target["layers.1.engram.embed.weight"], EMBED_EXPECTED[17:37, 5:24], rtol=0, atol=0)
    torch.testing.assert_close(target["layers.0.attention.wq_b.weight"], WQ_EXPECTED[1:3, 7:87], rtol=0, atol=0)


@pytest.mark.parametrize("sliced", [False, True], ids=["full", "odd-offset"])
def test_official_mx_weight_and_scale_both_multi_shard(cpu_npu_boundary, tmp_path, sliced):
    # Weight chunked over 3 files, scale over 2 different files, no Engram tables.
    weight = torch.ones(2, 96).to(torch.float8_e4m3fn)
    scale = torch.tensor([[126, 127, 128], [129, 128, 127]], dtype=torch.uint8)
    shards = {
        "model-00001-of-00004.safetensors": {"linear.weight": weight[:, :32].contiguous()},
        "model-00002-of-00004.safetensors": {"linear.weight": weight[:, 32:64].contiguous()},
        "model-00003-of-00004.safetensors": {
            "linear.weight": weight[:, 64:].contiguous(),
            "linear.scale": scale[:, :1].contiguous(),
        },
        "model-00004-of-00004.safetensors": {"linear.scale": scale[:, 1:].contiguous()},
    }
    offsets = {
        "model-00001-of-00004.safetensors": {"linear.weight": [0, 0]},
        "model-00002-of-00004.safetensors": {"linear.weight": [0, 32]},
        "model-00003-of-00004.safetensors": {"linear.weight": [0, 64], "linear.scale": [0, 0]},
        "model-00004-of-00004.safetensors": {"linear.scale": [0, 1]},
    }
    _write_package(tmp_path, shards, offsets)

    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    sizes = (2, 89) if sliced else (2, 96)
    target = {"linear.weight": torch.empty(*sizes)}
    planner = _RegionPlanner({"linear.weight": ((0, 7), sizes)}) if sliced else None
    dcp.load(target, storage_reader=reader, planner=planner)

    expected = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
    expected = expected[:, 7:] if sliced else expected
    torch.testing.assert_close(target["linear.weight"], expected, rtol=0, atol=0)


def test_official_rejects_orphan_fp8_without_scale_targeted(tmp_path):
    # A requested F8 weight with no .scale sidecar must fail closed.
    _write_package(
        tmp_path,
        {"model.safetensors": {"layers.0.attn.wq.weight": torch.ones(2, 32).to(torch.float8_e4m3fn)}},
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"layers.0.attn.wq.weight": torch.empty(2, 32)}
    with pytest.raises(CheckpointException, match="layers.0.attn.wq.weight"):
        dcp.load(target, storage_reader=reader)


def test_official_unrequested_orphan_fp8_stays_unread(cpu_npu_boundary, tmp_path):
    # An orphan elsewhere in the directory must not block loading a valid MX pair.
    qdata = torch.ones(2, 32).to(torch.float8_e4m3fn)
    scale = torch.full((2, 1), 127, dtype=torch.uint8)
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "layers.0.attn.wq.weight": qdata,
                "layers.0.attn.wq.scale": scale,
                "mtp.layers.0.attn.wq.weight": torch.ones(2, 32).to(torch.float8_e4m3fn),
            }
        },
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"layers.0.attn.wq.weight": torch.empty(2, 32)}
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["layers.0.attn.wq.weight"], qdata.float(), rtol=0, atol=0)


def test_official_unrequested_invalid_mx_pair_stays_unread(tmp_path):
    # Out-of-protocol pairs elsewhere must not block tensors that never reference them.
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "norm.weight": torch.tensor([1.0, 2.0]),
                "mtp.linear.weight": torch.ones(2, 32).to(torch.float8_e4m3fn),
                "mtp.linear.scale": torch.full((2, 1), 1.0),
            }
        },
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"norm.weight": torch.empty(2)}
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["norm.weight"], torch.tensor([1.0, 2.0]))
    assert "mtp.linear.weight" not in reader._mx_tensors


def test_official_rejects_invalid_mx_pair_when_targeted(tmp_path):
    # The deferred skip fails closed once the target references the pair.
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "norm.weight": torch.tensor([1.0, 2.0]),
                "mtp.linear.weight": torch.ones(2, 32).to(torch.float8_e4m3fn),
                "mtp.linear.scale": torch.full((2, 1), 1.0),
            }
        },
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"mtp.linear.weight": torch.empty(2, 32)}
    with pytest.raises(CheckpointException, match="does not support the quantization format"):
        dcp.load(target, storage_reader=reader)


def test_official_rejects_unsupported_quantized_format_when_targeted(cpu_npu_boundary, tmp_path):
    # E5M2 + .scale sidecar is outside both protocols; the Engram guard must fail closed.
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "layers.0.attention.wkv.weight": torch.ones(2, 32).to(torch.float8_e5m2),
                "layers.0.attention.wkv.scale": torch.full((2, 1), 127, dtype=torch.uint8),
            }
        },
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"layers.0.attention.wkv.weight": torch.empty(2, 32)}
    with pytest.raises(CheckpointException, match="does not support the quantization format"):
        dcp.load(target, storage_reader=reader)


@pytest.fixture
def official_dir_v2(tmp_path):
    """Official-package geometries: 32x32-tiled attn scale and I8-packed fp4 expert, both cross-shard."""
    shards = {
        "model-00001-of-00003.safetensors": {
            "layers.2.attn.wkv.weight": BLK_QDATA[:64],
            "layers.2.mlp.experts.7.w1.weight": FP4_QDATA,
        },
        "model-00002-of-00003.safetensors": {
            "layers.2.attn.wkv.weight": BLK_QDATA[64:],
            "layers.2.attn.wkv.scale": BLK_SCALE,
        },
        "model-00003-of-00003.safetensors": {"layers.2.mlp.experts.7.w1.scale": FP4_SCALE},
    }
    offsets = {
        "model-00002-of-00003.safetensors": {
            "layers.2.attn.wkv.weight": [64, 0],
            "layers.2.attn.wkv.scale": [0, 0],
        },
    }
    _write_package(tmp_path, shards, offsets)
    return tmp_path


def test_official_block_scale_and_fp4_metadata(official_dir_v2):
    reader = OfficialMxHuggingFaceStorageReader(str(official_dir_v2), table_shapes={})
    metadata = reader.read_metadata()

    attn = reader._mx_tensors["layers.2.attn.wkv.weight"]
    expert = reader._mx_tensors["layers.2.mlp.experts.7.w1.weight"]
    assert (attn.row_block, attn.packing) == (32, 1)
    assert (expert.row_block, expert.packing) == (1, 2)
    assert metadata.state_dict_metadata["layers.2.attn.wkv.weight"].size == (96, 64)
    assert metadata.state_dict_metadata["layers.2.mlp.experts.7.w1.weight"].size == (64, 64)
    assert metadata.state_dict_metadata["layers.2.mlp.experts.7.w1.weight"].properties.dtype == torch.float32
    assert "layers.2.attn.wkv.scale" not in metadata.state_dict_metadata
    assert "layers.2.mlp.experts.7.w1.scale" not in metadata.state_dict_metadata


def test_official_block_scale_and_fp4_load(official_dir_v2, cpu_npu_boundary):
    target = {
        "layers.2.attn.wkv.weight": torch.empty(96, 64),
        "layers.2.mlp.experts.7.w1.weight": torch.empty(64, 64),
    }
    dcp.load(target, storage_reader=OfficialMxHuggingFaceStorageReader(str(official_dir_v2), table_shapes={}))

    torch.testing.assert_close(target["layers.2.attn.wkv.weight"], BLK_EXPECTED, rtol=0, atol=0)
    torch.testing.assert_close(target["layers.2.mlp.experts.7.w1.weight"], FP4_EXPECTED, rtol=0, atol=0)


def test_official_router_gate_prefix_load(tmp_path):
    """A shrunken debug model reads the leading rows of the official aggregated router tensors."""
    bias = torch.arange(8, dtype=torch.float32)
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    _write_package(
        tmp_path,
        {"model.safetensors": {"layers.0.ffn.gate.bias": bias, "layers.0.ffn.gate.weight": weight}},
        {},
    )
    target = {
        "layers.0.ffn.gate.bias": torch.empty(5, dtype=torch.float32),
        "layers.0.ffn.gate.weight": torch.empty(5, 4, dtype=torch.bfloat16),
    }
    reader = OfficialMxHuggingFaceStorageReader(
        str(tmp_path),
        table_shapes={},
        expert_prefix_shapes={"layers.0.ffn.gate.bias": (5,), "layers.0.ffn.gate.weight": (5, 4)},
    )
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["layers.0.ffn.gate.bias"], bias[:5], rtol=0, atol=0)
    torch.testing.assert_close(target["layers.0.ffn.gate.weight"], weight[:5], rtol=0, atol=0)


def test_official_router_gate_prefix_beyond_checkpoint(tmp_path):
    """Fabricating rows the checkpoint does not have must fail closed."""
    bias = torch.arange(8, dtype=torch.float32)
    _write_package(tmp_path, {"model.safetensors": {"layers.0.ffn.gate.bias": bias}}, {})
    target = {"layers.0.ffn.gate.bias": torch.empty(10)}
    reader = OfficialMxHuggingFaceStorageReader(
        str(tmp_path), table_shapes={}, expert_prefix_shapes={"layers.0.ffn.gate.bias": (10,)}
    )
    with pytest.raises(CheckpointException, match="cannot serve"):
        dcp.load(target, storage_reader=reader)


def test_official_router_gate_prefix_full_size_and_unknown_key(tmp_path):
    """A prefix at the checkpoint's own size clips nothing; a key the package lacks is skipped."""
    bias = torch.arange(8, dtype=torch.float32)
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    _write_package(
        tmp_path, {"model.safetensors": {"layers.0.ffn.gate.bias": bias, "layers.0.ffn.gate.weight": weight}}, {}
    )
    target = {
        "layers.0.ffn.gate.bias": torch.empty(8, dtype=torch.float32),
        "layers.0.ffn.gate.weight": torch.empty(8, 4, dtype=torch.bfloat16),
    }
    reader = OfficialMxHuggingFaceStorageReader(
        str(tmp_path),
        table_shapes={},
        expert_prefix_shapes={"layers.0.ffn.gate.bias": (8,), "layers.9.ffn.gate.bias": (5,)},
    )
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["layers.0.ffn.gate.bias"], bias, rtol=0, atol=0)
    torch.testing.assert_close(target["layers.0.ffn.gate.weight"], weight, rtol=0, atol=0)


def test_official_router_gate_prefix_tail_mismatch(tmp_path):
    """A prefix whose trailing dimensions disagree with the checkpoint must fail closed."""
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    _write_package(tmp_path, {"model.safetensors": {"layers.0.ffn.gate.weight": weight}}, {})
    reader = OfficialMxHuggingFaceStorageReader(
        str(tmp_path), table_shapes={}, expert_prefix_shapes={"layers.0.ffn.gate.weight": (5, 6)}
    )
    with pytest.raises(ValueError, match="cannot serve"):
        reader.read_metadata()


def test_official_router_gate_prefix_multi_chunk_rejected(tmp_path):
    """Clipping a router tensor saved across shards is unsupported and must fail loudly."""
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    shards = {
        "model-00001-of-00002.safetensors": {"layers.0.ffn.gate.weight": weight[:4].contiguous()},
        "model-00002-of-00002.safetensors": {"layers.0.ffn.gate.weight": weight[4:].contiguous()},
    }
    offsets = {
        "model-00001-of-00002.safetensors": {"layers.0.ffn.gate.weight": [0, 0]},
        "model-00002-of-00002.safetensors": {"layers.0.ffn.gate.weight": [4, 0]},
    }
    _write_package(tmp_path, shards, offsets)
    reader = OfficialMxHuggingFaceStorageReader(
        str(tmp_path), table_shapes={}, expert_prefix_shapes={"layers.0.ffn.gate.weight": (5, 4)}
    )
    with pytest.raises(NotImplementedError, match="single-chunk"):
        reader.read_metadata()


def test_official_block_scale_tail_tile(cpu_npu_boundary, tmp_path):
    # 100 rows leave a 4-row tail in the last 32-row scale tile; narrow must stop at the tail.
    qdata = torch.full((100, 64), 2.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0], [1.0, 0.5], [0.25, 1.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu)
    expected = 2.0 * scale.float().repeat_interleave(32, 0)[:100].repeat_interleave(32, 1)
    _write_package(
        tmp_path,
        {"model.safetensors": {"layers.3.attn.wkv.weight": qdata, "layers.3.attn.wkv.scale": scale}},
        {},
    )
    target = {"layers.3.attn.wkv.weight": torch.empty(100, 64)}
    dcp.load(target, storage_reader=OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={}))
    torch.testing.assert_close(target["layers.3.attn.wkv.weight"], expected, rtol=0, atol=0)

    # Window inside the tail tile only: nonzero row_crop against the padded tile.
    window_target = {"layers.3.attn.wkv.weight": torch.empty(10, 64)}
    dcp.load(
        window_target,
        storage_reader=OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={}),
        planner=_RegionPlanner({"layers.3.attn.wkv.weight": ((90, 0), (10, 64))}),
    )
    torch.testing.assert_close(window_target["layers.3.attn.wkv.weight"], expected[90:100], rtol=0, atol=0)


def test_official_block_scale_windowed_rows(official_dir_v2, cpu_npu_boundary):
    # Rows 17-66 straddle two 32-row scale tiles and the weight's shard boundary at row 64.
    planner = _RegionPlanner({"layers.2.attn.wkv.weight": ((17, 5), (50, 40))})
    target = {"layers.2.attn.wkv.weight": torch.empty(50, 40)}
    dcp.load(
        target,
        storage_reader=OfficialMxHuggingFaceStorageReader(str(official_dir_v2), table_shapes={}),
        planner=planner,
    )

    torch.testing.assert_close(target["layers.2.attn.wkv.weight"], BLK_EXPECTED[17:67, 5:45], rtol=0, atol=0)


def test_official_scale_shape_between_grids_rejected(tmp_path):
    # (6, 2) is neither the rowwise (96, 2) nor the 32-row-blocked (3, 2) grid.
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "norm.weight": torch.tensor([1.0, 2.0]),
                "layers.0.attn.wkv.weight": torch.ones(96, 64).to(torch.float8_e4m3fn),
                "layers.0.attn.wkv.scale": torch.full((6, 2), 127, dtype=torch.uint8),
            }
        },
        {},
    )
    untargeted = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"norm.weight": torch.empty(2)}
    dcp.load(target, storage_reader=untargeted)  # malformed pair elsewhere must not block
    assert "layers.0.attn.wkv.weight" not in untargeted._mx_tensors

    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={})
    target = {"layers.0.attn.wkv.weight": torch.empty(96, 64)}
    with pytest.raises(CheckpointException, match="does not support the quantization format"):
        dcp.load(target, storage_reader=reader)


def test_official_engram_shape_mismatch_fails(tmp_path):
    _write_package(
        tmp_path,
        {
            "model.safetensors": {
                "layers.1.engram.embed.weight": torch.ones(40, 32).to(torch.float8_e4m3fn),
                "layers.1.engram.embed.scale": torch.full((40, 1), 2.0).to(torch.float8_e8m0fnu),
            }
        },
        {},
    )
    reader = OfficialMxHuggingFaceStorageReader(str(tmp_path), table_shapes={"layers.1.engram.embed.weight": (48, 32)})
    with pytest.raises(ValueError, match="does not match expected shape"):
        reader.read_metadata()
