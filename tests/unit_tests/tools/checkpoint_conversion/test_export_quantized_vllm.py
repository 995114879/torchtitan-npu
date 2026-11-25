# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protect the vLLM Ascend export contract of the checkpoint conversion tools.

The tests cover the CPU-observable guarantees the exporter makes before a
checkpoint can be served: the plain payload dtype, the ``weight_scale``
geometry, the description and loadability checks in ``vllm_export_contract``, and
the vision completeness rules.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "torchtitan_npu"
    / "tools"
    / "checkpoint_conversion"
    / "export_quantized_vllm.py"
)


def _make_tool_package(root: Path) -> Path:
    """Mirror the package layout so ``python -m`` can reach the real tool files."""
    package = root / "torchtitan_npu"
    (package / "tools").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tools" / "checkpoint_conversion").symlink_to(_SCRIPT.parent)
    return root


@pytest.fixture(scope="module")
def export():
    # The script imports ``vllm_export_contract``/``vllm_export_specs`` as top-level
    # modules when loaded by path, so its directory is on sys.path only for the load.
    original_path = list(sys.path)
    had_specs = "vllm_export_specs" in sys.modules
    had_contract = "vllm_export_contract" in sys.modules
    had_dcp_configs = "dcp_quant_configs" in sys.modules
    spec = importlib.util.spec_from_file_location("export_quantized_vllm_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    yield module
    if not had_specs:
        sys.modules.pop("vllm_export_specs", None)
    if not had_contract:
        sys.modules.pop("vllm_export_contract", None)
    if not had_dcp_configs:
        sys.modules.pop("dcp_quant_configs", None)


@pytest.fixture(scope="module")
def contract(export):
    return sys.modules["vllm_export_contract"]


def test_python_m_entrypoint_prints_help(tmp_path):
    """``python -m`` bootstraps the tool directory before its sibling imports.

    The real product package imports the NPU stack a CPU test cannot load, so
    the layout is mirrored with the real tool directory linked in.
    """
    package_root = _make_tool_package(tmp_path / "package")

    completed = subprocess.run(
        [sys.executable, "-m", "torchtitan_npu.tools.checkpoint_conversion.export_quantized_vllm", "--help"],
        cwd=package_root,
        env={**os.environ, "PYTHONPATH": str(package_root)},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_vision_prefixes_are_discovered_from_hf_index(contract, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(json.dumps({"vision_config": {"depth": 2}}))
    (assets / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.mm_projector.layers.0.weight": "model-00001.safetensors",
                    "model.language_model.layers.0.weight": "model-00001.safetensors",
                }
            }
        )
    )
    prefixes = contract.vision_prefixes_from_hf_assets(assets)
    assert "model.mm_projector." in prefixes
    assert "model.language_model." not in prefixes
    assert contract.has_vision_tower({"model.mm_projector.layers.0.weight": torch.zeros(1)}, prefixes)


def test_restore_native_hf_dtypes_from_safetensors_header(export, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    save_file(
        {
            "layers.0.attn_norm.weight": torch.zeros(2, dtype=torch.bfloat16),
            "layers.0.ffn.gate.bias": torch.zeros(2, dtype=torch.float32),
            "layers.0.mlp.fp4_payload": torch.zeros(2, dtype=torch.uint8),
            "layers.0.router.tid2eid": torch.zeros(2, dtype=torch.int64),
            "layers.0.router.narrowed_ids": torch.zeros(2, dtype=torch.int32),
        },
        str(assets / "model.safetensors"),
    )
    state = {
        "layers.0.attn_norm.weight": torch.zeros(2, dtype=torch.float32),
        "layers.0.ffn.gate.bias": torch.zeros(2, dtype=torch.bfloat16),
        "layers.0.mlp.fp4_payload": torch.zeros(2, dtype=torch.float32),
        "layers.0.router.tid2eid": torch.zeros(2, dtype=torch.int32),
        "layers.0.router.narrowed_ids": torch.zeros(2, dtype=torch.int64),
    }
    export._restore_native_hf_dtypes(state, assets)
    assert state["layers.0.attn_norm.weight"].dtype is torch.bfloat16
    assert state["layers.0.ffn.gate.bias"].dtype is torch.float32
    # A floating DCP tensor cannot be converted to packed FP4 bytes by a dtype cast.
    assert state["layers.0.mlp.fp4_payload"].dtype is torch.float32
    # Integer restore only widens: narrowing would silently wrap the restored ids.
    assert state["layers.0.router.tid2eid"].dtype is torch.int64
    assert state["layers.0.router.narrowed_ids"].dtype is torch.int64


class _FakeWeightConfig:
    """The weight-config surface the export helpers read; no quant kernels are involved."""

    def __init__(self, elem_dtype, block_size=32):
        self.elem_dtype = elem_dtype
        self.block_size = block_size
        self.mxfp4_fake_quantize_config = None


class _FakeSwap:
    """Stand-in for ``ParamSwapConfig``: only its two configs reach the helpers."""

    def __init__(self, elem_dtype):
        self.weight_config = _FakeWeightConfig(elem_dtype)
        self.activation_config = None


_MXFP8_FQN = "layers.0.attn.wq_a.weight"
_ENGRAM_FQN = "layers.1.engram.embed.weight"
_ENGRAM_SCALE = "layers.1.engram.embed.scale"


def _mxfp8_plan():
    return [(_FakeSwap(torch.float8_e4m3fn), None)]


def _write_export(tmp_path, contract, *, state, quantized=None, plan=None, config, shard_state=None, engram_fqns=()):
    """Write config.json, one shard, and the real description for the validator."""
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    if config is not None:
        (output_dir / "config.json").write_text(json.dumps(config))
    save_file(state if shard_state is None else shard_state, str(output_dir / "model.safetensors"))
    quantized = {_MXFP8_FQN: 0} if quantized is None else quantized
    plan = _mxfp8_plan() if plan is None else plan
    description = contract.build_quant_model_description(state, quantized, plan, engram_fqns=engram_fqns)
    contract.write_quant_model_description(output_dir, description)
    return output_dir, ["model.safetensors"], description


def _validate(
    contract,
    output_dir,
    shard_names,
    state,
    description,
    *,
    quantized=None,
    plan=None,
    model_name="deepseek_v4",
    keep_scale_tail_dim=False,
    needs_vision=False,
    input_sizes=None,
):
    contract.validate_export(
        output_dir,
        shard_names,
        state,
        {_MXFP8_FQN: 0} if quantized is None else quantized,
        _mxfp8_plan() if plan is None else plan,
        description,
        model_name=model_name,
        keep_scale_tail_dim=keep_scale_tail_dim,
        needs_vision=needs_vision,
        input_sizes=input_sizes,
    )


_V4_CONFIG = {"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}
_V41_CONFIG = {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"], "text_config": {}}


def _mxfp8_state():
    # K = 128 -> 128 / 32 = 4 E8M0 blocks; FP8 qdata keeps [out, K].
    return {
        _MXFP8_FQN: torch.zeros(4, 128, dtype=torch.float8_e4m3fn),
        "layers.0.attn.wq_a.weight_scale": torch.zeros(4, 4, dtype=torch.uint8),
        "layers.0.attn_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "hc_head_fn": torch.zeros(2, dtype=torch.bfloat16),
    }


def _expect_rejected(contract, tmp_path, state, match, *, config=_V4_CONFIG, shard_state=None, **validate_kwargs):
    """Write the export, then assert ``validate_export`` rejects it with ``match``."""
    output_dir, shard_names, description = _write_export(
        tmp_path, contract, state=state, config=config, shard_state=shard_state
    )
    with pytest.raises(ValueError, match=match):
        _validate(contract, output_dir, shard_names, state, description, **validate_kwargs)


def test_export_scale_folds_packed_scale_pairs(export):
    # 4096 quantized elements hold 4096 / 32 = 128 blocks, packed as (64, 2).
    scale = torch.arange(4 * 64 * 2, dtype=torch.float32).reshape(4, 64, 2)

    folded = export._export_scale(scale, keep_scale_tail_dim=False, quant_dim_size=4096, block_size=32)

    assert folded.shape == (4, 128)
    torch.testing.assert_close(folded, scale.reshape(4, 128))


def test_export_scale_trims_kernel_block_padding(export):
    # 96 elements need 3 blocks; the kernel pads the packed block count to (2, 2).
    scale = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)

    folded = export._export_scale(scale, keep_scale_tail_dim=False, quant_dim_size=96, block_size=32)

    assert folded.shape == (4, 3)
    torch.testing.assert_close(folded, scale.reshape(4, 4)[:, :3])


def test_export_qdata_keeps_fp8_native_dtype(export):
    fp8_bits = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
    fp8 = fp8_bits.view(torch.float8_e4m3fn)

    exported = export._export_qdata(fp8)

    assert exported.dtype is torch.float8_e4m3fn
    assert torch.equal(exported.view(torch.uint8), fp8_bits)


def test_export_qdata_packs_fp4_as_uint8(export):
    fp4_bits = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
    fp4 = fp4_bits.view(torch.float4_e2m1fn_x2)

    exported = export._export_qdata(fp4)

    assert exported.dtype is torch.uint8
    assert exported.shape == fp4.shape
    assert torch.equal(exported, fp4_bits)


def test_split_fused_experts_splits_w13(export):
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    state = {"layers.0.ffn.experts.0.w13.weight": values}

    export._split_fused_experts(state)

    assert set(state) == {
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w3.weight",
    }
    torch.testing.assert_close(state["layers.0.ffn.experts.0.w1.weight"], values[:2])
    torch.testing.assert_close(state["layers.0.ffn.experts.0.w3.weight"], values[2:])


def test_split_fused_experts_rejects_mixed_layout(export):
    state = {
        "layers.0.ffn.experts.0.w13.weight": torch.zeros(4, 2),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(2, 2),
    }

    with pytest.raises(ValueError, match="cannot mix fused"):
        export._split_fused_experts(state)


def test_effective_weight_config_drives_description_type_and_group_size(contract):
    # The outer block size is 64; the nested FP4 config's block size must win.
    swap = _FakeSwap(torch.float8_e4m3fn)
    swap.weight_config.block_size = 64
    swap.weight_config.mxfp4_fake_quantize_config = _FakeWeightConfig(torch.float4_e2m1fn_x2, block_size=32)
    state = {
        "a.weight": torch.zeros(2, 64, dtype=torch.uint8),
        "a.weight_scale": torch.zeros(2, 2, dtype=torch.uint8),
    }

    assert contract.effective_weight_config(swap.weight_config).elem_dtype is torch.float4_e2m1fn_x2
    assert contract.quant_description_type(swap.weight_config) == "W4A8_MXFP"
    assert contract.build_quant_model_description(state, {"a.weight": 0}, [(swap, None)])["group_size"] == 32


def test_validate_export_uses_master_input_size_for_fp4_scale(contract, tmp_path):
    # K = 64 packs into 32 bytes; the recorded master K must drive the expected block
    # count instead of the packed payload's doubled length.
    fqn = "layers.0.ffn.experts.0.w1.weight"
    state = {
        fqn: torch.zeros(4, 32, dtype=torch.uint8),
        "layers.0.ffn.experts.0.w1.weight_scale": torch.zeros(4, 2, dtype=torch.uint8),
        "hc_head_fn": torch.zeros(2, dtype=torch.bfloat16),
    }
    quantized = {fqn: 0}
    plan = [(_FakeSwap(torch.float4_e2m1fn_x2), None)]
    output_dir, shard_names, description = _write_export(
        tmp_path, contract, state=state, quantized=quantized, plan=plan, config=_V4_CONFIG
    )

    _validate(
        contract, output_dir, shard_names, state, description, quantized=quantized, plan=plan, input_sizes={fqn: 64}
    )
    # Without a recorded master K the packed payload's doubled length is the fallback.
    _validate(contract, output_dir, shard_names, state, description, quantized=quantized, plan=plan)
    assert description[fqn] == "W4A8_MXFP"

    with pytest.raises(ValueError, match="weight_scale shape"):
        _validate(
            contract, output_dir, shard_names, state, description, quantized=quantized, plan=plan, input_sizes={fqn: 96}
        )


def test_validate_export_rejects_missing_hc_head_fn(contract, tmp_path):
    state = _mxfp8_state()
    del state["hc_head_fn"]

    _expect_rejected(contract, tmp_path, state, "hc_head_fn")


def test_validate_export_rejects_description_type_mismatch(contract, tmp_path):
    state = _mxfp8_state()
    output_dir, shard_names, description = _write_export(tmp_path, contract, state=state, config=_V4_CONFIG)
    description[_MXFP8_FQN] = "W4A8_MXFP"
    contract.write_quant_model_description(output_dir, description)

    with pytest.raises(ValueError, match=r"description type 'W4A8_MXFP' != 'W8A8_MXFP8'"):
        _validate(contract, output_dir, shard_names, state, description)


def test_validate_export_rejects_config_model_type_mismatch(contract, tmp_path):
    _expect_rejected(contract, tmp_path, _mxfp8_state(), "model_type", config=_V41_CONFIG)


def test_validate_export_rejects_shard_missing_tensor(contract, tmp_path):
    state = _mxfp8_state()
    shard_state = {name: tensor for name, tensor in state.items() if name != "layers.0.attn.wq_a.weight_scale"}

    _expect_rejected(contract, tmp_path, state, "missing from the shards", shard_state=shard_state)


def test_validate_export_rejects_missing_config(contract, tmp_path):
    _expect_rejected(contract, tmp_path, _mxfp8_state(), r"config\.json is missing", config=None)


def test_validate_export_rejects_multimodal_v41_without_markers(contract, tmp_path):
    state = _mxfp8_state()
    del state["hc_head_fn"]
    state["vision.patch_embed.proj.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)

    _expect_rejected(
        contract, tmp_path, state, "image_start", config=_V41_CONFIG, model_name="deepseek_v4_1", needs_vision=True
    )


def test_quantize_engram_tables_replaces_fp32_table_with_mxfp8(export):
    fqn = "layers.1.engram.embed.weight"
    state = {
        fqn: torch.zeros(5, 64, dtype=torch.float32),
        # Non-table tensors belong to ``quantize_state_dict``, not this helper.
        "layers.1.attn_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
    }
    chunk_shapes = []

    def _fake_quantize(chunk):
        chunk_shapes.append(tuple(chunk.shape))
        return (
            torch.zeros(chunk.shape, dtype=torch.float8_e4m3fn),
            torch.ones((chunk.shape[0], chunk.shape[1] // 32), dtype=torch.float8_e8m0fnu),
        )

    engram_fqns = export.quantize_engram_tables(state, quantize_rows=_fake_quantize, chunk_rows=2)

    scale_fqn = "layers.1.engram.embed.scale"
    assert engram_fqns == {fqn}
    assert chunk_shapes == [(2, 64), (2, 64), (1, 64)]
    assert state[fqn].dtype is torch.float8_e4m3fn
    assert state[scale_fqn].dtype is torch.float8_e8m0fnu
    assert state[scale_fqn].shape == (5, 2)
    assert state["layers.1.attn_norm.weight"].dtype is torch.bfloat16


def test_build_quant_model_description_marks_engram_quant_type(contract):
    state = {
        "layers.1.engram.embed.weight": torch.zeros(4, 64, dtype=torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.zeros(4, 2, dtype=torch.float8_e8m0fnu),
        "layers.1.engram.wkv.weight": torch.zeros(4, 64, dtype=torch.float8_e4m3fn),
        "layers.1.engram.wkv.weight_scale": torch.zeros(4, 2, dtype=torch.uint8),
        # Elementwise gate parameters carry no quantization.
        "layers.1.engram.q_weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }

    description = contract.build_quant_model_description(
        state,
        {"layers.1.engram.wkv.weight": 0},
        _mxfp8_plan(),
        engram_fqns={"layers.1.engram.embed.weight"},
    )

    assert description["layers.1.engram.embed.quant_type"] == "MXFP8"
    assert "layers.1.engram.embed.weight" not in description
    assert "layers.1.engram.embed.scale" not in description
    assert description["layers.1.engram.wkv.weight"] == "W8A8_MXFP8"
    assert description["layers.1.engram.wkv.weight_scale"] == "W8A8_MXFP8"
    assert description["layers.1.engram.q_weight"] == "FLOAT"


@dataclass
class _StubMXQuantizeConfig:
    """The config surface ``quant_description_type`` reads: dense MXFP8."""

    block_size: int = 32
    elem_dtype: torch.dtype = torch.float8_e4m3fn
    round_mode: str = field(default="rint", compare=False)
    scale_alg: int = field(default=1, compare=False)
    dst_type_max: float = 0.0
    mxfp4_fake_quantize_config: object = None
    fsdp_prequantize: bool = field(default=False, compare=False)
    fsdp_prequantize_dtype: str = field(default="fp8", compare=False)


_StubBlockMXQuantizeConfig = type("BlockMXQuantizeConfig", (_StubMXQuantizeConfig,), {})


def _stub_recipe_imports(monkeypatch):
    """Replace ``build_recipe_plan``'s call-time imports so the suffix wiring is testable.

    ``torchao_npu`` and the training adaptor are absent in main-repo CI, and the
    recipe matrix itself is training-owned. The stub matrix keeps only the dense
    entry, which is the filter this test observes.
    """

    class _FilterKeys:
        DENSE = "dense"

    class _ModelType:
        DSV4 = "v4"
        DSV4_1 = "v41"
        QWEN3_5 = "qwen3_5"

    class _Spec:
        def __init__(self, filter_fn):
            self.swap_config = SimpleNamespace(weight_config=_StubMXQuantizeConfig(), activation_config=None)
            self.filter = filter_fn

    filters = types.ModuleType("torchao_npu.quantization.filters")
    filters.match_fqn_suffix = lambda *suffixes: lambda _param, fqn: fqn.endswith(suffixes)
    filters.match_fqn_regex = lambda pattern: lambda _param, fqn: re.fullmatch(pattern, fqn) is not None

    quant_configs = types.ModuleType("torchao_npu.quantization.quant_configs")
    quant_configs.MXQuantizeConfig = _StubMXQuantizeConfig
    quant_configs.BlockMXQuantizeConfig = _StubBlockMXQuantizeConfig

    configs = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.configs")
    configs._NAME_TO_TYPE = {
        "deepseek_v4": _ModelType.DSV4,
        "deepseek_v4_1": _ModelType.DSV4_1,
        "qwen3_5": _ModelType.QWEN3_5,
    }
    configs.get_model_config = lambda _model_type: SimpleNamespace(
        recipe_matrix={"mix": lambda model_filters, _ctx: [_Spec(model_filters[_FilterKeys.DENSE])]}
    )

    modules = {
        "torchao_npu": types.ModuleType("torchao_npu"),
        "torchao_npu.quantization": types.ModuleType("torchao_npu.quantization"),
        "torchao_npu.quantization.filters": filters,
        "torchao_npu.quantization.quant_configs": quant_configs,
        "torchtitan_npu": types.ModuleType("torchtitan_npu"),
        "torchtitan_npu.extensions": types.ModuleType("torchtitan_npu.extensions"),
        "torchtitan_npu.extensions.adaptor": types.ModuleType("torchtitan_npu.extensions.adaptor"),
        "torchtitan_npu.extensions.adaptor.torchao": types.ModuleType("torchtitan_npu.extensions.adaptor.torchao"),
        "torchtitan_npu.extensions.adaptor.torchao.configs": configs,
    }
    recipes = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.recipes")
    recipes.RecipeContext = lambda **kwargs: SimpleNamespace(**kwargs)
    modules[recipes.__name__] = recipes
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_v41_dense_plan_covers_engram_gate(export, monkeypatch):
    """The Engram context gate is a quantized Linear in the serving checkpoint."""
    _stub_recipe_imports(monkeypatch)

    def _matches(plan, fqn):
        return any(filter_fn(None, fqn) for _swap, filter_fn in plan)

    v41_plan = export.build_recipe_plan("mix", model_name="deepseek_v4_1", enable_mxfp4_qat=False)
    assert _matches(v41_plan, "layers.1.engram.wkv.weight")
    for fqn in ("layers.1.engram.q_weight", "layers.1.engram.k_weight", "layers.1.engram.embed.weight"):
        assert not _matches(v41_plan, fqn)

    # DeepSeek-V4 has no Engram and keeps its own suffix set.
    v4_plan = export.build_recipe_plan("mix", model_name="deepseek_v4", enable_mxfp4_qat=False)
    assert not _matches(v4_plan, "layers.1.engram.wkv.weight")


def test_export_replays_dcp_configs_end_to_end(export, monkeypatch, tmp_path):
    """Default replay path through the entrypoint: extraction, mapping, quantization, and save."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(json.dumps(_V41_CONFIG))
    state = {
        "layers.0.attention.wq_a.weight": torch.zeros(4, 64, dtype=torch.bfloat16),
        "layers.0.moe.routed_experts.inner_experts.w1_EFD": torch.zeros(4, 64, dtype=torch.bfloat16),
        "layers.0.moe.routed_experts.inner_experts.w3_EFD": torch.zeros(4, 64, dtype=torch.bfloat16),
        "layers.0.engram.gate.wkv": torch.zeros(4, 64, dtype=torch.bfloat16),
        "layers.0.engram.gate.q_weight": torch.zeros(2, 2, dtype=torch.bfloat16),
    }

    class _Adapter:
        from_hf_map = {
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.wq_a.weight",
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "layers.{}.engram.wkv.weight": "layers.{}.engram.gate.wkv",
            "layers.{}.engram.q_weight": "layers.{}.engram.gate.q_weight",
        }

        def to_hf(self, loaded):
            to_hf_map = {value: key for key, value in self.from_hf_map.items()}
            converted = {}
            for fqn, value in loaded.items():
                if fqn.endswith(("w1_EFD", "w3_EFD")):
                    projection = fqn.split(".w")[-1][0]
                    for expert in (0, 1):
                        converted[f"layers.0.ffn.experts.{expert}.w{projection}.weight"] = value
                    continue
                converted[to_hf_map[re.sub(r"(\d+)", "{}", fqn, count=1)].format(0)] = value
            return converted

    dense_config = _StubMXQuantizeConfig()
    expert_config = _StubBlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=_StubMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, dst_type_max=6.0)
    )
    extracted = {
        "layers.0.attention.wq_a.weight": (dense_config, None),
        "layers.0.moe.routed_experts.inner_experts.w1_EFD": (expert_config, None),
        "layers.0.moe.routed_experts.inner_experts.w3_EFD": (expert_config, None),
    }
    spec = SimpleNamespace(
        name="deepseek_v4_1",
        model=SimpleNamespace(build=lambda: torch.nn.Linear(64, 4, bias=False)),
        state_dict_adapter=lambda *args: _Adapter(),
    )
    checkpoint = types.ModuleType("torchtitan.components.checkpoint")
    checkpoint.ModelWrapper = lambda built: built
    monkeypatch.setitem(sys.modules, "torchtitan.components.checkpoint", checkpoint)
    monkeypatch.setattr(export, "_get_model_spec", lambda *args: spec)
    monkeypatch.setattr(export, "_load_dcp_state_dict", lambda *args: state)
    _stub_dcp_config_imports(monkeypatch)
    monkeypatch.setattr(export._dcp_quant_configs, "extract_wrapper_configs", lambda *args: extracted)

    def fake_quantize(weight, swap, device, **kwargs):
        is_fp4 = getattr(swap.weight_config, "mxfp4_fake_quantize_config", None) is not None
        return (
            torch.zeros(
                (weight.shape[0], weight.shape[1] // 2 if is_fp4 else weight.shape[1]),
                dtype=torch.uint8 if is_fp4 else torch.float8_e4m3fn,
            ),
            torch.zeros((weight.shape[0], weight.shape[1] // 32), dtype=torch.uint8),
        )

    monkeypatch.setattr(export, "_quantize_to_plain", fake_quantize)

    def save_shards(tensors, output_dir, **kwargs):
        output_dir.mkdir()
        save_file(tensors, str(output_dir / "model.safetensors"))
        return ["model.safetensors"]

    monkeypatch.setattr(export, "_save_export_shards", save_shards)
    args = export._build_arg_parser().parse_args(
        [
            str(tmp_path / "dcp"),
            str(tmp_path / "output"),
            "--hf_assets_path",
            str(assets),
            "--model_name",
            "deepseek_v4_1",
            "--model_flavor",
            "flash",
        ]
    )

    export.export_quantized_vllm(args)

    description = json.loads((args.output_dir / "quant_model_description.json").read_text())
    assert description["layers.0.attn.wq_a.weight"] == "W8A8_MXFP8"
    assert description["layers.0.engram.wkv.weight"] == "W8A8_MXFP8"  # serving-only, reuses the dense config
    assert description["layers.0.engram.q_weight"] == "FLOAT"
    for expert in (0, 1):
        for projection in (1, 3):
            assert description[f"layers.0.ffn.experts.{expert}.w{projection}.weight"] == "W4A8_MXFP"
    assert (args.output_dir / "model.safetensors").is_file()


@pytest.mark.parametrize(
    "input_format, message",
    [("dcp", "carries no DCP-embedded quantization configs"), ("hf", "HF input has no embedded quantization configs")],
    ids=["dcp", "hf"],
)
def test_export_replay_requires_embedded_configs(export, monkeypatch, tmp_path, input_format, message):
    _stub_recipe_imports(monkeypatch)
    with pytest.raises(ValueError, match=f"{message}.*'mix'"):
        export._resolve_quantization_plan(
            recipe=None,
            input_format=input_format,
            model_name="deepseek_v4_1",
            hf_state_dict={},
            extracted={},
            sd_adapter=None,
            enable_mxfp4_qat=False,
            dst_type_max=0.0,
            input_dir=tmp_path,
        )


@pytest.mark.parametrize("qat, dst_type_max", [(True, 0.0), (False, 1.0)], ids=["qat", "dst-type-max"])
def test_resolve_quantization_plan_rejects_recipe_flags_when_replaying(export, tmp_path, qat, dst_type_max):
    with pytest.raises(ValueError, match="recipe flags were also provided"):
        export._resolve_quantization_plan(
            recipe=None,
            input_format="dcp",
            model_name="deepseek_v4_1",
            hf_state_dict={},
            extracted={"layers.0.weight": (object(), object())},
            sd_adapter=object(),
            enable_mxfp4_qat=qat,
            dst_type_max=dst_type_max,
            input_dir=tmp_path,
        )


def _stub_dcp_config_imports(monkeypatch):
    """Replace the ``torchao_npu`` config surface the replay planner imports at call time."""

    class _ParamSwapConfig:
        def __init__(self, *, weight_config, activation_config):
            self.weight_config = weight_config
            self.activation_config = activation_config

    quant_configs = types.ModuleType("torchao_npu.quantization.quant_configs")
    quant_configs.MXQuantizeConfig = _StubMXQuantizeConfig
    quant_configs.BlockMXQuantizeConfig = _StubBlockMXQuantizeConfig
    configs = types.ModuleType("torchao_npu.configs")
    configs.ParamSwapConfig = _ParamSwapConfig
    monkeypatch.setitem(sys.modules, "torchao_npu", types.ModuleType("torchao_npu"))
    monkeypatch.setitem(sys.modules, "torchao_npu.quantization", types.ModuleType("torchao_npu.quantization"))
    monkeypatch.setitem(sys.modules, "torchao_npu.quantization.quant_configs", quant_configs)
    monkeypatch.setitem(sys.modules, "torchao_npu.configs", configs)


def test_dcp_plan_rejects_uncovered_quantizable_weights(export, monkeypatch):
    _stub_dcp_config_imports(monkeypatch)

    class _Adapter:
        from_hf_map = {"layers.{}.attn.wq_a.weight": "layers.{}.attention.wq_a.weight"}

    state = {
        "layers.0.attn.wq_a.weight": torch.zeros(4, 64, dtype=torch.bfloat16),
        "layers.1.attn.wq_a.weight": torch.zeros(4, 64, dtype=torch.bfloat16),
    }
    extracted = {"layers.0.attention.wq_a.weight": (_StubMXQuantizeConfig(), None)}

    with pytest.raises(ValueError, match="no quantization config for 1 quantizable weight"):
        export._build_dcp_plan(state, extracted, _Adapter(), "deepseek_v4_1")


def test_dcp_plan_fuses_split_qwen35_projections(export, monkeypatch):
    """Training wraps in_proj_q/k/v separately; the export quantizes the fused in_proj_qkv."""
    _stub_dcp_config_imports(monkeypatch)

    class _Adapter:
        from_hf_map = {"model.language_model.layers.{}.linear_attn.in_proj_qkv.weight": None}

    config = _StubMXQuantizeConfig()
    extracted = {f"layers.0.attn.in_proj_{projection}.weight": (config, None) for projection in ("q", "k", "v")}
    fused_fqn = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    state = {fused_fqn: torch.zeros(12, 64, dtype=torch.bfloat16)}

    plan = export._build_dcp_plan(state, extracted, _Adapter(), "qwen3_5")

    assert len(plan) == 1
    _swap, filter_fn = plan[0]
    assert filter_fn(None, fused_fqn)
    assert not filter_fn(None, "model.language_model.layers.0.linear_attn.in_proj_z.weight")


def test_dcp_plan_rejects_mismatched_fused_qwen35_configs(export, monkeypatch):
    _stub_dcp_config_imports(monkeypatch)

    class _Adapter:
        from_hf_map = {"model.language_model.layers.{}.linear_attn.in_proj_qkv.weight": None}

    q_config = _StubMXQuantizeConfig()
    k_config = replace(q_config, dst_type_max=1.0)
    extracted = {
        "layers.0.attn.in_proj_q.weight": (q_config, None),
        "layers.0.attn.in_proj_k.weight": (k_config, None),
        "layers.0.attn.in_proj_v.weight": (q_config, None),
    }
    fused_fqn = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    state = {fused_fqn: torch.zeros(12, 64, dtype=torch.bfloat16)}

    with pytest.raises(ValueError, match="fuse into one HF weight"):
        export._build_dcp_plan(state, extracted, _Adapter(), "qwen3_5")


def test_dcp_plan_rejects_incomplete_fused_qwen35_configs(export, monkeypatch):
    """Missing split configs would quantize that slice with its sibling's config."""
    _stub_dcp_config_imports(monkeypatch)

    class _Adapter:
        from_hf_map = {"model.language_model.layers.{}.linear_attn.in_proj_qkv.weight": None}

    config = _StubMXQuantizeConfig()
    extracted = {f"layers.0.attn.in_proj_{projection}.weight": (config, None) for projection in ("q", "k")}
    fused_fqn = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    state = {fused_fqn: torch.zeros(12, 64, dtype=torch.bfloat16)}

    with pytest.raises(ValueError, match=r"embeds no quantization config for layers\.0\.attn\.in_proj_v\.weight"):
        export._build_dcp_plan(state, extracted, _Adapter(), "qwen3_5")


@dataclass(frozen=True)
class _StorageIndex:
    """Hashable stand-in for the ``.metadata`` storage key (pickled, so it must live at module level)."""

    fqn: str


@pytest.mark.parametrize("field_name, value", [("round_mode", "floor"), ("scale_alg", 0)])
def test_dcp_config_comparison_keeps_kernel_options(export, field_name, value):
    original = _StubMXQuantizeConfig()
    changed = replace(original, **{field_name: value})

    assert original == changed  # torchao-style dataclass equality ignores these fields.
    assert not export._same_quantization_config(original, changed)
    assert not export._same_quantization_config(
        _StubMXQuantizeConfig(mxfp4_fake_quantize_config=original),
        _StubMXQuantizeConfig(mxfp4_fake_quantize_config=changed),
    )


def test_quantize_state_dict_quantizes_engram_gate(export, monkeypatch):
    def dense_filter(_param, fqn):
        return fqn.endswith(export.get_quantization_export_rule("deepseek_v4_1").dense_weight_suffixes)

    plan = [(_FakeSwap(torch.float8_e4m3fn), dense_filter)]
    state = {
        "layers.1.engram.wkv.weight": torch.zeros(64, 128, dtype=torch.bfloat16),
        "layers.1.engram.q_weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }

    def _fake_quantize_to_plain(weight, swap_config, device, *, keep_scale_tail_dim=False):
        assert weight.dtype is torch.bfloat16
        return (
            torch.zeros(weight.shape, dtype=torch.float8_e4m3fn),
            torch.zeros((weight.shape[0], weight.shape[1] // 32), dtype=torch.uint8),
        )

    monkeypatch.setattr(export, "_quantize_to_plain", _fake_quantize_to_plain)
    out, quantized = export.quantize_state_dict(state, plan, device="cpu")

    assert quantized == {"layers.1.engram.wkv.weight": 0}
    assert out["layers.1.engram.wkv.weight"].dtype is torch.float8_e4m3fn
    assert out["layers.1.engram.wkv.weight_scale"].dtype is torch.uint8
    assert out["layers.1.engram.wkv.weight_scale"].shape == (64, 4)
    assert out["layers.1.engram.q_weight"].dtype is torch.bfloat16


def _write_engram_index(output_dir, weight_map):
    (output_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def _engram_state(*, with_scale=True):
    state = {
        _ENGRAM_FQN: torch.zeros(4, 64, dtype=torch.float8_e4m3fn),
        "layers.1.attn_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
    }
    if with_scale:
        state[_ENGRAM_SCALE] = torch.zeros(4, 2, dtype=torch.float8_e8m0fnu)
    return state


def _validate_engram_export(contract, tmp_path, state):
    output_dir, shard_names, description = _write_export(
        tmp_path, contract, state=state, quantized={}, plan=[], config=_V41_CONFIG, engram_fqns={_ENGRAM_FQN}
    )
    _write_engram_index(output_dir, dict.fromkeys(state, shard_names[0]))
    return output_dir, shard_names, description


def test_validate_export_rejects_engram_without_scale(contract, tmp_path):
    state = _engram_state(with_scale=False)
    output_dir, shard_names, description = _validate_engram_export(contract, tmp_path, state)

    with pytest.raises(ValueError, match="missing Engram MXFP8 scale"):
        _validate(
            contract, output_dir, shard_names, state, description, quantized={}, plan=[], model_name="deepseek_v4_1"
        )


def test_validate_export_rejects_missing_vision_tower(contract, tmp_path):
    state = _mxfp8_state()
    # Markers alone do not make a tower: they must not satisfy the completeness check.
    state.update({name: torch.zeros(2) for name in ("image_start", "image_end", "image_newline")})

    _expect_rejected(
        contract, tmp_path, state, "vision tower", config=_V41_CONFIG, model_name="deepseek_v4_1", needs_vision=True
    )


def test_ensure_engram_index_writes_single_shard_index(export, tmp_path):
    state = _engram_state()
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    save_file(state, str(output_dir / "model.safetensors"))

    export._ensure_engram_index(output_dir, ["model.safetensors"], {_ENGRAM_FQN})

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == dict.fromkeys(state, "model.safetensors")
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in state.values())


@pytest.mark.parametrize(
    "name",
    [
        "model.safetensors",
        "quant_model_weights.safetensors.index.json",
    ],
)
def test_reject_stale_weight_artifacts_rejects_previous_export(export, tmp_path, name):
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / name).write_bytes(b"")

    with pytest.raises(ValueError, match="export into an empty directory") as excinfo:
        export._reject_stale_weight_artifacts(output_dir)

    assert name in str(excinfo.value)


def _write_config(tmp_path, config):
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps(config))
    return output_dir


def _v41_vision_state(*, moe_layers, bias_layers):
    state = _mxfp8_state()
    state.update(
        {
            "vision.patch_embed.proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
            "image_start": torch.zeros(2),
            "image_end": torch.zeros(2),
            "image_newline": torch.zeros(2),
        }
    )
    for layer in moe_layers:
        state[f"layers.{layer}.ffn.experts.0.w1.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    for layer in bias_layers:
        state[f"layers.{layer}.ffn.gate.bias_vl"] = torch.zeros(2)
    return state


def test_validate_export_rejects_missing_vision_bias_per_moe_layer(contract, tmp_path):
    state = _v41_vision_state(moe_layers=(0, 1), bias_layers=(0,))
    output_dir, shard_names, description = _write_export(tmp_path, contract, state=state, config=_V41_CONFIG)

    with pytest.raises(ValueError, match=r"layers\.1\.ffn\.gate\.bias_vl"):
        _validate(contract, output_dir, shard_names, state, description, model_name="deepseek_v4_1", needs_vision=True)


def test_validate_export_rejects_vision_config_without_weights(contract, tmp_path):
    config = {**_V41_CONFIG, "vision_n_layers": 2}

    _expect_rejected(contract, tmp_path, _mxfp8_state(), "vision_n_layers", config=config, model_name="deepseek_v4_1")


def test_save_failure_removes_partial_shards(export, tmp_path, monkeypatch):
    """A mid-write I/O error must not leave shards a rerun then rejects."""
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}")

    def failing_save(_state_dict, out, max_shard_size):
        # The real writer saves shards one by one; the first ones are already on
        # disk when a full disk aborts the next, and the caller never learns
        # their names.
        (Path(out) / "model-00001-of-00002.safetensors").write_bytes(b"partial")
        (Path(out) / "model-00002-of-00002.safetensors").write_bytes(b"partial")
        (Path(out) / "model.safetensors.index.json").write_text("")
        raise OSError(28, "No space left on device")

    fake_export = types.ModuleType("torchao_npu.serialization.export")
    fake_export.save_hf_safetensors = failing_save
    monkeypatch.setitem(sys.modules, "torchao_npu", types.ModuleType("torchao_npu"))
    monkeypatch.setitem(sys.modules, "torchao_npu.serialization", types.ModuleType("torchao_npu.serialization"))
    monkeypatch.setitem(sys.modules, "torchao_npu.serialization.export", fake_export)

    with pytest.raises(OSError, match="No space left"):
        export._save_export_shards({}, output_dir, max_shard_size="5GB")

    assert sorted(path.name for path in output_dir.iterdir()) == ["config.json"]


def test_copy_hf_assets_skips_weights_and_indexes(export, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    ):
        (assets / name).write_text("{}")
    (assets / "model.safetensors").write_bytes(b"")
    output_dir = tmp_path / "export"
    output_dir.mkdir()

    export._copy_hf_assets(assets, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == ["config.json", "tokenizer.json"]
