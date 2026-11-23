# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from dataclasses import replace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files

from torchtitan_npu.models.gemma3.state_dict_adapter import Gemma3StateDictAdapter

from .fixtures import hf_parameter_names, recipe_model_config


def test_native_export_maps_each_parameter_family_and_layer():
    config = recipe_model_config()
    # Checkpoint shapes must follow each child config, including a wider second MLP.
    config.layers[1].feed_forward.w1.out_features = 48
    config.layers[1].feed_forward.w2.in_features = 48
    config.layers[1].feed_forward.w3.out_features = 48
    model = config.build()
    adapter = Gemma3StateDictAdapter(model.config)
    native = model.state_dict()
    for index, (name, tensor) in enumerate(native.items()):
        if name != "lm_head.weight":
            tensor.fill_(index + 0.25)
    native["lm_head.weight"] = native["tok_embeddings.weight"]
    names = hf_parameter_names(len(model.config.layers))

    exported = adapter.to_hf(native)

    assert set(exported) == {hf for name, hf in names.items() if name != "lm_head.weight"}
    for name, hf_name in names.items():
        if name != "lm_head.weight":
            torch.testing.assert_close(exported[hf_name], native[name], rtol=0, atol=0)


def test_hf_import_rejects_duplicate_standalone_and_composite_text_keys(model, hf_reference):
    source = {name: tensor.detach().clone() for name, tensor in hf_reference.state_dict().items()}
    source["language_model.model.embed_tokens.weight"] = source["model.embed_tokens.weight"].clone()

    with pytest.raises(ValueError, match=re.escape("duplicate HF mapping for model.embed_tokens.weight")):
        Gemma3StateDictAdapter(model.config).from_hf(source)


@pytest.mark.parametrize(
    ("composite", "with_index", "filename"),
    [
        (True, True, "model-00001-of-00001.safetensors"),
        (True, False, "model-00001-of-00001.safetensors"),
        (False, False, "model.safetensors"),
    ],
    ids=["composite-indexed", "composite-unindexed-shard", "standalone-single-file"],
)
def test_safetensors_reader_loads_requested_text_weights(
    tmp_path, model, hf_reference, composite, with_index, filename
):
    prefix = "language_model.model" if composite else "model"
    names = hf_parameter_names(len(model.config.layers), prefix=prefix)
    # The source comes from a pinned HF implementation, not adapter.to_hf().
    source = {}
    for name, value in hf_reference.state_dict().items():
        key = f"language_model.{name}" if composite else name
        source[key] = value.detach().clone()
    if composite:
        source["vision_tower.encoder.weight"] = torch.tensor([[101.0, 102.0]])
        source["multi_modal_projector.weight"] = torch.tensor([[201.0]])
    save_file(source, tmp_path / filename)
    if with_index:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(source, filename)}), encoding="utf-8"
        )
    adapter = Gemma3StateDictAdapter(model.config, hf_assets_path=str(tmp_path))

    if with_index:
        assert adapter.fqn_to_index_mapping is not None
        assert set(adapter.fqn_to_index_mapping) == set(names.values())
    else:
        assert adapter.fqn_to_index_mapping is None

    loaded = adapter.to_hf(model.state_dict())
    dcp.load(loaded, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)), no_dist=True)
    native = adapter.from_hf(loaded)
    model.load_state_dict(native, strict=True)

    assert set(loaded) == {hf for name, hf in names.items() if name != "lm_head.weight"}
    assert set(model.state_dict()) == set(names)
    assert native["lm_head.weight"] is native["tok_embeddings.weight"]
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, source[names[name]], rtol=0, atol=0)


def test_adapter_selects_layout_from_actual_initial_path(tmp_path, model, hf_reference):
    assets_path = tmp_path / "composite-assets"
    initial_path = tmp_path / "standalone-weights"
    assets_path.mkdir()
    initial_path.mkdir()

    composite_source = {
        f"language_model.{name}": tensor.detach().clone() + 1
        for name, tensor in hf_reference.state_dict().items()
    }
    standalone_source = {name: tensor.detach().clone() for name, tensor in hf_reference.state_dict().items()}
    save_file(composite_source, assets_path / "model.safetensors")
    save_file(standalone_source, initial_path / "model.safetensors")

    adapter = Gemma3StateDictAdapter(model.config, str(assets_path))
    converted = adapter.to_hf(model.state_dict())
    dcp.load(converted, storage_reader=adapter.get_hf_storage_reader(str(initial_path)), no_dist=True)
    native = adapter.from_hf(converted)

    assert "model.embed_tokens.weight" in converted
    assert "language_model.model.embed_tokens.weight" not in converted
    model.load_state_dict(native, strict=True)
    names = hf_parameter_names(len(model.config.layers))
    for name, hf_name in names.items():
        if hf_name == "lm_head.weight":
            continue
        torch.testing.assert_close(model.state_dict()[name], standalone_source[hf_name], rtol=0, atol=0)


@pytest.mark.parametrize("composite", [False, True], ids=["standalone", "composite"])
def test_tied_indexed_export_removes_omitted_lm_head_from_writer_mapping(tmp_path, model, composite):
    prefix = "language_model.model" if composite else "model"
    names = hf_parameter_names(len(model.config.layers), prefix=prefix)
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names.values(), shard)}), encoding="utf-8"
    )
    adapter = Gemma3StateDictAdapter(model.config, hf_assets_path=str(tmp_path))

    exported = adapter.to_hf(model.state_dict())
    head_key = "language_model.lm_head.weight" if composite else "lm_head.weight"
    assert head_key not in exported
    assert adapter.fqn_to_index_mapping is not None
    assert head_key not in adapter.fqn_to_index_mapping

    sharded = tmp_path / "sharded"
    output = tmp_path / "output"
    output.mkdir()
    dcp.save(
        exported,
        storage_writer=HuggingFaceStorageWriter(
            str(sharded),
            save_distributed=True,
            fqn_to_index_mapping=adapter.fqn_to_index_mapping,
            enable_consolidation=False,
        ),
    )
    consolidate_safetensors_files(
        input_dir=str(sharded),
        output_dir=str(output),
        fqn_to_index_mapping=adapter.fqn_to_index_mapping,
    )
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert head_key not in weight_map


@pytest.mark.parametrize("composite", [False, True], ids=["standalone", "composite"])
def test_single_file_loads_untied_lm_head(tmp_path, recipe, hf_reference, composite):
    config = replace(recipe.model_spec.model, enable_weight_tying=False)
    prefix = "language_model." if composite else ""
    source = {prefix + name: tensor.detach().clone() for name, tensor in hf_reference.state_dict().items()}
    source[prefix + "lm_head.weight"].add_(1.0)
    save_file(source, tmp_path / "model.safetensors")
    adapter = Gemma3StateDictAdapter(config, hf_assets_path=str(tmp_path))
    model = config.build()

    loaded = adapter.to_hf(model.state_dict())
    dcp.load(loaded, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)), no_dist=True)
    native = adapter.from_hf(loaded)
    model.load_state_dict(native, strict=True)
    exported = adapter.to_hf(model.state_dict())

    assert set(exported) == set(source)
    torch.testing.assert_close(model.lm_head.weight, source[prefix + "lm_head.weight"], rtol=0, atol=0)
    assert model.lm_head.weight is not model.tok_embeddings.weight
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, source[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("composite", "hf_key", "native_key"),
    [
        (False, "lm_head.weight", "lm_head.weight"),
        (True, "model.layers.0.self_attn.q_proj.weight", "layers.0.attention.qkv_linear.wq.weight"),
    ],
    ids=["standalone-head", "composite-attention-weight"],
)
def test_hf_import_reports_missing_source_and_native_names(recipe, hf_reference, composite, hf_key, native_key):
    config = replace(recipe.model_spec.model, enable_weight_tying=False)
    prefix = "language_model." if composite else ""
    source = {prefix + name: tensor.detach().clone() for name, tensor in hf_reference.state_dict().items()}
    source.pop(prefix + hf_key)
    adapter = Gemma3StateDictAdapter(config)

    with pytest.raises(ValueError) as error:
        adapter.from_hf(source)

    message = str(error.value)
    assert prefix + hf_key in message
    assert native_key in message


@pytest.mark.parametrize("failure", ["unknown", "shape", "tied"], ids=["unknown", "shape", "tied"])
def test_hf_import_rejects_incompatible_weights(recipe, hf_reference, failure):
    source = {name: tensor.detach().clone() for name, tensor in hf_reference.state_dict().items()}
    if failure == "unknown":
        source["model.layers.0.unknown.weight"] = torch.ones(1)
    elif failure == "shape":
        source["model.layers.0.self_attn.q_proj.weight"] = torch.ones(1)
    else:
        source["lm_head.weight"].add_(1.0)

    with pytest.raises(ValueError, match=failure):
        Gemma3StateDictAdapter(recipe.model_spec.model).from_hf(source)
