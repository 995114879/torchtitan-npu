# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 SFT integration case with a generated local HF checkpoint."""

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

from tests.integration_tests import OverrideDefinitions

# Keep model/runtime imports local so suite collection does not activate the plugin.
GEMMA3_C4_STEPS = 30
_SHARED_TOKENIZER_DIR = Path(__file__).parents[1] / "assets" / "deepseek_v3"
_FIXTURE_VOCAB_SIZE = 2008
_FIXTURE_INITIALIZER_STD = 0.02


def build_gemma3_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            test_name="gemma3_c4_fsdp2",
            test_descr="Gemma3 C4 text SFT, FullAC, FSDP2, HF load and DCP save",
            ngpu=2,
            use_golden=False,
            check_loss=False,
            timeout=600,
            expected_steps=[tuple(range(1, GEMMA3_C4_STEPS + 1))],
            env_vars={
                "MODULE": "tests.integration_tests.gemma3",
                "CONFIG": "gemma3_c4_fsdp2",
                "TRAIN_FILE": "tests.integration_tests.gemma3",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            train_args=(
                "--metrics.enable_tensorboard",
                "--metrics.log_freq=1",
                "--dataloader.dataset=c4_test",
                "--dataloader.dataset-path=tests/assets/c4_test",
                "--training.disable-cuda-graphs",
            ),
            override_args=[()],
        ),
    ]


def _text_config():
    from torchtitan_npu.models.gemma3 import debug_gemma3_text_config

    config = debug_gemma3_text_config()
    return replace(
        config,
        vocab_size=_FIXTURE_VOCAB_SIZE,
        bos_token_id=2002,
        eos_token_id=2003,
        pad_token_id=2001,
        num_hidden_layers=2,
        layer_types=config.layer_types[:2],
    )


def _model_config():
    import torch

    from torchtitan_npu.models.gemma3 import _make_gemma3_config

    config = _make_gemma3_config(_text_config())
    # The fixture is initialized before being written as a synthetic HF
    # snapshot. Match the HF Gemma initializer instead of nn.Embedding's
    # default unit-variance reset, which makes a tied random lm_head produce
    # logits around sqrt(hidden_size) times too large for a useful smoke.
    return replace(
        config,
        tok_embeddings=replace(
            config.tok_embeddings,
            param_init={"weight": partial(torch.nn.init.normal_, mean=0.0, std=_FIXTURE_INITIALIZER_STD)},
        ),
    )


def gemma3_c4_fsdp2():
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.distributed.activation_checkpoint import FullAC

    from torchtitan_npu.models.gemma3.config_registry import debug_gemma3_fsdp2

    base = debug_gemma3_fsdp2()
    assert base.model_spec is not None
    model = _model_config()
    # Keep the production SFT recipe at its 1e-5 learning rate. The reduced
    # randomly initialized smoke model uses the historical debug rate so a
    # 30-step integration run exercises visible parameter updates as well as
    # the load/FSDP/FullAC/DCP path.
    debug_optimizer = replace(
        base.optimizer,
        param_groups=[
            replace(group, optimizer_kwargs={**group.optimizer_kwargs, "lr": 1e-3})
            for group in base.optimizer.param_groups
        ],
    )
    return replace(
        base,
        model_spec=replace(base.model_spec, model=model),
        loss=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size),
        optimizer=debug_optimizer,
        # Preserve the integration smoke's two warmup steps through the
        # upstream scheduler config.
        lr_scheduler=replace(base.lr_scheduler, warmup_steps=2),
        training=replace(base.training, seq_len=64, steps=GEMMA3_C4_STEPS, global_batch_size=2),
        activation_checkpoint=FullAC.Config(),
        checkpoint=replace(
            base.checkpoint,
            enable=True,
            interval=GEMMA3_C4_STEPS,
            last_save_model_only=False,
            initial_load_in_hf=True,
            initial_load_path=None,
        ),
    )


def _write_fixture(root: Path):
    # Each rank owns its fixture directory before the process group exists.
    import torch
    from safetensors.torch import save_file

    from torchtitan_npu.models.gemma3.state_dict_adapter import Gemma3StateDictAdapter

    root.mkdir(parents=True, exist_ok=False)
    model_config = _model_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = model_config.build()
        model.init_states()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    text_state = Gemma3StateDictAdapter(model_config).to_hf(expected)
    state = {f"language_model.{key}": value for key, value in text_state.items()}
    state["vision_tower.unused.weight"] = torch.ones(2, 2)
    state["multi_modal_projector.unused.weight"] = torch.ones(2, 2)
    save_file(state, root / "model.safetensors")
    config_payload = _text_config().to_dict()
    (root / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    _write_tokenizer(root)
    return expected


def _write_tokenizer(root: Path):
    """Reuse the repository's byte-level tokenizer for all Gemma3 fixtures."""
    if not _SHARED_TOKENIZER_DIR.is_dir():
        raise FileNotFoundError("Gemma3 fixture tokenizer assets are missing")
    shutil.copy2(_SHARED_TOKENIZER_DIR / "tokenizer.json", root / "tokenizer.json")
    tokenizer_config = json.loads((_SHARED_TOKENIZER_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    added_tokens = tokenizer_config.setdefault("added_tokens_decoder", {})
    for token_id, content in {
        2004: "<start_of_turn>",
        2005: "<end_of_turn>",
        2006: "user",
        2007: "model",
    }.items():
        added_tokens[str(token_id)] = {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")


def _assert_loaded_model(trainer, expected):
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    model = trainer.model_parts[0]
    assert torch.distributed.get_world_size() == trainer.parallel_dims.dp_shard == 2
    assert isinstance(model, FSDPModule)
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers.values())
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert set(model.state_dict()) == set(expected)
    for name, parameter in model.state_dict().items():
        assert isinstance(parameter, DTensor), name
        assert parameter.dtype == torch.float32, name
        torch.testing.assert_close(parameter.full_tensor().cpu(), expected[name], rtol=0, atol=0)


def _assert_updated_model(trainer, expected, steps: int, label: str):
    import torch

    model = trainer.model_parts[0]
    assert trainer.step == steps
    assert model.tok_embeddings.weight is model.lm_head.weight
    parameters = [
        p
        for optimizer in trainer.optimizers
        for group in optimizer.param_groups
        for p in group["params"]
    ]
    assert len(parameters) == len({id(p) for p in parameters}) == len(list(model.parameters()))
    for optimizer in trainer.optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                assert state["step"].item() == steps
                for moment in ("exp_avg", "exp_avg_sq"):
                    assert state[moment].dtype == torch.float32, moment
                    assert torch.isfinite(state[moment].to_local()).all(), moment
    embedding = model.tok_embeddings.weight.full_tensor().detach().cpu()
    assert not torch.equal(embedding, expected["tok_embeddings.weight"])
    print(
        f"{label} rank={torch.distributed.get_rank()} HF_LOAD FSDP2 FULL_AC UPDATE_{steps} PASS",
        flush=True,
    )


def _assert_checkpoint(checkpoint: Path, expected, steps: int, label: str):
    import torch
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata
    assert set(expected) <= metadata.keys()
    step_keys = [key for key in metadata if key.startswith("optimizer.") and key.endswith(".step")]
    assert len(step_keys) == len(expected) - 1  # embedding and lm_head share one optimizer state
    assert any(key.startswith("dataloader.") for key in metadata)
    assert any(key.startswith("lr_scheduler.") for key in metadata)
    assert "train_state.step" in metadata
    step_states = {key: torch.empty(metadata[key].size, dtype=metadata[key].properties.dtype) for key in step_keys}
    dcp.load(step_states, checkpoint_id=checkpoint, no_dist=True)
    assert all(value.item() == steps for value in step_states.values())
    assert list(checkpoint.glob("*.distcp"))
    print(f"{label.replace('SFT', 'DCP')} optimizer_steps={len(step_states)} step={steps} PASS", flush=True)


def main():
    import torch.distributed as dist
    from torchtitan.config import ConfigManager
    from torchtitan.tools.logging import init_logger

    import torchtitan_npu  # noqa: F401

    init_logger()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dump_folder", required=True)
    args, _ = parser.parse_known_args()
    root = Path(args.dump_folder).resolve() / "fixture" / f"rank-{os.environ['RANK']}"
    expected = _write_fixture(root)
    config = ConfigManager().parse_args([*sys.argv[1:], "--hf-assets-path", str(root)])
    steps = config.training.steps
    label = "GEMMA3_C4_SFT"
    trainer = config.build()
    load = trainer.checkpointer.load

    def load_and_check(*args, **kwargs):
        result = load(*args, **kwargs)
        assert result, "The initial HF checkpoint was not loaded"
        _assert_loaded_model(trainer, expected)
        return result

    try:
        with patch.object(trainer.checkpointer, "load", side_effect=load_and_check):
            trainer.train()
        _assert_updated_model(trainer, expected, steps, label)
        if dist.get_rank() == 0:
            _assert_checkpoint(
                Path(config.dump_folder) / config.checkpoint.folder / f"step-{steps}", expected, steps, label
            )
        dist.barrier()
    finally:
        trainer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
