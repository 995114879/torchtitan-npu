# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE, AsyncMode
from torchtitan.components.checkpointer.base import ModelWrapper

from tests.unit_tests.extensions.torchtitan.experiments.torchft.service_helpers import service_session
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.checkpoint import LastWordsCheckpointManager
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.rng import RNG_STATE, RNGCheckpointState
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.worker import (
    LastWordsConfig,
    LastWordsWorkerController,
    atomic_json,
    read_json,
)
from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import FaultTolerantTrainerEx


def test_normal_training_completes_after_checkpoint_drain(monkeypatch):
    events = []
    monkeypatch.setattr(FaultTolerantTrainerEx.__mro__[1], "train", lambda self: events.append("train"))
    trainer = FaultTolerantTrainerEx.__new__(FaultTolerantTrainerEx)
    trainer.close = lambda: None
    trainer.checkpointer = SimpleNamespace(finish_pending_save=lambda: events.append("drain"))
    trainer.last_words = SimpleNamespace(finish_training=lambda: events.append("finished"))

    trainer.train()

    assert events == ["train", "drain", "finished"]


class DictState:
    def __init__(self, value):
        self.value = dict(value)

    def state_dict(self):
        return self.value

    def load_state_dict(self, value):
        self.value = dict(value)


class CheckpointAdamW(torch.optim.AdamW):
    def _refresh_cached_state_dict(self):
        # Plain AdamW exposes live state; the TorchFT container refreshes a cache here.
        pass


@pytest.fixture
def saved_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0, -4.0])))
    model._register_state_dict_hook(lambda _, state, *_args: {key: value.clone() for key, value in state.items()})
    optimizer = CheckpointAdamW([model.weight], lr=0.02, foreach=False)
    model.weight.grad = torch.tensor([2.0, -3.0])
    optimizer.step()
    state = {"step": 5, "ntokens_seen": 160, "last_words_num_updates": 4}
    cursor = DictState({"position": torch.tensor(20)})

    manager = LastWordsCheckpointManager.__new__(LastWordsCheckpointManager)
    manager.enable = False  # The isolated CPU manager owns no upstream destructor resources.
    manager.folder, manager.pg = str(tmp_path), None
    manager.last_words_signature, manager.last_words_state = "contract", state
    manager._last_words_cursor_anchors = {}
    manager._last_words_upload_error = manager._last_words_pending_cursor = manager._last_words_pending_binding = None
    manager.save_future = manager.staging_future = None
    manager.async_mode = AsyncMode.DISABLED
    manager.last_words_cursor_current = manager.enable_ft_dataloader_checkpoints = False
    manager.ft_manager = manager.ema_optimizer = None
    manager.keep_latest_k = 0
    manager.verify_hash_manifest = False
    manager.exclude_from_loading = []
    manager.states = {
        MODEL: ModelWrapper(model),
        OPTIMIZER: optimizer,
        LR_SCHEDULER: DictState({"last_epoch": 4}),
        TRAIN_STATE: DictState(state),
    }
    rng = RNGCheckpointState(torch.device("cpu"), 0)
    rng.capture()
    manager.rng_state = rng
    manager.ft_states = {DATALOADER: cursor, RNG_STATE: rng}
    with service_session(tmp_path, "launch-1", num_replicas=1, world_size=1) as directory:
        monkeypatch.setenv("LAST_WORDS_SESSION_FILE", str(directory / "replica-0.json"))
        config = LastWordsConfig(enable=True, run_id="launch-1", heartbeat_timeout_seconds=30, timeout_seconds=100)
        with closing(
            LastWordsWorkerController(config, folder=str(tmp_path), replica_id=0, num_replicas=1, rank=0, world_size=1)
        ) as control:
            manager.last_words = control
            path = tmp_path / "ft-replicat-0" / config.run_id / "step-3"
            dcp.save(manager.ft_states, checkpoint_id=path)
            atomic_json(
                path / "last_words_cursor.json", manager._cursor_record(dict(state, step=3, last_words_num_updates=2))
            )
            control.request_terminal_stop("save at the completed boundary")
            target = manager.save_last_words(state)
    assert read_json(target / "service-result.json")["success"] is True
    try:
        yield manager, target, model, optimizer, cursor, state
    finally:
        rng.restore()


@pytest.mark.parametrize("legacy", [False, True], ids=["v2-automatic", "v1-explicit"])
def test_emergency_checkpoint_restores_complete_training_state(saved_checkpoint, legacy):
    manager, target, model, optimizer, cursor, state = saved_checkpoint
    expected_weight = model.weight.detach().clone()
    expected_rng = torch.get_rng_state().clone()
    if legacy:
        atomic_json(target / "manifest.json", dict(read_json(target / "manifest.json"), format_version=1))
        (target / "service-result.json").unlink()
        manager.last_words.config.resume_path = str(target)
    manager.last_words.config.run_id = "restart"
    with torch.no_grad():
        model.weight.fill_(25)
        optimizer.state[model.weight]["exp_avg"].fill_(100)
    manager.states[LR_SCHEDULER].value["last_epoch"] = 0
    manager.states[TRAIN_STATE].value["step"] = 0
    cursor.value["position"].fill_(99)
    torch.rand(1)

    assert manager._load()

    assert torch.equal(model.weight, expected_weight)
    torch.testing.assert_close(optimizer.state[model.weight]["exp_avg"], torch.tensor([0.2, -0.3]))
    assert optimizer.state[model.weight]["step"].item() == 1
    assert manager.states[LR_SCHEDULER].value == {"last_epoch": 4}
    assert manager.states[TRAIN_STATE].value == state
    assert cursor.value["position"].item() == 20
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_external_emergency_resume_can_save_and_reload_periodic_checkpoint(saved_checkpoint, monkeypatch):
    manager, target, model, _, cursor, state = saved_checkpoint
    manager.folder = str(Path(manager.folder) / "next-job")
    manager.last_words.config.run_id = "restart"
    manager.last_words.config.resume_path = str(target)
    manager.last_words_stop_requested = lambda: False
    manager._should_save = lambda *_: True
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    assert manager._load()
    assert manager._save(state["step"])

    binding = read_json(Path(manager.folder) / "step-5/last_words_binding.json")
    assert binding["state"] == state
    assert (Path(manager.folder) / binding["anchors"]["0"] / ".metadata").is_file()
    manager.last_words.config.resume_path = ""
    manager.last_words.config.run_id = "second-restart"
    with torch.no_grad():
        model.weight.fill_(50)
    cursor.value["position"].fill_(1000)

    assert manager._load()
    torch.testing.assert_close(model.weight, torch.tensor([0.9798, -3.9792]))
    assert cursor.value["position"].item() == 20
    assert manager.states[TRAIN_STATE].value == state
