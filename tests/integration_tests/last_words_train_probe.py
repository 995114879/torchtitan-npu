# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Inject software faults in actual TorchFT training and record full state."""

import faulthandler
import hashlib
import importlib.util
import json
import os
import pickle
import runpy
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpoint import LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE
from torchtitan.components.optimizer import OptimizersContainer

# The baseline arm imports production code from an older checkout. Load this
# harness's state implementation explicitly without changing those imports.
_rng = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "torchtitan_npu/extensions/torchtitan/experiments/torchft/last_words/rng.py"
    )
)


def full_states(trainer):
    # Read the live optimizer through the base container, bypassing TorchFT's
    # scalar cache. Both pre-feature and current checkpointers use this layout.
    states = {key: trainer.checkpointer.states[key] for key in (MODEL, LR_SCHEDULER, TRAIN_STATE)}
    states[OPTIMIZER] = OptimizersContainer.state_dict(trainer.optimizers)
    if "ema" in trainer.checkpointer.states:
        states["ema"] = trainer.checkpointer.states["ema"]
    return trainer.checkpointer._flattened_model_states_sd(states)


def load_initial_model(trainer):
    states = full_states(trainer)
    dcp.load(states, checkpoint_id=Path(trainer.checkpointer.folder) / "step-0", process_group=trainer.checkpointer.pg)
    trainer.checkpointer.states[MODEL].load_state_dict(states)
    trainer.optimizers.load_state_dict(states[OPTIMIZER])
    return states


def assert_initial_progress(trainer):
    progress = trainer.state_dict()
    if progress["step"] != 0 or progress.get("last_words_num_updates", 0) != 0:
        raise AssertionError("Numerical initial checkpoint must precede every optimizer update")


def _cursor_states(trainer):
    states = dict(trainer.checkpointer.ft_states)
    key = _rng["RNG_STATE"]
    if key not in states:
        states[key] = _rng["RNGCheckpointState"](trainer.device, dist.get_rank())
        states[key].capture()
    path = Path(trainer.checkpointer.folder) / f"ft-replicat-{trainer.ft_manager.replica_id}" / "step-0"
    return states, states[key], path


def save_initial_cursor(trainer):
    states, rng, path = _cursor_states(trainer)
    rng.capture()
    dcp.save(states, checkpoint_id=path, process_group=trainer.checkpointer.pg)
    return rng


def load_initial_cursor(trainer):
    states, rng, path = _cursor_states(trainer)
    dcp.load(states, checkpoint_id=path, process_group=trainer.checkpointer.pg)
    rng.restore()
    return rng


def digest(value):
    hasher = hashlib.sha256()

    def visit(item):
        if hasattr(item, "state_dict"):
            visit(item.state_dict())
        elif isinstance(item, DTensor):
            visit(item.to_local())
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(str((tensor.shape, tensor.dtype)).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                hasher.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, bytes):
            visit(pickle.loads(item))
        else:
            hasher.update(str(item).encode())

    visit(value)
    return hasher.hexdigest()


def record(trainer, name, value):
    replica = trainer.ft_manager.replica_id
    rank = torch.distributed.get_rank()
    folder = Path(os.environ["LAST_WORDS_TEST_EVIDENCE"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-replica-{replica}-rank-{rank}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def initial_state_digest(trainer, rng):
    states = full_states(trainer)
    progress = dict(states[TRAIN_STATE].state_dict())
    # The enabled trainer adds this counter to an otherwise identical step-zero state.
    progress.pop("last_words_num_updates", None)
    states[TRAIN_STATE] = progress
    states.pop("torchft", None)
    return {"model": digest(states), "cursor": digest(trainer.dataloader.state_dict()), "rng": digest(rng)}


def install_hooks(trainer):
    scenario = os.environ["LAST_WORDS_TEST_SCENARIO"]
    phase = os.environ["LAST_WORDS_TEST_PHASE"]
    record(
        trainer,
        f"source-{phase}",
        {
            "trainer": trainer_module.__file__,
            "checkpoint": sys.modules[type(trainer.checkpointer).__module__].__file__,
        },
    )

    metrics_log = trainer.metrics_processor.log

    def metrics(step, avg_loss, max_loss, grad_norm, **kwargs):
        value = {"loss": float(avg_loss), "grad_norm": float(grad_norm)}
        record(trainer, f"metrics-{phase}-step-{step}", value)
        return metrics_log(step, avg_loss, max_loss, grad_norm, **kwargs)

    trainer.metrics_processor.log = metrics
    if scenario == "numerical":
        return
    if phase not in ("fault", "resume"):
        return

    original_update = trainer._step_optimizer_and_scheduler

    def completed_update():
        result = original_update()
        if trainer._last_words_completed["step"] == trainer.step:
            record(
                trainer,
                f"completed-step-{trainer.step}",
                {
                    "model": digest(trainer.checkpointer._full_states()),
                    "cursor": digest(trainer.dataloader.state_dict()),
                    "rng": digest(trainer._last_words_rng_state),
                    "progress": trainer.state_dict(),
                },
            )
        return result

    trainer._step_optimizer_and_scheduler = completed_update


def injected_step(trainer, iterator):
    phase = os.environ["LAST_WORDS_TEST_PHASE"]
    scenario = os.environ["LAST_WORDS_TEST_SCENARIO"]
    fault_step = int(os.environ.get("LAST_WORDS_TEST_FAULT_STEP", "3"))
    if not hasattr(trainer, "_last_words_test_installed"):
        trainer._last_words_test_installed = True
        # Capture the actual runtime in every real rank, including formal arms.
        import importlib.metadata

        import torchft
        import torchft._torchft as torchft_native

        version = importlib.metadata.version("torchft")
        if version != "0.2.0" or hasattr(torchft_native, "LastWordsClient"):
            raise AssertionError("Service acceptance must use unmodified upstream TorchFT 0.2.0")
        record(
            trainer,
            f"execution-controls-{phase}",
            {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_config": trainer.config.debug.deterministic,
                "seed_config": trainer.config.debug.seed,
                "torch_initial_seed": torch.initial_seed(),
                "compile_enabled": trainer.config.compile.enable,
                "compile_backend": trainer.config.compile.backend,
                "compile_components": trainer.config.compile.components,
                "checkpoint_interval": trainer.config.checkpoint.interval,
                "checkpoint_keep_latest_k": trainer.config.checkpoint.keep_latest_k,
                "checkpoint_last_save_model_only": trainer.config.checkpoint.last_save_model_only,
                "model_profile": os.environ.get("LAST_WORDS_MODEL_PROFILE", "original"),
                "compile_cache_scope": "attempt-rank-v2"
                if os.environ.get("LAST_WORDS_MODEL_PROFILE") == "small-v1"
                else "original",
                "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "ascendc_cache": os.environ.get("TORCHINDUCTOR_NPU_EXT_CACHE_DIR"),
                "model_dimensions": _model_config.describe_model(trainer.config),
                "hccl_connect_timeout": os.environ.get("HCCL_CONNECT_TIMEOUT"),
                "hccl_host_socket_port_range": os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE"),
                "hccl_if_base_port": os.environ.get("HCCL_IF_BASE_PORT"),
            },
        )
        record(
            trainer,
            "torchft-runtime",
            {
                "version": version,
                "package": torchft.__file__,
                "native": torchft_native.__file__,
                "native_sha256": hashlib.sha256(Path(torchft_native.__file__).read_bytes()).hexdigest(),
                "custom_last_words_rpc": False,
            },
        )
        if phase == "resume":
            live_rng = _rng["RNGCheckpointState"](trainer.device, dist.get_rank())
            live_rng.capture()
            record(
                trainer,
                "loaded-resume",
                {
                    "model": digest(trainer.checkpointer._full_states()),
                    "cursor": digest(trainer.dataloader.state_dict()),
                    "rng": digest(trainer._last_words_rng_state),
                    "live_rng": digest(live_rng),
                    "progress": trainer.state_dict(),
                    "next_step": trainer.step,
                },
            )
        if phase == "init":
            trainer.step = 0
            rng = save_initial_cursor(trainer)
            record(trainer, "initial-saved", initial_state_digest(trainer, rng))
            if trainer.ft_manager.replica_id == 0:
                states = full_states(trainer)
                # Export the known zero counter for both test schemas. Disabled
                # trainers omit it, while enabled full-state loading requires it.
                progress = dict(states[TRAIN_STATE].state_dict())
                assert_initial_progress(trainer)
                states[TRAIN_STATE] = dict(progress, last_words_num_updates=0)
                states["torchft"] = dict(step=0, batches_committed=0)
                states["dataloader"] = trainer.dataloader
                dcp.save(
                    states,
                    checkpoint_id=Path(trainer.checkpointer.folder) / "step-0",
                    process_group=trainer.checkpointer.pg,
                )
            raise SystemExit(0)
        if scenario == "numerical":
            loop_step = trainer.step
            load_initial_model(trainer)
            trainer.ft_manager.manager.load_state_dict(dict(step=0, batches_committed=0))
            rng = load_initial_cursor(trainer)
            assert_initial_progress(trainer)
            record(trainer, f"initial-loaded-{phase}", initial_state_digest(trainer, rng))
            record(trainer, f"initial-native-{phase}", trainer.ft_manager.manager.state_dict())
            trainer.step = loop_step
        install_hooks(trainer)

    if phase == "fault" and trainer.step == fault_step:
        target = trainer.ft_manager.replica_id == 1
        rank = torch.distributed.get_rank()
        if scenario == "emergency_save":
            trainer.checkpointer.maybe_wait_for_saving()
            evidence = Path(os.environ["LAST_WORDS_TEST_EVIDENCE"])
            record(trainer, "boundary", {"step": trainer.step - 1})
            if target and rank == 0:
                deadline = time.monotonic() + 120
                while len(list(evidence.glob("boundary-*.json"))) != 4:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Ranks did not reach injection boundary")
                    time.sleep(0.05)
                record(trainer, "injected", {"point": "boundary_process_exit", "step": trainer.step})
                os._exit(77)
            while not trainer.last_words.requested:
                time.sleep(0.05)

    def observed_data(source):
        microbatch = 0
        while True:
            batch = next(source)
            verify_inputs = os.environ.get("LAST_WORDS_VERIFY_INPUTS") == "1"
            value = {"batch": digest(batch)}
            if verify_inputs:
                record(trainer, f"input-step-{trainer.step}-microbatch-{microbatch}", value)
            if phase == "fault":
                record(trainer, f"data-step-{trainer.step}-microbatch-{microbatch}", value)
            microbatch += 1
            yield batch

    result = original(trainer, observed_data(iterator))
    return result


def run_training():
    # Keep imports and monkeypatches out of CPU initial-state helper checks.
    global original, trainer_module, _model_config
    from torchtitan.train import main

    from torchtitan_npu.extensions.torchtitan.experiments.torchft import trainer as trainer_module
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import FaultTolerantTrainerEx

    # Only test helpers come from this harness. Production imports continue to use
    # the checkout selected by PYTHONPATH, including the pre-PR numerical baseline.
    _config_name = "tests.integration_tests.last_words_small_config"
    _config_spec = importlib.util.spec_from_file_location(
        _config_name, Path(__file__).with_name("last_words_small_config.py")
    )
    _model_config = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_name] = _model_config
    _config_spec.loader.exec_module(_model_config)
    original = FaultTolerantTrainerEx.train_step
    FaultTolerantTrainerEx.train_step = injected_step
    diagnostics = os.environ.get("LAST_WORDS_TEST_FAULTHANDLER") == "1"
    if diagnostics:
        faulthandler.dump_traceback_later(90, repeat=True)
    try:
        main()
    finally:
        FaultTolerantTrainerEx.train_step = original
        if diagnostics:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    run_training()
