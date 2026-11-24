# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit NPU TorchFT trainer configuration and normal training lifecycle."""

import contextlib
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.checkpoint.api import CheckpointException
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer

from torchtitan_npu.extensions.torchtitan.experiments.torchft.ft_manager import FTManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.checkpoint import LastWordsCheckpointManager
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.checkpoint_schema import (
    checkpoint_config_signatures,
    tokenizer_asset_digest,
)
from torchtitan_npu.extensions.torchtitan.experiments.torchft.last_words.worker import (
    LastWordsExitError,
    LastWordsStopRequestedError,
    LastWordsWorkerController,
)
from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import (
    ManagerEx,
)
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.step_recovery import (
    StepRecovery,
    reset_failed_iteration,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

from .last_words.rng import RNG_STATE, RNGCheckpointState


class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer):
    config: "FaultTolerantTrainerEx.Config"  # pyrefly: ignore [bad-override]
    optimizers: TorchFTOptimizersContainerEx  # pyrefly: ignore [bad-override]
    checkpointer: LastWordsCheckpointManager  # pyrefly: ignore [bad-override]
    ft_manager: FTManagerEx  # pyrefly: ignore [bad-override]
    last_words: LastWordsWorkerController | None

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):  # pyrefly: ignore [bad-override]
        optimizer: TorchFTOptimizersContainerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=TorchFTOptimizersContainerEx.Config
        )
        fault_tolerance: FTManagerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=FTManagerEx.Config
        )
        checkpoint: LastWordsCheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=lambda: LastWordsCheckpointManager.Config(
                enable=True, enable_ft_dataloader_checkpoints=True
            )
        )

        def _post_init_optimizer(self) -> None:
            # The TorchFT optimizer config has neither materialize() nor NPU Muon settings.
            pass

    def __init__(self, config: Config) -> None:
        self.last_words = None
        self._last_words_rng_state = None
        self._last_words_num_updates = 0
        self._last_words_completed = None
        self._last_words_updating = False
        self._last_words_attempt_active = False
        self._ft_step_active = False
        self._validate_supported_config(config)
        signatures = None
        if config.fault_tolerance.last_words.enable:
            signatures = checkpoint_config_signatures(config)
        super().__init__(config)
        ft_manager = self.ft_manager
        if not isinstance(ft_manager, FTManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires FTManagerEx")
        manager = ft_manager.manager
        if not isinstance(manager, ManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires ManagerEx")
        self.recovery = StepRecovery(
            ft_manager.process_group.recovery,
            manager=manager,
            model_parts=self.model_parts,
            train_state=self,
            lr_schedulers=self.lr_schedulers,
            max_consecutive_recoveries=config.fault_tolerance.max_consecutive_hccl_recoveries,
            reset_auxiliary_state=self._reset_auxiliary_state,
        )
        if signatures is not None:
            if getattr(self, "ema", None) is not None or getattr(getattr(self, "ema_optimizer", None), "enable", False):
                raise ValueError("TorchFT last words does not support EMA weights")
            if not isinstance(self.checkpointer, LastWordsCheckpointManager):
                raise TypeError("Last words requires LastWordsCheckpointManager")
            self.checkpointer.tokenizer_sha256 = tokenizer_asset_digest(config.hf_assets_path)
            self._last_words_rng_state = RNGCheckpointState(self.device, dist.get_rank())
            self._last_words_rng_state.capture()
            self.checkpointer.rng_state = self._last_words_rng_state
            self.checkpointer.ft_states[RNG_STATE] = self._last_words_rng_state
            self.last_words = LastWordsWorkerController(
                config.fault_tolerance.last_words,
                folder=self.checkpointer.folder,
                replica_id=ft_manager.replica_id,
                num_replicas=ft_manager.group_size,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                on_stop=self._request_fixed_last_words_stop,
            )
            self.optimizers.last_words = self.last_words
            self.checkpointer.last_words = self.last_words
            self.checkpointer.last_words_signature = signatures[0]
            self.checkpointer.last_words_resume_signatures = signatures
            # Cursor DCP uploads run in background threads on checkpointer.pg.
            # Control collectives must never share their ordering domain.
            self.checkpointer._last_words_control_pg = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=config.fault_tolerance.last_words.timeout_seconds)
            )
            ft_manager.process_group.recovery.last_words_request = self.last_words.request_terminal_stop

            def stop_requested():
                assert self.last_words is not None
                return self.last_words.requested

            ft_manager.process_group.recovery.last_words_stop_requested = stop_requested
            manager.last_words_stop_requested = stop_requested
            manager.last_words_request = self.last_words.request_terminal_stop

    def _request_fixed_last_words_stop(self):
        control = self.last_words
        if control is None:
            return  # Coordinator construction can observe a peer before assignment.
        completed = self._last_words_completed
        updates = completed.get("last_words_num_updates", completed["step"]) if completed is not None else 0
        if (
            threading.current_thread() is control._thread
            and updates == 0
            and not self._last_words_updating
            and not self.ft_manager.manager.commit_authorized
        ):
            # Initial compilation cannot create a completed cursor. Advertise
            # this CPU fact instead of consuming the save deadline waiting for
            # compilation/first communication. The external job manager owns
            # terminal teardown. A local training-thread error still finishes
            # its iteration cleanup before its usual explicit save refusal.
            control.update_status("ineligible")
            return
        self.ft_manager.process_group.recovery.request_last_words_stop()

    @staticmethod
    def _validate_supported_config(config: Config) -> None:
        FTManagerEx.validate_config(config.fault_tolerance)
        if config.fault_tolerance.last_words.enable:
            if config.fault_tolerance.enable_elastic:
                raise ValueError("Last words is save-only; disable elastic recovery")
            if not os.environ.get("LAST_WORDS_SESSION_FILE"):
                raise ValueError("Last words requires a service-issued LAST_WORDS_SESSION_FILE")
        if not config.checkpoint.enable or not config.checkpoint.enable_ft_dataloader_checkpoints:
            raise ValueError("NPU TorchFT requires checkpoint and per-replica dataloader checkpointing")

        if config.fault_tolerance.last_words.enable:
            if getattr(getattr(config, "ema_weights", None), "enable", False):
                raise ValueError("TorchFT last words does not support EMA weights")
            required = {"model", "optimizer", "lr_scheduler", "train_state", "dataloader", "torchft"}
            excluded = required.intersection(config.checkpoint.exclude_from_loading)
            if excluded:
                raise ValueError(f"Last words complete recovery cannot exclude training state: {sorted(excluded)}")

        model_spec = config.model_spec
        if (
            model_spec is None
            or not isinstance(model_spec.model, DeepSeekV4Model.Config)
            or not (model_spec.flavor == "deepseek_v4_flash" or model_spec.flavor.startswith("deepseek_v4_flash_"))
        ):
            raise ValueError("NPU TorchFT currently supports DeepSeek-V4 Flash recipes only")

        parallelism = config.parallelism
        if parallelism.data_parallel_replicate_degree != 1 and not parallelism.extensions.disable_expert_replication:
            raise ValueError(
                "NPU TorchFT requires data_parallel_replicate_degree=1 unless disable_expert_replication is enabled"
            )
        if parallelism.data_parallel_shard_degree <= 1 or parallelism.expert_parallel_degree <= 1:
            raise ValueError("NPU TorchFT currently requires both FSDP and expert parallelism")
        if (
            parallelism.tensor_parallel_degree != 1
            or parallelism.context_parallel_degree != 1
            or parallelism.pipeline_parallel_degree != 1
        ):
            raise ValueError("NPU TorchFT currently supports FSDP+EP only; TP, CP and PP are unsupported")

    @torch.no_grad()
    def _reset_auxiliary_state(self) -> None:
        # Auxiliary-loss modules expose a local accumulator and shared step metrics.
        # Inspect that state without coupling recovery to a model's patch class.
        for part in self.model_parts:
            for module in part.modules():
                accumulator = module._buffers.get("instance_acc")
                step_metrics = getattr(type(module), "group_acc", None)
                if isinstance(accumulator, torch.Tensor) and isinstance(step_metrics, dict):
                    accumulator.zero_()
                    step_metrics.clear()

    def init_distributed(self):
        dist_utils.set_spmd_backend(self.config.parallelism.spmd_backend)
        return super().init_distributed()

    def train_step(self, data_iterator) -> None:
        if self.last_words is not None:
            self._last_words_train_step(data_iterator)
            return
        self._ft_step_active = True
        try:
            self.recovery.run_step(self._train_step, data_iterator)
        finally:
            self._ft_step_active = False

    def state_dict(self):
        if self.last_words is not None and self._last_words_completed is not None and not self._last_words_updating:
            return dict(self._last_words_completed)
        state = super().state_dict()
        if self.last_words is not None:
            state["last_words_num_updates"] = self._last_words_num_updates
        elif getattr(self, "_ft_step_active", False) and not self.ft_manager.manager.commit_authorized:
            # The outer loop increments before zero_grad starts native healing.
            # A peer snapshot at this boundary contains the preceding state.
            state["step"] -= 1
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._last_words_num_updates = state_dict.get("last_words_num_updates", state_dict["step"])
        if self.last_words is not None:
            self._last_words_completed = dict(state_dict, last_words_num_updates=self._last_words_num_updates)
        if self._last_words_attempt_active or getattr(self, "_ft_step_active", False):
            self.step = state_dict["step"] + 1

    def _last_words_train_step(self, data_iterator):
        control = self.last_words
        assert control is not None
        if self._last_words_completed is None:
            self._last_words_completed = dict(self.state_dict(), step=self.step - 1)
            self.checkpointer.last_words_cursor_current = True
        if self.checkpointer.last_words_stop_requested():
            raise LastWordsStopRequestedError
        control.update_status("forward_backward", state=self._last_words_completed)
        self.checkpointer.last_words_cursor_current = False
        self._last_words_updating = False
        self._last_words_attempt_active = True
        attempt_failed = False
        exhausted = None
        result = None
        try:
            result = self._train_step(data_iterator)
        except DataloaderExhaustedError as error:
            exhausted = error
        except (Exception, CheckpointException) as error:
            control.request_terminal_stop(f"{type(error).__name__}: {error}")
            # The optimizer wrapper marks this before the first optimizer/pre-hook.
            if self._last_words_updating or self.ft_manager.manager.last_words_restore_incomplete:
                control.update_status("ineligible")
                raise LastWordsExitError("Optimizer or scheduler update incomplete; replica cannot dump") from error
            self._discard_last_words_attempt()
            attempt_failed = True
        finally:
            self._last_words_attempt_active = False
        manager = self.ft_manager.manager
        assert isinstance(manager, ManagerEx)
        if exhausted is None and not attempt_failed and not manager.commit_authorized:
            control.request_terminal_stop("TorchFT rejected an uncommitted attempt")
            self._discard_last_words_attempt()
        # Recoverable failures and normal returns must enter the same vote.
        # Otherwise one shard can wait for frozen files while its peer waits
        # in this collective, losing an otherwise complete healthy replica.
        if self.checkpointer.last_words_stop_requested():
            raise LastWordsStopRequestedError
        if exhausted is not None:
            # The upstream loop cancels this partial step and finishes normally.
            raise exhausted
        return result

    def _discard_last_words_attempt(self):
        try:
            self.ft_manager.process_group.recovery.prepare_for_terminal_save()
            reset_failed_iteration(self.model_parts)
            self._reset_auxiliary_state()
            torch_npu.npu.synchronize()
        except Exception as error:
            assert self.last_words is not None
            self.last_words.update_status("ineligible")
            raise LastWordsExitError("Device state unavailable for emergency save") from error

    def _finish_last_words(self):
        assert self._last_words_completed is not None and self.last_words is not None
        manager = self.ft_manager.manager
        if self._last_words_updating or manager.last_words_restore_incomplete or manager.commit_unknown:
            self.last_words.request_terminal_stop("Incomplete update/healing or unknown commit")
            self.last_words.update_status("ineligible")
            raise LastWordsExitError("Incomplete update/healing or unknown commit; replica cannot dump")
        if self._last_words_rng_state is None:
            self.last_words.update_status("ineligible")
            raise LastWordsExitError("Completed random streams are unavailable; replica cannot dump")
        # This snapshot is updated only at a committed boundary or by DCP load.
        self._last_words_rng_state.restore()
        self.load_state_dict(self._last_words_completed)
        self.checkpointer.save_last_words(self._last_words_completed)

    def train(self):
        try:
            try:
                result = super().train()
                if self.last_words is not None:
                    self.checkpointer.finish_pending_save()
                    self.last_words.finish_training()
            except LastWordsStopRequestedError:
                self._finish_last_words()
                return
        except BaseException:
            if self.last_words is not None:
                with contextlib.suppress(Exception):
                    self.last_words.fail_training()
            raise
        return result

    def _mark_last_words_updating(self):
        self._last_words_updating = True

    def close(self):
        control = getattr(self, "last_words", None)
        try:
            return super().close()
        finally:
            if control is not None:
                control.close()

    def _step_optimizer_and_scheduler(self):
        if self.last_words is not None:
            # Container pre-hooks mutate persistent MoE bias before step's body.
            # Vote first, then include those hooks in the protected transaction.
            if not self.ft_manager.manager.should_commit():
                return
            self._mark_last_words_updating()
            self.last_words.update_status("updating")
            self.optimizers.last_words_pre_authorized = True
        try:
            self.optimizers.step()
        finally:
            if self.last_words is not None:
                self.optimizers.last_words_pre_authorized = False
        self.lr_schedulers.step()
        if self.last_words is not None:
            torch_npu.npu.synchronize()
            assert self._last_words_rng_state is not None
            self._last_words_rng_state.capture()
            self._last_words_num_updates += 1
            self._last_words_completed = dict(self.state_dict())
            self.checkpointer.last_words_state = self._last_words_completed
            self.checkpointer.last_words_cursor_current = True
            self._last_words_updating = False
            self.last_words.update_status("completed", state=self._last_words_completed)

    def _train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
        """Keep the v0.3 FT loop, but finish metrics before authorizing updates."""
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        should_log = self.metrics_processor.should_log(self.step)
        parallel_dims = self.parallel_dims

        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pipeline_parallel_microbatches):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)

        global_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            global_valid_tokens = dist_utils.dist_sum_tensor(global_valid_tokens, parallel_dims.get_mesh("batch"))
        AuxLoss.set_step_denominator(global_valid_tokens)

        accumulated_loss: torch.Tensor | None = None
        for microbatches in microbatch_groups:
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(self.device)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device))
            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            loss = self.forward_backward_step(
                input_dict=fwd_bwd_input_dict,
                labels=fwd_bwd_labels,
                global_valid_tokens=global_valid_tokens,
            )
            if should_log:
                loss = loss.detach()
                if accumulated_loss is None:
                    accumulated_loss = loss.clone()
                else:
                    accumulated_loss.add_(loss)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        if not should_log:
            self._step_optimizer_and_scheduler()
            return

        assert accumulated_loss is not None
        if parallel_dims.dp_cp_enabled:
            ft_pg = self.ft_manager.loss_sync_pg
            assert ft_pg is not None, "Synchronous TorchFT requires a loss process group"
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
            global_avg_loss = dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg) / ft_pg.size()
            global_max_loss = dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
                ft_pg,
            )
        else:
            global_avg_loss = global_max_loss = accumulated_loss.item()
            global_ntokens_seen = self.ntokens_seen

        # No new cross-replica collective may be issued after a successful vote.
        self._step_optimizer_and_scheduler()
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics={"n_tokens_seen": global_ntokens_seen, "lr": lr},
        )
