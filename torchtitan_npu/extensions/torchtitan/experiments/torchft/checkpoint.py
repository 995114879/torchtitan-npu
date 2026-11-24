# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""TorchFT checkpoint handling for scalar optimizer state and scheduler restoration."""

import torch
from torchtitan.components.checkpoint import LR_SCHEDULER, OPTIMIZER
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer


class TorchFTCheckpointManagerEx(TorchFTCheckpointManager):
    def _prepare_save_state(self, state_dict):
        optimizer = state_dict.get(OPTIMIZER)
        if isinstance(optimizer, TorchFTOptimizersContainer):
            optimizer._refresh_cached_state_dict()  # pyrefly: ignore [missing-attribute]

    def dcp_save(self, state_dict, *args, **kwargs):
        self._prepare_save_state(state_dict)
        return super().dcp_save(state_dict, *args, **kwargs)

    def dcp_load(self, state_dict, checkpoint_id, from_hf=False, from_quantized=False):
        result = super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)
        schedulers = state_dict.get(LR_SCHEDULER)
        if (
            isinstance(state_dict.get(OPTIMIZER), TorchFTOptimizersContainer)
            and type(schedulers) is LRSchedulersContainer
        ):
            # Legacy periodic checkpoints can contain an older optimizer lr. The
            # built-in LambdaLR has already reconstructed each group's rate at last_epoch.
            for scheduler in schedulers:
                for group, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr(), strict=True):
                    if isinstance(group["lr"], torch.Tensor):
                        group["lr"].fill_(lr)
                    else:
                        group["lr"] = lr
        return result
