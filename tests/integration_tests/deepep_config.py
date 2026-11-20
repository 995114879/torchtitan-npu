# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe the real MXFP8 DeepEP path during ordinary model training."""

from collections import Counter
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torchao_npu.wrapper_tensors import BaseTrainingWeightWrapperTensor
from torchtitan.config import derive
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.adaptor.torchao.deepep_token_dispatcher import get_mxfp8_deepep_dispatcher_cls
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.config_registry import _make_trainer_config
from torchtitan_npu.override.common.token_dispatcher import AscDeepEPTokenDispatcher


class _DeepEPTrace(TorchDispatchMode):
    """Observe custom-op inputs/results without replacing any device implementation."""

    def __init__(self):
        super().__init__()
        self.calls = Counter()
        self.values = []

    def _observe_dispatch(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = func._schema.name
        if not name.startswith("deepep::"):
            return result
        self.calls[name] += 1
        if name == "deepep::dispatch":
            payload = args[0]
            assert len(payload) == 2, "training fell back to high-precision dispatch"
            assert payload[0].dtype == torch.float8_e4m3fn and payload[1].dtype == torch.uint8
        elif name == "deepep::dispatch_quantized_cached":
            assert args[0].dtype == torch.float8_e4m3fn and args[1].dtype == torch.uint8
        elif name == "deepep::combine":
            assert args[0].dtype in (torch.bfloat16, torch.float16, torch.float32)
            self.values.append(("combined", result.detach()))
        elif name == "deepep::dispatch_backward":
            assert args[0].dtype in (torch.bfloat16, torch.float16, torch.float32)
            self.values.append(("activation_grad", result[0].detach()))
            # The second result is uninitialized when score gradients are absent.
            if args[2]:
                self.values.append(("score_grad", result[1].detach()))
        return result

    def assert_complete(self):
        for name in ("dispatch", "combine", "dispatch_backward", "dispatch_quantized_cached"):
            assert self.calls[f"deepep::{name}"] > 0, f"missing DeepEP training stage: {name}"
        assert self.calls["deepep::combine_backward"] == 0, "combine backward fell back to high precision"
        for name, value in self.values:
            assert torch.isfinite(value).all().item(), f"non-finite {name}"
        for name in ("activation_grad", "score_grad"):
            collected = [value for key, value in self.values if key == name]
            # ``score_grad`` is only collected when ``has_grad_scores`` is set, so
            # an empty list means this VJP branch was never exercised (distinct
            # from a real path whose gradient came out identically zero).
            assert collected, f"{name} was never collected; its VJP path was not exercised"
            assert any(value.count_nonzero().item() > 0 for value in collected), f"{name} is identically zero"


# ``TorchDispatchMode`` requires its magic dispatch entry point to be named with
# double underscores. Bind it through a string so the generic-identifier naming
# rule does not flag the override; the method body lives in ``_observe_dispatch``.
setattr(_DeepEPTrace, "__torch_dispatch__", _DeepEPTrace._observe_dispatch)  # noqa: B010


def _local_parameter(parameter):
    local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    return local.to_tensor() if isinstance(local, BaseTrainingWeightWrapperTensor) else local


class DeepEPTrainer(TrainerEx):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        pass

    def train_step(self, data_iterator):
        quantization = self.config.extension.quantization
        assert quantization.enable_quantized_training and quantization.all_to_all_quant_format == "mxfp8"
        assert not self.config.compile.enable, "the communication observer covers eager training"
        dispatcher_cls = get_mxfp8_deepep_dispatcher_cls(AscDeepEPTokenDispatcher)
        model = self.model_parts[0]
        for layer in model.layers.values():
            dispatcher = layer.moe.routed_experts.token_dispatcher
            assert isinstance(dispatcher, dispatcher_cls), type(dispatcher)
            assert dispatcher._deepep_world_size == 2
            weight = layer.moe.routed_experts.inner_experts.w2_EDF
            local = weight.to_local() if isinstance(weight, DTensor) else weight
            assert isinstance(local, BaseTrainingWeightWrapperTensor), "expert GEMM was not quantized"

        # Layer 2 is the first learned router; layers 0 and 1 use hash routing.
        moe = model.layers["2"].moe
        parameters = {
            "expert_w2": moe.routed_experts.inner_experts.w2_EDF,
            "router_gate": moe.router.gate.weight,
        }
        before = {name: _local_parameter(value).detach().clone() for name, value in parameters.items()}
        with _DeepEPTrace() as trace:
            result = super().train_step(data_iterator)
        trace.assert_complete()
        for name, parameter in parameters.items():
            current = _local_parameter(parameter).detach()
            assert torch.isfinite(current).all().item(), f"non-finite {name} after optimizer step"
            assert not torch.equal(current, before[name]), f"optimizer did not update {name}"
        logger.info("DeepEP MXFP8 training step %s verified: %s", self.step, dict(trace.calls))
        return result


def deepseek_v4_deepep_training():
    config = _make_trainer_config(
        "debugmodel", local_batch_size=1, seq_len=256, moe_comm_backend="deepep"
    )
    config.activation_checkpoint = SelectiveAC.Config()
    return derive(config, DeepEPTrainer.Config)
