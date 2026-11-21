# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cover the default CP compile policy and the V4 activation checkpoint wrapping order."""

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC

import torchtitan_npu.models.deepseek_v4.parallelize as v4_parallel
from torchtitan_npu.models.deepseek_v4.compile_policy import (
    compiles_under_cp,
    skip_compile_under_cp,
)


@dataclass
class _CompileConfig:
    enable: bool = True


def _dims(cp: int) -> SimpleNamespace:
    # The policy reads one attribute of ``ParallelDims``; the rest is the caller's business.
    return SimpleNamespace(cp=cp)


@pytest.mark.parametrize(
    ("cp", "a5", "expected"),
    [
        (1, True, True),
        (1, False, True),
        (2, True, True),
        (2, False, False),
        (8, False, False),
    ],
)
def test_compiles_table(cp, a5, expected):
    assert compiles_under_cp(_dims(cp), a5=a5) is expected


def test_skip_compile_turns_it_off_and_says_why(caplog):
    config = _CompileConfig(enable=True)

    with caplog.at_level(logging.WARNING):
        returned = skip_compile_under_cp(object(), config, _dims(2), a5=False)

    assert returned is not config
    assert returned.enable is False
    assert "Skipping torch.compile" in caplog.text
    # The reason names the architecture fact, not just the degree.
    assert "arch22" in caplog.text


def test_skip_compile_keeps_the_object_when_it_may_compile():
    config = _CompileConfig(enable=True)
    # Same object back: the caller's config is shared with the recipe that built it.
    assert skip_compile_under_cp(object(), config, _dims(2), a5=True) is config
    assert skip_compile_under_cp(object(), config, _dims(1), a5=False) is config


def test_skip_compile_is_silent_when_not_asked():
    config = _CompileConfig(enable=False)
    assert skip_compile_under_cp(object(), config, _dims(2), a5=False) is config


@pytest.mark.parametrize(
    ("cp", "ac", "compile_block_first"),
    [(2, FullAC.Config(), True), (1, FullAC.Config(), False), (2, SelectiveAC.Config(), False)],
)
def test_cp_ac_compile_target(monkeypatch, cp, ac, compile_block_first):
    model = torch.nn.Module()
    block = torch.nn.Linear(2, 2)
    model.layers = torch.nn.ModuleDict({"0": block})
    config = CompileConfig(enable=True, backend="eager", components=["model"])

    def apply_upstream(model, *, compile_config, ac_config, parallel_dims, **kwargs):
        ac_config.build().apply(model)
        if compile_config.enable:
            v4_parallel.apply_compile(model, compile_config=compile_config, parallel_dims=parallel_dims)
        return model

    monkeypatch.setattr(v4_parallel, "parallelize_deepseekv3", apply_upstream)
    v4_parallel.parallelize_deepseek_v4(
        model,
        parallel_dims=SimpleNamespace(cp=cp, tp_enabled=False),
        training=TrainingConfig(),
        parallelism=SimpleNamespace(),
        compile_config=config,
        ac_config=ac,
        dump_folder="",
    )
    wrapper = model.layers["0"]
    assert isinstance(wrapper, CheckpointWrapper)
    assert wrapper._checkpoint_wrapped_module is block
    assert (block._compiled_call_impl is not None) == compile_block_first
    assert (wrapper._compiled_call_impl is None) == compile_block_first
    assert config.enable
