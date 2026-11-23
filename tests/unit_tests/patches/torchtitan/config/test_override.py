# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for folded FQN ranges in grouped override logs (#6315)."""

import importlib
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field

import pytest
from torchtitan.config import OverrideConfig, apply_overrides, derive, override
from torchtitan.config.configurable import Configurable
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.tools.logging import logger as titan_logger

import torchtitan_npu  # noqa: F401
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

pytestmark = pytest.mark.cpu

_override_engine = importlib.import_module("torchtitan.config.override")


class _ComponentA(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 128


class _DerivedComponentA(_ComponentA):
    @dataclass(kw_only=True, slots=True)
    class Config(_ComponentA.Config):
        pass


class _ComponentB(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        extra: int = 0


class _Parent(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        child: _ComponentA.Config = field(default_factory=_ComponentA.Config)
        children: list[_ComponentA.Config] = field(default_factory=list)


class _TaggedRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass


@pytest.fixture
def isolated_override_registry():
    saved = _override_engine._REGISTRY.copy()
    yield
    _override_engine._REGISTRY.clear()
    _override_engine._REGISTRY.update(saved)


def _swap_to_b(cfg: _ComponentA.Config, *, extra: int) -> _ComponentB.Config:
    return _ComponentB.Config(dim=cfg.dim, extra=extra)


def _swap_rmsnorm(cfg: RMSNorm.Config) -> _TaggedRMSNorm.Config:
    return derive(cfg, _TaggedRMSNorm.Config)


def test_package_import_folds_contiguous_and_gapped_fqns_in_grouped_override_logs(
    isolated_override_registry,
    caplog,
) -> None:
    target = f"{__name__}._swap_to_b"
    override(target=_ComponentA.Config)(_swap_to_b)

    config_mod = importlib.import_module("torchtitan.config")
    trainer_mod = importlib.import_module("torchtitan.trainer")
    assert config_mod.apply_overrides is sys.modules["torchtitan.config.override"].apply_overrides
    assert trainer_mod.apply_overrides is config_mod.apply_overrides

    parent_cfg = _Parent.Config(
        children=[
            _ComponentA.Config(),
            _ComponentA.Config(),
            _ComponentA.Config(),
            _DerivedComponentA.Config(),
            _ComponentA.Config(),
            _ComponentA.Config(),
        ]
    )

    with caplog.at_level(logging.INFO, logger=titan_logger.name):
        replacements = apply_overrides(OverrideConfig(imports=[(target, {"extra": 256})]), parent_cfg)

    override_messages = [
        rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(("[Override] ", "Applied "))
    ]
    assert override_messages == [
        f"[Override] {target}: _ComponentA.Config -> _ComponentB.Config with extra=256"
        " matched=6 fqns=['child', 'children.{0..2,4,5}']",
        f"[Override] {target}: _DerivedComponentA.Config -> _ComponentB.Config with extra=256"
        " matched=1 fqns=['children.3']",
        "Applied 7 override(s)",
    ]
    assert replacements == [
        f"[Override] {target}: child _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.0 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.1 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.2 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.3 _DerivedComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.4 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.5 _ComponentA.Config -> _ComponentB.Config with extra=256",
    ]
    assert isinstance(parent_cfg.child, _ComponentB.Config)
    assert all(isinstance(node, _ComponentB.Config) and node.extra == 256 for node in parent_cfg.children)


def test_apply_overrides_folds_layer_fqns_on_deepseek_v4_debugmodel(
    isolated_override_registry,
    caplog,
) -> None:
    target = f"{__name__}._swap_rmsnorm"
    override(target=RMSNorm.Config)(_swap_rmsnorm)

    cfg = deepseek_v4_debugmodel()
    assert cfg.model_spec is not None
    matched_nodes = list(cfg.traverse(RMSNorm.Config, recurse=True))
    expected_fqns = [fqn for fqn, _, _, _ in matched_nodes]
    expected_counts = Counter(type(node).__qualname__ for _, node, _, _ in matched_nodes)
    num_layers = len(cfg.model_spec.model.layers)
    assert num_layers >= 2
    layer_range = f"{{0..{num_layers - 1}}}"
    assert any(fqn.startswith("model_spec.model.layers.0.") for fqn in expected_fqns)

    with caplog.at_level(logging.INFO, logger=titan_logger.name):
        replacements = apply_overrides(OverrideConfig(imports=[target]), cfg)

    assert len(replacements) == len(expected_fqns)
    grouped_msgs = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(f"[Override] {target}: ")]
    assert len(grouped_msgs) == len(expected_counts)
    for cls_name, count in expected_counts.items():
        msg = next(m for m in grouped_msgs if f"{cls_name} -> _TaggedRMSNorm.Config" in m)
        assert f"matched={count} " in msg
    rmsnorm_msg = next(m for m in grouped_msgs if "RMSNorm.Config -> _TaggedRMSNorm.Config" in m)
    unitscale_msg = next(m for m in grouped_msgs if "UnitScaleRMSNorm.Config -> _TaggedRMSNorm.Config" in m)
    assert f"model_spec.model.layers.{layer_range}.attention_norm" in rmsnorm_msg
    assert "model_spec.model.layers.0.attention_norm" not in rmsnorm_msg
    assert f"model_spec.model.layers.{layer_range}.attention.q_head_norm" in unitscale_msg
