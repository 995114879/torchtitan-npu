# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Effective override reporting through the real upstream registry and traversal."""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
from torchtitan.config.configurable import Configurable

engine = importlib.import_module("torchtitan.config.override")


@dataclass(kw_only=True, slots=True)
class AttentionConfig(Configurable.Config):
    num_heads: int = 8


@dataclass(kw_only=True, slots=True)
class QuantizedAttentionConfig(AttentionConfig):
    backend: str = "quantized"
    recipe: str = "mxfp4"


@dataclass(kw_only=True, slots=True)
class FusedAttentionConfig(AttentionConfig):
    backend: str = "ascendc"


@dataclass(kw_only=True, slots=True)
class QuantizationSettings:
    enable_quantized_training: bool = True
    li_quantization: str = "mxfp4"


@dataclass(kw_only=True, slots=True)
class StackConfig(Configurable.Config):
    attention: AttentionConfig
    other: AttentionConfig
    extension: object = None


def fused(cfg, **kwargs):
    return engine.derive(cfg, FusedAttentionConfig, **kwargs)


def exact_only(cfg):
    return engine.derive(cfg, FusedAttentionConfig)


def _load_override_module(module_name: str, layer: str = "extensions"):
    path = Path(__file__).resolve().parents[5] / f"torchtitan_npu/{layer}/torchtitan/config/override.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def observed(monkeypatch):
    backport_name = "torchtitan_npu.patches.torchtitan.config.override"
    if backport_name not in sys.modules:
        monkeypatch.setitem(sys.modules, backport_name, _load_override_module(backport_name, "patches"))
    registry = engine._REGISTRY.copy()
    bindings = [
        (module, vars(module).get("apply_overrides"))
        for name, module in tuple(sys.modules.items())
        if module is not None
        and name.startswith(("torchtitan.", "torchtitan_npu."))
        and "apply_overrides" in vars(module)
    ]
    module = _load_override_module("observed_override_impl")
    engine.override(target=AttentionConfig, fqns=["attention"])(fused)
    engine.override(target=AttentionConfig, fqns=["other"], exact=True)(exact_only)
    yield module
    engine._REGISTRY.clear()
    engine._REGISTRY.update(registry)
    for target, original in bindings:
        target.apply_overrides = original  # pyrefly: ignore [missing-attribute]


def test_mixed_matches_report_final_backend_and_quantization(observed, caplog, monkeypatch):
    cfg = StackConfig(
        attention=AttentionConfig(),
        other=QuantizedAttentionConfig(),
        extension=SimpleNamespace(quantization=QuantizationSettings()),
    )
    imports = [f"{__name__}.fused", f"{__name__}.exact_only"]
    calls = []

    @wraps(fused)
    def counted(cfg):
        calls.append(cfg.num_heads)
        return fused(cfg)

    monkeypatch.setattr(engine._REGISTRY[f"{__name__}.fused"], "factory", counted)
    assert importlib.import_module("torchtitan.config").apply_overrides is engine.apply_overrides

    with caplog.at_level("INFO"):
        result = engine.apply_overrides(engine.OverrideConfig(imports=imports), cfg)

    assert isinstance(cfg.attention, FusedAttentionConfig)
    assert isinstance(cfg.other, QuantizedAttentionConfig)
    assert len(result) == 1
    assert calls == [8]
    assert "matched=1 fqns=['attention']" in caplog.text
    assert "[Override zero match]" in caplog.text
    assert "QuantizedAttentionConfig" in caplog.text
    assert "quantization takeover" in caplog.text
    assert "'backend': 'ascendc'" in caplog.text
    assert "'recipe': 'mxfp4'" in caplog.text
    assert "'num_heads': 8" in caplog.text
    assert "[Override]" in caplog.text
    assert "[Override entry]" not in caplog.text
    assert "[Override quantization]" in caplog.text
    assert "'enable_quantized_training': True" in caplog.text
    assert "'li_quantization': 'mxfp4'" in caplog.text


def test_success_logs_group_equal_replacements_and_keep_factory_kwargs(observed, caplog, monkeypatch):
    factory = f"{__name__}.fused"
    monkeypatch.setattr(engine._REGISTRY[factory], "fqns", None)
    cfg = StackConfig(
        attention=AttentionConfig(),
        other=AttentionConfig(),
        extension=QuantizedAttentionConfig(),
    )

    with caplog.at_level("INFO"):
        result = engine.apply_overrides(engine.OverrideConfig(imports=[(factory, {"num_heads": 16})]), cfg)

    messages = [
        record.getMessage() for record in caplog.records if record.getMessage().startswith(("[Override] ", "Applied "))
    ]
    assert messages == [
        f"[Override] {factory}: AttentionConfig -> FusedAttentionConfig with num_heads=16"
        " matched=2 fqns=['attention', 'other']",
        f"[Override] {factory}: QuantizedAttentionConfig -> FusedAttentionConfig with num_heads=16"
        " matched=1 fqns=['extension']",
        "Applied 3 override(s)",
    ]
    assert result == [
        f"[Override] {factory}: attention AttentionConfig -> FusedAttentionConfig with num_heads=16",
        f"[Override] {factory}: other AttentionConfig -> FusedAttentionConfig with num_heads=16",
        f"[Override] {factory}: extension QuantizedAttentionConfig -> FusedAttentionConfig with num_heads=16",
    ]


def test_invalid_entry_is_resolution_failure_without_mutation(observed, caplog):
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())
    with pytest.raises(ValueError, match="no registered"):
        engine.apply_overrides(engine.OverrideConfig(imports=[f"{__name__}.missing"]), cfg)
    assert type(cfg.attention) is AttentionConfig
    assert "[Override resolution failed]" in caplog.text
    assert "[Override zero match]" not in caplog.text


def test_conflict_is_reported_before_any_factory_runs(observed, caplog):
    engine._REGISTRY[f"{__name__}.exact_only"].fqns = ["attention"]
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())
    with pytest.raises(ValueError, match="both claim"):
        engine.apply_overrides(engine.OverrideConfig(imports=[f"{__name__}.fused", f"{__name__}.exact_only"]), cfg)
    assert type(cfg.attention) is AttentionConfig
    assert "[Override conflict]" in caplog.text
    assert "[Override zero match]" not in caplog.text


def test_empty_imports_preserve_tree_and_report_effective_config(observed, caplog):
    cfg = StackConfig(attention=AttentionConfig(), other=QuantizedAttentionConfig())
    with caplog.at_level("INFO"):
        result = engine.apply_overrides(engine.OverrideConfig(), cfg)
    assert result == []
    assert type(cfg.attention) is AttentionConfig
    assert type(cfg.other) is QuantizedAttentionConfig
    assert "[Override zero match]" not in caplog.text
    assert "[Override configured]" in caplog.text


@pytest.mark.parametrize("drift", ["missing-helper", "signature-change"])
def test_diagnostic_api_drift_preserves_replacements(observed, caplog, monkeypatch, drift):
    helpers = SimpleNamespace(
        **{
            name: getattr(engine, name)
            for name in ("_resolve_target", "_resolve_active", "_collect_claims", "_check_node_conflicts")
        }
    )
    if drift == "missing-helper":
        del helpers._collect_claims
    else:
        helpers._collect_claims = lambda: []
    monkeypatch.setattr(observed, "_override", helpers)
    calls = []

    def counted(cfg):
        calls.append(cfg.num_heads)
        return fused(cfg)

    monkeypatch.setattr(engine._REGISTRY[f"{__name__}.fused"], "factory", counted)
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())

    with caplog.at_level("INFO"):
        result = observed.apply_overrides(engine.OverrideConfig(imports=[f"{__name__}.fused"]), cfg)

    assert isinstance(cfg.attention, FusedAttentionConfig)
    assert len(result) == 1
    assert calls == [8]
    assert "[Override diagnostics unavailable] entry inspection failed" in caplog.text
    assert "[Override]" in caplog.text


@pytest.mark.parametrize("invalid", ["unknown-entry", "conflicting-claims"])
def test_diagnostic_fallback_preserves_upstream_validation(observed, monkeypatch, invalid):
    monkeypatch.setattr(observed, "_override", SimpleNamespace())
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())
    if invalid == "unknown-entry":
        imports, message = [f"{__name__}.missing"], "no registered"
    else:
        engine._REGISTRY[f"{__name__}.exact_only"].fqns = ["attention"]
        imports, message = [f"{__name__}.fused", f"{__name__}.exact_only"], "both claim"

    with pytest.raises(ValueError, match=message):
        observed.apply_overrides(engine.OverrideConfig(imports=imports), cfg)

    assert type(cfg.attention) is AttentionConfig


def test_summary_failure_preserves_completed_replacements(observed, caplog, monkeypatch):
    def unavailable_summary(config_root):
        raise AttributeError("upstream summary traversal changed")

    monkeypatch.setattr(observed, "_config_summary", unavailable_summary)
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())

    result = observed.apply_overrides(engine.OverrideConfig(imports=[f"{__name__}.fused"]), cfg)

    assert isinstance(cfg.attention, FusedAttentionConfig)
    assert len(result) == 1
    assert "config summary failed after upstream replacements completed" in caplog.text


def test_factory_failure_is_not_swallowed_by_diagnostics(observed, caplog, monkeypatch):
    factory = f"{__name__}.fused"
    monkeypatch.setattr(engine._REGISTRY[factory], "fqns", None)
    seen = 0

    def failing_second_factory(cfg):
        nonlocal seen
        seen += 1
        if seen == 1:
            observed.logger.info("[Override] custom factory note: node=%d", seen)
            return fused(cfg)
        raise RuntimeError("factory could not build replacement")

    monkeypatch.setattr(engine._REGISTRY[factory], "factory", failing_second_factory)
    cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())

    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="factory could not build replacement"):
        observed.apply_overrides(engine.OverrideConfig(imports=[factory]), cfg)

    assert isinstance(cfg.attention, FusedAttentionConfig)
    assert type(cfg.other) is AttentionConfig
    assert "[Override] custom factory note: node=1" in caplog.text
    assert "Applied " not in caplog.text


def test_reexec_module_does_not_duplicate_reports(observed, caplog):
    _load_override_module("observed_override_reloaded")
    cfg = StackConfig(attention=AttentionConfig(), other=QuantizedAttentionConfig())
    imports = [f"{__name__}.fused", f"{__name__}.exact_only"]

    with caplog.at_level("INFO"):
        result = engine.apply_overrides(engine.OverrideConfig(imports=imports), cfg)

    assert len(result) == 1
    assert caplog.text.count("[Override zero match]") == 1
    assert caplog.text.count("Applied 1 override(s)") == 1


def test_same_thread_nested_apply_keeps_call_logs_separate(observed, caplog, monkeypatch):
    factory = f"{__name__}.fused"
    inner_cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())
    outer_cfg = StackConfig(attention=AttentionConfig(), other=AttentionConfig())
    nested = False

    @wraps(fused)
    def nested_factory(cfg):
        nonlocal nested
        if not nested:
            nested = True
            engine.apply_overrides(engine.OverrideConfig(imports=[factory]), inner_cfg)
        return fused(cfg)

    monkeypatch.setattr(engine._REGISTRY[factory], "factory", nested_factory)

    with caplog.at_level("INFO"):
        outer_result = engine.apply_overrides(engine.OverrideConfig(imports=[factory]), outer_cfg)

    messages = [
        record.getMessage() for record in caplog.records if record.getMessage().startswith(("[Override] ", "Applied "))
    ]
    expected_line = f"[Override] {factory}: AttentionConfig -> FusedAttentionConfig matched=1 fqns=['attention']"
    assert messages == [
        expected_line,
        "Applied 1 override(s)",
        expected_line,
        "Applied 1 override(s)",
    ]
    assert isinstance(inner_cfg.attention, FusedAttentionConfig)
    assert isinstance(outer_cfg.attention, FusedAttentionConfig)
    assert outer_result == [f"[Override] {factory}: attention AttentionConfig -> FusedAttentionConfig"]
