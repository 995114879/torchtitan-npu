# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for TorchAO-NPU converter log grouping and lossless FQN folding."""

import logging
import re
from collections import Counter

import pytest

from torchtitan_npu.extensions.adaptor.torchao.logging_utils import (
    GroupedConverterLogger,
    compact_fqns,
)


def _expand_brace_segment(segment: str) -> list[str]:
    """Independent oracle expanding a brace-range segment back into literal indices."""
    if not (segment.startswith("{") and segment.endswith("}")):
        return [segment]
    expanded: list[str] = []
    for token in segment[1:-1].split(","):
        if ".." in token:
            start_str, end_str = token.split("..", 1)
            expanded.extend(str(i) for i in range(int(start_str), int(end_str) + 1))
        else:
            expanded.append(token)
    return expanded


def _expand_compact_fqn(pattern: str) -> list[str]:
    """Expand one folded FQN pattern back into every concrete FQN it represents."""
    results = [""]
    for segment in re.findall(r"\{[^}]+\}|[^.]+", pattern):
        choices = _expand_brace_segment(segment)
        results = [f"{prefix}.{choice}" if prefix else choice for prefix in results for choice in choices]
    return results


def _expand_compact_fqns(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in patterns:
        expanded.extend(_expand_compact_fqn(pattern))
    return expanded


@pytest.mark.parametrize(
    ("fqns", "expected"),
    [
        ([], []),
        (["tok_embeddings", "output"], ["tok_embeddings", "output"]),
        (["layers.0.attention.wq_a"], ["layers.0.attention.wq_a"]),
        (["layers.0.attention.wq_a", "layers.1.attention.wq_a"], ["layers.{0,1}.attention.wq_a"]),
        (
            [f"layers.{i}.attention.wq_a" for i in range(43)],
            ["layers.{0..42}.attention.wq_a"],
        ),
        (
            [f"layers.{i}.moe.routed_experts" for i in (1, 3, 4, 5, 8, 9)],
            ["layers.{1,3..5,8,9}.moe.routed_experts"],
        ),
        (
            ["layers.2.attention.wo", "layers.0.attention.wo", "layers.1.attention.wo"],
            ["layers.{0..2}.attention.wo"],
        ),
        (
            ["layers.00.attention.wo", "layers.01.attention.wo"],
            ["layers.00.attention.wo", "layers.01.attention.wo"],
        ),
        (
            ["layers.0.w", "layers.0.w", "layers.1.w", "layers.1.w"],
            ["layers.{0,1}.w", "layers.{0,1}.w"],
        ),
    ],
    ids=[
        "empty",
        "no-numeric-segments",
        "single-layer",
        "two-consecutive-layers",
        "contiguous-43-layers",
        "sparse-ranges",
        "unsorted-input",
        "leading-zeros-preserved",
        "duplicate-fqns-preserved",
    ],
)
def test_compact_fqns_folds_ranges_losslessly(fqns, expected):
    actual = compact_fqns(fqns)
    assert actual == expected
    assert Counter(_expand_compact_fqns(actual)) == Counter(fqns)


def test_compact_fqns_folds_nested_numeric_segments_and_preserves_template_order():
    layer_major = [
        f"model_spec.model.layers.{layer}.attention.{proj}" for layer in range(4) for proj in ("wq_a", "wq_b", "wo")
    ]
    nested_uniform = [f"layers.{layer}.experts.{expert}.w1" for layer in range(3) for expert in range(4)]
    nested_ragged = [
        "layers.0.experts.0.w2",
        "layers.0.experts.1.w2",
        "layers.0.experts.2.w2",
        "layers.1.experts.0.w2",
        "layers.1.experts.1.w2",
    ]
    fixed_trailing_index = [f"layers.{layer}.branches.0" for layer in range(3)]

    assert compact_fqns(layer_major) == [
        "model_spec.model.layers.{0..3}.attention.wq_a",
        "model_spec.model.layers.{0..3}.attention.wq_b",
        "model_spec.model.layers.{0..3}.attention.wo",
    ]
    assert compact_fqns(nested_uniform) == ["layers.{0..2}.experts.{0..3}.w1"]
    assert compact_fqns(nested_ragged) == [
        "layers.0.experts.{0..2}.w2",
        "layers.1.experts.{0,1}.w2",
    ]
    assert compact_fqns(fixed_trailing_index) == ["layers.{0..2}.branches.0"]

    for sample in (layer_major, nested_uniform, nested_ragged, fixed_trailing_index):
        assert Counter(_expand_compact_fqns(compact_fqns(sample))) == Counter(sample)


def test_grouped_converter_logger_aggregates_info_and_keeps_debug_details(caplog):
    class _DummyConverter:
        pass

    converter_logger = GroupedConverterLogger(_DummyConverter())
    with caplog.at_level(logging.DEBUG):
        for layer in range(3):
            converter_logger.record(
                "Linear.Config -> NpuQuantizedLinear.Config",
                f"model_spec.model.layers.{layer}.attention.wq_a",
            )
            converter_logger.record(
                "Linear.Config -> NpuQuantizedLinear.Config",
                f"model_spec.model.layers.{layer}.attention.wo",
            )
            converter_logger.record(
                "GroupedExperts.Config -> NpuQuantizedGroupedExperts.Config",
                f"model_spec.model.layers.{layer}.moe.routed_experts.inner_experts",
            )
        converter_logger.flush()
        # A second flush after clearing must not duplicate records.
        converter_logger.flush()

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    converter_name = f"{_DummyConverter.__module__}.{_DummyConverter.__qualname__}"
    assert info_messages == [
        f"[Converter] {converter_name}: Linear.Config -> NpuQuantizedLinear.Config matched=6 "
        "fqns=['model_spec.model.layers.{0..2}.attention.wq_a', 'model_spec.model.layers.{0..2}.attention.wo']",
        f"[Converter] {converter_name}: GroupedExperts.Config -> NpuQuantizedGroupedExperts.Config matched=3 "
        "fqns=['model_spec.model.layers.{0..2}.moe.routed_experts.inner_experts']",
    ]
    assert len(debug_messages) == 9
    assert (
        f"[Converter] {converter_name}: model_spec.model.layers.1.attention.wo "
        "Linear.Config -> NpuQuantizedLinear.Config"
    ) in debug_messages


def test_grouped_converter_logger_empty_flush_is_silent(caplog):
    class _DummyConverter:
        pass

    converter_logger = GroupedConverterLogger(_DummyConverter())
    with caplog.at_level(logging.DEBUG):
        converter_logger.flush()

    assert caplog.records == []


def test_grouped_converter_logger_supports_separate_debug_detail(caplog):
    class _DummyConverter:
        pass

    converter_logger = GroupedConverterLogger(_DummyConverter())
    with caplog.at_level(logging.DEBUG):
        for layer in range(3):
            converter_logger.record(
                "fsdp_prequantize=enabled",
                f"layers.{layer}.attention.wq_a",
                debug_detail="fsdp_prequantize=enabled (whitelist .attention.wq_a)",
            )
            converter_logger.record(
                "fsdp_prequantize=enabled",
                f"layers.{layer}.attention.wq_b",
                debug_detail="fsdp_prequantize=enabled (whitelist .attention.wq_b)",
            )
        converter_logger.flush()

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    converter_name = f"{_DummyConverter.__module__}.{_DummyConverter.__qualname__}"
    assert info_messages == [
        f"[Converter] {converter_name}: fsdp_prequantize=enabled matched=6 "
        "fqns=['layers.{0..2}.attention.wq_a', 'layers.{0..2}.attention.wq_b']"
    ]
    assert len(debug_messages) == 6
    assert (
        f"[Converter] {converter_name}: layers.1.attention.wq_a fsdp_prequantize=enabled (whitelist .attention.wq_a)"
    ) in debug_messages
    assert (
        f"[Converter] {converter_name}: layers.2.attention.wq_b fsdp_prequantize=enabled (whitelist .attention.wq_b)"
    ) in debug_messages
