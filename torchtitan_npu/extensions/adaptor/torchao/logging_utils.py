# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped logging helpers for TorchAO-NPU config converters."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence


def _is_canonical_index(segment: str) -> bool:
    """Return True for canonical non-negative integer segments without leading zeros."""
    return segment.isascii() and segment.isdigit() and (segment == "0" or not segment.startswith("0"))


def _format_run(start: int, end: int) -> str:
    if end == start:
        return str(start)
    if end == start + 1:
        return f"{start},{end}"
    return f"{start}..{end}"


def _format_index_runs(indices: list[int]) -> str:
    """Format sorted unique integers into a lossless brace-range expression."""
    if len(indices) == 1:
        return str(indices[0])
    parts: list[str] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(_format_run(start, prev))
        start = prev = value
    parts.append(_format_run(start, prev))
    return "{" + ",".join(parts) + "}"


def _fold_at_numeric_from_right(items: list[list[str]], from_right: int) -> tuple[list[list[str]], bool]:
    """Fold segmented FQNs sharing prefix/suffix around their ``from_right``-th numeric segment."""
    buckets: dict[tuple[tuple[str, ...], tuple[str, ...]], list[tuple[int, int, list[str]]]] = defaultdict(list)
    emitted: list[tuple[int, list[str]]] = []
    for pos, segments in enumerate(items):
        num_positions = [idx for idx, seg in enumerate(segments) if _is_canonical_index(seg)]
        if len(num_positions) < from_right:
            emitted.append((pos, segments))
            continue
        idx_pos = num_positions[-from_right]
        template = (tuple(segments[:idx_pos]), tuple(segments[idx_pos + 1 :]))
        buckets[template].append((pos, int(segments[idx_pos]), segments))

    changed = False
    for (prefix, suffix), entries in buckets.items():
        unique_entries: list[tuple[int, int, list[str]]] = []
        seen_indices: set[int] = set()
        for entry in entries:
            if entry[1] in seen_indices:
                emitted.append((entry[0], entry[2]))
            else:
                seen_indices.add(entry[1])
                unique_entries.append(entry)

        if len(unique_entries) >= 2:
            changed = True
            sorted_indices = sorted(entry[1] for entry in unique_entries)
            folded = [*prefix, _format_index_runs(sorted_indices), *suffix]
            emitted.append((unique_entries[0][0], folded))
        else:
            emitted.append((unique_entries[0][0], unique_entries[0][2]))

    emitted.sort(key=lambda entry: entry[0])
    return [segments for _, segments in emitted], changed


def compact_fqns(fqns: Sequence[str]) -> list[str]:
    """Losslessly fold repeated numeric FQN segments into brace ranges.

    Preserves first-seen template order and folds inner numeric segments before
    outer ones so nested indexed paths such as ``layers.{0..42}.experts.{0..7}.w1``
    collapse cleanly without dropping or merging distinct paths.
    """
    current = [fqn.split(".") for fqn in fqns]
    while len(current) > 1:
        max_numeric = max(
            (sum(1 for seg in segments if _is_canonical_index(seg)) for segments in current),
            default=0,
        )
        if max_numeric == 0:
            break
        merged = False
        for from_right in range(1, max_numeric + 1):
            current, changed = _fold_at_numeric_from_right(current, from_right)
            if changed:
                merged = True
                break
        if not merged:
            break
    return [".".join(segments) for segments in current]


class GroupedConverterLogger:
    """Aggregate per-node converter logs at INFO while preserving DEBUG details."""

    def __init__(self, converter: object) -> None:
        cls = converter if isinstance(converter, type) else type(converter)
        self._name = f"{cls.__module__}.{cls.__qualname__}"
        self._groups: dict[str, list[str]] = defaultdict(list)

    def record(self, detail: str, fqn: str, *, debug_detail: str | None = None) -> None:
        logger.debug("[Converter] %s: %s %s", self._name, fqn, debug_detail or detail)
        self._groups[detail].append(fqn)

    def flush(self) -> None:
        for detail, fqns in self._groups.items():
            logger.info(
                "[Converter] %s: %s matched=%d fqns=%s",
                self._name,
                detail,
                len(fqns),
                compact_fqns(fqns),
            )
        self._groups.clear()
