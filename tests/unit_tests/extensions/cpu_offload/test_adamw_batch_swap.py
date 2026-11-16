# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace

import torch

import torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw as adamw_module
from torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw import (
    CpuOffloadAdamW,
    _AdamWEntry,
)


class _RecordingStaging:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.stream = object()
        self.h2d_batches: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], object]] = []

    def submit_h2d_batch(self, sources, destinations, *, stream):
        sources = tuple(sources)
        destinations = tuple(destinations)
        self.h2d_batches.append((sources, destinations, stream))
        for source, destination in zip(sources, destinations, strict=True):
            destination.copy_(source)
        return SimpleNamespace(wait_on=lambda _stream: None)


def _optimizer(staging: _RecordingStaging) -> CpuOffloadAdamW:
    optimizer = object.__new__(CpuOffloadAdamW)
    optimizer._staging = staging
    optimizer._clip_channel = None
    optimizer._offload_states = True
    optimizer._compute_device = torch.device("cpu")
    optimizer._work_buffers = {}
    optimizer._batch_work_buffers = {}
    return optimizer


def _entry(index: int, *, dtype: torch.dtype = torch.float32) -> _AdamWEntry:
    parameter = torch.arange(4, dtype=dtype) + index
    gradient = torch.full_like(parameter, 10 + index)
    exp_avg = torch.full_like(parameter, 20 + index)
    exp_avg_sq = torch.full_like(parameter, 30 + index)
    state = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    return _AdamWEntry(
        parameter=parameter,
        group={},
        state=state,
        canonical=(parameter, gradient, exp_avg, exp_avg_sq),
    )


def test_enqueue_batch_copies_all_active_states_in_one_batch() -> None:
    staging = _RecordingStaging()
    optimizer = _optimizer(staging)
    entries = (_entry(0), _entry(1))

    batch = optimizer._enqueue_batch(0, entries, compute_stream=object())

    assert len(staging.h2d_batches) == 1
    sources, destinations, stream = staging.h2d_batches[0]
    assert stream is staging.stream
    expected_sources = tuple(
        tensor for entry in entries for tensor in (entry.canonical[1], entry.canonical[0], *entry.canonical[2:])
    )
    assert len(sources) == len(expected_sources)
    assert all(source is expected for source, expected in zip(sources, expected_sources, strict=True))
    assert len({tensor.data_ptr() for tensor in destinations}) == len(destinations)
    for source, destination in zip(sources, destinations, strict=True):
        torch.testing.assert_close(destination, source)

    assert batch.entries == entries
    assert len(batch.works) == len(entries)
    for entry, work in zip(entries, batch.works, strict=True):
        torch.testing.assert_close(work.parameter, entry.canonical[0])
        torch.testing.assert_close(work.gradient, entry.canonical[1])
        torch.testing.assert_close(work.exp_avg, entry.canonical[2])
        torch.testing.assert_close(work.exp_avg_sq, entry.canonical[3])


def test_enqueue_batch_splits_copy_groups_at_dtype_boundary() -> None:
    staging = _RecordingStaging()
    optimizer = _optimizer(staging)
    entries = (_entry(0), _entry(1, dtype=torch.float64), _entry(2))

    optimizer._enqueue_batch(0, entries, compute_stream=object())

    assert len(staging.h2d_batches) == 2
    expected_groups = (
        tuple(
            tensor
            for entry in (entries[0], entries[2])
            for tensor in (entry.canonical[1], entry.canonical[0], *entry.canonical[2:])
        ),
        (entries[1].canonical[1], entries[1].canonical[0], *entries[1].canonical[2:]),
    )
    actual_groups = tuple(sources for sources, _destinations, _stream in staging.h2d_batches)
    assert len(actual_groups) == len(expected_groups)
    for actual, expected in zip(actual_groups, expected_groups, strict=True):
        assert len(actual) == len(expected)
        assert all(source is expected_source for source, expected_source in zip(actual, expected, strict=True))
    for sources, destinations, stream in staging.h2d_batches:
        assert stream is staging.stream
        assert len({source.dtype for source in sources}) == 1
        assert all(source.dtype == destination.dtype for source, destination in zip(sources, destinations, strict=True))
        for source, destination in zip(sources, destinations, strict=True):
            torch.testing.assert_close(destination, source)


def test_entry_batches_respect_work_buffer_limit(monkeypatch) -> None:
    staging = _RecordingStaging()
    optimizer = _optimizer(staging)
    monkeypatch.setattr(adamw_module, "_BATCH_BYTES", 100)
    entries = (_entry(0), _entry(1), _entry(2))

    batches = tuple(optimizer._iter_entry_batches(entries))

    assert tuple(len(batch) for batch in batches) == (1, 1, 1)
    assert tuple(batch[0] for batch in batches) == entries


def test_enqueue_batch_reuses_cached_gradient_without_work_buffer() -> None:
    staging = _RecordingStaging()
    optimizer = _optimizer(staging)
    entry = _entry(0)
    optimizer._clip_channel = SimpleNamespace(
        take_gradient=lambda parameter, gradient: entry.canonical[1],
    )

    batch = optimizer._enqueue_batch(0, (entry,), compute_stream=object())

    assert all(key[1] != 1 for key in batch.buffer_keys)
    assert len(staging.h2d_batches[0][0]) == 3
