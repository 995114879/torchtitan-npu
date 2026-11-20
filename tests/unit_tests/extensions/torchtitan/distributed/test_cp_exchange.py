# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bounded CP all-to-all against the flat one, on real Gloo ranks."""

import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single as flat

# The four-rank splits the DeepSeek-V4 debug model produced on the NPUs (one supernode is owed
# nothing, which is what used to drop a collective), and a ragged six-rank table.
_SPLITS = {
    "npu": ((0, 254, 0, 0), (0, 0, 254, 0), (0, 0, 0, 254), (0, 0, 0, 0)),
    "ragged": (
        (3, 0, 5, 2, 0, 1),
        (0, 4, 0, 1, 2, 0),
        (2, 2, 0, 6, 0, 3),
        (1, 0, 3, 0, 4, 0),
        (0, 5, 0, 2, 0, 1),
        (2, 0, 1, 0, 3, 0),
    ),
}


def _cp_mesh(shape, names):
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", shape, mesh_dim_names=names)["cp"]


def _compare(mesh, send, recv, *, dtype=torch.float64):
    """Run both exchanges over the same rows and require the same values and gradients."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange

    rank = mesh.get_local_rank()
    rows = torch.arange(sum(send), dtype=dtype).reshape(-1, 1).repeat(1, 4) + 100 * rank
    weight = torch.randn(sum(recv), 4, dtype=dtype, generator=torch.Generator().manual_seed(rank))
    wanted, bounded = rows.clone().requires_grad_(), rows.clone().requires_grad_()
    expected = flat(wanted, output_split_sizes=recv, input_split_sizes=send, group=mesh)
    actual = cp_exchange.bounded_all_to_all(bounded, output_split_sizes=recv, input_split_sizes=send, group=mesh)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    assert torch.isfinite(actual).all()  # a zero-weight arc on the payload used to turn this NaN
    assert torch.equal(actual, expected)
    assert torch.equal(bounded.grad, wanted.grad)


def _worker(rank: int, rendezvous: str, case: str, intra: int) -> None:
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange
    from torchtitan_npu.models.deepseek_v4 import context_parallel

    splits = _SPLITS[case]
    send, recv = list(splits[rank]), [splits[peer][rank] for peer in range(len(splits))]
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=len(splits))
    try:
        mesh = _cp_mesh((len(splits),), ("cp",))
        cp_exchange.install(intra)
        # The model code keeps calling `all_to_all_single` and never learns it is patched.
        assert context_parallel.all_to_all_single is cp_exchange.bounded_all_to_all
        _compare(mesh, send, recv)
        _compare(mesh, send, recv, dtype=torch.float16)  # a real run's payload is half precision
    finally:
        dist.destroy_process_group()


def _subgroup_worker(rank: int, rendezvous: str, intra: int) -> None:
    """Two data-parallel replicas, so the CP axis of the second one does not start at rank 0."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange
    from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
    from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8)
    try:
        mesh = _cp_mesh((2, 4), ("dp", "cp"))
        cp_exchange.install(intra)
        # doc_chunk: every rank takes its own slice of each document.
        width = 256 // mesh.size()
        permutation = torch.cat(
            [
                torch.arange(doc * 256 + slot * width, doc * 256 + (slot + 1) * width)
                for slot in range(mesh.size())
                for doc in range(2)
            ]
        )
        metadata = DeepSeekV4Metadata.from_positions(torch.cat([torch.arange(256)] * 2).reshape(1, -1), ratio=1)
        with cp_shard_context_set(cp_mesh=mesh, load_balancer="doc_chunk"):
            local = metadata.per_doc_cp_metadata(
                permutation=permutation,
                cp_size=mesh.size(),
                cp_rank=mesh.get_local_rank(),
                window_size=64,
            )
        # The splits are the ones the production metadata build produced for this subgroup.
        assert local.xchg is not None
        _compare(mesh, local.xchg.cmp.send_split, local.xchg.cmp.recv_split)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo is unavailable")
@pytest.mark.parametrize(("case", "intra"), [("npu", 2), ("ragged", 2)])
def test_the_bounded_exchange_matches_the_flat_one(case: str, intra: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(
            _worker, args=(f"file://{tmp}/r", case, intra), nprocs=len(_SPLITS[case]), join=True
        )


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo is unavailable")
def test_a_cp_axis_that_does_not_start_at_rank_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_subgroup_worker, args=(f"file://{tmp}/r", 2), nprocs=8, join=True)


def test_a_caller_that_is_not_a_bounded_cp_exchange_passes_through(monkeypatch) -> None:
    """The EP token dispatcher calls `all_to_all_single(counts, None, None, group=...)`."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange

    seen = []

    def fake_flat(*args, **kwargs):
        seen.append((args, kwargs))
        return "flat"

    monkeypatch.setattr(cp_exchange, "_FLAT", fake_flat)
    monkeypatch.setattr(cp_exchange, "_DEGREE", 2)  # the switch is on, as in a real run
    counts, ep_mesh = torch.zeros(4, dtype=torch.int32), object()
    assert cp_exchange.bounded_all_to_all(counts, None, None, group=ep_mesh) == "flat"
    assert cp_exchange.bounded_all_to_all(counts, None, None, ep_mesh, "tag") == "flat"
    assert seen == [((counts, None, None, ep_mesh, ""), {}), ((counts, None, None, ep_mesh, "tag"), {})]


def _cp_axis(size: int):
    """A CP mesh stand-in: only the two facts the exchange reads before it plans."""
    return SimpleNamespace(mesh_dim_names=("cp",), size=lambda: size)


def test_only_a_cp_axis_that_has_to_split_refuses_a_traced_region(monkeypatch) -> None:
    """`cp` inside one supernode keeps compiling: the flat collective needs no host plan."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange

    monkeypatch.setattr(cp_exchange, "_DEGREE", 4)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(cp_exchange, "_FLAT", lambda *args, **kwargs: "flat")
    axis_in_one_supernode = _cp_axis(4)
    assert cp_exchange.bounded_all_to_all(torch.zeros(2), [2], [2], axis_in_one_supernode) == "flat"
    assert cp_exchange.bounded_all_to_all(torch.zeros(2), None, None, _cp_axis(8)) == "flat"
    with pytest.raises(RuntimeError, match="without model compilation"):
        cp_exchange.bounded_all_to_all(torch.zeros(2), [2], [2], _cp_axis(8))


def test_a_default_run_after_a_bounded_run_goes_flat(monkeypatch) -> None:
    """`install(-1)` revokes the bound: the substituted symbol must not keep the old degree."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange

    monkeypatch.setattr(cp_exchange, "_FLAT", lambda *args, **kwargs: "flat")
    monkeypatch.setattr(cp_exchange, "_DEGREE", -1)
    cp_exchange.install(2)
    assert cp_exchange._DEGREE == 2
    cp_exchange.install(-1)
    assert cp_exchange._DEGREE == -1
    assert not cp_exchange._is_bounded_axis(_cp_axis(8))
    assert cp_exchange.bounded_all_to_all(torch.zeros(2), [2], [2], _cp_axis(8)) == "flat"


def test_the_mesh_hook_builds_the_domains_once(monkeypatch) -> None:
    """The hook may fire per mesh resolution; the communicators are created once per mesh."""
    from torchtitan_npu.extensions.torchtitan.distributed import cp_exchange

    created = []

    def group(ranks):
        created.append(tuple(ranks))
        return f"group{len(created)}"

    monkeypatch.setattr(cp_exchange, "_group", group)
    monkeypatch.setattr(cp_exchange, "_DEGREE", 2)
    mesh = SimpleNamespace(
        mesh_dim_names=("cp",), mesh=torch.arange(8), size=lambda: 8, get_local_rank=lambda: 4
    )
    first = cp_exchange._domain_of(mesh)
    second = cp_exchange._domain_of(mesh)
    assert first is second  # cached, so a second run does not create the groups again
    assert created == [(4, 5), (0, 2, 4, 6)]  # supernode, then the same slot everywhere
