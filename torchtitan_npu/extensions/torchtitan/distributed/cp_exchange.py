# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded CP all-to-all: keep each CP exchange inside one supernode.

An AllToAllV may not span a 128-NPU supernode, and the CP axis cannot be shaped to avoid it, so
this swaps ``all_to_all_single`` for one all-gather across the ranks holding the same slot in
each supernode plus one all-to-all inside it.  The model code is untouched: the substitution is
a symbol patch, and a run that leaves ``--parallelism.extensions.context-parallel-intra-degree``
at ``-1`` never reaches this module.

The padding and the slice boundaries come from the column's split table, which has to be read on
the host, so the exchange is not traceable with ``torch.compile(fullgraph=True)``: it refuses a
traced region instead of letting the graph disagree with the flat exchange.
"""

import sys
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as functional
from torch.distributed._functional_collectives import all_gather_single
from torchtitan.tools.logging import logger

_FLAT = functional.all_to_all_single
"""The collective the CP modules reach for; the wrapper falls back to it."""
_PAD_ALIGN = 128  # Round the payload up, so the shapes repeat across layers.
_DEGREE = -1  # CP ranks one all-to-all may span; `install` sets it from the run's config.


@dataclass(frozen=True, slots=True)
class CpDomain:
    """The two communicators one CP exchange may use."""

    size: int  # CP ranks in total.
    intra: int
    rank: int  # Index into the CP axis, i.e. the local rank of its mesh.
    inside: dist.ProcessGroup  # This rank's supernode.
    column: dist.ProcessGroup  # The same slot in every supernode.

    @property
    def start(self) -> int:
        """First peer of this rank's supernode."""
        return self.rank // self.intra * self.intra

    @property
    def column_size(self) -> int:
        return self.size // self.intra


def install(intra_degree: int) -> None:
    """Bound every CP all-to-all of this run to one supernode (once, from the trainer's init).

    A non-positive degree revokes an earlier bound: the substituted symbol stays in place but
    falls through to the flat collective, so a second run in the same process is not silently
    held to the first run's degree.
    """
    global _DEGREE
    if intra_degree < 1:
        _DEGREE = -1
        return
    _DEGREE = intra_degree
    # Every module that already imported the collective, plus the module future imports read.
    symbol = "all_to_all_single"
    setattr(functional, symbol, bounded_all_to_all)
    for module in list(sys.modules.values()):
        if getattr(module, symbol, None) is _FLAT:
            setattr(module, symbol, bounded_all_to_all)
    logger.info(f"CP all-to-all is bounded to {intra_degree} ranks per supernode")
    _build_domains_when_the_cp_mesh_is_resolved()


_HOOKED = False


def _build_domains_when_the_cp_mesh_is_resolved() -> None:
    """Build the communicators outside the model, where ``dist.new_group`` is safe to call."""
    global _HOOKED
    if _HOOKED:  # A second install in this process must not stack wrappers.
        return
    _HOOKED = True
    from torchtitan.distributed.parallel_dims import ParallelDims

    resolve = ParallelDims.get_optional_mesh

    def get_optional_mesh(self, dims, **kwargs):
        mesh = resolve(self, dims, **kwargs)
        if mesh is not None and mesh.mesh_dim_names == ("cp",):
            _domain_of(mesh)
        return mesh

    ParallelDims.get_optional_mesh = get_optional_mesh


def _group(ranks: list[int]) -> dist.ProcessGroup:
    return cast("dist.ProcessGroup", dist.new_group(ranks, use_local_synchronization=True))


@cache
def _domain(ranks: tuple[int, ...], intra: int, rank: int) -> CpDomain:
    first = rank // intra * intra
    return CpDomain(
        size=len(ranks),
        intra=intra,
        rank=rank,
        inside=_group(list(ranks[first : first + intra])),
        column=_group(list(ranks[rank % intra :: intra])),
    )


def _is_bounded_axis(group) -> bool:
    """Whether this group is a CP axis wider than one supernode, i.e. an exchange to split."""
    if _DEGREE < 1 or getattr(group, "mesh_dim_names", None) != ("cp",):
        return False
    size = group.size()
    return size > _DEGREE and not size % _DEGREE


def _domain_of(group) -> CpDomain | None:
    """The domains of a CP axis, or ``None`` when that axis is one supernode already."""
    if not _is_bounded_axis(group):
        return None
    ranks = tuple(int(rank) for rank in group.mesh.flatten().tolist())
    return _domain(ranks, _DEGREE, group.get_local_rank())


def bounded_all_to_all(
    input: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
    group: Any = None,
    tag: str = "",
) -> torch.Tensor:
    """``all_to_all_single`` kept inside one supernode: same signature, same arguments, same results.

    A caller that is not a bounded CP exchange -- the EP token dispatcher's counting exchange, for
    one -- reaches the flat collective with its own arguments untouched.  A bounded exchange makes
    two collectives of its own, so it ignores ``tag``; no CP call site passes one today.
    """
    if output_split_sizes is None or input_split_sizes is None:
        return _FLAT(input, output_split_sizes, input_split_sizes, group, tag)
    if _is_bounded_axis(group) and torch.compiler.is_compiling():
        # Only a CP axis that actually has to be split reads the plan on the host; a CP axis that
        # fits in one supernode keeps the flat collective and stays traceable.
        raise RuntimeError(
            "the bounded CP exchange reads the column's split table on the host, which torch.compile "
            "cannot trace: run this job without model compilation, or drop "
            "--parallelism.extensions.context-parallel-intra-degree"
        )
    domain = _domain_of(group)
    if domain is None:
        return _FLAT(input, output_split_sizes, input_split_sizes, group, tag)
    return _exchange(input, domain, send=input_split_sizes, recv=output_split_sizes)


def _table(domain: CpDomain, send: list[int], device) -> tuple:
    """Every column member's send split: the one cross-rank input the exchange needs."""
    gathered = all_gather_single(torch.tensor(list(send), device=device, dtype=torch.int64), 0, domain.column)
    return tuple(tuple(mate.tolist()) for mate in gathered.reshape(domain.column_size, domain.size))


def _exchange(rows, domain: CpDomain, *, send: list[int], recv: list[int]):
    """One exchange through the two domains, returning the flat one's rows in its order."""
    intra, start, columns = domain.intra, domain.start, domain.column_size
    mine = domain.rank // intra
    table = _table(domain, send, rows.device)
    # A member publishes everything it addresses outside its own supernode; the widest one sets
    # the payload shape every member pads to.
    pad = -(-max(sum(s) - sum(s[g * intra : (g + 1) * intra]) for g, s in enumerate(table)) // _PAD_ALIGN)
    pad *= _PAD_ALIGN
    low, high = sum(send[:start]), sum(send[: start + intra])  # This supernode's own rows.
    # A member publishes its neighbours in peer order, so this supernode's run starts after the
    # peers below it that the member does not keep to itself.
    cursor = {
        group: sum(table[group][peer] for peer in range(start) if peer // intra != group)
        for group in range(columns)
        if group != mine
    }
    arrived = rows.new_zeros((columns, 0, *rows.shape[1:]))
    if pad:
        published = torch.cat((rows[:low], rows[high:]))
        padding = published.new_zeros((pad - published.size(0), *rows.shape[1:]))
        arrived = all_gather_single(torch.cat((published, padding)), 0, domain.column).reshape(
            columns, pad, *rows.shape[1:]
        )
    # Hand each supernode peer what it is owed: this rank's own rows first, then the column's
    # members in order, which is the order the receiving block expects.  A supernode owed
    # nothing still slices (an empty slice keeps the gather's adjoint in the graph, and its
    # collective has to run on every column member).
    out, sizes = [], []
    for offset in range(intra):
        peer = start + offset
        piece = [rows[sum(send[:peer]) : sum(send[:peer]) + send[peer]]]
        for group in range(columns):
            if group != mine:
                piece.append(arrived[group, cursor[group] : cursor[group] + table[group][peer]])
                cursor[group] += table[group][peer]
        out += piece
        sizes.append(sum(item.size(0) for item in piece))
    received = _FLAT(
        torch.cat(out),
        output_split_sizes=[sum(recv[group * intra + offset] for group in range(columns)) for offset in range(intra)],
        input_split_sizes=sizes,
        group=domain.inside,
    )
    # A slot delivers its own supernode's rows first, then the other supernodes ascending.
    pieces, at = {}, 0
    for offset in range(intra):
        for group in (mine, *[group for group in range(columns) if group != mine]):
            pieces[group, offset] = received[at : at + recv[group * intra + offset]]
            at += recv[group * intra + offset]
    # The flat exchange returns its peers' rows in rank order, which is supernode-major.
    return torch.cat([pieces[group, offset] for group in range(columns) for offset in range(intra)])
