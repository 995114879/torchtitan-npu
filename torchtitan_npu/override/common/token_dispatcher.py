# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NPU fused MoE token dispatcher.

Replaces the pure-PyTorch local token reordering (argsort + index_select) in
``AllToAllTokenDispatcher._local_reorder`` and the combine-time
score multiply + ``deterministic_scatter_add`` with the fused NPU kernels
``npu_moe_token_permute`` / ``npu_moe_token_unpermute``. For EP>1, the
rank-major -> expert-major reroute after all-to-all is replaced by
``npu_moe_re_routing``.

Semantics of the upstream metadata fields under this override:

- ``token_indices_experts_sorted_N`` stores the *inverse* permutation Q
  returned by ``npu_moe_token_permute`` (the double-argsorted forward
  permutation), NOT the forward argsort P of the upstream ``_local_reorder``.
  Q is exactly the ``sorted_indices`` input consumed by
  ``npu_moe_token_unpermute``.
- ``topk_scores_experts_sorted_N`` stores the original ``topk_scores_TK``
  ``(T, K)`` tensor, NOT the per-row gathered scores of the upstream
  ``_local_reorder``. ``npu_moe_token_unpermute`` expects ``probs`` shaped
  ``(T, K)``.
- ``routed_scores_R`` is populated when ``absorb_router_scores=True``. In that
  case scores are permuted alongside tokens, transported through EP, and
  consumed by grouped experts before the down projection; combine then skips
  the post-W2 multiply.

The EP=1 path no longer falls back to ``LocalTokenDispatcher``: dispatch is a
local permute and combine is a local unpermute, both fused into the NPU
kernels.

Precision: when pre-W2 absorption is disabled, ``npu_moe_token_unpermute``
multiplies the routing scores and sums over top-K internally in fp32
regardless of the input dtypes, and returns an output matching the first
argument's dtype. This is at least as precise as the upstream
fp32-score-multiply + bf16 ``scatter_add``.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch_npu
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import current_spmd_mesh, maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.token_dispatcher import BaseEPTokenDispatcher

from torchtitan_npu.extensions.torchtitan.ep_process_group import get_deepep_group
from torchtitan_npu.ops.ascendc.moe_re_routing import npu_moe_re_routing
from torchtitan_npu.ops.ascendc.moe_token_unpermute import npu_moe_token_unpermute
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)

if TYPE_CHECKING:
    from cann_ops_transformer import ElasticBuffer
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan_npu.ops.ascendc import moonep
    from torchtitan_npu.ops.ascendc.deepep import (
        DispatchHandle,
        ElasticBufferHandle,
    )


def _stable_argsort_expert_ids(values_1d: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Exact stable argsort for expert ids in ``[0, num_experts)``.

    ``torch.argsort(..., stable=True)`` does not survive dynamo fake-eval
    under the spmd_types patch stack (aten sort kernels fail inside
    FakeTensor evaluation).  The composite key ``expert_id * N + position``
    is strictly ordered by ``(expert, position)``, so an ascending
    full-width topk over the keys yields the identical stable permutation
    in O(N) time and memory (no ``[N, E]`` one-hot).
    """
    n = values_1d.numel()
    keys = values_1d.to(torch.int64) * n + torch.arange(n, device=values_1d.device)
    return keys.topk(n, largest=False, sorted=True).indices


def _compile_friendly_local_reorder(
    self,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Traceable replacement for the dispatcher ``_local_reorder``.

    Upstream uses ``torch.argsort(stable=True)``, which does not trace
    under fullgraph=True with the spmd_types stack.  One shared body for
    the Local and AllToAll dispatchers (identical upstream); the returned
    permutation matches upstream exactly, so eager and compiled runs stay
    bit-exact.
    """
    token_indices_experts_sorted_N = _stable_argsort_expert_ids(
        topk_expert_ids_TK.view(-1).to(torch.int64), self.num_experts
    )
    topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[token_indices_experts_sorted_N]
    token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
    routed_input_ND = x_TD[token_indices_experts_sorted_N]
    return (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    )


def apply_compile_friendly_local_reorder() -> None:
    """Opt the dispatchers into the traceable stable reorder (compile only).

    Called by the V4.1 compile assembly (models/deepseek_v4_1/parallelize)
    when torch.compile is enabled; eager runs keep the upstream
    ``torch.argsort(stable=True)`` helper.
    """
    LocalTokenDispatcher._local_reorder = _compile_friendly_local_reorder
    AllToAllTokenDispatcher._local_reorder = _compile_friendly_local_reorder


class AscAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher for EP>1 with NPU fused permute/unpermute kernels.

    Overrides only ``dispatch`` and ``combine``. The all-to-all exchange
    helpers (``_token_count_exchange``, ``_sync_token_count_exchange``,
    ``_dispatch_token_exchange``, ``_combine_token_exchange``) and
    ``_permute`` / ``_unpermute`` (rank-major <-> expert-major) are inherited
    from ``AllToAllTokenDispatcher``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def _dispatch_activation_exchange(
        self,
        routed_input_ND: torch.Tensor,
        pg: "ProcessGroup",
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        """Exchange activations; router scores use ``_dispatch_token_exchange``."""
        return self._dispatch_token_exchange(routed_input_ND, pg, output_splits, input_splits)

    def _reroute_activation(self, routed_input_RD, expert_token_num_per_rank, routed_scores_rank_major_R):
        return npu_moe_re_routing(
            routed_input_RD,
            expert_token_num_per_rank,
            per_token_scales=routed_scores_rank_major_R,
        )

    def _combine_activation_exchange(
        self,
        routed_output_RD,
        restore_indices,
        pg,
        input_splits,
        output_splits,
    ):
        """Restore rank-major order and exchange expert outputs."""
        routed_output_rank_major = npu_moe_token_unpermute(
            routed_output_RD,
            restore_indices,
            probs=None,
        )
        return self._combine_token_exchange(
            routed_output_rank_major,
            pg,
            input_splits,
            output_splits,
        )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        ``npu_moe_token_permute`` replaces ``_local_reorder``: it fuses the
        argsort + ``index_select`` into one kernel and returns the inverse
        permutation Q instead of the forward argsort P.

        When ``ep_mesh`` is None (EP=1), returns the local permute result
        directly with ``LocalDispatchMetadata`` — no all-to-all.

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        # Fused NPU permute: replaces _local_reorder (argsort + index_select).
        # sorted_indices_N is the inverse permutation Q; npu_moe_token_unpermute
        # consumes Q directly in combine.
        routed_input_ND, sorted_indices_N = torch_npu.npu_moe_token_permute(
            x_TD,
            topk_expert_ids_TK,
        )

        if self.absorb_router_scores:
            routed_scores_ND, _ = torch_npu.npu_moe_token_permute(
                topk_scores_TK.reshape(-1, 1),
                topk_expert_ids_TK.reshape(-1, 1),
            )
            routed_scores_ND = routed_scores_ND.reshape(-1)
        else:
            routed_scores_ND = None

        # EP=1: local dispatch — no all-to-all needed.
        if self.ep_mesh is None:
            return (
                routed_input_ND,
                num_local_tokens_per_expert_E,
                LocalDispatchMetadata(
                    token_indices_experts_sorted_N=sorted_indices_N,
                    topk_scores_experts_sorted_N=topk_scores_TK,
                    routed_scores_R=routed_scores_ND,
                ),
            )

        ep_size = self.ep_mesh.size()

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():  # sparse mesh reinterpret
            for axis in ["dp", "cp", "tp"]:
                spmd.mutate_type(num_local_tokens_per_expert_E, axis, src=spmd.P, dst=spmd.V)

        # generate the input splits and output splits for all-to-all
        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
                    num_local_tokens_per_expert_E, spmd.current_mesh()
                )
                routed_input_ND = spmd.reinterpret_mesh(routed_input_ND, spmd.current_mesh())
                if routed_scores_ND is not None:
                    routed_scores_ND = spmd.reinterpret_mesh(routed_scores_ND, spmd.current_mesh())

            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E,
                    pg,
                    ep_size,
                )
                (
                    num_global_tokens_per_local_expert_E,
                    input_splits_list,
                    output_splits_list,
                ) = self._sync_token_count_exchange(
                    num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e,
                    ep_size,
                )

            routed_input_RD = self._dispatch_activation_exchange(
                routed_input_ND,
                pg,
                output_splits_list,
                input_splits_list,
            )
            routed_scores_rank_major_R = (
                self._dispatch_token_exchange(
                    routed_scores_ND.reshape(-1, 1),
                    pg,
                    output_splits_list,
                    input_splits_list,
                ).reshape(-1)
                if routed_scores_ND is not None
                else None
            )
            # Fused CANN reroute replaces inherited _permute. The returned
            # restore indices are consumed by the matching unpermute op in
            # combine and by the reroute autograd formula.
            expert_token_num_per_rank = num_global_tokens_per_local_expert_E.view(
                ep_size,
                -1,
            )
            (
                routed_input_RD,
                permuted_scales,
                restore_indices,
                num_global_tokens_per_local_expert_e,
            ) = self._reroute_activation(
                routed_input_RD,
                expert_token_num_per_rank,
                routed_scores_rank_major_R,
            )
            routed_scores_R = permuted_scales if routed_scores_ND is not None else None

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=sorted_indices_N,
            topk_scores_experts_sorted_N=topk_scores_TK,
            input_shape=routed_input_RD.shape,
            permuted_indices=restore_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            routed_scores_R=routed_scores_R,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + fused unpermute.

        ``npu_moe_token_unpermute`` replaces the upstream score multiply +
        ``deterministic_scatter_add``: the kernel scatters each routed row
        back to its original token, multiplies by the routing score, and sums
        over top-K, accumulating internally in fp32. The output dtype follows
        the first argument.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        probs = (
            torch.ones_like(metadata.topk_scores_experts_sorted_N)
            if self.absorb_router_scores and metadata.routed_scores_R is not None
            else metadata.topk_scores_experts_sorted_N
        )

        # EP=1: fused NPU unpermute — no all-to-all to reverse.
        if self.ep_mesh is None:
            return npu_moe_token_unpermute(
                routed_output_RD,
                metadata.token_indices_experts_sorted_N,
                probs=probs,
            )

        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            routed_output_RD = self._combine_activation_exchange(
                routed_output_RD,
                metadata.permuted_indices,
                pg,
                metadata.input_splits,
                metadata.output_splits,
            )

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            # dense mesh reinterpret
            routed_output_RD = spmd.reinterpret_mesh(routed_output_RD, spmd.current_mesh())

        # Fused NPU unpermute: scatter + score x probs + sum over top-K.
        return npu_moe_token_unpermute(
            routed_output_RD,
            metadata.token_indices_experts_sorted_N,
            probs=probs,
        )


@dataclass(slots=True)
class DispatchState:
    """Dispatcher-owned state required by the ElasticBuffer combine call."""

    recv_scores: torch.Tensor
    handle: "DispatchHandle"
    routed_scores_R: torch.Tensor | None = None  # noqa: N815


class AscDeepEPTokenDispatcher(DeepEPTokenDispatcher):
    """Use a process-global CANN ElasticBuffer on its exclusive DeepEP communication domain."""

    _buffers: ClassVar[dict[tuple["ProcessGroup", int, int, int, int], "ElasticBuffer"]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(DeepEPTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        # DeepEPTokenDispatcher.__init__ imports the CUDA DeepEP package.  The
        # NPU implementation uses CANN ElasticBuffer instead, so initialize the
        # shared EP state directly while retaining DeepEP's Config contract.
        BaseEPTokenDispatcher.__init__(self, config)
        self.absorb_router_scores = config.absorb_router_scores
        # Keep the generic EP extension from replacing the source mesh before
        # this dispatcher obtains its buffer-configured DeepEP group.
        self._uses_custom_ep_process_group = True
        self.hidden_dim = cast("int", config.hidden_dim)
        self.num_max_tokens_per_rank = cast("int", config.num_max_tokens_per_rank)
        self._deepep_group: ProcessGroup
        self._deepep_world_size: int
        self._deepep_buffer_handle: ElasticBufferHandle

        from torchtitan_npu.ops.ascendc import deepep  # noqa: F401

    @classmethod
    def get_buffer_and_group(
        cls,
        ep_mesh: "DeviceMesh",
        *,
        hidden_dim: int,
        num_max_tokens_per_rank: int,
        top_k: int,
    ) -> tuple["ElasticBuffer", "ProcessGroup"]:
        """Reuse buffers with the same communication group, device and capacity."""
        deepep_group = get_deepep_group(ep_mesh)
        key = (deepep_group, torch_npu.npu.current_device(), hidden_dim, num_max_tokens_per_rank, top_k)
        if key in cls._buffers:
            return cls._buffers[key], deepep_group

        from cann_ops_transformer import ElasticBuffer

        buffer = ElasticBuffer(
            deepep_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden_dim,
            num_topk=top_k,
        )
        cls._buffers[key] = buffer
        return buffer, deepep_group

    def init_buffer(self) -> None:
        """Initialize the process-global ElasticBuffer on the dispatcher's exclusive EP group."""
        ep_mesh = self.ep_mesh
        if ep_mesh is None:
            raise ValueError("NPU DeepEP requires expert_parallel_degree > 1.")
        hidden_dim = cast("int", self.hidden_dim)
        num_max_tokens_per_rank = self.num_max_tokens_per_rank

        buffer, deepep_group = self.get_buffer_and_group(
            ep_mesh,
            hidden_dim=hidden_dim,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            top_k=self.top_k,
        )
        from torchtitan_npu.ops.ascendc.deepep import ElasticBufferHandle

        self._deepep_buffer_handle = ElasticBufferHandle(buffer)
        self._deepep_group = deepep_group
        self._deepep_world_size = deepep_group.size()

    def _dispatch_activation_exchange(self, x_td, topk_expert_ids_tk, topk_scores_tk, *, num_experts: int):
        """Communication seam shared by high-precision and quantized dispatchers."""
        recv_payload, recv_scores, counts, handle = torch.ops.deepep.dispatch(
            [x_td],
            topk_expert_ids_tk,
            topk_scores_tk,
            num_experts,
            num_experts // self._deepep_world_size,
            self._deepep_buffer_handle,
        )
        return recv_payload[0], recv_scores, counts, handle

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, "DispatchState"]:
        del num_local_tokens_per_expert_E
        return self.dispatch_experts(x_TD, topk_scores_TK, topk_expert_ids_TK, num_experts=self.num_experts)

    def dispatch_experts(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        *,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, "DispatchState"]:
        """Use the existing transport for real or virtual expert identifiers."""
        topk_expert_ids_TK = topk_expert_ids_TK.to(torch.int32).contiguous()
        topk_scores_TK = topk_scores_TK.float().contiguous()
        (
            hidden_states_RD,
            recv_scores,
            num_global_tokens_per_local_expert_e,
            handle,
        ) = self._dispatch_activation_exchange(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_experts=num_experts,
        )
        return (
            hidden_states_RD,
            num_global_tokens_per_local_expert_e,
            DispatchState(
                recv_scores=recv_scores,
                handle=handle,
                routed_scores_R=recv_scores if self.absorb_router_scores else None,
            ),
        )

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: "DispatchState",
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        # In cached-dispatch mode, combine_backward calls
        # ElasticBuffer.dispatch(handle=...), which requires topk_idx/topk_weights=None
        # and cannot consume grad_recv_scores as new routing weights. Keep score
        # weighting outside combine so autograd first computes grad(recv_scores);
        # dispatch_backward then passes it to ElasticBuffer.combine(..., topk_weights=...)
        # to recover grad(topk_weights).
        weighted_states = routed_output_RD
        if not self.absorb_router_scores:
            weighted_states = routed_output_RD * metadata.recv_scores.to(routed_output_RD.dtype).reshape(-1, 1)

        return self._combine_activation_exchange(weighted_states, metadata, x_TD.shape[0])

    def _combine_activation_exchange(self, routed_output_rd, metadata, num_tokens):
        """Combine already weighted expert outputs; metadata supplies routing only."""
        combined_td = torch.ops.deepep.combine(
            routed_output_rd,
            metadata.handle,
            num_tokens,
            # Preserve the dispatch row-count source for compiled backward.
            metadata.recv_scores,
            self._deepep_buffer_handle,
        )
        return combined_td


@dataclass(kw_only=True, slots=True)
class MoonEPDispatchState(DispatchState):
    """DeepEP dispatch state, MoonEP plan and this call's staged expert weights."""

    virtual_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    staged_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class AscMoonEPTokenDispatcher(AscDeepEPTokenDispatcher):
    """Add MoonEP planning and weight transport to the NPU DeepEP path."""

    @dataclass(kw_only=True, slots=True)
    class Config(AscDeepEPTokenDispatcher.Config):
        pass

    _contexts: ClassVar[dict[tuple["ProcessGroup", int, int, int], "moonep.MoonEPContextHandle"]] = {}

    def __init__(self, config: Config):
        super().__init__(config)
        self.expert_hidden_dim = config.expert_hidden_dim
        self.context: moonep.MoonEPContextHandle | None = None

    def init_buffer(self) -> None:
        if self.ep_mesh is None or self.ep_mesh.size() <= 1:
            raise ValueError("MoonEP requires expert_parallel_degree > 1.")
        hidden_dim = self.hidden_dim
        expert_hidden_dim = self.expert_hidden_dim
        if hidden_dim is None or expert_hidden_dim is None:
            raise ValueError("MoonEP requires hidden_dim and expert_hidden_dim to size its communication buffer.")
        width = hidden_dim * expert_hidden_dim
        num_ranks = self.ep_mesh.size()
        super().init_buffer()
        key = (
            self._deepep_group,
            torch_npu.npu.current_device(),
            self.num_experts,
            width,
        )
        if key not in self._contexts:
            from ops import moonep_context  # pyrefly: ignore [missing-import]
            from torch_npu._C._distributed_c10d import ProcessGroupHCCL  # pyrefly: ignore [missing-import]

            from torchtitan_npu.ops.ascendc import moonep

            num_local_experts = self.num_experts // num_ranks
            # Reserve 3 weight groups (w1/gate, w3/up, w2/down), 1 reusable communication slot,
            # and gradient storage sized for FP32 (4 bytes/element).
            need = moonep_context.HEAP_BASE + moonep_context.reduce_region_end(
                num_ranks, num_local_experts, width, num_groups=3, slots=1, grad_bytes_per_elem=4
            )
            options = ProcessGroupHCCL.Options()
            # The backend's memory test reports usable heap bytes at about twice
            # the configured MiB; divide by two and keep 2 MiB of config headroom.
            options.hccl_config = {"hccl_buffer_size": math.ceil(need / (2 * 1024**2)) + 2}
            # Weight transport must not share ElasticBuffer's communicator/heap.
            context_group = dist.new_group(
                ranks=dist.get_process_group_ranks(self._deepep_group),
                backend="hccl",
                group_desc="asc_moonep_weights",
                use_local_synchronization=True,
                pg_options=options,
            )
            context = moonep_context.create_moonep_context(group=context_group)
            # ElasticBuffer already exists; context creation checked isolation,
            # and this private weight group is never used by ElasticBuffer.
            # Skip plan's redundant GC scan: isinstance on an expired weak proxy
            # can raise ReferenceError during compiled execution.
            context.eb_scan_done = True
            moonep_context.check_heap_capacity(context, need, "MoonEP weights and FP32 gradient slots")
            # The backend context retains context_group through its group attribute.
            self._contexts[key] = moonep.MoonEPContextHandle(context)
        self.context = self._contexts[key]

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, MoonEPDispatchState]:
        """Plan virtual experts, prefetch weights, then dispatch tokens."""
        from torchtitan_npu.ops.ascendc import moonep

        assert self.context is not None and self.ep_mesh is not None
        virtual_ids, cu_seqlens, experts_to_copy = moonep.plan(
            num_local_tokens_per_expert_E,
            topk_expert_ids_TK,
            self.ep_mesh.size(),
            self.context,
        )
        # Keep prefetch and token dispatch separate for future stream overlap.
        with maybe_set_sparse_mesh():
            staged_weights = self.prefetch_expert_weights(expert_weights, experts_to_copy)
        recv, counts, metadata = super().dispatch_experts(
            x_TD,
            topk_scores_TK,
            virtual_ids,
            num_experts=2 * self.num_experts,
        )
        return (
            recv,
            counts,
            MoonEPDispatchState(
                virtual_ids=virtual_ids,
                cu_seqlens=cu_seqlens,
                experts_to_copy=experts_to_copy,
                staged_weights=staged_weights,
                recv_scores=metadata.recv_scores,
                handle=metadata.handle,
                routed_scores_R=metadata.routed_scores_R,
            ),
        )

    def prefetch_expert_weights(
        self,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        experts_to_copy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefetch the supplied BF16 local SPMD gate/up/down weights."""
        from torchtitan_npu.ops.ascendc import moonep

        assert self.context is not None
        return moonep.prefetch_weight(
            weights[0].contiguous(),
            weights[1].contiguous(),
            weights[2].contiguous(),
            experts_to_copy,
            self.context,
        )


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="NPU fused npu_moe_token_permute/unpermute for MoE dispatch/combine",
)
def asc(
    cfg: AllToAllTokenDispatcher.Config,
) -> AscAllToAllTokenDispatcher.Config:
    return derive(cfg, AscAllToAllTokenDispatcher.Config)


@override(
    target=DeepEPTokenDispatcher.Config,
    description=("Use cann_ops_transformer.ElasticBuffer for DeepEP dispatch/combine on NPU"),
)
def asc_deepep(
    cfg: DeepEPTokenDispatcher.Config,
) -> AscDeepEPTokenDispatcher.Config:
    """Select CANN without discarding an already-specialized NPU dispatcher."""
    # TorchAO converts before Trainer applies overrides. Re-deriving an MXFP8
    # config to the base NPU config would drop its quantized dispatch/combine.
    if isinstance(cfg, AscDeepEPTokenDispatcher.Config):
        return cfg
    return derive(cfg, AscDeepEPTokenDispatcher.Config)


@override(
    target=DeepEPTokenDispatcher.Config,
    description=(
        "Use MoonEP planning and weight transport on the NPU DeepEP path; "
        "requires the existing EP process-group integration"
    ),
)
def asc_moonep(cfg: DeepEPTokenDispatcher.Config) -> AscMoonEPTokenDispatcher.Config:
    """Select MoonEP transport; the expert implementation is chosen separately.

    MoonEP does not support PP, TP or quantized dispatch/combine.
    """
    # TorchAO converts before overrides; derive would discard its quantized transport.
    if hasattr(cfg, "mxfp8_quant_params"):
        raise ValueError("MoonEP does not support quantized dispatch/combine; disable all_to_all_quant_format.")
    return derive(cfg, AscMoonEPTokenDispatcher.Config)
