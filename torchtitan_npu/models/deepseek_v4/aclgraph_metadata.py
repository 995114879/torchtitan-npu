# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed document frames and persistent storage for explicitly selected ACLGraph models."""

from dataclasses import replace

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass, tree_map_with_path

from .metadata import DeepSeekV4Metadata


def build_v4_metadata(positions, ratio, *, document_capacity):
    """Validate the original row, then add empty leading document slots for replay."""
    metadata = DeepSeekV4Metadata.from_positions(positions, ratio)
    boundaries = metadata.q.cu_seqlens
    documents = boundaries.numel() - 1
    if document_capacity < documents:
        raise ValueError(
            f"Packed batch has {documents} documents; fixed capacity is {document_capacity}. "
            "Check the dataloader's per_doc_alignment and the configured batch size and sequence length."
        )
    # Leading empty slots preserve token coordinates and safe overlap indices.
    padding = document_capacity - documents

    def pad(value):
        return torch.cat((value.new_zeros(padding), value))

    q = replace(metadata.q, cu_seqlens=pad(boundaries), seqused=pad(metadata.q.seqused))
    cmp_k = metadata.cmp_k
    if cmp_k is not None:
        cmp_k = replace(
            cmp_k,
            cu_seqlens=pad(cmp_k.cu_seqlens),
            seqused=pad(cmp_k.seqused),
            residual=None if cmp_k.residual is None else pad(cmp_k.residual),
        )
    return replace(metadata, cu_seqlens=q.cu_seqlens, q=q, swa_k=q, cmp_k=cmp_k)


@torch.no_grad()
def reuse_metadata_buffers(metadata, buffers):
    """Copy new per-geometry records into stable storage, preserving tensor aliases."""
    aliases = {}

    def reuse(path, value):
        if not isinstance(value, torch.Tensor):
            return value
        if id(value) in aliases:
            return aliases[id(value)]
        saved = buffers.get(path)
        if saved is None or (saved.shape, saved.dtype, saved.device) != (value.shape, value.dtype, value.device):
            saved = buffers[path] = torch.empty_like(value)
            torch._dynamo.mark_static_address(saved, guard=True)
        saved.copy_(value)
        aliases[id(value)] = saved
        return saved

    return tree_map_with_path(reuse, metadata)


if DeepSeekV4Metadata not in SUPPORTED_NODES:
    register_dataclass(
        DeepSeekV4Metadata, serialized_type_name=f"{DeepSeekV4Metadata.__module__}.{DeepSeekV4Metadata.__name__}"
    )
