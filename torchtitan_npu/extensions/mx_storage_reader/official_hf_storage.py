# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Official MXFP8 release directory: Engram row/block MX + generic fixed-format MX + plain tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.futures import Future

from .engram import EngramHuggingFaceStorageReader
from .hf_storage import MXHuggingFaceStorageReader

if TYPE_CHECKING:
    import os

    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.planner import LoadPlanner


class OfficialMxHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Read an official MXFP8 release directory: Engram MX, fixed-format MX pairs and plain tensors.

    Requests are routed per FQN to the sub-reader that produced the metadata
    entry, so scales may sit in other shards and tensors may be chunked. Tensors
    the target never references (MTP, vision) stay unread.
    """

    def __init__(
        self,
        path: str,
        *,
        table_shapes: dict[str, tuple[int, int]],
        expert_prefix_shapes: dict[str, tuple[int, ...]] | None = None,
        thread_count: int = 4,
        npu_max_inflight: int = 2,
    ) -> None:
        super().__init__(path=path, thread_count=thread_count)
        self._engram_reader = EngramHuggingFaceStorageReader(path, table_shapes=table_shapes, from_quantized=True)
        self._mx_reader = MXHuggingFaceStorageReader(path, thread_count=thread_count, npu_max_inflight=npu_max_inflight)
        self._expert_prefix_shapes = dict(expert_prefix_shapes or {})
        self._mx_tensors: dict[str, Any] = {}

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        super().reset(checkpoint_id)
        self._mx_reader.reset(checkpoint_id)
        self._engram_reader.reset(checkpoint_id)
        self._mx_tensors = {}

    def read_metadata(self) -> Metadata:
        engram_metadata = self._engram_reader.read_metadata()
        # Engram scale grids would be misread by the MX reader's fixed-format discovery.
        engram_keys = {name for name in self._engram_reader._raw_metadata if ".engram." in name}
        # Out-of-protocol pairs elsewhere in the directory are skipped, not rejected.
        mx_metadata = self._mx_reader.read_metadata(skip=engram_keys, defer_unsupported=True)
        self._mx_tensors = dict(self._mx_reader._mx_tensors)

        # dcp calls set_up_storage_reader only on this outer reader; the Engram sub-reader needs it for its guard.
        self._engram_reader.set_up_storage_reader(engram_metadata, False)

        merged = engram_metadata
        merged_storage = merged.storage_data
        for key, descriptor in self._mx_tensors.items():
            mx_entry = mx_metadata.state_dict_metadata.get(key)
            if mx_entry is None:
                raise AssertionError(f"MX reader lost track of {key}")
            merged.state_dict_metadata[key] = mx_entry
            merged.state_dict_metadata.pop(descriptor.scale_key, None)
        for index, info in mx_metadata.storage_data.items():
            if index.fqn in self._mx_tensors:
                merged_storage[index] = info
        self.storage_data = merged_storage
        self._apply_expert_prefixes(merged)
        return merged

    def _apply_expert_prefixes(self, metadata: Metadata) -> None:
        """Advertise shared router tensors at the model's (smaller) expert count.

        A shrunken debug model loads the leading experts of the official
        ``ffn.gate.*`` tensors; the trailing expert rows are never requested.
        Per-expert tensors already subset by name, so only these aggregated
        single-chunk tensors need the metadata clip.
        """
        for fqn, shape in self._expert_prefix_shapes.items():
            entry = metadata.state_dict_metadata.get(fqn)
            if entry is None:
                continue
            if not isinstance(entry, TensorStorageMetadata):
                raise ValueError(f"HF expert tensor {fqn} is not a plain tensor")
            size = tuple(entry.size)
            target = tuple(shape)
            if size[1:] != target[1:] or target[0] > size[0]:
                raise ValueError(f"HF expert tensor {fqn} with shape {size} cannot serve model shape {target}")
            if target[0] == size[0]:
                continue
            if len(entry.chunks) != 1:
                raise NotImplementedError(
                    f"Prefix load supports single-chunk expert tensors only, {fqn} has {len(entry.chunks)} chunks"
                )
            chunk = entry.chunks[0]
            entry.chunks = [ChunkStorageMetadata(torch.Size(chunk.offsets), torch.Size(target))]
            entry.size = torch.Size(target)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        mx_requests: list = []
        other_requests: list = []
        for req in plan.items:
            if req.storage_index.fqn in self._mx_tensors:
                mx_requests.append(req)
            else:
                other_requests.append(req)

        if mx_requests:
            self._mx_reader.read_data(LoadPlan(mx_requests), planner)
        if other_requests:
            self._engram_reader.read_data(LoadPlan(other_requests), planner)

        future: Future[None] = Future()
        future.set_result(None)
        return future
