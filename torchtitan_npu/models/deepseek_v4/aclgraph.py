# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-model preparation owned by the enabled ACLGraph pipeline."""

from functools import wraps
from types import MethodType

from .aclgraph_metadata import reuse_metadata_buffers


def _fixed_attention_metadata(self, positions, **kwargs):
    metadata = {}
    for layer in self.config.layers:
        inner = layer.attention.inner_attention
        key = inner.attention_metadata_key
        if key not in metadata:
            metadata[key] = inner.build_aclgraph_metadata(
                positions, document_capacity=self._aclgraph_document_capacity, **kwargs
            )
    return metadata


def prepare_aclgraph_model(model, *, document_capacity):
    """Adapt one built model before tracing, preserving parameter and mesh ownership."""
    if model.config.mtp_layers:
        raise ValueError("DeepSeek-V4 ACLGraph does not support MTP yet; set num_mtp_layers=0 or disable ACLGraph")
    if hasattr(model, "_static_metadata_buffers"):
        return
    if document_capacity < 1:
        raise ValueError("ACLGraph document capacity must be positive")
    for layer in model.config.layers:
        if not callable(getattr(layer.attention.inner_attention, "build_aclgraph_metadata", None)):
            raise ValueError("The Attention backend does not provide ACLGraph metadata preparation")
    original_preprocess = model.preprocess_inputs

    @wraps(original_preprocess)
    def preprocess(*args, **kwargs):
        inputs, labels, extra_kwargs = original_preprocess(*args, **kwargs)
        if "attention_metadata" in extra_kwargs:
            extra_kwargs = reuse_metadata_buffers(extra_kwargs, model._static_metadata_buffers)
        return inputs, labels, extra_kwargs

    model._aclgraph_document_capacity = document_capacity
    model._static_metadata_buffers = {}
    model._get_attention_metadata = MethodType(_fixed_attention_metadata, model)
    model.preprocess_inputs = preprocess
