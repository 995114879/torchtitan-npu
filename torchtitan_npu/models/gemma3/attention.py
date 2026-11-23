# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 RoPE and scaling on TorchTitan grouped-query attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchtitan.models.common.attention import FlexAttention, GQAttention
from torchtitan.models.common.rope import CosSinRoPE

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


class Gemma3RotaryEmbedding(CosSinRoPE):
    """Gemma3 RoPE with local/global frequency rules and compact buffers."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        scaling_factor: float = 1.0

    def __init__(self, config: Config):
        if config.dim <= 0 or config.dim % 2:
            raise ValueError(f"head_dim must be a positive even integer, got {config.dim}")
        if config.scaling != "none":
            raise ValueError("Gemma3 RoPE uses scaling_factor; upstream Llama/YaRN scaling is unsupported")
        self.head_dim = config.dim
        self.theta = float(config.theta)
        self.scaling_factor = float(config.scaling_factor)
        if not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0:
            raise ValueError("RoPE scaling_factor must be positive and finite")
        if isinstance(config.max_seq_len, bool) or not isinstance(config.max_seq_len, int) or config.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be a positive integer, got {config.max_seq_len!r}")
        super().__init__(config)

    def _precompute_cache(self) -> torch.Tensor:
        """Keep inverse frequencies in a non-persistent module buffer."""

        # ``RoPE._init_self_buffers`` supplies the requested device through a
        # ``torch.device`` context; leaving factory calls device-less also
        # handles ``to_empty`` targets and explicit CPU reinitialization.
        inverse_frequency = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        inverse_frequency = inverse_frequency / self.scaling_factor
        if "inverse_frequency" in self._buffers:
            self.inverse_frequency = inverse_frequency
        else:
            self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        # RoPE.init_states() expects a cache buffer.  Gemma3 computes the
        # position-specific cosine/sine values lazily in _reshape_cache().
        return torch.empty((0, 2 * self.head_dim), dtype=torch.float32)

    def _reshape_cache(self, query: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Build the upstream cosine/sine cache for the requested positions."""

        batch_size, sequence_length = query.shape[:2]
        if positions is None:
            positions = torch.arange(sequence_length, device=query.device, dtype=torch.long).expand(batch_size, -1)
        elif positions.shape != (batch_size, sequence_length):
            raise ValueError("positions must match query batch and sequence dimensions")
        positions = positions.to(device=query.device, dtype=torch.float32)
        inverse_frequency = self.inverse_frequency.to(device=query.device)
        frequencies = torch.einsum("bs,d->bsd", positions, inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return cache.unsqueeze(2)


def apply_gemma3_softcap(
    scores: torch.Tensor,
    softcap: float | None,
) -> torch.Tensor:
    """Apply the same optional tanh cap to attention scores or output logits."""

    if softcap is None:
        return scores
    if not isinstance(softcap, (int, float)) or softcap <= 0:
        raise ValueError(f"softcap must be positive or None, got {softcap!r}")
    cap = float(softcap)
    return cap * torch.tanh(scores / cap)


class Gemma3FlexAttention(FlexAttention):
    """Supply Gemma3's optional score cap to the upstream FlexAttention backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        softcap: float

    def __init__(self, config: Config):
        super().__init__(config)
        if config.softcap <= 0:
            raise ValueError("attention softcap must be positive")
        cap = config.softcap

        def cap_score(score, batch, head, query_index, key_index):
            return apply_gemma3_softcap(score, cap)

        self.score_mod = cap_score

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q_blnh: torch.Tensor,
        k_blnh: torch.Tensor,
        v_blnh: torch.Tensor,
        *,
        attention_masks: BlockMask,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            q_blnh,
            k_blnh,
            v_blnh,
            attention_masks=attention_masks,
            score_mod=self.score_mod,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )


class Gemma3Attention(GQAttention):
    """Use the shared GQA pipeline with Gemma3's query scaling."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        n_kv_heads: int | None
        head_dim: int | None
        query_pre_attn_scalar: int | float
        layer_type: str
        sliding_window: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.scaling = config.query_pre_attn_scalar**-0.5
