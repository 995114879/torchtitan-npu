# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 text decoder with shared full/sliding masks and tied weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, cast

import torch
from torch.nn.attention.flex_attention import or_masks
from torchtitan.models.common.attention import (
    AttentionMasksType,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.protocols.module import Module

from .attention import Gemma3Attention, apply_gemma3_softcap

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


class Gemma3ScaledEmbedding(Embedding):
    """Gemma token embedding with the model-width scale applied at runtime."""

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        padding_idx: int | None = None

    embed_scale: torch.Tensor

    def __init__(self, config: Config):
        super().__init__(config)
        # TorchTitan 0.3.0 does not pass padding_idx to nn.Embedding.
        self.padding_idx = config.padding_idx
        self._fill_padding_idx_with_zero()
        self.register_buffer(
            "embed_scale",
            torch.tensor(self.embedding_dim**0.5),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Restore the runtime scale after TorchTitan materializes the module."""

        if buffer_device is None:
            buffer_device = self.embed_scale.device
        self.embed_scale = torch.full(
            (),
            self.embedding_dim**0.5,
            dtype=self.embed_scale.dtype,
            device=buffer_device,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore [bad-param-name-override]
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3RMSNorm(RMSNorm):
    """Gemma's zero-centered RMSNorm with float32 normalization statistics."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        eps: float = 1e-6

    def reset_parameters(self) -> None:
        # Gemma stores a zero-centered scale and applies (1 + weight).
        torch.nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + cast("float", self.eps))
        normalized = normalized * (1.0 + self.weight.float())
        return normalized.type_as(x)


class Gemma3MLP(FeedForward):
    """GELU-tanh gated MLP using Gemma's three bias-free projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        activation: GELU.Config
        # Satisfy the patched parent constructor without exposing SwiGLU options.
        swiglu_limit: float = field(default=0.0, init=False)

    def __init__(self, config: Config):
        if config.activation.approximate != "tanh":
            raise ValueError("Gemma3MLP requires GELU approximate='tanh'")
        super().__init__(config)
        self.activation = config.activation.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class Gemma3DecoderLayer(Module):
    """Gemma3 decoder layer with four norms and two residual sublayers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Gemma3Attention.Config
        feed_forward: Gemma3MLP.Config
        input_layernorm: RMSNorm.Config
        post_attention_layernorm: RMSNorm.Config
        pre_feedforward_layernorm: RMSNorm.Config
        post_feedforward_layernorm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_mask_key = (
            "sliding_window_mask" if config.attention.layer_type == "sliding_attention" else "basic_mask"
        )
        self.feed_forward = config.feed_forward.build()
        self.input_layernorm = config.input_layernorm.build()
        self.post_attention_layernorm = config.post_attention_layernorm.build()
        self.pre_feedforward_layernorm = config.pre_feedforward_layernorm.build()
        self.post_feedforward_layernorm = config.post_feedforward_layernorm.build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: dict[str, BlockMask],
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_masks[self.attention_mask_key],
            positions,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma3Model(Decoder):
    """Gemma3 model hierarchy with a causal-LM output contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[Gemma3DecoderLayer.Config]
        max_position_embeddings: int = 131072
        final_logit_softcapping: float | None = None

        def validate_sequence_length(self, sequence_length: int) -> None:
            if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
                raise ValueError("sequence length must be a positive integer")
            if sequence_length < 1:
                raise ValueError(f"sequence length must be positive, got {sequence_length}")
            if sequence_length > self.max_position_embeddings:
                raise ValueError(
                    "sequence length exceeds max_position_embeddings: "
                    f"{sequence_length} > {self.max_position_embeddings}"
                )

        def update_from_config(self, *, config, **kwargs) -> None:
            self.validate_sequence_length(config.training.seq_len)
            parallelism = config.parallelism
            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError("Weight tying is not supported with Pipeline Parallel.")
            if parallelism.tensor_parallel_degree > 1:
                if self.layers[0].attention.n_heads % parallelism.tensor_parallel_degree:
                    raise ValueError("tensor_parallel_degree must divide num_attention_heads")
                if self.layers[0].attention.n_kv_heads % parallelism.tensor_parallel_degree:
                    raise ValueError("tensor_parallel_degree must divide num_key_value_heads")
            sequence_length = config.training.seq_len
            self.layers = [
                dataclass_replace(
                    layer,
                    attention=dataclass_replace(
                        layer.attention,
                        rope=dataclass_replace(layer.attention.rope, max_seq_len=sequence_length),
                    ),
                )
                for layer in self.layers
            ]

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, int]:
            self.validate_sequence_length(seq_len)
            from torchtitan.models.utils import get_dense_model_nparams_and_flops

            head_dim = self.layers[0].attention.head_dim
            assert head_dim is not None
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * head_dim,
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def _retie_weights(self) -> None:
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._retie_weights()
        return result

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        """Initialize the module tree using TorchTitan's recursive contract."""
        self._retie_weights()
        super().init_states(buffer_device=buffer_device)

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, BlockMask]:
        """Build one full and one sliding BlockMask per batch, shared by layers."""
        config = cast("Gemma3Model.Config", self.config)
        if padding_mask is None:
            document_mask = get_efficient_causal_mask_mod_for_packed_document(positions)
        else:
            # Padding resets positions too, but must not start a new document.
            document_positions = torch.where(padding_mask, positions, 1)
            document_mask = get_document_mask_mod(document_positions)

        base_mods = [get_causal_mask_mod(), document_mask]
        if padding_mask is not None:
            valid_padding_mask = padding_mask

            def valid_key(batch, head, query_index, key_index):
                return valid_padding_mask[batch, key_index] & (
                    positions[batch, key_index] <= positions[batch, query_index]
                )

            base_mods.append(valid_key)

        masks: dict[str, BlockMask] = {}
        for layer in config.layers:
            attention = layer.attention
            is_sliding = attention.layer_type == "sliding_attention"
            mask_key = "sliding_window_mask" if is_sliding else "basic_mask"
            if mask_key in masks:
                continue
            mask_mods = list(base_mods)
            if is_sliding:
                window_mask = get_sliding_window_mask_mod(attention.sliding_window)
                if padding_mask is not None:

                    def padded_query(batch, head, query_index, key_index):
                        return ~valid_padding_mask[batch, query_index]

                    # Ignored query rows retain their document's valid history.
                    window_mask = or_masks(window_mask, padded_query)
                mask_mods.append(window_mask)
            masks[mask_key] = self._create_flex_attention_mask(positions, attention, mask_mods)
        return masks

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = cast("Gemma3Model.Config", self.config)
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [batch, sequence], got {tuple(tokens.shape)}")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"tokens must contain integer token IDs, got {tokens.dtype}")
        sequence_length = tokens.shape[1]
        config.validate_sequence_length(sequence_length)
        if padding_mask is not None:
            if padding_mask.shape != tokens.shape:
                raise ValueError("padding_mask must match tokens [batch, sequence]")
            padding_mask = padding_mask.to(device=tokens.device, dtype=torch.bool)

        if positions is None:
            positions = torch.arange(
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.long,
            ).expand(tokens.shape[0], -1)
        elif positions.ndim != 2 or positions.shape != tokens.shape:
            raise ValueError("positions must match tokens [batch, sequence]")
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions, padding_mask=padding_mask)
        hidden_states = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            hidden_states = layer(
                hidden_states,
                positions=positions,
                attention_masks=cast("dict[str, BlockMask]", attention_masks),
            )
        hidden_states = self.norm(hidden_states)
        logits = hidden_states if self._skip_lm_head else self.lm_head(hidden_states)
        if self._skip_lm_head:
            return logits
        return apply_gemma3_softcap(
            logits,
            config.final_logit_softcapping,
        )
