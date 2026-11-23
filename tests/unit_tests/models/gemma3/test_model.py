# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask
from torchtitan.models.common.attention import FlexAttention
from torchtitan.protocols.module import Module

from torchtitan_npu.models.gemma3 import _make_gemma3_layer_config
from torchtitan_npu.models.gemma3.attention import Gemma3RotaryEmbedding
from torchtitan_npu.models.gemma3.model import Gemma3MLP, Gemma3RMSNorm

from .fixtures import recipe_model_config, recipe_text_config


def test_recipe_loss_backward_updates_shared_embedding_once(recipe, model):
    tokens = torch.tensor([[2, 7, 3, 5, 11, 13, 17]])
    labels = torch.tensor([[-100, -100, 5, 11, 13, 17]])
    logits = model(tokens[:, :-1])
    loss, _ = recipe.loss.build(compile_config=recipe.compile)(logits, labels, (labels != -100).sum())
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert logits.shape == (1, 6, model.config.vocab_size)
    assert logits.dtype == torch.float32

    optimizer = replace(recipe.optimizer, implementation="for-loop").build(model_parts=[model])
    before = model.tok_embeddings.weight.detach().clone()
    loss.backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.lm_head.weight.grad is model.tok_embeddings.weight.grad
    parameters = [
        p
        for item in optimizer
        for group in item.param_groups
        for p in group["params"]
    ]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    optimizer.step()
    assert not torch.equal(before, model.tok_embeddings.weight)
    assert model.lm_head.weight is model.tok_embeddings.weight


def test_positions_must_match_tokens_at_model_boundary(model):
    tokens = torch.tensor([[2, 7, 9, 11]])
    with pytest.raises(ValueError, match="positions must match"):
        model(tokens, positions=torch.tensor([[0, 1, 2]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_recipe_materialization_restores_embedding_scale_and_shared_weight(dtype):
    config = recipe_model_config(pad_token_id=3)
    with torch.device("meta"):
        model = config.build()
    assert model.lm_head.weight is model.tok_embeddings.weight
    model.to(dtype)
    assert model.lm_head.weight is model.tok_embeddings.weight
    model.to_empty(device="cpu")
    assert model.lm_head.weight is model.tok_embeddings.weight
    model.tok_embeddings.embed_scale.fill_(float("nan"))

    model.init_states(buffer_device=torch.device("cpu"))

    embedding = model.tok_embeddings
    assert embedding.embed_scale.dtype == dtype
    assert model.lm_head.weight is embedding.weight
    assert set(embedding.state_dict()) == {"weight"}
    assert torch.count_nonzero(embedding.weight[3]) == 0
    actual = embedding(torch.tensor([[3, 1]]))
    expected = torch.stack((torch.zeros(16, dtype=dtype), embedding.weight[1] * 4)).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.float()
    assert model.lm_head.weight is embedding.weight


def test_forward_enforces_model_position_limit():
    # No decoder layers are needed to test the root model limit independently,
    # avoiding quadratic attention allocation.
    config = replace(recipe_model_config(max_position_embeddings=4097), layers=[])
    model = config.build().float()
    model.init_states(buffer_device=torch.device("cpu"))

    tokens = torch.full((1, 4097), 2, dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (1, 4097, config.vocab_size)

    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(torch.zeros((1, 4098), dtype=torch.long))


def test_mlp_matches_independent_gated_formula_and_gradients(model):
    mlp = model.layers["0"].feed_forward
    assert isinstance(mlp, Gemma3MLP)
    assert mlp.activation.approximate == "tanh"
    assert mlp.swiglu_limit == 0.0
    assert all(projection.bias is None for projection in (mlp.w1, mlp.w2, mlp.w3))

    values = torch.randn(2, 3, model.config.dim, requires_grad=True)
    reference_input = values.detach().clone().requires_grad_()
    weights = [p.detach().clone().requires_grad_() for p in (mlp.w1.weight, mlp.w2.weight, mlp.w3.weight)]

    actual = mlp(values)
    gate, down, up = weights
    expected = F.linear(
        F.gelu(F.linear(reference_input, gate), approximate="tanh") * F.linear(reference_input, up), down
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(values.grad, reference_input.grad, rtol=0, atol=0)
    for actual_weight, expected_weight in zip((mlp.w1.weight, mlp.w2.weight, mlp.w3.weight), weights, strict=True):
        torch.testing.assert_close(actual_weight.grad, expected_weight.grad, rtol=0, atol=0)


def test_recipe_rejects_unsupported_mlp_activation():
    with pytest.raises(ValueError, match="only supports hidden_activation='gelu_pytorch_tanh'"):
        _make_gemma3_layer_config(replace(recipe_text_config(), hidden_activation="silu"), layer_idx=0)


def test_decoder_residual_order_matches_hand_calculated_result(recipe):
    class AffineStub(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            multiplier: float = 1.0
            increment: float = 0.0

        def __init__(self, config: Config):
            super().__init__()
            self.multiplier, self.increment = config.multiplier, config.increment

        def forward(self, values, *args, **kwargs):
            return values * self.multiplier + self.increment

    layer_config = _make_gemma3_layer_config(recipe_text_config(), layer_idx=0)
    layer = layer_config.build()
    layer.input_layernorm = AffineStub.Config(multiplier=2).build()
    layer.attention = AffineStub.Config(increment=1).build()
    layer.post_attention_layernorm = AffineStub.Config(multiplier=3).build()
    layer.pre_feedforward_layernorm = AffineStub.Config(multiplier=5).build()
    layer.feed_forward = AffineStub.Config(increment=7).build()
    layer.post_feedforward_layernorm = AffineStub.Config(multiplier=11).build()

    actual = layer(
        torch.ones(1, 1, recipe.model_spec.model.dim),
        attention_masks={"sliding_window_mask": None},
    )

    # 1 -> 2 -> 3 -> 9 -> 10 -> 50 -> 57 -> 627 -> 637.
    torch.testing.assert_close(actual, torch.full_like(actual, 637))


def test_decoder_child_config_changes_stay_local_after_materialization():
    text_config = recipe_text_config()
    configs = [_make_gemma3_layer_config(text_config, layer_idx=index) for index in range(2)]
    norm_eps = {
        "input_layernorm": 2e-6,
        "post_attention_layernorm": 3e-6,
        "pre_feedforward_layernorm": 5e-6,
        "post_feedforward_layernorm": 7e-6,
    }
    for name, eps in norm_eps.items():
        getattr(configs[0], name).eps = eps
    configs[0].attention.qk_norm.eps = 11e-6
    configs[0].feed_forward.w1.param_init["weight"] = partial(torch.nn.init.constant_, val=0.5)
    configs[0].feed_forward.w2.param_init["weight"] = partial(torch.nn.init.constant_, val=-0.75)

    with torch.device("meta"):
        first, second = [config.build() for config in configs]
    for layer in (first, second):
        layer.to_empty(device="cpu")
        layer.init_states(buffer_device=torch.device("cpu"))

    for name, eps in norm_eps.items():
        assert isinstance(getattr(first, name), Gemma3RMSNorm)
        assert getattr(first, name).eps == eps
        assert getattr(second, name).eps == 1e-6
    assert first.attention.q_norm.eps == first.attention.k_norm.eps == 11e-6
    assert second.attention.q_norm.eps == second.attention.k_norm.eps == 1e-6
    torch.testing.assert_close(first.feed_forward.w1.weight, torch.full((32, 16), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(first.feed_forward.w2.weight, torch.full((16, 32), -0.75), rtol=0, atol=0)
    assert first.feed_forward.w3.weight.abs().max() <= 0.25
    # The other layer keeps the Kaiming-uniform bound 1 / sqrt(hidden_size).
    assert second.feed_forward.w1.weight.abs().max() <= 0.25


def test_untied_head_materializes_and_updates_independently(recipe):
    config = recipe_model_config(tie_word_embeddings=False)
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.tok_embeddings.weight.fill_(float("nan"))
        model.lm_head.weight.fill_(float("nan"))
    model.init_states(buffer_device=torch.device("cpu"))
    assert model.lm_head.weight is not model.tok_embeddings.weight
    weights = (model.tok_embeddings.weight, model.lm_head.weight)
    assert all(torch.isfinite(weight).all() for weight in weights)
    before = [weight.detach().clone() for weight in weights]
    optimizer = replace(recipe.optimizer, implementation="for-loop").build(model_parts=[model])
    parameter_ids = [
        id(p)
        for item in optimizer
        for group in item.param_groups
        for p in group["params"]
    ]
    assert all(parameter_ids.count(id(weight)) == 1 for weight in weights)

    logits = model(torch.tensor([[2, 5, 7, 11]]))
    F.cross_entropy(logits.flatten(0, 1), torch.tensor([5, 7, 11, 1])).backward()
    assert all(weight.grad is not None and torch.count_nonzero(weight.grad) > 0 for weight in weights)
    optimizer.step()

    for original, weight in zip(before, weights, strict=True):
        assert not torch.equal(original, weight)
    assert model.lm_head.weight is not model.tok_embeddings.weight


def test_rmsnorm_float32_statistics_and_zero_centered_scale():
    norm = Gemma3RMSNorm.Config(normalized_shape=4).build().half()

    assert isinstance(norm, Gemma3RMSNorm)
    assert norm.eps == 1e-6
    assert torch.count_nonzero(norm.weight) == 0

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.0, 0.5, -0.25, 1.0]))
    values = torch.tensor([[1024.0, -512.0, 256.0, -128.0]], dtype=torch.float16, requires_grad=True)
    reference_input = values.detach().float().requires_grad_()
    reference_weight = norm.weight.detach().float().requires_grad_()

    actual = norm(values)
    expected = reference_input * torch.rsqrt(reference_input.square().mean(-1, keepdim=True) + 1e-6)
    expected = (expected * (1 + reference_weight)).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    # The half input accumulates the two normalization gradient branches with
    # half rounding; the independent reference accumulates them in float32.
    torch.testing.assert_close(values.grad, reference_input.grad.half(), rtol=2e-3, atol=2e-6)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad.half(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "rope_options, frequencies",
    [
        ({"theta": 1_000_000.0, "scaling_factor": 8.0}, [0.125, 0.000125]),
        ({"theta": 10_000.0}, [1.0, 0.01]),
    ],
    ids=["global-linear", "local-default"],
)
def test_rope_config_restores_compact_buffers_and_rotates_each_batch(dtype, rope_options, frequencies):
    with torch.device("meta"):
        rope = Gemma3RotaryEmbedding.Config(dim=4, max_seq_len=131072, **rope_options).build().to(dtype)
    rope.to_empty(device="cpu")
    rope.inverse_frequency.fill_(float("nan"))

    rope.init_states()

    expected_frequency = torch.tensor(frequencies, dtype=torch.float32)
    torch.testing.assert_close(rope.inverse_frequency, expected_frequency, rtol=0, atol=0)
    assert rope.cache.shape == (0, 8)
    assert rope.config.max_seq_len == 131072
    assert not rope.state_dict()

    positions = torch.tensor([[0, 2, 1], [3, 0, 4]])
    query = torch.linspace(-2, 3, 48).reshape(2, 3, 2, 4).to(dtype).requires_grad_()
    key = torch.linspace(-1, 2, 24).reshape(2, 3, 1, 4).to(dtype).requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    angles = positions[:, :, None, None].float() * expected_frequency
    cos, sin = angles.cos(), angles.sin()
    expected = []
    for values in (reference_query, reference_key):
        first, second = values.float().chunk(2, dim=-1)
        expected.append(torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(dtype))

    actual_query, actual_key = rope(query, key, positions)

    torch.testing.assert_close(actual_query, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected[1], rtol=0, atol=0)
    (actual_query.sum() + actual_key.sum()).backward()
    (expected[0].sum() + expected[1].sum()).backward()
    torch.testing.assert_close(query.grad, reference_query.grad, rtol=0, atol=0)
    torch.testing.assert_close(key.grad, reference_key.grad, rtol=0, atol=0)


def test_attention_backend_applies_query_scale_softcap_and_gqa(monkeypatch):
    def dense_flex_attention(
        query,
        key,
        value,
        *,
        score_mod,
        block_mask,
        scale,
        enable_gqa,
        return_aux,
        kernel_options,
    ):
        if enable_gqa:
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        scores = query @ key.transpose(-1, -2)
        if scale is not None:
            scores = scores * scale
        if score_mod is not None:
            scores = score_mod(scores, None, None, None, None)
        allowed = create_mask(
            block_mask.mask_mod,
            query.shape[0],
            query.shape[1],
            query.shape[2],
            key.shape[2],
            device=query.device,
        )
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        return probabilities @ value, None

    monkeypatch.setattr(FlexAttention, "_compiled_flex_attn", dense_flex_attention)
    model = recipe_model_config(query_pre_attn_scalar=9, sliding_window=3, attn_logit_softcapping=1.25).build()
    attention = model.layers["0"].attention
    mask = model.get_attention_masks(torch.arange(4).unsqueeze(0))["sliding_window_mask"]
    query = torch.linspace(-2, 2, 64).reshape(1, 4, 4, 4).requires_grad_()
    key = torch.linspace(-1, 1, 32).reshape(1, 4, 2, 4).requires_grad_()
    value = torch.linspace(0, 2, 32).reshape(1, 4, 2, 4).requires_grad_()
    reference = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]

    actual = attention.inner_attention(
        query, key, value, attention_masks=mask, scale=attention.scaling, enable_gqa=attention.enable_gqa
    )
    q, k, v = reference
    q = q.transpose(1, 2)
    k = k.repeat_interleave(2, dim=2).transpose(1, 2)
    v = v.repeat_interleave(2, dim=2).transpose(1, 2)
    scores = 1.25 * torch.tanh((q @ k.transpose(-1, -2)) / 3 / 1.25)
    allowed = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    expected = (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for tensor, reference_tensor in zip((query, key, value), reference, strict=True):
        torch.testing.assert_close(tensor.grad, reference_tensor.grad, rtol=1e-5, atol=1e-6)


def test_attention_mask_causal_window_padding_and_document_boundaries():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    padding = torch.tensor([[True, True, True, True, True, False]])
    model = recipe_model_config(sliding_window=2).build()
    masks = model.get_attention_masks(positions, padding_mask=padding)
    global_mask = create_mask(masks["basic_mask"].mask_mod, 1, 1, 6, 6, device="cpu")
    local_mask = create_mask(masks["sliding_window_mask"].mask_mod, 1, 1, 6, 6, device="cpu")
    global_expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    local_expected = global_expected.clone()
    local_expected[2, 0] = False
    for actual, allowed in ((global_mask, global_expected), (local_mask, local_expected)):
        torch.testing.assert_close(actual[0, 0], allowed, rtol=0, atol=0)


def test_attention_mask_does_not_start_a_document_at_padding_reset():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]])
    padding = torch.tensor([[True, True, True, True, True, True, False, False]])
    model = recipe_model_config(sliding_window=2).build()
    mask = model.get_attention_masks(positions, padding_mask=padding)["basic_mask"]
    allowed = create_mask(mask.mask_mod, 1, 1, 8, 8, device="cpu")[0, 0]

    # The zero reset at index 6 is structural padding, so padded query rows
    # retain the second document's valid history rather than becoming a third
    # document with no unmasked keys.
    assert allowed[6, 3]
    assert not allowed[6, 0]
    assert allowed[7, 4]
    assert not allowed[7, 0]
    assert not allowed[6:, 6:].any()


def test_long_padding_keeps_valid_sliding_attention_logits_unchanged(model):
    tokens = torch.tensor([[2, 7, 3, 5, 11, 13]])
    valid_length = tokens.shape[1]
    sliding_window = model.config.layers[0].attention.sliding_window
    padding_length = valid_length + sliding_window
    padded_tokens = F.pad(tokens, (0, padding_length))
    # Packer positions restart at zero and then increase through the padding.
    positions = torch.cat((torch.arange(valid_length), torch.arange(padding_length))).unsqueeze(0)
    padding_mask = torch.arange(padded_tokens.shape[1]).unsqueeze(0) < valid_length
    masks = model.get_attention_masks(positions, padding_mask=padding_mask)
    sequence_length = padded_tokens.shape[1]
    allowed = create_mask(masks["sliding_window_mask"].mask_mod, 1, 1, sequence_length, sequence_length, device="cpu")
    assert not allowed[0, 0, valid_length - 1, 0]
    # The last padding query is beyond every real key's local window.
    assert allowed[0, 0, -1, :valid_length].all()
    assert not allowed[..., valid_length:].any()

    padded_logits = model(padded_tokens, positions=positions, padding_mask=padding_mask)
    valid_logits = model(tokens, positions=positions[:, :valid_length])

    assert torch.isfinite(padded_logits).all()
    torch.testing.assert_close(padded_logits[:, :valid_length], valid_logits, rtol=1e-5, atol=1e-5)


def test_model_all_padding_attention_rows_remain_finite(model):
    tokens = torch.zeros((1, 16), dtype=torch.long)
    positions = torch.arange(16).unsqueeze(0)
    padding = torch.zeros_like(tokens, dtype=torch.bool)

    logits = model(tokens, positions=positions, padding_mask=padding)

    assert torch.isfinite(logits).all()


def test_model_does_not_infer_padding_from_real_eos_tokens(model):
    tokens = torch.tensor([[2, 1, 7]])
    positions = torch.tensor([[0, 1, 0]])
    explicit_valid = torch.ones_like(tokens, dtype=torch.bool)

    actual = model(tokens, positions=positions, padding_mask=explicit_valid)
    implicit = model(tokens, positions=positions)
    torch.testing.assert_close(actual, implicit, rtol=0, atol=0)


@pytest.mark.parametrize("skip_lm_head", [False, True], ids=["logits", "hidden-states"])
def test_model_applies_final_softcap_only_to_logits(model, skip_lm_head):
    model.config.final_logit_softcapping = 0.25
    model._skip_lm_head = skip_lm_head
    normalized = []
    with model.norm.register_forward_hook(lambda module, args, output: normalized.append(output)):
        actual = model(torch.tensor([[2, 7, 3, 11, 13]]))
    hidden_states = normalized[0]
    assert hidden_states.abs().max() > 0.25

    if skip_lm_head:
        expected = hidden_states
        assert actual.shape == (1, 5, model.config.dim)
    else:
        raw_logits = F.linear(hidden_states, model.lm_head.weight)
        assert raw_logits.abs().max() > 0.25
        expected = 0.25 * torch.tanh(raw_logits / 0.25)
        assert actual.shape == (1, 5, model.config.vocab_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
