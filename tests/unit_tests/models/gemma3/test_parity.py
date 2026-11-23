# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchtitan_npu.models.gemma3.state_dict_adapter import Gemma3StateDictAdapter

from .fixtures import hf_parameter_names


def test_hf_logits_loss_and_gradients_after_meta_materialization(recipe, hf_reference):
    # Both attention types are present; sequence length exceeds the local window.
    # The fixture deliberately keeps query_pre_attn_scalar distinct from head_dim.
    config = recipe.model_spec.model
    with torch.device("meta"):
        actual_model = config.build()
    actual_model.to_empty(device="cpu")
    actual_model.init_states(buffer_device=torch.device("cpu"))
    adapter = Gemma3StateDictAdapter(config)
    actual_model.load_state_dict(adapter.from_hf(hf_reference.state_dict()), strict=True)
    names = hf_parameter_names(len(config.layers))
    assert set(actual_model.state_dict()) == set(names)
    assert actual_model.lm_head.weight is actual_model.tok_embeddings.weight

    # Independent frequency values protect the non-persistent buffers rebuilt by
    # init_states: global theta=1e6, factor=8; local theta=1e4, factor=1.
    for index, layer in enumerate(actual_model.layers.values()):
        expected_frequency = [0.125, 0.000125] if index == 1 else [1.0, 0.01]
        torch.testing.assert_close(layer.attention.rope.inverse_frequency, torch.tensor(expected_frequency))
    torch.testing.assert_close(actual_model.tok_embeddings.embed_scale, torch.tensor(4.0))

    tokens = torch.tensor([[2, 7, 3, 5, 11, 13, 17], [2, 19, 5, 23, 29, 31, 7]])
    positions = torch.arange(7).expand_as(tokens)
    labels = torch.tensor([[-100, -100, 5, 11, 13, 17, 1], [-100, 5, 23, 29, 31, 7, 1]])
    actual = actual_model(tokens, positions=positions)
    expected = hf_reference(tokens, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten(), ignore_index=-100)
    reference_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten(), ignore_index=-100)
    torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-5)
    loss.backward()
    reference_loss.backward()
    reference_parameters = dict(hf_reference.named_parameters(remove_duplicate=False))
    for name, parameter in actual_model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameters[names[name]].grad, rtol=1e-4, atol=1e-4)
