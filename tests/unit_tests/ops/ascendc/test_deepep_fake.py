# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape, autograd and checkpoint contracts for DeepEP dispatch payloads.

FakeTensor and small CPU kernels exercise the custom-op boundary only; they
do not validate CANN kernels or communication between ranks.
"""

import importlib
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


@pytest.fixture(scope="module")
def deepep():
    # PyTorch has moved these private opaque-object symbols between releases.
    # Adapt the test import surface before loading production ``deepep.py``;
    # the production module deliberately remains unchanged.
    opaque_object = importlib.import_module("torch._library.opaque_object")
    if not hasattr(opaque_object, "CustomClassBase"):
        # Recent PyTorch versions moved the base class to
        # ``torch._custom_class_base``.  Re-export the real class only for this
        # test import; a plain Python stand-in is insufficient because newer
        # registration code checks ``CustomClassBaseMeta``.
        try:
            custom_class_base = importlib.import_module("torch._custom_class_base").CustomClassBase
        except (ImportError, AttributeError):
            pytest.skip("PyTorch build lacks the opaque-object custom-op API")
        opaque_object.CustomClassBase = custom_class_base
    if not hasattr(opaque_object, "register_opaque_type"):
        # ``register_custom_class`` is the non-deprecated spelling in newer
        # builds and accepts the same ``typ="reference"`` compatibility
        # value used by deepep.py.
        register_custom_class = getattr(opaque_object, "register_custom_class", None)
        if register_custom_class is None:
            pytest.skip("PyTorch build lacks the opaque-object registration API")
        opaque_object.register_opaque_type = register_custom_class

    # The shared conftest installs a call-recorder fake in place of the real
    # ``cann_ops_transformer`` package. Give it the two symbols this module
    # imports at module level so the production module is importable on CPU.
    # Adding attributes to the one-time recorder fake needs no restoration.
    cann = sys.modules["cann_ops_transformer"]
    if not hasattr(cann, "ElasticBuffer"):
        cann.ElasticBuffer = type("ElasticBuffer", (), {})
    if not hasattr(cann, "EPHandle"):
        cann.EPHandle = type("EPHandle", (), {})

    return importlib.import_module("torchtitan_npu.ops.ascendc.deepep")


def test_deepep_quantized_cached_op_dispatches_to_fake(deepep):
    buffer = deepep.ElasticBufferHandle(value=None)
    handle = deepep.DispatchHandle(value=None)
    with FakeTensorMode():
        qdata = torch.empty((4, 32), dtype=torch.uint8)
        scale = torch.empty((4, 8), dtype=torch.uint8)
        recv_qdata, recv_scale = torch.ops.deepep.dispatch_quantized_cached(qdata, scale, handle, 6, buffer)
    assert recv_qdata.shape == (6, 32)
    assert recv_qdata.dtype == torch.uint8
    assert recv_scale.shape == (6, 8)
    assert recv_scale.dtype == torch.uint8


@pytest.mark.parametrize("num_permuted_tokens", [0, 6])
@pytest.mark.parametrize("hidden", [32, 64])
def test_deepep_cached_fake_preserves_payload_dimensions(deepep, num_permuted_tokens, hidden):
    buffer = deepep.ElasticBufferHandle(value=None)
    handle = deepep.DispatchHandle(value=None)
    with FakeTensorMode():
        qdata = torch.empty((4, hidden), dtype=torch.uint8)
        scale = torch.empty((4, hidden // 4), dtype=torch.uint8)
        recv_qdata, recv_scale = deepep._dispatch_quantized_cached_fake(
            qdata, scale, handle, num_permuted_tokens, buffer
        )
    assert recv_qdata.shape == (num_permuted_tokens, hidden)
    assert recv_qdata.dtype == torch.uint8
    assert recv_scale.shape == (num_permuted_tokens, hidden // 4)
    assert recv_scale.dtype == torch.uint8


@pytest.mark.parametrize("quantized", [False, True], ids=["high-precision", "mxfp8-payload"])
def test_deepep_dispatch_fake_uses_shared_dynamic_receive_count(deepep, quantized):
    env = ShapeEnv()
    buffer = deepep.ElasticBufferHandle(value=None)

    def run(payload, topk_idx, topk_w, buffer):
        return torch.ops.deepep.dispatch(payload, topk_idx, topk_w, 8, 4, buffer)

    with FakeTensorMode(shape_env=env):
        tokens = env.create_unbacked_symint()
        torch._constrain_as_size(tokens, min=4)
        dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        payload = [torch.empty((tokens, 64), dtype=dtype)]
        if quantized:
            payload.append(torch.empty((tokens, 2), dtype=torch.uint8))
        topk_idx = torch.empty((tokens, 2), dtype=torch.int32)
        topk_w = torch.empty((tokens, 2), dtype=torch.float32)
        graph = make_fx(run, tracing_mode="symbolic")(payload, topk_idx, topk_w, buffer)

    dispatch_node = next(node for node in graph.graph.nodes if node.target == torch.ops.deepep.dispatch.default)
    recv_payload, recv_scores, counts, handle = dispatch_node.meta["val"]
    assert len(recv_payload) == (2 if quantized else 1)
    assert recv_payload[0].dtype == dtype
    assert recv_payload[0].shape[1] == 64
    assert recv_scores.dtype == torch.float32
    assert counts.dtype == torch.int64
    assert isinstance(recv_payload[0].shape[0], torch.SymInt)
    for tensor in [*recv_payload, recv_scores]:
        assert isinstance(tensor.shape[0], torch.SymInt)
        assert tensor.shape[0] == recv_payload[0].shape[0]
    if quantized:
        assert recv_payload[1].dtype == torch.uint8
        assert recv_payload[1].shape[1] == 2
    assert counts.shape == (4,)
    assert isinstance(handle, deepep.DispatchHandle)


@pytest.fixture
def cpu_dispatch_kernels(deepep):
    """Two tokens, each sent to experts 0 and 1, in expert-major order."""
    calls = []
    handle = deepep.DispatchHandle(value=None)

    def dispatch(payload, topk_idx, topk_weights, num_experts, num_local_experts, buffer):
        calls.append("dispatch")
        rows = torch.tensor([0, 1, 0, 1])
        score_slots = torch.tensor([0, 2, 1, 3])
        return (
            [tensor.index_select(0, rows) for tensor in payload],
            topk_weights.flatten().index_select(0, score_slots),
            torch.tensor([2, 2], dtype=torch.int64),
            handle,
        )

    def dispatch_backward(grad_hidden, grad_scores, has_grad_scores, topk_idx, saved_handle, buffer):
        calls.append("dispatch_backward")
        assert saved_handle is handle
        grad_x = torch.stack((grad_hidden[0] + grad_hidden[2], grad_hidden[1] + grad_hidden[3]))
        grad_weights = grad_scores[[0, 2, 1, 3]].reshape(2, 2) if has_grad_scores else torch.empty((2, 2))
        return grad_x, grad_weights

    library = torch.library.Library("deepep", "IMPL", "CPU")
    try:
        library.impl("dispatch", dispatch)
        library.impl("dispatch_backward", dispatch_backward)
        yield calls
    finally:
        library._destroy()


def test_deepep_dispatch_tensor_list_autograd_restores_activation_and_score_gradients(deepep, cpu_dispatch_kernels):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    scores = torch.tensor([[0.2, 0.8], [0.3, 0.7]], requires_grad=True)
    topk_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    buffer = deepep.ElasticBufferHandle(value=None)

    received, recv_scores, _, _ = torch.ops.deepep.dispatch([x], topk_idx, scores, 2, 2, buffer)
    loss = (received[0] * torch.arange(1.0, 13.0).reshape(4, 3)).sum()
    loss = loss + (recv_scores * torch.tensor([2.0, 3.0, 5.0, 7.0])).sum()
    loss.backward()

    torch.testing.assert_close(
        received[0], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    torch.testing.assert_close(recv_scores, torch.tensor([0.2, 0.3, 0.8, 0.7]))
    torch.testing.assert_close(x.grad, torch.tensor([[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]]))
    torch.testing.assert_close(scores.grad, torch.tensor([[2.0, 5.0], [3.0, 7.0]]))
    assert cpu_dispatch_kernels == ["dispatch", "dispatch_backward"]


@pytest.fixture
def mxfp8_deepep(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "experiments" / "torchao-npu"))
    return pytest.importorskip("torchao_npu.ops.mxfp8_deepep_payload")


@pytest.fixture
def mxfp8_deepep_adaptor(mxfp8_deepep):
    return importlib.import_module("torchtitan_npu.extensions.adaptor.torchao.deepep_token_dispatcher")


def _mxfp8_dispatcher(mxfp8_deepep_adaptor, *, absorb_router_scores=False):
    from torchtitan_npu.override.common.token_dispatcher import AscDeepEPTokenDispatcher

    cls = mxfp8_deepep_adaptor.get_mxfp8_deepep_dispatcher_cls(AscDeepEPTokenDispatcher)
    dispatcher = object.__new__(cls)
    dispatcher.num_experts = 4
    dispatcher._deepep_world_size = 2
    dispatcher._deepep_buffer_handle = object()
    dispatcher.mxfp8_quant_params = (32, "rint", 0, 0.0)
    dispatcher.absorb_router_scores = absorb_router_scores
    return dispatcher


class _FakeMXTensor(torch.Tensor):
    """Tensor subclass used to observe the MXFP8 gradient representation."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)


def _patch_fake_combine_transport(monkeypatch, mxfp8_deepep_adaptor, grad_values):
    """Make cached dispatch return a known routed gradient for VJP tests."""
    qdata = torch.empty((grad_values.shape[0], grad_values.shape[1]), dtype=torch.uint8)
    scale = torch.empty((grad_values.shape[0], 2), dtype=torch.uint8)

    def fake_quantize(activation, quant_params):
        del quant_params
        return qdata[: activation.shape[0]], scale[: activation.shape[0]]

    def fake_cached(qdata_arg, scale_arg, handle_arg, row_count, buffer_arg):
        del qdata_arg, scale_arg, handle_arg, row_count, buffer_arg
        return qdata, scale

    def fake_wire(qdata_arg, scale_arg, dtype, quant_params, requires_grad):
        del qdata_arg, scale_arg, dtype, quant_params, requires_grad
        return _FakeMXTensor(grad_values)

    def fake_combine(activation, handle, num_tokens, row_count_source, buffer):
        del handle, row_count_source, buffer
        return activation.new_zeros((num_tokens, activation.shape[1]))

    # The adaptor imports these helpers into its own module namespace, so patch
    # that namespace rather than the transport-only payload module fixture.
    monkeypatch.setattr(mxfp8_deepep_adaptor, "_quantize_payload", fake_quantize)
    monkeypatch.setattr(mxfp8_deepep_adaptor, "_wire_tensor", fake_wire)
    monkeypatch.setattr(torch.ops.deepep, "dispatch_quantized_cached", fake_cached, raising=False)
    monkeypatch.setattr(torch.ops.deepep, "combine", fake_combine, raising=False)


def test_mxfp8_deepep_payload_layout(monkeypatch, mxfp8_deepep):
    qdata = torch.arange(2 * 64, dtype=torch.uint8).reshape(2, 64)
    # block_size=32 over a width of 64 gives two MX blocks per row, which the
    # quantizer packs into (T, (num_blocks + 1) // 2, 2) E8M0 bytes: (2, 1, 2).
    scale = torch.arange(2 * 1 * 2, dtype=torch.uint8).reshape(2, 1, 2)
    activation = torch.randn(2, 64, dtype=torch.bfloat16)

    def fake_quantize(activation, *params):
        return qdata, scale

    monkeypatch.setattr(mxfp8_deepep, "_mxfp8_quantize_or_empty", fake_quantize)
    packed_qdata, packed_scale = mxfp8_deepep._quantize_payload(activation, (32, "rint", 0, 0.0))
    torch.testing.assert_close(packed_qdata, qdata)
    torch.testing.assert_close(packed_scale, scale.reshape(2, 2))


def test_mxfp8_deepep_wire_tensor_unpacks_scale(monkeypatch, mxfp8_deepep):
    seen = []

    def fake_wire(qdata, scale, dtype, quant_params, requires_grad):
        seen.append((qdata, scale, dtype, quant_params, requires_grad))
        return qdata

    monkeypatch.setattr(mxfp8_deepep, "_mxfp8_wire_tensor", fake_wire)
    qdata = torch.empty((3, 64), dtype=torch.uint8)
    packed_scale = torch.empty((3, 4), dtype=torch.uint8)
    mxfp8_deepep._wire_tensor(qdata, packed_scale, torch.bfloat16, (32, "rint", 0, 0.0), True)
    assert seen[0][0] is qdata
    assert seen[0][1].shape == (3, 2, 2)
    assert seen[0][2:] == (torch.bfloat16, (32, "rint", 0, 0.0), True)


def test_mxfp8_deepep_dispatch_seam_quantizes_payload(monkeypatch, mxfp8_deepep_adaptor):
    dispatcher = _mxfp8_dispatcher(mxfp8_deepep_adaptor)
    qdata = torch.ones((4, 3), dtype=torch.uint8)
    scale = torch.ones((4, 2), dtype=torch.uint8)
    handle = object()
    calls = []

    def fake_wire(qdata, scale, dtype, quant_params, requires_grad):
        calls.append(("wire", qdata, scale, dtype, quant_params, requires_grad))
        return qdata.float()

    def fake_dispatch(payload, topk_idx, topk_scores, *args):
        calls.append(("dispatch", payload, topk_idx, topk_scores, args))
        return [qdata, scale], topk_scores.reshape(-1), torch.tensor([2, 2]), handle

    monkeypatch.setattr(mxfp8_deepep_adaptor, "_quantize_payload", lambda *_: (qdata, scale))
    monkeypatch.setattr(mxfp8_deepep_adaptor, "_wire_tensor", fake_wire)
    monkeypatch.setattr(torch.ops.deepep, "dispatch", fake_dispatch, raising=False)

    x = torch.randn(2, 3)
    topk_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    topk_scores = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
    routed, scores, counts, received_handle = dispatcher._dispatch_activation_exchange(
        x, topk_idx, topk_scores, num_experts=dispatcher.num_experts
    )

    _, payload, received_idx, received_scores, args = calls[0]
    assert len(payload) == 2 and payload[0] is qdata and payload[1] is scale
    assert received_idx is topk_idx and received_scores is topk_scores
    assert args == (4, 2, dispatcher._deepep_buffer_handle)
    assert received_handle is handle and counts.requires_grad is False
    assert calls[1][0] == "wire"
    assert calls[1][1] is qdata and calls[1][2] is scale
    assert calls[1][5] is False
    torch.testing.assert_close(routed, qdata.float())
    torch.testing.assert_close(scores, topk_scores.reshape(-1))


def test_mxfp8_deepep_dispatch_vjp_routes_activation_and_score_gradients(monkeypatch, mxfp8_deepep_adaptor):
    ascendc_ops = importlib.import_module("torchtitan_npu.ops.ascendc.deepep")
    calls = []

    def fake_wire(qdata, scale, dtype, quant_params, requires_grad):
        del scale, dtype, quant_params, requires_grad
        return qdata.float()

    def fake_backward(ctx, grad_hidden, grad_scores, *_):
        calls.append((grad_hidden, grad_scores, ctx.handle))
        return grad_hidden.new_full((2, 3), 2.0), None, grad_scores.new_full((2, 2), 3.0), None, None, None

    monkeypatch.setattr(mxfp8_deepep_adaptor, "_wire_tensor", fake_wire)
    monkeypatch.setattr(ascendc_ops, "_dispatch_backward", fake_backward)

    x = torch.randn(2, 3, requires_grad=True)
    topk_scores = torch.zeros((2, 2), requires_grad=True)
    routed, scores, counts = mxfp8_deepep_adaptor._MXFP8DeepEPDispatch.apply(
        x,
        torch.ones((4, 3), dtype=torch.uint8),
        torch.ones((4, 2), dtype=torch.uint8),
        torch.tensor([0.2, 0.3, 0.8, 0.7]),
        torch.tensor([2, 2]),
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        topk_scores,
        object(),
        object(),
        (32, "rint", 0, 0.0),
    )
    (routed.sum() + scores.sum()).backward()

    assert counts.requires_grad is False
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))
    torch.testing.assert_close(topk_scores.grad, torch.full_like(topk_scores, 3.0))
    assert calls[0][0].shape == (4, 3) and calls[0][1].shape == (4,)


def test_mxfp8_deepep_combine_vjp_uses_cached_quantized_dispatch(monkeypatch, mxfp8_deepep_adaptor):
    calls = {}
    qdata = torch.empty((4, 3), dtype=torch.uint8)
    scale = torch.empty((4, 2), dtype=torch.uint8)

    def fake_quantize(activation, quant_params):
        calls["quantize"] = (activation, quant_params)
        return qdata[: activation.shape[0]], scale[: activation.shape[0]]

    def fake_cached(qdata_arg, scale_arg, handle_arg, row_count, buffer_arg):
        calls["cached"] = (qdata_arg, scale_arg, handle_arg, row_count, buffer_arg)
        return qdata, scale

    def fake_wire(qdata_arg, scale_arg, dtype, quant_params, requires_grad):
        calls["wire"] = (qdata_arg, scale_arg, dtype, quant_params, requires_grad)
        return torch.full((4, 3), 5.0, dtype=dtype)

    def fake_combine(activation, handle_arg, num_tokens, row_count_source, buffer_arg):
        calls["combine"] = (activation, handle_arg, num_tokens, row_count_source, buffer_arg)
        return activation.new_zeros((num_tokens, activation.shape[1]))

    monkeypatch.setattr(mxfp8_deepep_adaptor, "_quantize_payload", fake_quantize)
    monkeypatch.setattr(mxfp8_deepep_adaptor, "_wire_tensor", fake_wire)
    monkeypatch.setattr(torch.ops.deepep, "dispatch_quantized_cached", fake_cached, raising=False)
    monkeypatch.setattr(torch.ops.deepep, "combine", fake_combine, raising=False)

    activation = torch.randn(4, 3, requires_grad=True)
    row_count_source = torch.empty(4)
    handle = object()
    buffer = object()
    quant_params = (32, "rint", 0, 0.0)
    output = mxfp8_deepep_adaptor._MXFP8DeepEPCombine.apply(
        activation, row_count_source, handle, 2, buffer, quant_params
    )
    output.sum().backward()

    assert calls["combine"][1:] == (handle, 2, row_count_source, buffer)
    assert calls["quantize"][0].shape == (2, 3)
    assert calls["cached"][2:] == (handle, 4, buffer)
    assert calls["wire"][2:] == (torch.float32, quant_params, False)
    torch.testing.assert_close(activation.grad, torch.full_like(activation, 5.0))


def test_mxfp8_deepep_pre_w2_vjp_keeps_mxfp8_gradient_for_gmm(monkeypatch, mxfp8_deepep_adaptor):
    """Pre-W2 experts must receive the wire MXFP8 dY from combine backward."""
    from torchtitan_npu.override.common.token_dispatcher import DispatchState

    grad_values = torch.tensor(
        [[2.0, -1.0, 0.5], [0.5, 3.0, -2.0], [-2.0, 4.0, 1.0], [1.0, -0.5, 2.0]]
    )
    _patch_fake_combine_transport(monkeypatch, mxfp8_deepep_adaptor, grad_values)
    dispatcher = _mxfp8_dispatcher(mxfp8_deepep_adaptor, absorb_router_scores=True)

    gmm_gradients = []

    class FakeGroupedMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, activation, weight):  # pyrefly: ignore [bad-override]
            ctx.save_for_backward(weight)
            return activation @ weight.transpose(0, 1)

        @staticmethod
        def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
            (weight,) = ctx.saved_tensors
            assert isinstance(grad_output, _FakeMXTensor)
            gmm_gradients.append(grad_output)
            # The real grouped-MM dgrad consumes the MXTensor directly.  The
            # fake converts only after observing that representation so this
            # test remains CPU-only while still exercising a matrix dgrad.
            grad_output_hp = grad_output.as_subclass(torch.Tensor)
            return grad_output_hp @ weight, None

    expert_input = torch.randn(4, 3, requires_grad=True)
    weight = torch.tensor(
        [[1.0, 0.0, 0.5], [-0.5, 2.0, 0.0], [0.25, -1.0, 1.5]]
    )
    routed_output = FakeGroupedMM.apply(expert_input, weight)
    metadata = DispatchState(
        recv_scores=torch.tensor([0.2, 0.3, 0.8, 0.7]),
        handle=object(),
    )
    combined = dispatcher.combine(routed_output, metadata, torch.empty((2, 3)))
    combined.sum().backward()

    assert len(gmm_gradients) == 1
    assert isinstance(gmm_gradients[0], _FakeMXTensor)
    torch.testing.assert_close(gmm_gradients[0].as_subclass(torch.Tensor), grad_values)
    torch.testing.assert_close(expert_input.grad, grad_values @ weight)


def test_mxfp8_deepep_post_w2_vjp_dequantizes_activation_and_score_gradients(
    monkeypatch, mxfp8_deepep, mxfp8_deepep_adaptor
):
    """Post-W2 score multiplication must consume dequantized combine dY."""
    from torchtitan_npu.override.common.token_dispatcher import DispatchState

    grad_values = torch.tensor(
        [[2.0, -1.0, 0.5], [0.5, 3.0, -2.0], [-2.0, 4.0, 1.0], [1.0, -0.5, 2.0]]
    )
    _patch_fake_combine_transport(monkeypatch, mxfp8_deepep_adaptor, grad_values)
    # ``_DequantizeGradient`` is defined in the payload module and resolves
    # its class/function globals there; patch both namespaces used by the
    # adaptor and by that autograd Function.
    monkeypatch.setattr(mxfp8_deepep, "MXTensor", _FakeMXTensor)
    monkeypatch.setattr(mxfp8_deepep_adaptor, "MXTensor", _FakeMXTensor)
    dequantized = []

    def fake_dequantize(value):
        dequantized.append(value)
        return value.as_subclass(torch.Tensor)

    monkeypatch.setattr(mxfp8_deepep, "_mxfp8_dequantize_tensor", fake_dequantize)
    monkeypatch.setattr(mxfp8_deepep_adaptor, "_mxfp8_dequantize_tensor", fake_dequantize)
    dispatcher = _mxfp8_dispatcher(mxfp8_deepep_adaptor, absorb_router_scores=False)

    routed_values = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]],
        requires_grad=True,
    )
    recv_scores = torch.tensor([0.2, 0.3, 0.8, 0.7], requires_grad=True)
    metadata = DispatchState(recv_scores=recv_scores, handle=object())
    combined = dispatcher.combine(routed_values, metadata, torch.empty((2, 3)))
    upstream = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])
    (combined * upstream).sum().backward()

    expected_routed_grad = grad_values * recv_scores.detach().reshape(-1, 1)
    expected_score_grad = (grad_values * routed_values.detach()).sum(dim=-1)
    torch.testing.assert_close(routed_values.grad, expected_routed_grad)
    torch.testing.assert_close(recv_scores.grad, expected_score_grad)
    assert len(dequantized) == 1
    assert isinstance(dequantized[0], _FakeMXTensor)


@pytest.mark.parametrize("absorb_router_scores", [False, True], ids=["post-w2-score", "pre-w2-score"])
def test_mxfp8_deepep_combine_keeps_score_multiply_outside_transport(
    monkeypatch, mxfp8_deepep_adaptor, absorb_router_scores
):
    from torchtitan_npu.override.common.token_dispatcher import DispatchState

    dispatcher = _mxfp8_dispatcher(mxfp8_deepep_adaptor, absorb_router_scores=absorb_router_scores)

    seen = []

    def fake_combine(activation, handle, num_tokens, row_count_source, buffer):
        del handle, num_tokens, row_count_source, buffer
        seen.append(activation.detach().clone())
        return activation[:2]

    monkeypatch.setattr(torch.ops.deepep, "combine", fake_combine, raising=False)
    # Keep this test focused on the forward placement of score multiplication;
    # the dedicated pre-W2 and post-W2 VJP tests above cover backward behavior.
    routed_output = torch.ones((4, 3), requires_grad=True)
    recv_scores = torch.tensor([0.2, 0.3, 0.8, 0.7])
    metadata = DispatchState(recv_scores=recv_scores, handle=object())
    result = dispatcher.combine(routed_output, metadata, torch.empty((2, 3)))

    expected = routed_output * recv_scores.reshape(-1, 1) if not absorb_router_scores else routed_output
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(result, expected[:2])


@pytest.mark.parametrize("quantized", [False, True], ids=["high-precision", "mxfp8-payload"])
def test_default_selective_ac_reuses_dispatch_tensor_list_outputs(deepep, cpu_dispatch_kernels, quantized):
    from torchtitan.distributed.activation_checkpoint import SelectiveAC

    save_ops = SelectiveAC.Config().build().get_save_ops()
    assert torch.ops.deepep.dispatch.default in save_ops
    buffer = deepep.ElasticBufferHandle(value=None)
    topk_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    scores = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
    forward_calls = []

    def region(x):
        forward_calls.append(1)
        with torch.no_grad():
            # Bytes stand in for encoded FP8 data; this checks SAC reuse, not quantization.
            payload = [x.to(torch.uint8), torch.ones((2, 2), dtype=torch.uint8)] if quantized else [x]
            received, _, _, _ = torch.ops.deepep.dispatch(payload, topk_idx, scores, 2, 2, buffer)
        # An uncached saved intermediate forces checkpoint recomputation even
        # though the transport itself intentionally runs under no_grad.
        return (x.square() * received[0][:2].float()).sum()

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    result = checkpoint(
        region,
        x,
        use_reentrant=False,
        context_fn=partial(create_selective_checkpoint_contexts, list(save_ops)),
        early_stop=False,
    )
    result.backward()

    torch.testing.assert_close(result, torch.tensor(441.0))
    torch.testing.assert_close(x.grad, torch.tensor([[2.0, 8.0, 18.0], [32.0, 50.0, 72.0]]))
    assert len(forward_calls) == 2
    assert cpu_dispatch_kernels == ["dispatch"]
