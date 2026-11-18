# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn.functional as F
from _support.testing_utils import target_devices
from torchao.float8.float8_utils import compute_error
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao_npu.ops.float8_ops import float8_rowwise_fake_quantize
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
    Float8TrainingWeightWrapperTensor,
    MXTrainingWeightWrapperTensor,
)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            BaseTrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_shape, op_func",
    [
        # select / slice
        ((4, 64, 128), lambda x: x[0]),
        ((4, 64, 128), lambda x: x[0:2]),
        ((4, 64, 128), lambda x: x[1:]),
        ((4, 64, 128), lambda x: x[:3]),
        ((4, 64, 128), lambda x: x[::2]),
        ((4, 64, 128), lambda x: x[:, 0]),
        # index
        ((4, 64, 128), lambda x: x[torch.tensor([0, 2])]),
        ((4, 64, 128), lambda x: x[torch.tensor([True, False, True, False])]),
        # _unsafe_index -- internal ATen op triggered by advanced indexing
        (
            (4, 64, 128),
            lambda x: torch.ops.aten._unsafe_index.Tensor(x, [torch.tensor([0, 2])]),
        ),
        # unsqueeze
        ((4, 64, 128), lambda x: x.unsqueeze(0)),
        # new_zeros
        ((4, 64, 128), lambda x: x.new_zeros(2, 64, 128)),
        # as_strided
        ((4, 64, 128), lambda x: x.as_strided((2, 64, 128), (16384, 128, 1))),
        # transpose
        ((4, 64, 128), lambda x: x.transpose(0, 1)),
        # detach
        ((4, 64, 128), lambda x: x.detach()),
        # clone
        ((4, 64, 128), lambda x: x.clone()),
        # view
        ((4, 64, 128), lambda x: x.view(4, 2, 32, 128)),
        # permute
        ((4, 64, 128), lambda x: x.permute(1, 0, 2)),
        # _to_copy
        ((4, 64, 128), lambda x: x.to(dtype=torch.float16)),
        # squeeze.dim -- needs singleton dim
        ((1, 64, 128), lambda x: x.squeeze(0)),
        # squeeze (no dim) -- needs singleton dim
        ((4, 1, 128), lambda x: x.squeeze()),
        # t -- needs 2D shape
        ((64, 128), lambda x: x.t()),
        # split -- returns tuple
        ((8, 64, 128), lambda x: torch.split(x, 2)),
    ],
)
@pytest.mark.parametrize("device", target_devices)
def test_wrapper_preserves_subclass(wrapper_cls, weight_config, act_config, weight_shape, op_func, device):
    """All ops in _ops_to_preserve_subclass return the wrapper subclass.

    ``c10d.scatter_.default`` requires distributed runtime and is excluded here.
    ``copy_`` is tested separately in ``test_wrapper_dispatch_copy_`` (in-place op).
    ``pin_memory`` is tested separately in ``test_pin_memory_preserves_subclass``.
    """

    def apply_assertions(result, ref_result):
        assert isinstance(result, wrapper_cls)
        assert result.weight_config is weight_config
        assert result.activation_config is act_config
        assert torch.equal(result._data, ref_result)

    weight = torch.randn(*weight_shape, device=device)
    wrapper = wrapper_cls(weight, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = op_func(wrapper)
        ref_result = op_func(wrapper._data)

        if isinstance(result, tuple):
            for r, ref in zip(result, ref_result, strict=True):
                apply_assertions(r, ref)
        else:
            apply_assertions(result, ref_result)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            BaseTrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
    ],
)
def test_pin_memory_preserves_subclass(wrapper_cls, weight_config, act_config):
    """pin_memory preserves the wrapper subclass."""

    weight = torch.randn(4, 64, 128, device="cpu")
    wrapper = wrapper_cls(weight.clone(), activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        pinned_wrapper = wrapper.pin_memory()
        pinned_weight = weight.pin_memory()

    assert pinned_weight.is_pinned(), "The reference weight tensor should be pinned."
    assert pinned_wrapper.is_pinned(), "The resulting wrapper tensor should be pinned."
    assert isinstance(pinned_wrapper, wrapper_cls), "The wrapper class should be preserved after being pinned."
    assert pinned_wrapper.weight_config is weight_config
    assert pinned_wrapper.activation_config is act_config
    assert torch.equal(pinned_wrapper, pinned_weight)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            BaseTrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
    ],
)
def test_wrapper_dispatch_copy_(wrapper_cls, weight_config, act_config, device):
    """copy_ via __torch_dispatch__ returns self and updates _data in-place."""
    w = torch.randn(4, 64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        w2 = torch.randn(4, 64, 128, device=device)
        wrapper2 = wrapper_cls(w2, activation_config=act_config, weight_config=weight_config)
        result = wrapper.copy_(wrapper2)
        assert result is wrapper
        assert torch.equal(wrapper._data, w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            BaseTrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            MXQuantizeConfig(),
        ),
    ],
)
def test_wrapper_dispatch_detach(wrapper_cls, weight_config, act_config, device):
    """detach via __torch_dispatch__ creates a new wrapper with shared configs and detached _data."""
    w = torch.randn(4, 64, 128, requires_grad=True, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = wrapper.detach()
        assert type(result) is wrapper_cls
        assert result._data.requires_grad is False
        assert result.weight_config is weight_config
        assert result.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "call_fn, a_shape, w_shape, kwargs",
    [
        (lambda a, w: torch.mm(a, w.T), (16, 64), (128, 64), {}),
        (lambda a, w: torch.matmul(a, w.T), (16, 64), (128, 64), {}),
        (lambda a, w: torch.bmm(a, w.transpose(-2, -1)), (4, 16, 64), (4, 128, 64), {}),
        (lambda a, w: F.linear(a, w), (16, 64), (128, 64), {}),
        (
            lambda a, w, *, bias: torch.addmm(bias, a, w.T),
            (16, 64),
            (128, 64),
            {"bias_shape": (128,)},
        ),
        (
            lambda a, w, *, offs: torch._grouped_mm(a, w.transpose(-2, -1), offs=offs),
            (16, 1024),
            (4, 2048, 1024),
            {"offs": torch.tensor([4, 8, 12, 16], dtype=torch.int32)},
        ),
    ],
)
def test_wrapper_torch_function_disabled(call_fn, a_shape, w_shape, kwargs, device):
    """BaseTrainingWeightWrapperTensor.__torch_function__ passes through without fake quant."""
    w = torch.randn(*w_shape, device=device)
    wrapper = BaseTrainingWeightWrapperTensor(w, weight_config=Float8FakeQuantizeConfig())

    a = torch.randn(*a_shape, device=device)
    resolved = {}
    for k, v in kwargs.items():
        if k.endswith("_shape"):
            resolved[k[: -len("_shape")]] = torch.randn(*v, device=device)
        elif isinstance(v, torch.Tensor):
            resolved[k] = v.to(device)
        else:
            resolved[k] = v

    result = call_fn(a, wrapper, **resolved)
    ref_result = call_fn(a, w, **resolved)
    assert torch.equal(result, ref_result), "Base class should pass through without fake quantization"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
        ),
    ],
)
def test_activation_qat_empty_input(wrapper_cls, weight_config, act_config, device):
    """Activation fake quant is skipped for empty tensors (expert with 0 tokens)."""
    w = torch.randn(128, 64, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)
    a = torch.randn(0, 64, device=device)
    out = torch.mm(a, param.T)
    assert out.shape == (0, 128), "Output should be empty when input is empty"
    assert out.numel() == 0


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize(
    "call_fn, a_shape, w_shape, bias_shape",
    [
        (lambda a, w, b: torch.addmm(b, a, w.T), (32, 64), (128, 64), (128,)),
        (lambda a, w, b: F.linear(a, w, b), (32, 64), (128, 64), (128,)),
    ],
)
def test_bias_bypass(wrapper_cls, weight_config, act_config, call_fn, a_shape, w_shape, bias_shape, device):
    """Wrapped bias is unconditionally bypassed in __torch_function__."""
    a = torch.randn(*a_shape, dtype=torch.bfloat16, device=device)
    w_wrapped = wrapper_cls(
        torch.randn(*w_shape, dtype=torch.bfloat16, device=device),
        weight_config=weight_config,
        activation_config=act_config,
    )

    bias = torch.randn(*bias_shape, dtype=torch.bfloat16, device=device)
    bias_wrapped = wrapper_cls(bias, weight_config=weight_config, activation_config=act_config)
    out_wrapped = call_fn(a, w_wrapped, bias_wrapped)
    out_ref = call_fn(a, w_wrapped, bias)
    assert torch.equal(out_wrapped, out_ref), "bias should not be fake-quantized"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("granularity", [PerRow(dim=-1), PerRow(dim=-2)])
def test_float8_rowwise_fake_quantize(granularity, device):
    """float8_rowwise_fake_quantize applies FP8 fake quantization."""
    weight_config = Float8FakeQuantizeConfig()
    if granularity.dim in (-2, 0):
        w = torch.randn(2048, 1024, device=device).T
    else:
        w = torch.randn(1024, 2048, device=device)
    result = float8_rowwise_fake_quantize(w, weight_config, granularity)
    assert type(result) is torch.Tensor
    assert result.shape == w.shape
    assert result.dtype == w.dtype
    sqnr = compute_error(result, w)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > 30, f"SQNR too low ({sqnr:.1f} dB)"


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config, sqnr_threshold",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
            26,
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            MXQuantizeConfig(),
            18,
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            MXQuantizeConfig(),
            15,
        ),
    ],
)
@pytest.mark.parametrize(
    "call_fn, a_shape, w_shape, bias_shape, out_shape",
    [
        # A_shape: (tokens, in_features) or (batch, tokens, in_features)
        # w_shape: (in_features, out_features), linear weight is (out_features, in_features)
        # bias_shape: () for no bias, (out_features,) for bias
        # out_shape: (tokens, out_features) or (batch, tokens, out_features)
        (lambda a, w, bias: torch.mm(a, w.T), (32, 1024), (2048, 1024), (), (32, 2048)),
        (
            lambda a, w, bias: torch.bmm(a, w.transpose(-2, -1)),
            (4, 32, 1024),
            (4, 2048, 1024),
            (),
            (4, 32, 2048),
        ),
        (lambda a, w, bias: F.linear(a, w), (32, 1024), (2048, 1024), (), (32, 2048)),
        (
            lambda a, w, bias: F.linear(a, w, bias),
            (32, 1024),
            (2048, 1024),
            (2048,),
            (32, 2048),
        ),
        (
            lambda a, w, bias: torch.matmul(a, w.T),
            (32, 1024),
            (2048, 1024),
            (),
            (32, 2048),
        ),
        (
            lambda a, w, bias: torch.addmm(bias, a, w.T),
            (32, 1024),
            (2048, 1024),
            (2048,),
            (32, 2048),
        ),
    ],
)
@pytest.mark.parametrize("device", target_devices)
def test_op_fake_quantize(
    wrapper_cls,
    weight_config,
    act_config,
    sqnr_threshold,
    call_fn,
    a_shape,
    w_shape,
    bias_shape,
    out_shape,
    device,
):
    """__torch_function__ fake-quantizes weight/activation and produces good SQNR."""

    activation_tensor = torch.randn(*a_shape, device=device, dtype=torch.bfloat16)
    weight_tensor = torch.randn(*w_shape, device=device, dtype=torch.bfloat16)

    # Prepare the wrapper tensor
    activation = torch.nn.Parameter(activation_tensor.clone())
    weight = torch.nn.Parameter(
        wrapper_cls(
            weight_tensor.clone(),
            activation_config=act_config,
            weight_config=weight_config,
        )
    )

    bias = torch.nn.Parameter(torch.randn(bias_shape, device=device))
    ref_bias = torch.nn.Parameter(bias.data.clone())

    # Prepare the reference
    ref_activation = torch.nn.Parameter(activation_tensor.clone())
    ref_weight = torch.nn.Parameter(weight_tensor.clone())

    # Run the function call
    learning_rate = 1  # set learning rate to 1 to ensure noises in new weights are not suppressed or amplified

    opt_params = [weight, bias] if bias_shape else [weight]
    optimizer = torch.optim.SGD(opt_params, lr=learning_rate)
    out = call_fn(activation, weight, bias)

    ref_opt_params = [ref_weight, ref_bias] if bias_shape else [ref_weight]
    ref_optimizer = torch.optim.SGD(ref_opt_params, lr=learning_rate)
    ref_out = call_fn(ref_activation, ref_weight, ref_bias)

    assert out.shape == out_shape
    assert out.shape == ref_out.shape

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    target = torch.ones_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()
    ref_loss = F.mse_loss(ref_out, target)
    ref_loss.backward()

    assert activation.grad is not None
    activation_grad_sqnr = compute_error(activation.grad, ref_activation.grad)
    assert activation_grad_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert activation_grad_sqnr > sqnr_threshold, f"Input grad SQNR too low ({activation_grad_sqnr:.1f} dB)"

    assert weight.grad is not None
    weight_grad_sqnr = compute_error(weight.grad, ref_weight.grad)
    assert weight_grad_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert weight_grad_sqnr > sqnr_threshold, f"Weight grad SQNR too low ({weight_grad_sqnr:.1f} dB)"

    if bias_shape:
        assert bias.grad is not None
        bias_grad_sqnr = compute_error(bias.grad, ref_bias.grad)
        assert bias_grad_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
        assert bias_grad_sqnr > sqnr_threshold, f"Bias grad SQNR too low ({bias_grad_sqnr:.1f} dB)"

    # Update weights
    optimizer.step()
    ref_optimizer.step()

    assert not torch.equal(weight_tensor, weight), "weight should be updated"
    assert not torch.equal(weight_tensor, ref_weight), "ref_weight is not updated. Bug in test_op_fake_quantize"
    new_weight_sqnr = compute_error(weight, ref_weight)
    assert new_weight_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert new_weight_sqnr > sqnr_threshold, f"New weight SQNR too low ({new_weight_sqnr:.1f} dB)"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config, sqnr_threshold",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 20),
        (
            Float8TrainingWeightWrapperTensor,
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
            20,
        ),
        (
            MXTrainingWeightWrapperTensor,
            MXQuantizeConfig(),
            MXQuantizeConfig(),
            15,
        ),
        (
            BlockMXTrainingWeightWrapperTensor,
            BlockMXQuantizeConfig(),
            MXQuantizeConfig(),
            17,
        ),
    ],
)
def test_op_grouped_mm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    """grouped_mm with fake-quantized weight produces good forward+backward SQNR."""

    s, e, k, n = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    a = torch.randn(s, k, dtype=torch.bfloat16, requires_grad=True, device=device)
    w = torch.randn(e, n, k, dtype=torch.bfloat16, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=device)
    a_ref = a.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(a_ref, w_ref.transpose(-2, -1), offs=offs)
    out = torch._grouped_mm(a, param.transpose(-2, -1), offs=offs)
    assert out.shape == (s, n)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.backward(torch.ones_like(ref_out))
    out.backward(torch.ones_like(out))
    assert a.grad is not None
    assert compute_error(a.grad, a_ref.grad) > sqnr_threshold, (
        f"Input grad SQNR too low ({compute_error(a.grad, a_ref.grad):.1f} dB)"
    )
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, (
        f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"
    )
