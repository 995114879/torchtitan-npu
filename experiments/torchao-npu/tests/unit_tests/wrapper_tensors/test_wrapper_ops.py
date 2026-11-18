# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re

import pytest
import torch
import torch.nn.functional as F
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.utils import TorchAOBaseTensor
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
    Float8TrainingWeightWrapperTensor,
    MXTrainingWeightWrapperTensor,
)


def _init_single_rank_gloo_mesh(pg_file):
    """Init a single-rank gloo process group plus a 1-D DeviceMesh.

    spmd_types >= 0.2 resolves axis names against an ambient ``DeviceMesh``
    (via ``set_current_mesh``); a bare ``ProcessGroup`` is no longer accepted.
    ``dist.init_process_group`` returns None, so build the mesh explicitly and
    return it.
    """
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=pg_file.as_uri())
    return dist.init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
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
def test_spmd_shard_preserves_subclass(wrapper_cls, weight_config, act_config, tmp_path):
    """spmd.shard (I -> S) preserves the wrapper through the full parameter lifecycle.

    The spmd_types backend shards parameters with the ``replicate_to_varying``
    custom op. If that op is missing from ``_ops_to_preserve_subclass``, the
    parameter silently degrades to a plain tensor and no quantization transform
    ever fires. CPU-only: the custom op is device-agnostic, only CPU + gloo are
    needed.
    """
    spmd = pytest.importorskip("spmd_types")
    import torch.distributed as dist

    pg_file = tmp_path / "spmd_wrapper_store"
    try:
        mesh = _init_single_rank_gloo_mesh(pg_file)
        weight = torch.randn(8, 64, 128)
        wrapper = wrapper_cls(weight, activation_config=act_config, weight_config=weight_config)

        with spmd.set_current_mesh(mesh):
            sharded = spmd.shard(wrapper, "dp", src=spmd.I, dst=spmd.S(0))
        assert isinstance(sharded, wrapper_cls)
        assert sharded.weight_config is weight_config
        assert sharded.activation_config is act_config
        assert torch.equal(sharded.to_tensor(), weight)

        # Parameter lifecycle: nn.Parameter -> empty_like (to_empty) -> in-place init
        param = torch.nn.Parameter(sharded)
        assert isinstance(param, wrapper_cls)

        moved = torch.empty_like(param)
        assert isinstance(moved, wrapper_cls)

        torch.nn.init.normal_(moved, 0.0, 0.02)
        assert isinstance(moved, wrapper_cls)
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            pg_file.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
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
@pytest.mark.parametrize("rank", [0, 1])
def test_spmd_replicate_to_varying_preserves_subclass(wrapper_cls, weight_config, act_config, rank):
    """The replicate_to_varying custom op re-wraps its output with the split data."""
    pytest.importorskip("spmd_types")
    weight = torch.randn(8, 64, 128)
    wrapper = wrapper_cls(weight, activation_config=act_config, weight_config=weight_config)

    result = torch.ops.spmd_types.replicate_to_varying.default(wrapper, 2, 0, rank, stack=False)
    assert isinstance(result, wrapper_cls)
    assert result.weight_config is weight_config
    assert result.activation_config is act_config
    assert torch.equal(result.to_tensor(), torch.chunk(weight, 2, dim=0)[rank])


def _prequantized_wrapper(shape=(4, 64, 128)):
    """Wrap a hand-built ``BlockMXTensor`` without running a quantization kernel."""
    config = BlockMXQuantizeConfig()
    act_config = MXQuantizeConfig()
    quant_block_size = config.block_size * 2
    rows, cols = shape[-2], shape[-1]
    qdata = torch.randint(0, 256, shape, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale1 = torch.ones((*shape[:-2], rows, cols // quant_block_size, 2), dtype=torch.float8_e8m0fnu)
    scale2 = torch.ones((*shape[:-2], rows // quant_block_size, cols, 2), dtype=torch.float8_e8m0fnu)
    data = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config, act_config)
    wrapper = BlockMXTrainingWeightWrapperTensor(data, weight_config=config, activation_config=act_config)
    return wrapper, data


@pytest.mark.parametrize(
    "shape, op_func",
    [
        ((4, 64, 128), lambda x: x.clone()),
        ((4, 64, 128), lambda x: x.detach()),
        ((4, 64, 128), lambda x: x.view(2, 2, 64, 128)),
        ((4, 64, 128), lambda x: x.transpose(-1, -2)),
        ((2, 2, 64, 128), lambda x: x.permute(1, 0, 2, 3)),
        ((4, 64, 128), lambda x: x.to(torch.float16)),
        ((64, 128), lambda x: x.t()),
        ((4, 64, 128), lambda x: x.as_strided((4, 64, 128), (8192, 128, 1))),
    ],
    ids=["clone", "detach", "view", "transpose_trailing", "permute_leading", "to_fp16", "t", "as_strided"],
)
def test_wrapper_preserves_subclass_with_prequantized_data(shape, op_func):
    """The preserved ops re-wrap a quantized ``_data`` and keep its payload."""
    wrapper, data = _prequantized_wrapper(shape)

    with torch._C.DisableTorchFunctionSubclass():
        result = op_func(wrapper)
        reference = op_func(data)

    assert isinstance(result, BlockMXTrainingWeightWrapperTensor)
    assert isinstance(result._data, BlockMXTensor)
    assert result.weight_config is wrapper.weight_config
    assert result.activation_config is wrapper.activation_config
    for name in BlockMXTensor.tensor_data_names:
        assert torch.equal(getattr(result._data, name).view(torch.uint8), getattr(reference, name).view(torch.uint8)), (
            f"{name} changed"
        )


def test_wrapper_copy_in_place_with_prequantized_data():
    """``copy_`` writes the components into the target wrapper and returns it."""
    target, target_data = _prequantized_wrapper()
    source, source_data = _prequantized_wrapper()

    result = target.copy_(source)

    assert result is target
    for name in BlockMXTensor.tensor_data_names:
        assert torch.equal(
            getattr(target_data, name).view(torch.uint8), getattr(source_data, name).view(torch.uint8)
        ), f"{name} was not copied"


@pytest.mark.parametrize(
    "shape, op_func",
    [
        ((4, 64, 128), lambda x: x[0]),
        ((4, 64, 128), lambda x: x[0:2]),
        ((4, 64, 128), lambda x: x[torch.tensor([0, 2])]),
        ((4, 64, 128), lambda x: x.unsqueeze(0)),
        ((4, 64, 128), lambda x: x.new_zeros(2, 64, 128)),
        ((4, 64, 128), lambda x: torch.split(x, 2)),
        ((4, 64, 128), lambda x: x.pin_memory()),
    ],
    ids=["select", "slice", "index", "unsqueeze", "new_zeros", "split", "pin_memory"],
)
def test_wrapper_prequantized_dispatch_rejections(shape, op_func):
    """Ops outside ``BlockMXTensor``'s whitelist raise, naming the quantized class."""
    wrapper, _ = _prequantized_wrapper(shape)

    with pytest.raises(NotImplementedError, match="BlockMXTensor"), torch._C.DisableTorchFunctionSubclass():
        op_func(wrapper)


@pytest.mark.parametrize(
    "shape, op_func, error",
    [
        ((4, 64, 128), lambda x: x.transpose(0, 1), ValueError),
        ((4, 64, 128), lambda x: x.view(4, 2, 32, 128), ValueError),
        ((4, 64, 128), lambda x: x.as_strided((2, 64, 128), (8192, 128, 1)), NotImplementedError),
        ((4, 64, 128), lambda x: x.to(torch.float8_e4m3fn), ValueError),
    ],
    ids=["transpose_crosses_boundary", "view_moves_block_boundary", "as_strided_changes_layout", "to_low_precision"],
)
def test_wrapper_prequantized_dispatch_rejects_invalid_arguments(shape, op_func, error):
    """Argument combinations ``BlockMXTensor`` cannot represent are rejected."""
    wrapper, _ = _prequantized_wrapper(shape)

    with pytest.raises(error), torch._C.DisableTorchFunctionSubclass():
        op_func(wrapper)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    ],
)
def test_wrapper_dispatch_non_preserved_op(wrapper_cls, weight_config, act_config, func):
    """Dispatch of an op not in _ops_to_preserve_subclass returns a plain tensor."""
    w = torch.randn(4, 64, 128, device="cpu")
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = func(wrapper, wrapper)
    assert type(result) is torch.Tensor
    assert not isinstance(result, BaseTrainingWeightWrapperTensor)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        torch.mm,
        F.linear,
    ],
)
def test_wrapper_torch_function_arg_a_is_wrapper(wrapper_cls, weight_config, act_config, func):
    """__torch_function__ asserts A is not a wrapper."""
    w = torch.randn(64, 128, device="cpu")
    w1 = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)
    w2 = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)
    with pytest.raises(AssertionError, match=rf"^A should not be a {wrapper_cls.__name__}$"):
        func(w1, w2)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
@pytest.mark.parametrize("func", [torch.addmm, F.linear])
def test_wrapper_torch_function_arg_b_not_wrapper(wrapper_cls, weight_config, act_config, func):
    """__torch_function__ asserts B is a wrapped weight."""
    bias = wrapper_cls(
        torch.randn(128, device="cpu"),
        weight_config=weight_config,
        activation_config=act_config,
    )
    a = torch.randn(16, 64, device="cpu")
    b = torch.randn(64, 128, device="cpu")
    with pytest.raises(AssertionError, match=rf"^B should be a {wrapper_cls.__name__}$"):
        func(bias, a, b) if func is torch.addmm else func(a, b, bias)


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
@pytest.mark.parametrize(
    "a_shape, w_shape, call_fn, kwargs",
    [
        ((16, 64), (128, 64), lambda a, w: torch.mm(a, w.T), {}),
        ((16, 64), (128, 64), lambda a, w: torch.matmul(a, w.T), {}),
        (
            (16, 64),
            (128, 64),
            lambda a, w, *, bias: torch.addmm(bias, a, w.T),
            {"bias_shape": (128,)},
        ),
        ((16, 64), (128, 64), lambda a, w: F.linear(a, w), {}),
        (
            (8, 1024),
            (4, 2048, 1024),
            lambda a, w: torch._grouped_mm(a, w.transpose(-2, -1)),
            {},
        ),
    ],
)
def test_wrapper_torch_function_activation_quantized_tensor(
    wrapper_cls, weight_config, act_config, a_shape, w_shape, call_fn, kwargs
):
    """__torch_function__ asserts activation is not a TorchAOBaseTensor when act_config is set."""

    class DummyTensor(TorchAOBaseTensor):
        @classmethod
        def __torch_function__(cls, func, types, args, kwargs=None):
            if func in (
                torch.mm,
                torch.bmm,
                torch.addmm,
                torch.matmul,
                torch._grouped_mm,
                F.linear,
            ):
                return NotImplemented
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))

    a = torch.randn(*a_shape, device="cpu").as_subclass(DummyTensor)
    b = wrapper_cls(
        torch.randn(*w_shape, device="cpu"),
        activation_config=act_config,
        weight_config=weight_config,
    )
    expected_match = (
        r"^When an activation config is specified, the activation must not be a quantized tensor, got "
        + re.escape(str(type(a)))
        + "$"
    )

    resolved = {}
    for k, v in kwargs.items():
        if k.endswith("_shape"):
            resolved[k[: -len("_shape")]] = torch.randn(*v, device="cpu")
        else:
            resolved[k] = v

    with pytest.raises(AssertionError, match=expected_match):
        call_fn(a, b, **resolved)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockMXTrainingWeightWrapperTensor, BlockMXQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_wrapper_torch_function_non_fake_quant_op(wrapper_cls, weight_config, act_config):
    """__torch_function__ on a non-fake-quant op passes through without fake quantization."""
    w1 = torch.randn(64, 128, device="cpu")
    w2 = torch.randn(64, 128, device="cpu")
    wrapper = wrapper_cls(w2.clone(), weight_config=weight_config, activation_config=act_config)
    result = torch.add(w1, wrapper)
    expected = torch.add(w1, w2)
    assert type(result) is torch.Tensor
    assert not isinstance(result, TorchAOBaseTensor)
    assert torch.equal(result, expected)
