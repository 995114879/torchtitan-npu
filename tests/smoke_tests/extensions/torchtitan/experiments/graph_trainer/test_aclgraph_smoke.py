# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-NPU capture ownership and fixed document frame replay."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.config import apply_overrides
from torchtitan.experiments.graph_trainer.registry import POST_INIT_HOOKS

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import _load_graph_pass_components
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import aclgraph as aclgraph_module
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.aclgraph_runtime import (
    ACLGraphWrapper,
    _CaptureCoordinator,
)
from torchtitan_npu.models.deepseek_v4.aclgraph_metadata import reuse_metadata_buffers
from torchtitan_npu.models.deepseek_v4.config_registry import graph_trainer_deepseek_v4_debugmodel
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("mutate", [False, True], ids=["retained-view", "eager-source-write"])
def test_aclgraph_preserves_views_of_eager_results(npu_device, mutate):
    torch_npu.npu.set_device(npu_device)

    def step(x):
        stable = x * 7
        y = x * 2
        view = y.view(-1)
        if mutate:
            y.add_(3)
        return stable, y, view, view.neg()

    gm = make_fx(step, tracing_mode="fake")(torch.ones(2, device=npu_device))
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.mul.Tensor and node.args[1] == 2:
            node.meta["custom"] = {"module_fqn": "layers.0.moe.routed_experts"}
    compiled = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    assert [part["eager"] for part in compiled.meta["aclgraph_partitions"]] == [False, True, False]
    retained = []
    try:
        for values in ([2., 3.], [10., 20.], [100., 200.], [-1., 4.]):
            source = torch.tensor(values)
            actual = compiled(source.to(npu_device))
            y = 2 * source + (3 if mutate else 0)
            retained.append((actual, (7 * source, y, y, -y)))
            assert actual[1].untyped_storage().data_ptr() == actual[2].untyped_storage().data_ptr()
            for observed, expected in retained:
                torch.testing.assert_close(tuple(value.cpu() for value in observed), expected, rtol=0, atol=0)
        assert all(wrapper.graph is not None for wrapper in compiled._aclgraph_wrappers)
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


@pytest.mark.parametrize("unaligned", [False, True], ids=["variable-prefix", "unaligned-storage"])
def test_aclgraph_stages_sliced_buffers(npu_device, unaligned):
    torch_npu.npu.set_device(npu_device)

    def step(x):
        return x.as_strided((1,), (1,), 1).neg() if unaligned else x.neg()

    def make_input(size):
        if unaligned:
            raw = torch.zeros(10, dtype=torch.uint8, device=npu_device)
            value = raw[:8].view(torch.float32)
            value.copy_(torch.tensor([size, size + 1], dtype=torch.float32, device=npu_device))
            return value
        return torch.arange(size, size * 2, dtype=torch.float32, device=npu_device)[:2]

    gm = make_fx(step, tracing_mode="fake")(make_input(4))
    compiled = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    retained = []
    try:
        for size in (4, 6, 8, 10):
            actual = compiled(make_input(size))
            expected = -torch.tensor([size + 1] if unaligned else [size, size + 1], dtype=torch.float32)
            retained.append((actual, expected))
            for observed, reference in retained:
                torch.testing.assert_close(observed.cpu(), reference, rtol=0, atol=0)
        assert len(compiled._aclgraph_wrappers) == 1
        assert compiled._aclgraph_wrappers[0].graph is not None
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


def test_aclgraph_preserves_output_storage_views(npu_device):
    torch_npu.npu.set_device(npu_device)

    def step(x):
        base = 2 * x
        sliced = base.as_strided((2, 2), (4, 1), 5)
        return {"views": [sliced, base.t(), base.view(torch.int32), sliced], "base": base}

    gm = make_fx(step, tracing_mode="fake")(torch.ones(4, 4, device=npu_device))
    compiled = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    retained = []
    try:
        for start in (0, 10, -20, 30):
            source = torch.arange(start, start + 16, dtype=torch.float32).reshape(4, 4)
            actual = compiled(source.to(npu_device))
            assert actual["views"][0] is actual["views"][3]
            assert all(
                view.untyped_storage().data_ptr() == actual["base"].untyped_storage().data_ptr()
                for view in actual["views"]
            )
            assert actual["views"][1].stride() == (1, 4)
            assert actual["views"][0].storage_offset() == 5
            assert actual["views"][0].stride() == (4, 1)
            actual["views"][0].add_(100)
            reference = 2 * source
            reference.as_strided((2, 2), (4, 1), 5).add_(100)
            retained.append((actual, reference))
            torch_npu.npu.synchronize()
            for observed, expected in retained:
                torch.testing.assert_close(observed["base"].cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(observed["views"][2].cpu(), expected.view(torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(observed["views"][0].cpu(), expected[1:3, 1:3], rtol=0, atol=0)
        assert len(compiled._aclgraph_wrappers) == 1
        assert compiled._aclgraph_wrappers[0].graph is not None
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


@pytest.mark.parametrize("eager_mutation", [False, True], ids=["captured-write", "eager-write"])
def test_aclgraph_output_aliases_preserve_sgd_updates(npu_device, eager_mutation):
    torch_npu.npu.set_device(npu_device)

    def joint(x, weight, none_grad):
        loss = (x * weight).square().sum()
        gradient, = torch.autograd.grad(loss, weight)
        view = gradient.view(4, 4)
        gradient.add_(0.25)
        return loss, none_grad, gradient, view, gradient * 2

    weight = torch.ones(16, device=npu_device, requires_grad=True)
    example_inputs = (torch.ones_like(weight), weight, None)
    gm = make_fx(joint, tracing_mode="fake")(*example_inputs)
    for node in gm.graph.nodes:
        if eager_mutation and node.target == torch.ops.aten.add_.Tensor:
            node.meta["custom"] = {"module_fqn": "layers.0.moe.routed_experts"}
    compiled = aclgraph_module.aclgraph_pass(gm, example_inputs, static_input_indices=[1, 2])
    reference_weight = weight.detach().clone().requires_grad_()
    retained = []
    try:
        for step in range(4):
            x = torch.tensor([1., -0.5, 0.25, -1.] * 4, device=npu_device) * (1 + step % 3)
            loss, none_grad, gradient, view, doubled = compiled(x, weight, None)
            assert none_grad is None
            reference_loss = (x * reference_weight).square().sum()
            reference_gradient, = torch.autograd.grad(reference_loss, reference_weight)
            reference_gradient.add_(0.25)
            torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
            torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0)
            torch.testing.assert_close(view, reference_gradient.view(4, 4), rtol=0, atol=0)
            torch.testing.assert_close(doubled, reference_gradient * 2, rtol=0, atol=0)
            assert gradient.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
            retained.append((gradient, reference_gradient))
            with torch.no_grad():
                weight.add_(gradient, alpha=-1 / 256)
                reference_weight.add_(reference_gradient, alpha=-1 / 256)
            torch.testing.assert_close(weight, reference_weight, rtol=0, atol=0)
        for gradient, expected in retained:
            torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        assert [part["eager"] for part in compiled.meta["aclgraph_partitions"]] == (
            [False, True, False] if eager_mutation else [False]
        )
        assert all(wrapper.graph is not None for wrapper in compiled._aclgraph_wrappers)
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


def test_aclgraph_preserves_returned_input_views(npu_device):
    torch_npu.npu.set_device(npu_device)

    def step(x):
        return x.view(-1), x.neg() * 7

    gm = make_fx(step, tracing_mode="fake")(torch.ones(2, device=npu_device))
    compiled = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    retained = []
    try:
        for values in ([1., 2.], [2., 3.], [3., 4.], [-1., 5.]):
            source = torch.tensor(values)
            device_source = source.to(npu_device)
            actual = compiled(device_source)
            torch_npu.npu.synchronize()
            retained.append((actual, (source, -7 * source)))

            assert actual[0].data_ptr() == device_source.data_ptr()
            for observed, expected in retained:
                torch.testing.assert_close(tuple(value.cpu() for value in observed), expected, rtol=0, atol=0)
        assert compiled._aclgraph_wrappers
        assert all(wrapper.graph is not None for wrapper in compiled._aclgraph_wrappers)
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


def test_aclgraph_preserves_sliced_input_storage_offsets(npu_device):
    torch_npu.npu.set_device(npu_device)

    def step(x):
        return x.as_strided((1,), (1,), 2).neg(), x.neg() * 7

    example = torch.arange(11., 15., device=npu_device)[1:]
    gm = make_fx(step, tracing_mode="fake")(example)
    compiled = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    retained = []
    try:
        for start in (11., 21., 31., 41.):
            base = torch.arange(start, start + 4)
            actual = compiled(base.to(npu_device)[1:])
            torch_npu.npu.synchronize()
            retained.append((actual, (torch.tensor([-start - 2]), -7 * base[1:])))

            for observed, expected in retained:
                torch.testing.assert_close(tuple(value.cpu() for value in observed), expected, rtol=0, atol=0)
        assert compiled._aclgraph_wrappers
        assert all(wrapper.graph is not None for wrapper in compiled._aclgraph_wrappers)
    finally:
        aclgraph_module._close_aclgraph_trainer(
            SimpleNamespace(_traced_step=SimpleNamespace(gm=compiled)), lambda: None,
        )


@torch.no_grad()
@pytest.mark.parametrize("shared", [False, True], ids=["private", "shared"])
def test_aclgraph_replays_sliced_region_inputs(npu_device, shared):
    torch_npu.npu.set_device(npu_device)
    coordinator = _CaptureCoordinator()
    wrapper = ACLGraphWrapper(
        lambda x: x.as_strided((1,), (1,), 2).neg(),
        static_input_indices=[], capture_coordinator=coordinator,
        reusable_input_indices=[0] if shared else [],
    )
    coordinator.wrappers.append(wrapper)
    retained = []
    try:
        for start in (11., 21., 31., 41.):
            source = torch.arange(start, start + 4, device=npu_device)[1:]
            retained.append((wrapper(source), torch.tensor([-start - 2])))
            torch_npu.npu.synchronize()

            for actual, expected in retained:
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert wrapper.graph is not None
    finally:
        wrapper.teardown()
        coordinator.teardown()


@torch.no_grad()
def test_aclgraph_replays_compression_with_capacity_for_a_16k_sequence(npu_device):
    torch_npu.npu.set_device(npu_device)
    config = graph_trainer_deepseek_v4_debugmodel()
    config.compile.pass_pipeline = "mutation-functionalization+aclgraph"
    config.training.local_batch_size = 1
    config.training.seq_len = 16384
    # The reference complex cache cannot be indexed on NPU; use the existing
    # real-valued RoPE implementation for both eager and captured compression.
    config.override.imports = ["torchtitan_npu.override.common.rope.workaround"]
    apply_overrides(config.override, config)
    with torch.device("meta"):
        model = config.model_spec.model.build()
    trainer = SimpleNamespace(
        config=config, model_parts=[model], close=lambda: None,
        parallel_dims=SimpleNamespace(pp_enabled=False, cp_enabled=False, tp_enabled=False),
    )
    _load_graph_pass_components(config.compile.pass_pipeline)
    POST_INIT_HOOKS[config.compile.pass_pipeline](trainer)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        compressor = model.config.layers[2].attention.compressor.build()
        compressor.init_states(buffer_device=torch.device("cpu"))
        source = torch.randn(1, 16384, compressor.wkv.in_features).to(npu_device)
    compressor = compressor.to(npu_device)
    key = model.config.layers[2].attention.inner_attention.attention_metadata_key

    def compress(x, metadata):
        return compressor(x, metadata.position_k, cu_seqlens=metadata.swa_k.cu_seqlens)

    coordinator = _CaptureCoordinator()
    wrapper = ACLGraphWrapper(compress, static_input_indices=[0, 1], capture_coordinator=coordinator)
    coordinator.wrappers.append(wrapper)
    buffers = {}
    captured_graph = None
    try:
        # Warm up and capture with 128 documents, then replay with 64 and 32.
        for length in (128, 128, 256, 512):
            positions = (torch.arange(16384, device=npu_device) % length).unsqueeze(0)
            metadata = reuse_metadata_buffers(model._get_attention_metadata(positions), buffers)[key]
            compact = DeepSeekV4Metadata.from_positions(positions, 4)
            expected = compressor(source, positions, cu_seqlens=compact.swa_k.cu_seqlens)

            actual = wrapper(source, metadata)

            torch_npu.npu.synchronize()
            assert metadata.q.cu_seqlens.shape == (129,)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            if captured_graph is not None:
                assert wrapper.graph is captured_graph
            if wrapper.graph is not None:
                captured_graph = wrapper.graph
        assert captured_graph is not None
    finally:
        torch_npu.npu.synchronize()
        wrapper.teardown()
        coordinator.teardown()


def test_aclgraph_releases_nested_producer_inputs_and_preserves_state(npu_device):
    torch_npu.npu.set_device(npu_device)
    source = torch.ones(4096, dtype=torch.bfloat16, device=npu_device)
    accumulator = torch.zeros((), dtype=torch.float32, device=npu_device)
    state = {"values": [accumulator]}
    coordinator = _CaptureCoordinator()
    capture_input_refs = []

    def produce(x):
        return {"chunks": tuple((2 * x).split(2048)), "history": x.square()}

    def consume(payload, state):
        if consumer.warmed_up and consumer.graph is None:
            capture_input_refs.extend(weakref.ref(value) for value in payload["chunks"])
        value = state["values"][0]
        value.add_(sum(chunk.sum(dtype=torch.float32) for chunk in payload["chunks"]))
        return {"state": value, "gradient": payload["chunks"][0] + 1}

    producer = ACLGraphWrapper(
        produce,
        static_input_indices=[0],
        clone_output_indices=[2],
        capture_coordinator=coordinator,
    )
    consumer = ACLGraphWrapper(
        consume,
        static_input_indices=[0, 1],
        borrowed_input_indices=[0],
        capture_coordinator=coordinator,
    )
    coordinator.wrappers.extend((producer, consumer))

    def run_step():
        payload = producer(source)
        return consumer({"chunks": payload["chunks"]}, state), payload["history"]

    retained_gradients = []
    retained_history = []
    captured_graphs = None
    try:
        for value, expected_state in ((1, 8192), (2, 24576), (3, 49152)):
            source.fill_(value)
            result, history = run_step()
            torch_npu.npu.synchronize()
            assert result["state"] is accumulator
            assert accumulator.item() == expected_state
            retained_gradients.append((result["gradient"], 2 * value + 1))
            retained_history.append((history, value**2))
            for gradient, expected in retained_gradients:
                torch.testing.assert_close(gradient.cpu(), torch.full((2048,), expected, dtype=torch.bfloat16))
            for saved, expected in retained_history:
                torch.testing.assert_close(saved.cpu(), torch.full((4096,), expected, dtype=torch.bfloat16))
            if value == 2:
                captured_graphs = (producer.graph, consumer.graph)
                gc.collect()
                assert len(capture_input_refs) == 2
                assert all(reference() is None for reference in capture_input_refs)
        assert captured_graphs is not None
        assert captured_graphs[0] is producer.graph
        assert captured_graphs[1] is consumer.graph
    finally:
        torch_npu.npu.synchronize()
        consumer.teardown()
        producer.teardown()
        coordinator.teardown()
