# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host partitioning and capture/replay ownership contracts on CPU."""

from contextlib import contextmanager
from operator import getitem
from types import SimpleNamespace
import weakref

import torch
import torch_npu
import pytest
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.traceback import annotate_fn, preserve_node_meta
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map
from torchtitan.config import apply_overrides, derive
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY, POST_INIT_HOOKS

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import aclgraph as aclgraph_module
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.aclgraph import (
    reinplace_loss_accumulation_pass,
)
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.aclgraph_runtime import (
    ACLGraphWrapper,
    _CaptureCoordinator,
    _PartitionInterpreter,
)
from torchtitan_npu.models.deepseek_v4.aclgraph_metadata import reuse_metadata_buffers
from torchtitan_npu.models.deepseek_v4.config_registry import (
    deepseek_v4_debugmodel,
    graph_trainer_deepseek_v4_debugmodel,
)
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override.deepseek_v4.asc_metadata import AscV4Metadata


@pytest.fixture
def mtp_aclgraph_trainer(monkeypatch):
    config = graph_trainer_deepseek_v4_debugmodel()
    mtp_config = deepseek_v4_debugmodel(num_mtp_layers=1)
    config.model_spec.model = derive(mtp_config.model_spec.model, type(config.model_spec.model))
    config.loss = mtp_config.loss
    config.compile.pass_pipeline = "mutation-functionalization+aclgraph"
    config.compile.ep_overlap.enabled = False
    with torch.device("meta"):
        model = config.model_spec.model.build()
    monkeypatch.setattr(torch_npu.npu, "get_task_queue_enable", lambda: 1, raising=False)
    return SimpleNamespace(
        config=config, model_parts=[model], close=lambda: None,
        parallel_dims=SimpleNamespace(pp_enabled=False, cp_enabled=False, tp_enabled=False),
    )


@pytest.mark.parametrize("force_load_balance", [False, True], ids=["real-routing", "balanced-routing"])
def test_aclgraph_preparation_rejects_mtp_before_modifying_the_model(mtp_aclgraph_trainer, force_load_balance):
    trainer = mtp_aclgraph_trainer
    model = trainer.model_parts[0]
    for layer in (*model.config.layers, *model.config.mtp_layers):
        layer.moe.router._debug_force_load_balance = force_load_balance
    layers = (*model.layers.values(), *model.mtp_layers)
    dispatchers = [layer.moe.routed_experts.token_dispatcher for layer in layers]
    preprocess = model.preprocess_inputs
    attention_metadata = model._get_attention_metadata

    with pytest.raises(ValueError, match="ACLGraph does not support MTP.*num_mtp_layers=0 or disable ACLGraph"):
        POST_INIT_HOOKS[trainer.config.compile.pass_pipeline](trainer)

    assert model.preprocess_inputs == preprocess
    assert model._get_attention_metadata == attention_metadata
    assert not hasattr(model, "_static_metadata_buffers")
    assert all(layer.moe.routed_experts.token_dispatcher is original for layer, original in zip(layers, dispatchers))


@pytest.mark.parametrize("pp_enabled", [False, True], ids=["pp1", "pp2"])
@pytest.mark.parametrize("disable_passes", [True, False], ids=["pass-disabled", "all-passes-disabled"])
def test_disabling_aclgraph_preserves_mtp_model_preparation(mtp_aclgraph_trainer, disable_passes, pp_enabled):
    trainer = mtp_aclgraph_trainer
    trainer.parallel_dims.pp_enabled = pp_enabled
    if disable_passes:
        trainer.config.compile.disable_passes.append("aclgraph_pass")
    else:
        trainer.config.compile.enable_passes = False
    model = trainer.model_parts[0]
    layers = (*model.layers.values(), *model.mtp_layers)
    dispatchers = [layer.moe.routed_experts.token_dispatcher for layer in layers]
    preprocess = model.preprocess_inputs
    attention_metadata = model._get_attention_metadata

    POST_INIT_HOOKS[trainer.config.compile.pass_pipeline](trainer)

    assert len(model.mtp_layers) == 1
    assert model.preprocess_inputs == preprocess
    assert model._get_attention_metadata == attention_metadata
    assert not hasattr(model, "_static_metadata_buffers")
    assert all(layer.moe.routed_experts.token_dispatcher is original for layer, original in zip(layers, dispatchers))


def test_aclgraph_rejects_pipeline_parallelism_before_preparation(monkeypatch):
    config = graph_trainer_deepseek_v4_debugmodel()
    config.compile.pass_pipeline = "mutation-functionalization+aclgraph"
    config.compile.ep_overlap.enabled = False
    effects = []

    def get_queue_mode():
        effects.append("read queue")
        return 2

    def original_pipeline(*args, **kwargs):
        effects.append("construct pipeline")
        return []

    monkeypatch.setattr(torch_npu.npu, "get_task_queue_enable", get_queue_mode, raising=False)
    monkeypatch.setattr(
        torch_npu.npu, "set_task_queue_enable", lambda mode: effects.append(("set queue", mode)), raising=False,
    )
    monkeypatch.setitem(PASS_PIPELINE_REGISTRY, "mutation-functionalization", original_pipeline)
    original_close = lambda: None
    trainer = SimpleNamespace(
        config=config,
        parallel_dims=SimpleNamespace(pp_enabled=True, cp_enabled=False),
        model_parts=[SimpleNamespace(prepare_aclgraph=lambda **kwargs: effects.append(("prepare model", kwargs)))],
        close=original_close,
    )

    with pytest.raises(
        ValueError, match="ACLGraph does not support pipeline parallelism; set PP=1 or disable ACLGraph"
    ):
        POST_INIT_HOOKS[config.compile.pass_pipeline](trainer)
    with pytest.raises(
        ValueError, match="ACLGraph does not support pipeline parallelism; set PP=1 or disable ACLGraph"
    ):
        PASS_PIPELINE_REGISTRY[config.compile.pass_pipeline](
            SimpleNamespace(num_static_inputs=0), config, parallel_dims=trainer.parallel_dims,
        )

    assert effects == []
    assert trainer.close is original_close


@pytest.mark.parametrize(
    "batch_size,seq_len,alignment,expected_capacity",
    [(1, 4096, 128, 32), (1, 16384, 128, 128), (2, 4096, 128, 64), (1, 4096, 256, 16), (1, 256, None, 256)],
    ids=["s4k", "s16k", "batch2", "align256", "unknown-loader"],
)
def test_aclgraph_preparation_sizes_document_frames_for_the_training_config(
    monkeypatch, batch_size, seq_len, alignment, expected_capacity,
):
    config = graph_trainer_deepseek_v4_debugmodel()
    config.compile.pass_pipeline = "mutation-functionalization+aclgraph"
    config.training.local_batch_size = batch_size
    config.training.seq_len = seq_len
    if alignment is None:
        config.dataloader = SimpleNamespace()
    else:
        config.dataloader.per_doc_alignment = alignment
    with torch.device("meta"):
        model = config.model_spec.model.build()
    trainer = SimpleNamespace(
        config=config, model_parts=[model], close=lambda: None,
        parallel_dims=SimpleNamespace(pp_enabled=False, cp_enabled=False, tp_enabled=False),
    )
    monkeypatch.setattr(torch_npu.npu, "get_task_queue_enable", lambda: 1, raising=False)

    dispatchers = [layer.moe.routed_experts.token_dispatcher for layer in model.layers.values()]
    POST_INIT_HOOKS[config.compile.pass_pipeline](trainer)
    assert all(
        layer.moe.routed_experts.token_dispatcher is original
        for layer, original in zip(model.layers.values(), dispatchers, strict=True)
    )

    # Consume the prepared frames in the real ratio-4 overlap compressor.
    # Full documents and twice-longer documents share the same frame storage.
    core = model.config.layers[2].attention.inner_attention
    buffers = {}
    address = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        compressor = model.config.layers[2].attention.compressor.build()
        compressor.init_states(buffer_device=torch.device("cpu"))
        x = torch.randn(batch_size, seq_len, compressor.wkv.in_features)
        for length in (alignment or 128, 2 * (alignment or 128)):
            positions = (torch.arange(seq_len) % length).repeat(batch_size, 1)
            metadata = reuse_metadata_buffers(
                model._get_attention_metadata(positions), buffers
            )[core.attention_metadata_key]
            compact = DeepSeekV4Metadata.from_positions(positions, 4)
            documents = batch_size * seq_len // length

            assert metadata.q.cu_seqlens.tolist() == (
                [0] * (expected_capacity - documents) + list(range(0, batch_size * seq_len + 1, length))
            )
            assert metadata.q.seqused.tolist() == [0] * (expected_capacity - documents) + [length] * documents
            actual = compressor(x, positions, cu_seqlens=metadata.swa_k.cu_seqlens)
            expected = compressor(x, positions, cu_seqlens=compact.swa_k.cu_seqlens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if address is not None:
                assert metadata.q.cu_seqlens.data_ptr() == address
            address = metadata.q.cu_seqlens.data_ptr()


@pytest.mark.parametrize("quant_mode", [None, 1], ids=["bf16", "fp8"])
def test_aclgraph_preprocess_builds_backend_metadata_from_fixed_frames(monkeypatch, dsv4, quant_mode):
    quantized = None
    if quant_mode is not None:
        quantized = pytest.importorskip("torchao_npu.quantized_modules.deepseek_v4.sparse_attention")

    recipe = graph_trainer_deepseek_v4_debugmodel()
    recipe.compile.pass_pipeline = "mutation-functionalization+aclgraph"
    recipe.training.local_batch_size = 1
    recipe.training.seq_len = 512
    recipe.dataloader.per_doc_alignment = 128
    recipe.override.imports = [
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"num_heads": 16, "index_num_heads": 8}),
    ]
    apply_overrides(recipe.override, recipe)
    if quantized is not None:
        for layer in recipe.model_spec.model.layers:
            layer.attention.inner_attention = derive(
                layer.attention.inner_attention, quantized.QuantizedV4SparseAttention.Config, quant_mode=quant_mode,
            )
    with torch.device("meta"):
        model = recipe.model_spec.model.build()
    parallel_dims = SimpleNamespace(pp_enabled=False, cp_enabled=False, tp_enabled=False)
    trainer = SimpleNamespace(config=recipe, model_parts=[model], close=lambda: None, parallel_dims=parallel_dims)
    monkeypatch.setattr(torch_npu.npu, "get_task_queue_enable", lambda: 1, raising=False)
    POST_INIT_HOOKS[recipe.compile.pass_pipeline](trainer)
    monkeypatch.setattr(dsv4.cann_ops, "calls", [])
    quant_calls = []

    def quant_metadata(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return torch.full((64, kwargs["cu_seqlens_q"].numel()), 17, dtype=torch.int32)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "quant_lightning_indexer_metadata", quant_metadata, raising=False
    )
    addresses = None
    tokens = torch.arange(512).unsqueeze(0)
    for lengths, boundaries in [((128, 128, 256), [0, 0, 128, 256, 512]), ((256, 256), [0, 0, 0, 256, 512])]:
        positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
        _, _, extra = model.preprocess_inputs(
            {"input": tokens, "positions": positions}, tokens, parallel_dims=parallel_dims,
            device=torch.device("cpu"), parallelism=SimpleNamespace(spmd_backend="partial_dtensor"),
        )
        entries = {entry.ratio: entry for entry in extra["attention_metadata"].values()}
        assert set(entries) == {0, 4, 128}
        for ratio, entry in entries.items():
            assert entry.cu_seqlens.tolist() == boundaries
            assert entry.cu_seqlens is entry.q.cu_seqlens
            assert entry.q.cu_seqlens.tolist() == boundaries
            assert entry.q.seqused.tolist() == [0] * (4 - len(lengths)) + list(lengths)
            assert entry.position_k is extra["positions"]
            if ratio:
                assert entry.cmp_k.cu_seqlens.tolist() == [value // ratio for value in boundaries]
            if quantized is not None and ratio == 4:
                assert type(entry) is quantized.QuantV4Metadata
                assert entry.quant_mode == quant_mode
                assert torch.all(entry.aicpu_metadata.lightning_indexer.fwd == 17)
                assert quant_calls[-1][0][4] == quant_mode
                assert quant_calls[-1][1]["cu_seqlens_q"].tolist() == boundaries
            else:
                assert type(entry) is AscV4Metadata
        assert entries[4].aicpu_metadata.lightning_indexer.bwd is not None
        current = [value.data_ptr() for value in tree_leaves(extra) if isinstance(value, torch.Tensor)]
        if addresses is not None:
            assert current == addresses
        addresses = current
    assert len(quant_calls) == (2 if quantized is not None else 0)
    assert sum(name == "sparse_flash_mla_metadata" for name, _, _ in dsv4.cann_ops.calls) == 6


@pytest.fixture
def capture(monkeypatch):
    """Simulate the device recording boundary; callers supply replay execution."""
    capture = SimpleNamespace(graph=None, recordings=[])
    monkeypatch.setattr(torch_npu.npu, "NPUGraph", lambda: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(torch_npu.npu, "graph_pool_handle", lambda: (0, 1))
    monkeypatch.setattr(torch_npu.npu, "synchronize", lambda: None)

    @contextmanager
    def record(graph, *, pool):
        capture.graph = graph
        capture.recordings.append(graph)
        try:
            yield
        finally:
            capture.graph = None

    monkeypatch.setattr(torch_npu.npu, "graph", record)
    return capture


@pytest.mark.parametrize("pp_enabled", [False, True], ids=["pp1", "pp2"])
def test_disabling_aclgraph_uses_original_pipeline_and_cpu_result(monkeypatch, pp_enabled):
    def original_pass(gm, _inputs):
        output = next(node for node in gm.graph.nodes if node.op == 'output')
        with gm.graph.inserting_before(output):
            scaled = gm.graph.call_function(torch.mul, (output.args[0], 3))
        output.args = (scaled,)
        gm.recompile()
        return gm

    monkeypatch.setitem(PASS_PIPELINE_REGISTRY, "mutation-functionalization", lambda *args, **kwargs: [original_pass])
    config = SimpleNamespace(compile=SimpleNamespace(
        enable_passes=True, disable_passes=['aclgraph_pass'],
        mode='aot_fx_trace', inductor_compilation='regional',
        ep_overlap=SimpleNamespace(enabled=False),
    ))
    passes = PASS_PIPELINE_REGISTRY['mutation-functionalization+aclgraph'](
        SimpleNamespace(num_static_inputs=0), config,
        parallel_dims=SimpleNamespace(pp_enabled=pp_enabled, cp_enabled=False),
    )
    assert passes == [original_pass]
    gm = torch.fx.symbolic_trace(lambda x: x + 2)
    value = torch.tensor([1., 5.])
    for graph_pass in passes:
        gm = graph_pass(gm, (value,))
    torch.testing.assert_close(gm(value), torch.tensor([9., 21.]), rtol=0, atol=0)
    assert not hasattr(gm, '_aclgraph_wrappers')


def test_aclgraph_staging_rejects_aliases_whose_source_follows_them():
    gm = torch.fx.symbolic_trace(lambda alias, source: alias + source)

    with pytest.raises(ValueError, match="earlier non-aliased staged inputs"):
        _PartitionInterpreter(gm, staged_input_indices=[0, 1], input_aliases={0: 1})


def test_aclgraph_wrapper_rejects_borrowed_inputs_without_static_addresses():
    with pytest.raises(ValueError, match="borrowed inputs must also be static"):
        ACLGraphWrapper(torch.neg, static_input_indices=[], borrowed_input_indices=[0])


@pytest.mark.parametrize("num_inputs", [0, 1, 2], ids=["no-input", "one-input", "two-inputs"])
def test_aclgraph_releases_borrowed_inputs_after_their_last_capture_use(capture, monkeypatch, num_inputs):
    mode = FakeTensorMode()
    inputs = [FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0")) for _ in range(num_inputs)]
    references = [weakref.ref(value) for value in inputs]
    # Simulate the non-owning native handle; the wrapper and FX liveness are real.
    monkeypatch.setattr(torch_npu._C, "_weak_ref_tensor", weakref.proxy)

    def check_released(*values):
        assert all(reference() is None for reference in references)
        return torch.tensor([7.])

    graph = torch.fx.Graph()
    consumed = [graph.call_function(torch.neg, (graph.placeholder(f"input_{i}"),)) for i in range(num_inputs)]
    graph.output(graph.call_function(check_released, tuple(consumed)))
    interpreter = torch.fx.Interpreter(torch.fx.GraphModule({}, graph))
    wrapper = ACLGraphWrapper(
        interpreter.run, static_input_indices=range(num_inputs), borrowed_input_indices=range(num_inputs),
        capture_runnable=interpreter.boxed_run, capture_boxed=True, clone_outputs=False,
        capture_coordinator=_CaptureCoordinator(),
    )
    wrapper.warmed_up = True

    result = wrapper.call_boxed(inputs)

    torch.testing.assert_close(result, torch.tensor([7.]), rtol=0, atol=0)
    assert inputs == []
    assert len(capture.recordings) == 1


@pytest.fixture
def replay_capture(capture, monkeypatch):
    """Execute captured CPU FX operations while simulating the native graph API."""
    original_run = torch.fx.Interpreter.boxed_run

    def boxed_run(interpreter, args):
        inputs = list(args)
        output = original_run(interpreter, args)
        if capture.graph is not None:
            def replay():
                current = torch.fx.Interpreter(interpreter.module).run(*inputs)
                for saved, value in zip(tree_leaves(output), tree_leaves(current), strict=True):
                    if isinstance(saved, torch.Tensor):
                        saved.copy_(value)

            capture.graph.replay = replay

            # Match native handles: distinct StorageImpls can point at the
            # same allocation. The replay closure above keeps CPU owners live.

            def non_owning_handle(value):
                if not isinstance(value, torch.Tensor):
                    return value
                storage = value.untyped_storage()
                handle = torch._C._construct_storage_from_data_pointer(
                    storage.data_ptr(), value.device, storage.nbytes(),
                )
                return torch.empty(0, dtype=value.dtype).set_(
                    handle, value.storage_offset(), value.shape, value.stride(),
                )

            return tree_map(non_owning_handle, output)
        return output

    monkeypatch.setattr(torch.fx.Interpreter, "boxed_run", boxed_run)
    return capture


@pytest.mark.parametrize(
    ("return_base", "eager_mutation"),
    [(False, True), (True, True), (True, False)],
    ids=["eager-write-view", "eager-write-alias-group", "captured-write-alias-group"],
)
def test_aclgraph_preserves_returned_aliases_without_extra_partitions(replay_capture, return_base, eager_mutation):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(2, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    # Lifted scalar inputs can lack val without making output aliases unknown.
    scale = graph.placeholder("scale")
    none_grad = graph.placeholder("none_grad")
    stable = graph.call_function(torch.ops.aten.mul.Scalar, (x, scale))
    stable.meta["val"] = value.clone()
    allocated = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    allocated.meta["val"] = value.clone()
    view = graph.call_function(torch.ops.aten.view.default, (allocated, [2]))
    view.meta["val"] = allocated.meta["val"].view(2)
    independent = graph.call_function(torch.ops.aten.add.Scalar, (stable, 1))
    independent.meta["val"] = value.clone()
    mutated = graph.call_function(torch.ops.aten.add_.Scalar, (allocated, 3))
    mutated.meta["val"] = allocated.meta["val"]
    if eager_mutation:
        mutated.meta["custom"] = {"module_fqn": "layers.0.moe.routed_experts"}
    final = graph.call_function(torch.ops.aten.mul.Scalar, (independent, 2))
    final.meta["val"] = value.clone()
    graph.output((final, none_grad, view, mutated) if return_base else (final, none_grad, view))
    gm = aclgraph_module.aclgraph_pass(
        torch.fx.GraphModule({}, graph), (value, 7, None), static_input_indices=[1, 2],
    )
    assert [part["eager"] for part in gm.meta["aclgraph_partitions"]] == (
        [False, True, False] if eager_mutation else [False]
    )

    retained = []
    for values in ([2., 3.], [4., 5.], [-2., 1.], [0., 6.]):
        source = torch.tensor(values)
        actual = gm(source, 7, None)
        expected = (14 * source + 2, None, 2 * source + 3)
        if return_base:
            expected += (2 * source + 3,)
        retained.append((actual, expected))

        if return_base:
            assert actual[2].untyped_storage().data_ptr() == actual[3].untyped_storage().data_ptr()
        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert gm._aclgraph_wrappers
    assert len(replay_capture.recordings) == len(gm._aclgraph_wrappers)


@pytest.mark.parametrize("mutate", [False, True], ids=["retained-view", "eager-source-write"])
def test_aclgraph_preserves_views_of_eager_results(replay_capture, mutate):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(2, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    stable = graph.call_function(torch.ops.aten.mul.Scalar, (x, 7))
    stable.meta["val"] = value.clone()
    source = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    source.meta.update(val=value.clone(), custom={"module_fqn": "layers.0.moe.routed_experts"})
    view = graph.call_function(torch.ops.aten.view.default, (source, [2]))
    view.meta["val"] = source.meta["val"].view(2)
    if mutate:
        written = graph.call_function(torch.ops.aten.add_.Scalar, (source, 3))
        written.meta["val"] = source.meta["val"]
    final = graph.call_function(torch.ops.aten.neg.default, (view,))
    final.meta["val"] = value.clone()
    graph.output((stable, source, view, final))
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])
    assert [part["eager"] for part in compiled.meta["aclgraph_partitions"]] == [False, True, False]

    retained = []
    for values in ([2., 3.], [10., 20.], [100., 200.], [-1., 4.]):
        x = torch.tensor(values)
        actual = compiled(x)
        y = 2 * x + (3 if mutate else 0)
        retained.append((actual, (7 * x, y, y, -y)))
        assert actual[1].untyped_storage().data_ptr() == actual[2].untyped_storage().data_ptr()
        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert len(replay_capture.recordings) == 2


def test_aclgraph_stages_fixed_prefixes_from_variable_storage(replay_capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta")[:2], torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    output = graph.call_function(torch.ops.aten.neg.default, (x,))
    output.meta["val"] = value.clone()
    graph.output(output)
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for size in (4, 6, 8, 10):
        source = torch.arange(size, size * 2, dtype=torch.float32)[:2]
        retained.append((compiled(source), -source))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(replay_capture.recordings) == 1
    assert compiled.forward.__self__._input_buffers[0].untyped_storage().nbytes() == 8


def test_aclgraph_preserves_unaligned_backing_storage(replay_capture):
    mode = FakeTensorMode()
    raw = torch.empty(10, dtype=torch.uint8, device="meta")
    value = FakeTensor(mode, raw[:8].view(torch.float32), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    view = graph.call_function(torch.ops.aten.as_strided.default, (x, [1], [1], 1))
    view.meta["val"] = value.as_strided((1,), (1,), 1)
    output = graph.call_function(torch.ops.aten.neg.default, (view,))
    output.meta["val"] = view.meta["val"].clone()
    graph.output(output)
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for start in (1., 3., -2., 5.):
        raw = torch.zeros(10, dtype=torch.uint8)
        source = raw[:8].view(torch.float32)
        source.copy_(torch.tensor([start, start + 1]))
        retained.append((compiled(source), torch.tensor([-start - 1])))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(replay_capture.recordings) == 1


@pytest.mark.parametrize("change", ["negative-view", "subclass"])
def test_aclgraph_rejects_dispatch_changes_before_replay(capture, change):
    class TensorSubclass(torch.Tensor):
        pass

    source = torch.tensor([1., 2., 3.])
    wrapped = ACLGraphWrapper(lambda x: 2 * x, static_input_indices=[0])
    wrapped(source)
    wrapped(source)
    changed = source._neg_view() if change == "negative-view" else source.as_subclass(TensorSubclass)

    with pytest.raises(ValueError, match="dispatch|Tensor type|guard"):
        wrapped(changed)
    torch.testing.assert_close(source, torch.tensor([1., 2., 3.]), rtol=0, atol=0)


@pytest.mark.parametrize("return_aliases", [False, True], ids=["single-view", "alias-group"])
def test_aclgraph_copies_nested_output_views_preserving_their_storage_layout(replay_capture, return_aliases):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, 4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    base = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    base.meta["val"] = value.clone()
    sliced = graph.call_function(torch.ops.aten.as_strided.default, (base, [2, 2], [4, 1], 5))
    sliced.meta["val"] = base.meta["val"].as_strided((2, 2), (4, 1), 5)
    transposed = graph.call_function(torch.ops.aten.t.default, (base,))
    transposed.meta["val"] = base.meta["val"].t()
    retyped = graph.call_function(torch.ops.aten.view.dtype, (base, torch.int32))
    retyped.meta["val"] = base.meta["val"].view(torch.int32)
    graph.output(
        {"views": [sliced, transposed, retyped, sliced], "base": base} if return_aliases else {"views": [sliced]}
    )
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])
    assert [part["eager"] for part in compiled.meta["aclgraph_partitions"]] == [False]

    retained = []
    for start in (0, 10, -20, 30):
        source = torch.arange(start, start + 16, dtype=torch.float32).reshape(4, 4)
        actual = compiled(source)
        expected_base = source * 2
        expected_slice = expected_base.as_strided((2, 2), (4, 1), 5)
        expected = {
            "views": [expected_slice, expected_base.t(), expected_base.view(torch.int32), expected_slice],
            "base": expected_base,
        } if return_aliases else {"views": [expected_slice]}
        if return_aliases:
            assert actual["views"][0] is actual["views"][3]
            assert all(
                view.untyped_storage().data_ptr() == actual["base"].untyped_storage().data_ptr()
                for view in actual["views"]
            )
            assert actual["views"][1].stride() == (1, 4)
        assert actual["views"][0].storage_offset() == 5
        assert actual["views"][0].stride() == (4, 1)
        # Caller writes affect the other returned aliases, but never a graph's
        # fixed buffers or an earlier step's outputs.
        actual["views"][0].add_(100)
        expected_slice.add_(100)
        retained.append((actual, expected))
        for observed, reference in retained:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_aclgraph_preserves_input_views_across_steps(replay_capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(2, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    view = graph.call_function(torch.ops.aten.view.default, (x, [2]))
    view.meta["val"] = value.view(2)
    negative = graph.call_function(torch.ops.aten.neg.default, (x,))
    negative.meta["val"] = value.clone()
    stable = graph.call_function(torch.ops.aten.mul.Scalar, (negative, 7))
    stable.meta["val"] = value.clone()
    graph.output((view, stable))
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for values in ([1., 2.], [2., 3.], [3., 4.], [-1., 5.]):
        source = torch.tensor(values)
        actual = gm(source)
        retained.append((actual, (source.clone(), -7 * source)))

        assert actual[0].data_ptr() == source.data_ptr()
        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert gm._aclgraph_wrappers
    assert len(replay_capture.recordings) == len(gm._aclgraph_wrappers)


def test_aclgraph_preserves_caller_mesh_outputs_without_disabling_capture(replay_capture):
    mesh = DeviceMesh("cpu", [0], _init_backend=False, _rank=0)
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x, mesh_input = graph.placeholder("x"), graph.placeholder("mesh")
    x.meta["val"], mesh_input.meta["val"] = value, mesh
    gradient = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    gradient.meta["val"] = value.clone()
    view = graph.call_function(torch.ops.aten.view.default, (gradient, [2, 2]))
    view.meta["val"] = gradient.meta["val"].view(2, 2)
    graph.output((mesh_input, gradient, view))
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[1])
    assert [part["eager"] for part in compiled.meta["aclgraph_partitions"]] == [False]

    for values in ([1., -2., 3., -4.], [4., 3., 2., 1.], [0., 1., 0., -1.]):
        source = torch.tensor(values)
        returned_mesh, actual, alias = compiled(source, mesh)
        assert returned_mesh is mesh
        assert actual.untyped_storage().data_ptr() == alias.untyped_storage().data_ptr()
        torch.testing.assert_close((actual, alias), (2 * source, (2 * source).view(2, 2)), rtol=0, atol=0)
    assert len(replay_capture.recordings) == 1


def test_aclgraph_preserves_empty_output_aliases(capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(0, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    allocated = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    allocated.meta["val"] = value.clone()
    view = graph.call_function(torch.ops.aten.view.default, (allocated, [0]))
    view.meta["val"] = allocated.meta["val"].view(0)
    graph.output((allocated, view))
    compiled = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    for _ in range(3):
        base, alias = compiled(torch.empty(0))
        assert torch._C._is_alias_of(base, alias)
        version = alias._version
        base.add_(1)
        assert alias._version == version + 1
    assert not capture.recordings


def test_aclgraph_preserves_absolute_offsets_of_sliced_inputs(replay_capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta")[1:], torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    view = graph.call_function(torch.ops.aten.as_strided.default, (x, [1], [1], 2))
    view.meta["val"] = value.as_strided((1,), (1,), 2)
    read = graph.call_function(torch.ops.aten.neg.default, (view,))
    read.meta["val"] = view.meta["val"].clone()
    negative = graph.call_function(torch.ops.aten.neg.default, (x,))
    negative.meta["val"] = value.clone()
    stable = graph.call_function(torch.ops.aten.mul.Scalar, (negative, 7))
    stable.meta["val"] = value.clone()
    graph.output((read, stable))
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for start in (11., 21., 31., 41.):
        base = torch.arange(start, start + 4)
        source = base[1:]
        actual = gm(source)
        retained.append((actual, (torch.tensor([-start - 2]), -7 * source)))

        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert gm._aclgraph_wrappers
    assert len(replay_capture.recordings) == len(gm._aclgraph_wrappers)


@pytest.mark.parametrize("shared", [False, True], ids=["private", "shared"])
def test_aclgraph_preserves_storage_offsets_in_staging(capture, shared):
    def step(x):
        output = torch.empty(1)

        def execute():
            output.copy_(x.as_strided((1,), (1,), 2).neg())

        if capture.graph is None:
            execute()
        else:
            capture.graph.replay = execute
        return output

    coordinator = _CaptureCoordinator()
    wrapped = ACLGraphWrapper(
        step, static_input_indices=[], capture_coordinator=coordinator,
        reusable_input_indices=[0] if shared else [],
    )
    coordinator.wrappers.append(wrapped)
    retained = []
    for start in (11., 21., 31., 41.):
        source = torch.arange(start, start + 4)[1:]
        retained.append((wrapped(source), torch.tensor([-start - 2])))

        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(capture.recordings) == 1


@pytest.mark.parametrize("change", ["storage_offset", "storage size"], ids=["offset", "storage-size"])
def test_aclgraph_rejects_changed_storage_offsets_before_replay(capture, change):
    wrapped = ACLGraphWrapper(
        torch.add, static_input_indices=[], capture_coordinator=_CaptureCoordinator(), reusable_input_indices=[0],
    )
    wrapped(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 3.]))
    wrapped(torch.tensor([4., 5., 6.]), torch.tensor([4., 5., 6.]))
    base = torch.tensor([11., 12., 13., 14.])
    source = base[1:] if change == "storage_offset" else base[:3]

    with pytest.raises(ValueError, match=change):
        wrapped(torch.tensor([21., 22., 23.]), source)

    torch.testing.assert_close(wrapped.args[0], torch.tensor([4., 5., 6.]), rtol=0, atol=0)
    torch.testing.assert_close(wrapped.args[1], torch.tensor([4., 5., 6.]), rtol=0, atol=0)


def test_aclgraph_reuses_region_inputs_without_overwriting_live_sources(replay_capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value

    def operation(target, args, *, eager=False):
        node = graph.call_function(target, args)
        node.meta["val"] = value.clone()
        if eager:
            node.meta["custom"] = {"module_fqn": "layers.0.moe.routed_experts"}
        return node

    a = operation(torch.ops.aten.add.Scalar, (x, 1), eager=True)
    b = operation(torch.ops.aten.mul.Scalar, (x, 2), eager=True)
    weighted = operation(torch.ops.aten.mul.Scalar, (b, 3))
    first = operation(torch.ops.aten.add.Tensor, (a, weighted))
    c = operation(torch.ops.aten.add.Scalar, (x, 5), eager=True)
    d = operation(torch.ops.aten.mul.Scalar, (x, 4), eager=True)
    weighted = operation(torch.ops.aten.mul.Scalar, (d, 7))
    result = operation(torch.ops.aten.add.Tensor, (first, c))
    result = operation(torch.ops.aten.add.Tensor, (result, weighted))
    result = operation(torch.ops.aten.add.Tensor, (result, a))
    graph.output((first, result))
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for values in ([1., 2., -3., 4.], [-4., 3., 2., -1.], [5., -2., 1., -3.], [0., 1., 2., 3.]):
        source = torch.tensor(values)
        retained.append((gm(source), (7 * source + 1, 37 * source + 7)))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    wrappers = gm._aclgraph_wrappers
    assert len(wrappers) == 2
    pointers = [
        [wrapper.args[i].data_ptr() for i in sorted(wrapper.reusable_input_indices)]
        for wrapper in wrappers
    ]
    assert all(len(addresses) > 1 and len(addresses) == len(set(addresses)) for addresses in pointers)
    assert set(pointers[0]) & set(pointers[1])
    coordinator = gm._aclgraph_capture_coordinator
    assert sum(len(buffers) for buffers in coordinator.shared_input_buffers.values()) < sum(map(len, pointers))
    assert len(replay_capture.recordings) == 2
    aclgraph_module._close_aclgraph_trainer(SimpleNamespace(_traced_step=SimpleNamespace(gm=gm)), lambda: None)
    assert not coordinator.shared_input_buffers


@pytest.mark.parametrize("output_kind", ["view", "nested-view", "opaque-view", "keyed-view", "unknown-view"])
def test_aclgraph_keeps_independent_inputs_when_output_storage_can_escape(replay_capture, output_kind):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    source = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    source.meta.update(val=value.clone(), custom={"module_fqn": "layers.0.moe.routed_experts"})
    view = graph.call_function(torch.ops.aten.view.default, (source, [4]))
    if output_kind != "unknown-view":
        view.meta["val"] = source.meta["val"].view(4)
    if output_kind == "nested-view":
        packed = graph.call_function(tuple, ([view],))
        packed.meta["val"] = (view.meta["val"],)
    elif output_kind == "opaque-view":
        packed = graph.call_function(SimpleNamespace, (), {"value": view})
        packed.meta["val"] = SimpleNamespace(value=view.meta["val"])
    elif output_kind == "keyed-view":
        packed = graph.call_function(dict, (((view, 1),),))
        packed.meta["val"] = {view.meta["val"]: 1}
    later = graph.call_function(torch.ops.aten.add.Scalar, (x, 3))
    later.meta.update(val=value.clone(), custom={"module_fqn": "layers.1.moe.routed_experts"})
    if output_kind == "nested-view":
        view = graph.call_function(getitem, (packed, 0))
        view.meta["val"] = source.meta["val"].view(4)
    elif output_kind == "opaque-view":
        view = graph.call_function(getattr, (packed, "value"))
        view.meta["val"] = source.meta["val"].view(4)
    elif output_kind == "keyed-view":
        iterator = graph.call_function(iter, (packed,))
        view = graph.call_function(next, (iterator,))
        view.meta["val"] = source.meta["val"].view(4)
    result = graph.call_function(torch.ops.aten.add.Tensor, (view, later))
    result.meta["val"] = value.clone()
    graph.output(result)
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])

    retained = []
    for values in ([1., 2., -3., 4.], [-4., 3., 2., -1.], [5., -2., 1., -3.]):
        source = torch.tensor(values)
        retained.append((gm(source), 3 * source + 3))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if output_kind in ("view", "nested-view"):
        # Views of eager allocations stay eager; only the final add is captured.
        (last,) = gm._aclgraph_wrappers
        assert last.reusable_input_indices
        assert len(replay_capture.recordings) == 1
    else:
        first, last = gm._aclgraph_wrappers
        assert not first.reusable_input_indices
        assert last.reusable_input_indices
        assert all(first.args[0].data_ptr() != last.args[i].data_ptr() for i in last.reusable_input_indices)
        assert len(replay_capture.recordings) == 2


def test_aclgraph_rejects_stream_changes_before_mutating_inputs(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: x.add_(1))
    coordinator = _CaptureCoordinator()
    object.__setattr__(gm, "_aclgraph_capture_coordinator", coordinator)
    interpreter = _PartitionInterpreter(gm)
    # The native stream query is isolated; CPU tensors cannot create NPU slots.
    coordinator._input_buffer_streams[torch.device("npu:0")] = object()
    monkeypatch.setattr(torch_npu.npu, "current_stream", lambda device: object())
    source = torch.tensor([2., -3.])

    with pytest.raises(ValueError, match="same replay stream across calls"):
        interpreter.run(source)

    torch.testing.assert_close(source, torch.tensor([2., -3.]), rtol=0, atol=0)


def test_aclgraph_rejects_actual_output_aliases_missing_from_metadata(replay_capture):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = value
    source = graph.call_function(torch.ops.aten.mul.Scalar, (x, 2))
    source.meta.update(val=value.clone(), custom={"module_fqn": "layers.0.moe.routed_experts"})
    view = graph.call_function(torch.ops.aten.view.default, (source, [4]))
    # Model inaccurate metadata from a producer that omitted its alias relation.
    view.meta["val"] = value.clone()
    graph.output(view)
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])
    source = torch.tensor([1., -2., 3., -4.])

    torch.testing.assert_close(gm(source), 2 * source, rtol=0, atol=0)
    assert gm._aclgraph_wrappers[0].reusable_input_indices
    with pytest.raises(ValueError, match="reusable input unexpectedly aliases a captured output"):
        gm(source)


def test_aclgraph_replays_fp8_container_inputs_without_tensor_value_comparison(monkeypatch, capture):
    class RejectFP8Equality(TorchDispatchMode):
        def dispatch(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.eq.Tensor, torch.ops.aten.ne.Tensor) and any(
                isinstance(value, torch.Tensor) and value.dtype == torch.float8_e4m3fn
                for value in tree_leaves(args)
            ):
                raise AssertionError("FP8 input guards must not compare tensor values")
            return func(*args, **(kwargs or {}))

    # Register the callback under PyTorch's required protocol name.
    monkeypatch.setattr(RejectFP8Equality, "__torch_dispatch__", RejectFP8Equality.dispatch)

    def step(payload):
        x = payload["quantized"][0][0]
        output = torch.empty_like(x, dtype=torch.float32)

        def execute():
            output.copy_(x.float() * payload["scale"])

        if capture.graph is None:
            execute()
        else:
            capture.graph.replay = execute
        return output

    coordinator = _CaptureCoordinator()
    wrapped = ACLGraphWrapper(step, static_input_indices=[], capture_coordinator=coordinator)
    coordinator.wrappers.append(wrapped)
    buffer = torch.empty(2, dtype=torch.float8_e4m3fn)
    for values in ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
        buffer.copy_(torch.tensor(values))
        # A fresh container/view shares the capture-time storage, but has new
        # values. Python container equality would invoke unsupported FP8 eq.
        payload = {"quantized": [(buffer.view(2),)], "scale": 2, "optional": None}
        with RejectFP8Equality():
            output = wrapped(payload)
        torch.testing.assert_close(output, 2 * torch.tensor(values), rtol=0, atol=0)

    assert len(capture.recordings) == 1
    # Container tensors are borrowed, so pointer/layout guards remain required.
    with pytest.raises(ValueError, match=r"input\[0\].*static address"):
        wrapped({**payload, "quantized": [(buffer.clone(),)]})
    with pytest.raises(ValueError, match=r"input\[0\].*shape"):
        wrapped({**payload, "quantized": [(buffer.view(1, 2),)]})
    payload["scale"] = 3
    with pytest.raises(ValueError, match=r"input\[0\].*scalar 2 -> 3"):
        wrapped(payload)


@pytest.mark.parametrize(
    "producer_owned,escaping", [(False, False), (True, False), (False, True)],
    ids=["staged", "producer-buffer", "escaping-container"],
)
def test_aclgraph_pipeline_partitions_host_reads_and_preserves_replayed_outputs(
    monkeypatch, capture, producer_owned, escaping
):
    mode = FakeTensorMode(shape_env=ShapeEnv())
    value = FakeTensor(mode, torch.empty(2, 2, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    x, weight, duplicate = (graph.placeholder(name) for name in ("x", "weight", "duplicate"))
    x.meta["val"], weight.meta["val"], duplicate.meta["val"] = value, value.clone(), value
    mm = graph.call_function(torch.ops.aten.mm.default, (x, weight))
    mm.meta["val"] = value.clone()
    alias = graph.call_function(torch.ops.aten.alias.default, (x,))
    alias.meta["val"] = value
    count = graph.call_function(torch.ops.aten.sum.default, (mm,), {"dtype": torch.int64})
    count.meta["val"] = FakeTensor(mode, torch.empty((), device="meta", dtype=torch.int64), torch.device("npu:0"))
    packed = graph.call_function(tuple, ([mm, weight],))
    packed.meta["val"] = (mm.meta["val"], weight.meta["val"])
    host = graph.call_function(torch.ops.aten._to_copy.default, (count,), {"device": torch.device("cpu")})
    host.meta["val"] = torch.empty((), dtype=torch.int64)
    scalar = graph.call_function(torch.ops.aten._local_scalar_dense.default, (host,))
    scalar.meta["val"] = mode.shape_env.create_unbacked_symint()
    unpacked = graph.call_function(getitem, (packed, 0))
    unpacked.meta["val"] = mm.meta["val"]
    combined = graph.call_function(torch.ops.aten.add.Tensor, (unpacked, alias))
    combined.meta["val"] = value.clone()
    combined = graph.call_function(torch.ops.aten.add.Tensor, (combined, duplicate))
    combined.meta["val"] = value.clone()
    scaled = graph.call_function(torch.ops.aten.mul.Scalar, (combined, scalar))
    scaled.meta.update(val=value.clone(), autograd_backward=True, custom={"module_fqn": "layers.1.attention"})
    final = graph.call_function(torch.ops.aten.mul.Scalar, (scaled, 3))
    final.meta.update(val=value.clone(), autograd_backward=True, custom={"module_fqn": "layers.0.attention"})
    graph.output((mm, scalar, final) if escaping else final)
    gm = torch.fx.GraphModule({}, graph)

    # Only native recording is simulated; run the real pass, staging and FX.
    original_boxed_run = torch.fx.Interpreter.boxed_run
    consumer_aliases = []

    def boxed_run(interpreter, args):
        inputs = list(args)
        output = original_boxed_run(interpreter, args)
        if capture.graph is not None:
            bindings = dict(
                zip((n.name for n in interpreter.graph.nodes if n.op == "placeholder"), inputs, strict=True)
            )
            if alias.name in bindings and duplicate.name in bindings:
                consumer_aliases.append(bindings[alias.name].data_ptr() == bindings[duplicate.name].data_ptr())

            def replay():
                result = torch.fx.Interpreter(interpreter.module).run(*inputs)
                for saved, current in zip(tree_leaves(output), tree_leaves(result), strict=True):
                    if isinstance(saved, torch.Tensor):
                        saved.copy_(current)

            capture.graph.replay = replay
        return output

    monkeypatch.setattr(torch.fx.Interpreter, "boxed_run", boxed_run)
    monkeypatch.setitem(PASS_PIPELINE_REGISTRY, "mutation-functionalization", lambda *_args, **_kwargs: [])
    config = SimpleNamespace(compile=SimpleNamespace(
        mode="aot_fx_trace", inductor_compilation="regional", ep_overlap=SimpleNamespace(enabled=False),
    ))
    for graph_pass in PASS_PIPELINE_REGISTRY["mutation-functionalization+aclgraph"](
        SimpleNamespace(num_static_inputs=0), config,
    ):
        gm = graph_pass(gm, (value, weight.meta["val"], value))
    assert [part["eager"] for part in gm.meta["aclgraph_partitions"]] == [False, True, False, True, False]
    assert [part["capture"] for part in gm.meta["aclgraph_partitions"]] == [True, False, True, False, True]

    real_weight = torch.diag(torch.tensor([2., 1.]))
    producer_buffer = torch.empty(2, 2)
    torch._dynamo.mark_static_address(producer_buffer, guard=True)
    if producer_owned:
        torch._dynamo.mark_static_address(real_weight, guard=True)
    retained = []
    values_by_step = ([[1., 2.], [3., 4.]], [[3., 4.], [1., 2.]], [[2., 1.], [2., 5.]])
    for values in values_by_step:
        source = torch.tensor(values)
        if producer_owned:
            producer_buffer.copy_(source)
            source = producer_buffer
        canonical = source.view_as(source) if producer_owned else source
        actual = gm(canonical, real_weight, source)
        # Each input has column sums (4, 6), so the host scalar stays 14.
        expected = 42 * torch.tensor(values) * torch.tensor([4., 3.])
        if escaping:
            expected = (torch.tensor(values) * torch.tensor([2., 1.]), 14, expected)
        retained.append((actual, expected))
        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    if escaping:
        for wrapper in gm._aclgraph_wrappers:
            for buffer in tree_leaves(wrapper.output):
                if isinstance(buffer, torch.Tensor):
                    buffer.fill_(-100)
        for observed, expected in retained:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert consumer_aliases == [True]
    if producer_owned:
        assert any(
            isinstance(arg, torch.Tensor) and arg.data_ptr() == producer_buffer.data_ptr()
            for wrapper in gm._aclgraph_wrappers for arg in wrapper.args
        )
    assert len(capture.recordings) == 3
    with pytest.raises(ValueError, match="shared input alias changed"):
        gm(source, real_weight, source._neg_view())
    aclgraph_module._close_aclgraph_trainer(SimpleNamespace(_traced_step=SimpleNamespace(gm=gm)), lambda: None)
    assert all(wrapper.graph is None and wrapper.args is None for wrapper in gm._aclgraph_wrappers)


def test_aclgraph_borrows_intermediates_from_a_region_with_an_escaping_gradient(capture, monkeypatch):
    mode = FakeTensorMode()
    value = FakeTensor(mode, torch.empty(4, device="meta"), torch.device("npu:0"))
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = value
    saved = graph.call_function(torch.ops.aten.mul.Scalar, (source, 2))
    saved.meta["val"] = value.clone()
    gradient = graph.call_function(torch.ops.aten.mul.Scalar, (saved, 3))
    gradient.meta["val"] = value.clone()
    expert = graph.call_function(torch.ops.aten.mul.Scalar, (gradient, 5))
    expert.meta.update(val=value.clone(), custom={"module_fqn": "layers.0.moe.routed_experts"})
    final = graph.call_function(torch.ops.aten.add.Tensor, (saved, expert))
    final.meta["val"] = value.clone()
    graph.output((gradient, final))
    original_run = torch.fx.Interpreter.boxed_run

    def boxed_run(interpreter, args):
        inputs = list(args)
        output = original_run(interpreter, args)
        if capture.graph is not None:
            def replay():
                current = torch.fx.Interpreter(interpreter.module).run(*inputs)
                for target, actual in zip(tree_leaves(output), tree_leaves(current), strict=True):
                    if isinstance(target, torch.Tensor):
                        target.copy_(actual)

            capture.graph.replay = replay
        return output

    monkeypatch.setattr(torch.fx.Interpreter, "boxed_run", boxed_run)
    gm = aclgraph_module.aclgraph_pass(torch.fx.GraphModule({}, graph), (), static_input_indices=[])
    regions = list(gm.children())
    producer, consumer = regions[0]._aclgraph_wrapper, regions[-1]._aclgraph_wrapper
    consumer_inputs = [node.name for node in regions[-1].graph.nodes if node.op == "placeholder"]
    saved_input = consumer_inputs.index(saved.name)
    assert saved_input in consumer.borrowed_input_indices
    producer_output = next(node for node in regions[0].graph.nodes if node.op == "output")
    outputs = [node.name for node in tree_leaves(producer_output.args)]
    saved_output = outputs.index(saved.name)
    retained = []
    for values in ([1., 2., 3., 4.], [4., 3., 2., 1.], [-1., 2., -3., 4.], [0., 1., 0., -1.]):
        x = torch.tensor(values)
        retained.append((gm(x), (6 * x, 32 * x)))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if producer.graph is not None:
            assert not producer.clone_outputs
            assert consumer.args[saved_input].data_ptr() == tree_leaves(producer.output)[saved_output].data_ptr()
    assert len(capture.recordings) == 2
    aclgraph_module._close_aclgraph_trainer(SimpleNamespace(_traced_step=SimpleNamespace(gm=gm)), lambda: None)


def test_aclgraph_keeps_dynamic_routed_expert_forward_and_backward_eager(capture, monkeypatch):
    @annotate_fn({"module_fqn": "layers.0.moe.routed_experts"})
    def experts(x):
        return x.masked_select(x > 0).square().sum()

    def joint(x):
        loss = experts(x * 2).square()
        return loss, torch.autograd.grad(loss, x)[0]

    with preserve_node_meta():
        gm = make_fx(joint, tracing_mode="symbolic")(torch.tensor([1., -1., 2., -2.], requires_grad=True))
    mode = FakeTensorMode()
    for node in gm.graph.nodes:
        if isinstance(value := node.meta.get("val"), torch.Tensor):
            node.meta["val"] = FakeTensor(mode, torch.empty_like(value, device="meta"), torch.device("npu:0"))
    original_run = torch.fx.Interpreter.boxed_run

    def boxed_run(interpreter, args):
        inputs = list(args)
        output = original_run(interpreter, args)
        if capture.graph is not None:
            def replay():
                current = torch.fx.Interpreter(interpreter.module).run(*inputs)
                for saved, value in zip(tree_leaves(output), tree_leaves(current), strict=True):
                    if isinstance(saved, torch.Tensor):
                        saved.copy_(value)

            capture.graph.replay = replay
        return output

    monkeypatch.setattr(torch.fx.Interpreter, "boxed_run", boxed_run)
    gm = aclgraph_module.aclgraph_pass(gm, (), static_input_indices=[])
    assert gm._aclgraph_wrappers
    assert any(part["eager"] for part in gm.meta["aclgraph_partitions"])
    for region in gm.children():
        if hasattr(region, "_aclgraph_wrapper"):
            assert not any(
                "routed_experts" in node.meta.get("custom", {}).get("module_fqn", "")
                for node in region.graph.nodes if node.op != "placeholder"
            )
    # The number of routed rows changes 2 -> 4 -> 1 -> 0 -> 3. The independent
    # formula checks both the loss and the input gradient of the joint graph.
    retained = []
    for values in ([1., -1., 2., -2.], [1., 2., 3., 4.], [-1., -2., 3., -4.], [-1., -2., -3., -4.], [2., 1., -1., 3.]):
        square_sum = sum(value * value for value in values if value > 0)
        expected = (torch.tensor(16. * square_sum**2), torch.tensor([
            64 * square_sum * value if value > 0 else 0. for value in values
        ]))
        retained.append((gm(torch.tensor(values)), expected))
        for actual, reference in retained:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert len(capture.recordings) == len(gm._aclgraph_wrappers)


def test_aclgraph_static_replay_preserves_state_and_rejects_changed_inputs(capture, monkeypatch):
    def step(x, accumulator, scale):
        accumulator = accumulator["values"][0]
        output = torch.empty_like(x)

        def execute():
            accumulator.add_(x.sum())
            output.copy_(scale * x + accumulator)

        # Simulate the external capture boundary, not the wrapper's copy logic.
        if capture.graph is None:
            execute()
        else:
            capture.graph.replay = execute
        return {"gradient": output, "state": (accumulator, scale)}

    coordinator = _CaptureCoordinator()
    wrapped = ACLGraphWrapper(step, static_input_indices=[1], capture_coordinator=coordinator)
    coordinator.wrappers.append(wrapped)
    accumulator = torch.tensor(0.0)
    state = {"values": [accumulator]}
    retained = []
    sequence = [
        ([1.0, 2.0], 2, 3.0, [5.0, 7.0]),
        ([3.0, 5.0], 2, 11.0, [17.0, 21.0]),
        ([-2.0, 4.0], 2, 13.0, [9.0, 21.0]),
    ]
    for values, scale, expected_state, expected in sequence:
        if wrapped.graph is not None:
            def unexpected_coordination(*args, **kwargs):
                pytest.fail("Static replay must not coordinate across ranks")

            monkeypatch.setattr(coordinator, "before_capture", unexpected_coordination)
            monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_coordination)
            monkeypatch.setattr(torch.distributed, "barrier", unexpected_coordination)
        result = wrapped(torch.tensor(values), state, scale)
        assert result["state"][0] is accumulator
        assert result["state"][1] == scale
        assert accumulator.item() == expected_state
        retained.append((result["gradient"], expected))
        for saved, saved_expected in retained:
            torch.testing.assert_close(saved, torch.tensor(saved_expected), rtol=0, atol=0)
    with pytest.raises(ValueError, match="static replay inputs changed.*shape"):
        wrapped(torch.tensor([1.0, 2.0, 3.0]), state, 2)
    with pytest.raises(ValueError, match="static replay inputs changed.*scalar"):
        wrapped(torch.tensor([1.0, 2.0]), state, 3)
    accumulator.data = accumulator.clone()
    with pytest.raises(ValueError, match="static replay inputs changed.*static address"):
        wrapped(torch.tensor([1.0, 2.0]), state, 2)
    assert accumulator.item() == 13.0
    assert len(capture.recordings) == 1


@pytest.mark.parametrize("retain_first", [False, True])
def test_loss_accumulation_preserves_gradients_views_and_live_readers(retain_first):
    def accumulate(*inputs):
        first = result = inputs[0].float()
        for value in inputs[1:]:
            result = result + value.float()
        return result.view(3, 2), first if retain_first else inputs[0]

    base = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    values = [base + i for i in range(8)]
    gm = make_fx(accumulate, tracing_mode="fake")(*values)
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.add.Tensor:
            node.meta["custom"] = {"module_fqn": "loss"}
    compiled = reinplace_loss_accumulation_pass(gm, values)
    args = [value.clone().requires_grad_() for value in values]
    result, retained = compiled(*args)
    expected = 8 * base.float() + 28
    torch.testing.assert_close(result, expected.reshape(3, 2), rtol=0, atol=0)
    torch.testing.assert_close(retained, base.to(retained.dtype), rtol=0, atol=0)
    result.square().sum().backward()
    for arg, value in zip(args, values, strict=True):
        torch.testing.assert_close(arg, value, rtol=0, atol=0)
        torch.testing.assert_close(arg.grad, (2 * expected).bfloat16(), rtol=0, atol=0)
    assert sum(node.target == torch.ops.aten.add_.Tensor for node in compiled.graph.nodes) == (0 if retain_first else 7)
