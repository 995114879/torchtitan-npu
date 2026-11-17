# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native AG multi-rank behavior and CPU contracts for RS allocation; not NPU allocator tests."""

import copy
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import torchtitan_npu.patches.torch.distributed.fsdp as fsdp_patch


def _run_two_rank_native_ag_worker(rank: int, rendezvous: str, enabled: bool) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    param_group_mod = fsdp_patch._fsdp_param_group
    orig_ag = param_group_mod.foreach_all_gather
    ag_calls = []

    def spy_foreach_all_gather(
        fsdp_params,
        group,
        async_op,
        all_gather_copy_in_stream,
        all_gather_stream,
        device,
        all_gather_comm,
    ):
        current = torch.get_device_module("cpu").current_stream()
        ag_calls.append((async_op, all_gather_copy_in_stream == current, all_gather_stream == current))
        return orig_ag(
            fsdp_params,
            group,
            async_op,
            all_gather_copy_in_stream,
            all_gather_stream,
            device,
            all_gather_comm,
        )

    param_group_mod.foreach_all_gather = spy_foreach_all_gather
    try:
        mesh = init_device_mesh("cpu", (2,))
        fsdp_patch._AG_ALLOC_ON_COMPUTE_STREAM = enabled
        fsdp_patch._RS_ALLOC_ON_COMPUTE_STREAM = False
        fsdp_patch.apply()

        with torch.random.fork_rng():
            torch.manual_seed(42)
            reference = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.SiLU(), torch.nn.Linear(6, 4))
        model = copy.deepcopy(reference)
        for module in (model[0], model[2], model):
            fully_shard(module, mesh=mesh)
            assert all(group.unshard_async_op == enabled for group in module._get_fsdp_state()._fsdp_param_groups)
        model[0].set_modules_to_forward_prefetch([model[2]])
        model[2].set_modules_to_backward_prefetch([model[0]])

        full_x = torch.arange(16, dtype=torch.float32).reshape(4, 4).div(9)
        local_x = full_x[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        ref_x = full_x.clone().requires_grad_()
        expected = reference(ref_x)
        actual = model(local_x)
        torch.testing.assert_close(actual, expected[rank * 2 : (rank + 1) * 2], rtol=0, atol=0)

        actual.square().sum().backward()
        expected.square().sum().backward()

        assert ag_calls, "Expected world_size=2 FSDP to invoke foreach_all_gather"
        assert all(call == (enabled, enabled, enabled) for call in ag_calls)
        torch.testing.assert_close(local_x.grad, ref_x.grad[rank * 2 : (rank + 1) * 2], rtol=1e-6, atol=1e-6)
        for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param.grad.full_tensor(), ref_param.grad / 2, rtol=1e-6, atol=1e-6)
    finally:
        param_group_mod.foreach_all_gather = orig_ag
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("enabled", [False, True], ids=["sync-ag", "native-async-ag"])
def test_native_ag_mode_preserves_forward_and_gradients_across_ranks(tmp_path, enabled):
    torch.multiprocessing.start_processes(
        _run_two_rank_native_ag_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", enabled),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_apply_toggles_param_group_init(monkeypatch):
    group_type = fsdp_patch._fsdp_param_group.FSDPParamGroup
    monkeypatch.setattr(fsdp_patch._fsdp_param_group, "FSDPParam", fsdp_patch.FSDPParam)
    monkeypatch.setattr(group_type, "__init__", group_type.__init__)
    monkeypatch.setattr(fsdp_patch.FSDPParam, "init_dtype_attrs", fsdp_patch.FSDPParam.init_dtype_attrs)
    monkeypatch.setattr(fsdp_patch._fsdp_collectives, "foreach_reduce", fsdp_patch._fsdp_collectives.foreach_reduce)
    monkeypatch.setattr(fsdp_patch._fsdp_param_group, "foreach_reduce", fsdp_patch._fsdp_param_group.foreach_reduce)
    monkeypatch.setattr(fsdp_patch.FSDPParam, "all_gather_inputs", fsdp_patch.FSDPParam.all_gather_inputs)

    monkeypatch.setattr(fsdp_patch, "_AG_ALLOC_ON_COMPUTE_STREAM", True)
    fsdp_patch.apply()
    assert group_type.__init__ is fsdp_patch._patched_param_group_init

    monkeypatch.setattr(fsdp_patch, "_AG_ALLOC_ON_COMPUTE_STREAM", False)
    fsdp_patch.apply()
    assert group_type.__init__ is fsdp_patch._ORIGINAL_PARAM_GROUP_INIT


@pytest.fixture
def streams(monkeypatch):
    events = []
    buffers = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

    owner, reduce, all_reduce = Stream("owner"), Stream("reduce"), Stream("all_reduce")
    state = SimpleNamespace(current=owner)

    @contextmanager
    def use_stream(stream):
        previous, state.current = state.current, stream
        try:
            yield
        finally:
            state.current = previous

    handle = SimpleNamespace(current_stream=lambda: state.current, stream=use_stream)

    def allocate(size, *, dtype, device):
        assert size == (8,) and dtype == torch.float32 and device == torch.device("cpu")
        buf = SimpleNamespace(record_stream=Mock())
        buffers.append(buf)
        events.append(("allocate", state.current.name))
        return buf

    comm = Mock(allocate=Mock(side_effect=allocate))
    monkeypatch.setattr(torch, "get_device_module", lambda device: handle)
    return SimpleNamespace(
        owner=owner,
        reduce=reduce,
        all_reduce=all_reduce,
        state=state,
        handle=handle,
        buffers=buffers,
        comm=comm,
        events=events,
    )


@pytest.mark.parametrize(
    ("side_caller", "use_extra"),
    [(False, False), (True, False), (True, True)],
    ids=["same-stream", "single-side-stream", "extra-all-reduce-stream"],
)
def test_rs_allocator_orders_reuse_and_records_consumer(streams, side_caller, use_extra):
    s = streams
    extra = (s.all_reduce, s.owner, s.reduce, None) if use_extra else ()
    allocator = fsdp_patch._ComputeStreamAllocator(s.comm, s.handle, extra)
    caller = s.reduce if side_caller else s.owner
    with s.handle.stream(caller):
        buf = allocator.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        assert buf is s.buffers[0]
        assert s.state.current is caller
        allocator(output_tensor=buf)
    assert s.events == [("allocate", "owner")] + ([("wait_stream", "reduce", "owner")] if side_caller else [])
    expected_recorded = ([s.reduce] + ([s.all_reduce] if use_extra else [])) if side_caller else []
    assert [call.args[0] for call in buf.record_stream.call_args_list] == expected_recorded
    s.comm.assert_called_once_with(output_tensor=buf)


def test_rs_disabled_preserves_native_arguments(monkeypatch):
    native = Mock(return_value=object())
    monkeypatch.setattr(fsdp_patch, "_RS_ALLOC_ON_COMPUTE_STREAM", False)
    monkeypatch.setattr(fsdp_patch, "_AG_ALLOC_ON_COMPUTE_STREAM", True)
    allocator = Mock(side_effect=AssertionError("Disabled RS must not wrap the comm"))
    monkeypatch.setattr(fsdp_patch, "_ComputeStreamAllocator", allocator)
    monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_REDUCE", native)
    params, grads = [SimpleNamespace(reduce_dtype=torch.float32)], [torch.ones(8)]
    assert fsdp_patch._patched_foreach_reduce(params, grads, marker="native") is native.return_value
    native.assert_called_once_with(params, grads, marker="native")
    allocator.assert_not_called()


@pytest.mark.parametrize(
    ("size", "factor", "all_reduce", "all_reduce_grads", "has_hook", "expected", "expect_all_reduce_record"),
    [
        (None, 1.0, False, False, False, None, False),
        (1, 1.0, False, False, False, None, False),
        (2, 1.0, False, False, False, 1.0, False),
        (1, 1.0, True, True, False, 1.0, True),
        (2, 1.0, True, False, False, 1.0, False),
        (2, 1.0, True, False, True, 1.0, False),
        (2, 1.0, False, False, True, 1.0, True),
        (1, 2.0, False, False, False, 2.0, False),
        (1, None, False, False, False, None, False),
    ],
    ids=[
        "no-group-unit",
        "single-rank-unit",
        "multi-rank-unit",
        "hsdp-sync-unit",
        "hsdp-accum-only-unit",
        "hsdp-accum-with-hook-unit",
        "custom-hook-unit",
        "custom-factor",
        "none-factor",
    ],
)
def test_rs_wrapper_preserves_scaling_and_native_streams(
    monkeypatch,
    streams,
    size,
    factor,
    all_reduce,
    all_reduce_grads,
    has_hook,
    expected,
    expect_all_reduce_record,
):
    s = streams
    monkeypatch.setattr(fsdp_patch, "_RS_ALLOC_ON_COMPUTE_STREAM", True)
    mixed_grads = [torch.ones(8, dtype=torch.bfloat16), torch.ones(8, dtype=torch.float32)]
    hook = (lambda _tensor: None) if has_hook else None
    arguments = dict(
        fsdp_params=[SimpleNamespace(reduce_dtype=torch.float32), SimpleNamespace(reduce_dtype=torch.float32)],
        unsharded_grads=mixed_grads,
        reduce_scatter_group=None if size is None else SimpleNamespace(size=lambda: size),
        reduce_scatter_stream=s.reduce,
        reduce_scatter_comm=s.comm,
        orig_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        device=torch.device("cpu"),
        gradient_divide_factor=factor,
        all_reduce_group=object() if all_reduce else None,
        all_reduce_stream=s.all_reduce,
        all_reduce_grads=all_reduce_grads,
        partial_reduce_output=None,
        all_reduce_hook=hook,
        force_sum_reduction_for_comms=False,
    )

    def native(*args, **kwargs):
        received = fsdp_patch._FOREACH_REDUCE_SIGNATURE.bind(*args, **kwargs).arguments
        for name, value in arguments.items():
            if name not in ("reduce_scatter_comm", "gradient_divide_factor"):
                assert received[name] is value
        assert all(grad.dtype == torch.float32 for grad in received["unsharded_grads"])
        assert received["gradient_divide_factor"] == expected
        received["reduce_scatter_comm"].allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        with s.handle.stream(received["reduce_scatter_stream"]):
            return received["reduce_scatter_comm"].allocate((8,), dtype=torch.float32, device=torch.device("cpu"))

    monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_REDUCE", native)
    out = fsdp_patch._patched_foreach_reduce(**arguments)
    assert out is s.buffers[1]
    assert s.events == [("allocate", "owner"), ("allocate", "owner"), ("wait_stream", "reduce", "owner")]
    s.buffers[0].record_stream.assert_not_called()
    expected_streams = [s.reduce, s.all_reduce] if expect_all_reduce_record else [s.reduce]
    assert [call.args[0] for call in s.buffers[1].record_stream.call_args_list] == expected_streams
