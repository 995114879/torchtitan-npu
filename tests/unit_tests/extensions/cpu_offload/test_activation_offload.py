# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU boundary scheduling with real checkpoint and FSDP input views."""

import copy
import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp._fully_shard._fsdp_param_group import RegisterPostBackwardFunction
from torch.utils.checkpoint import DefaultDeviceType


class CpuSwapBackend:
    """Emulate storage retirement/restoration; does not emulate NPU streams."""

    def __init__(self):
        self.tensors = {}
        self.host = {}
        self.calls = []
        self.restored_refs = []

    def register_tensor(self, tensor, name):
        self.tensors[name] = tensor.detach()
        self.calls.append(("register", name))

    def execute(self, name, action):
        self.calls.append((action, name))
        tensor = self.tensors[name]
        if action == "D2H":
            self.host[name] = tensor.clone()
            tensor.untyped_storage().resize_(0)
        elif action == "H2D":
            tensor.untyped_storage().resize_(tensor.numel() * tensor.element_size())
            tensor.copy_(self.host.pop(name))

    def pop_tensor_by_name(self, name, *, device_waited=False):
        assert device_waited
        assert self.calls[-1] == ("WAIT_DEVICE", name)
        tensor = self.tensors.pop(name)
        self.restored_refs.append(weakref.ref(tensor))
        return tensor

    def remove_tensor(self, name):
        self.tensors.pop(name, None)
        self.host.pop(name, None)


@pytest.fixture
def offload_module(monkeypatch):
    backend = CpuSwapBackend()
    # Isolate only package activation and the hardware transfer API. The
    # scheduler, checkpoint engine and FSDP autograd function remain real.
    for name in ("torchtitan_npu", "torchtitan_npu.extensions", "torchtitan_npu.extensions.novaswap"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    sys.modules["torchtitan_npu.extensions.novaswap"].swap_api = backend
    path = Path(__file__).resolve().parents[4] / "torchtitan_npu/extensions/cpu_offload/activation_offload.py"
    spec = importlib.util.spec_from_file_location("_activation_boundary_unit", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, backend


def test_fsdp_input_identity_view_is_swappable(offload_module):
    module, _ = offload_module
    value = torch.arange(24.0).reshape(3, 8).clone().requires_grad_()
    wrapped = RegisterPostBackwardFunction.apply(None, value)[0]
    parameter = nn.Parameter(value.detach().clone())

    assert wrapped._base is value
    assert module.ActivationBoundaryOffload._has_compact_activation_storage(wrapped)
    assert not module.ActivationBoundaryOffload._has_compact_activation_storage(parameter.view_as(parameter))
    assert not module.ActivationBoundaryOffload._has_compact_activation_storage(value[:1])
    assert not module.ActivationBoundaryOffload._has_compact_activation_storage(value.t())


def test_storage_probe_failure_is_filtered(offload_module):
    module, _ = offload_module

    class UnallocatedStorageTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, value.requires_grad)

        def untyped_storage(self):
            raise RuntimeError("storage is unavailable")

    value = UnallocatedStorageTensor(torch.ones(4, requires_grad=True))
    assert not module.ActivationBoundaryOffload._has_compact_activation_storage(value)


class NamedTensorOutput(NamedTuple):
    value: torch.Tensor


def test_output_aliases_supports_pytree_containers(offload_module):
    module, _ = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    candidate = torch.ones(4)

    assert manager._output_aliases(NamedTensorOutput(candidate), candidate)


def test_output_aliases_fails_closed_for_unknown_containers(offload_module):
    module, _ = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    candidate = torch.ones(4)

    class UnknownContainer:
        def __init__(self, value):
            self.value = value

    assert manager._output_aliases(UnknownContainer(candidate), candidate)


def test_close_restores_checkpoint_wrapper_forward(offload_module):
    module, _ = offload_module
    wrapped = checkpoint_wrapper(nn.Identity())
    original_forward = wrapped.forward
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)

    assert manager.install([wrapped]) == 1
    assert wrapped.forward.__func__ is not original_forward.__func__
    assert wrapped._activation_boundary_offload

    manager.close()

    assert wrapped.forward.__func__ is original_forward.__func__
    assert not hasattr(wrapped, "_activation_boundary_offload")


def test_pack_defers_swap_registration_until_forward_selection(offload_module, monkeypatch):
    module, backend = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    monkeypatch.setattr(manager, "_candidate", lambda value: isinstance(value, torch.Tensor))
    group = manager._begin_group()
    token = manager.pack(torch.ones(4, requires_grad=True))

    assert isinstance(token, module._ActivationToken)
    assert backend.calls == []

    manager._finish_group(group, torch.zeros(1))

    assert [action for action, _ in backend.calls] == ["register", "D2H"]
    manager._cleanup()


def test_resident_token_releases_scheduler_reference_after_unpack(offload_module, monkeypatch):
    module, _ = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    monkeypatch.setattr(manager, "_candidate", lambda value: isinstance(value, torch.Tensor))
    group = manager._begin_group()
    token = manager.pack(torch.ones(4, requires_grad=True))
    manager._finish_group(group, token.tensor)
    resident = token.tensor
    resident_ref = weakref.ref(resident)

    restored = manager.unpack(token)

    assert restored is resident
    assert token.tensor is None
    del resident, restored
    gc.collect()
    assert resident_ref() is None


def test_registration_error_is_not_silently_downgraded(offload_module, monkeypatch):
    module, backend = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    monkeypatch.setattr(manager, "_candidate", lambda value: isinstance(value, torch.Tensor))
    monkeypatch.setattr(backend, "register_tensor", lambda *_args: (_ for _ in ()).throw(RuntimeError("register")))
    group = manager._begin_group()
    manager.pack(torch.ones(4, requires_grad=True))

    with pytest.raises(RuntimeError, match="register"):
        manager._finish_group(group, torch.zeros(1))


def test_registration_rejection_keeps_activation_resident(offload_module, monkeypatch):
    module, backend = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    monkeypatch.setattr(manager, "_candidate", lambda value: isinstance(value, torch.Tensor))
    monkeypatch.setattr(backend, "register_tensor", lambda *_args: (_ for _ in ()).throw(ValueError("unsupported")))
    group = manager._begin_group()
    token = manager.pack(torch.ones(4, requires_grad=True))

    manager._finish_group(group, torch.zeros(1))

    assert token.state == "resident"
    assert backend.calls == []


def test_cleanup_failure_keeps_live_name_and_raises(offload_module, monkeypatch):
    module, backend = offload_module
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    manager._live_names.add("activation.failed")
    monkeypatch.setattr(backend, "remove_tensor", lambda _name: (_ for _ in ()).throw(RuntimeError("cleanup")))

    with pytest.raises(RuntimeError, match="Failed to clean"):
        manager._cleanup()
    assert "activation.failed" in manager._live_names


class FsdpInputBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.backward_count = 0

    def post_backward(self):
        self.backward_count += 1

    def forward(self, value):
        value = RegisterPostBackwardFunction.apply(self, value)[0]
        return self.block(value)


def test_checkpoint_fsdp_views_restore_values_and_gradients(offload_module, monkeypatch):
    module, backend = offload_module
    # CPU-only inputs otherwise inherit the default checkpoint accelerator.
    monkeypatch.setattr(DefaultDeviceType, "get_device_type", lambda: "cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        reference = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh())
    candidate = copy.deepcopy(reference)
    candidate = nn.Sequential(
        FsdpInputBlock(checkpoint_wrapper(nn.Sequential(candidate[0], candidate[1]))),
        FsdpInputBlock(checkpoint_wrapper(nn.Sequential(candidate[2], candidate[3]))),
    )
    manager = module.ActivationBoundaryOffload(min_tensor_bytes=1)
    # CPU tests replace only device eligibility, preserving the production
    # compact-storage and Parameter/view checks that caused the FSDP failure.
    monkeypatch.setattr(
        manager,
        "_candidate",
        lambda value: isinstance(value, torch.Tensor)
        and value.requires_grad
        and value.numel() > 0
        and manager._has_compact_activation_storage(value),
    )
    manager.install([candidate])
    value = torch.linspace(-0.9, 0.8, 32).reshape(4, 8).requires_grad_()
    expected_input = value.detach().clone().requires_grad_()
    expected = reference(expected_input)
    expected.square().sum().backward()

    with manager.microbatch():
        actual = candidate(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        gc.collect()
        assert all(ref() is None for ref in backend.restored_refs)

    torch.testing.assert_close(value.grad, expected_input.grad, rtol=0, atol=0)
    for param, expected_param in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.grad, expected_param.grad, rtol=0, atol=0)
    d2h = [name for action, name in backend.calls if action == "D2H"]
    h2d = [name for action, name in backend.calls if action == "H2D"]
    assert len(d2h) == 2
    assert h2d == d2h[::-1]
    assert len([action for action, _ in backend.calls if action == "WAIT_DEVICE"]) == 2
    assert all(action != "WAIT" for action, _ in backend.calls)
    assert manager._stats["prefetch"] == 1
    assert [block.backward_count for block in candidate] == [1, 1]
    assert backend.tensors == {}
    assert backend.host == {}
