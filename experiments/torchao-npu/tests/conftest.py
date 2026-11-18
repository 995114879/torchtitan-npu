# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import torch
from _support.compat import _install_op_plugin_current_version_shim

# CPU tests also import the CANN Python package. This shim never probes hardware.
_install_op_plugin_current_version_shim()


@pytest.fixture(autouse=True)
def _reset_test_state():
    """Keep compiler state and random seeds isolated without probing an NPU."""
    torch.manual_seed(42)
    torch._dynamo.reset()
    torch.compiler.reset()
    yield
    gc.collect()


@pytest.fixture
def mock_distributed_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12355")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
