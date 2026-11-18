# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import torch
import torch_npu  # noqa: F401


def configure_npu_tests():
    """Run before test-module imports, including device-valued parametrization."""
    if not torch.npu.is_available():
        raise pytest.UsageError("NPU is not available; select tests/unit_tests on CPU hosts.")


@pytest.fixture(autouse=True)
def reset_npu_state(_reset_test_state):
    torch.npu.manual_seed_all(42)
    yield
    gc.collect()
    torch.npu.empty_cache()
