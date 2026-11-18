# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType

import pytest


@pytest.fixture
def cpu_custom_ops(monkeypatch):
    """Isolate NPU registration and its Python consumers for one CPU test."""
    consumers = (
        "torchao_npu.ops.kv_cache_fake_quant",
        "torchao_npu.quantized_modules.deepseek_v4_1.compressor",
        "torchao_npu.quantized_modules.deepseek_v4_1.sparse_attention",
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "custom_ops", ModuleType("custom_ops"))
        for name in consumers:
            patch.delitem(sys.modules, name, raising=False)
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                patch.delattr(parent, child, raising=False)
        try:
            yield
        finally:
            # Import also caches children on their parent packages. Remove new
            # entries before monkeypatch restores any pre-existing real modules.
            for name in consumers:
                sys.modules.pop(name, None)
                parent_name, child = name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    vars(parent).pop(child, None)
