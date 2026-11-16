# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU boundary for the MX reader tests.

Replaces the NPU dequant kernel with the spec-level LUT/exp2 reference and
records its call shapes.
"""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cpu_npu_boundary(monkeypatch):
    calls = []

    def anti_quant(qdata, scale, *, axis, dst_type, src_type):
        calls.append((qdata.shape, scale.shape, axis, dst_type, src_type))
        if qdata.dtype == torch.uint8:
            # 0xE2 stores +1 and -4, low nibble first (packed MXFP4).
            lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
            values = lut[torch.stack((qdata & 15, qdata >> 4), dim=-1).long()].flatten(-2)
        else:
            values = qdata.float()
        # scale arrives as the backend's paired view (..., blocks/2, 2); flatten(-2) only
        # undoes that pairing, so per-row scales never mix across rows.
        scales = torch.exp2(scale.flatten(-2).float() - 127).repeat_interleave(32, -1)
        return values * scales[..., : values.shape[-1]]

    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    device_module = SimpleNamespace(
        Stream=lambda **kw: stream,
        Event=lambda: event,
        current_stream=lambda device: stream,
        stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(float4_e2m1fn_x2=object(), npu_anti_mx_quant=anti_quant),
    )
    return calls
