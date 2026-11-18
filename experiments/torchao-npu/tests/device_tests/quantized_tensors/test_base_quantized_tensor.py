# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the behavior :class:`BaseQuantizedTensor` gives every subclass."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_device_moves_the_components(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to("npu")

    assert y.qdata.device.type == "npu", f".to('npu') left qdata on {y.qdata.device}"
    assert y.scale.device.type == "npu", f".to('npu') left scale on {y.scale.device}"
    assert y.dtype is x.dtype, f".to('npu') changed the logical dtype to {y.dtype}"
    assert x.qdata.device.type == "cpu", f".to('npu') moved the original's qdata to {x.qdata.device}"

    # the move must not alter the values: moving the plain components ourselves
    # (bypassing MXTensor's handler) has to reproduce y's components exactly
    moved_qdata = qdata.to(y.qdata.device)
    assert torch.equal(y.qdata.view(torch.uint8), moved_qdata.view(torch.uint8)), ".to('npu') changed the qdata values"

    moved_scale = scale.to(y.scale.device)
    assert torch.equal(y.scale.view(torch.uint8), moved_scale.view(torch.uint8)), ".to('npu') changed the scale values"


def test_is_pinned_requires_the_components_to_agree():
    """One component cannot answer for the other's storage, so a split answer is an error."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata.pin_memory(), scale, torch.bfloat16, -1, config)

    assert x.qdata.is_pinned() and not x.scale.is_pinned(), "the premise: only the qdata is pinned"

    with pytest.raises(RuntimeError, match="agree on ``is_pinned``"):
        x.is_pinned()
