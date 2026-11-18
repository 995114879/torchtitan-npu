# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for :class:`torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor`."""

import pytest
import torch
import torch_npu
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.grouped_mx_tensor import GroupedMXTensor


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_from_hp_matches_npu_grouped_dynamic_mx_quant(elem_dtype):
    """``from_hp`` hands the config's parameters to the op and keeps its output as it is."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    offs = torch.tensor([16, 48, 64], device="npu", dtype=torch.int32)  # uneven groups

    y = GroupedMXTensor.from_hp(x, offs, config)
    qdata, scale = torch_npu.npu_grouped_dynamic_mx_quant(
        x,
        offs.to(torch.int32),
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        blocksize=config.block_size,
        scale_alg=config.scale_alg,
    )

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"
    assert y.quant_axis == 0, f"quant_axis tracked as {y.quant_axis}, expected 0 (dim -2)"
    expected_pack_axis = y.ndim - 1 if elem_dtype is torch.float4_e2m1fn_x2 else None
    assert y.pack_axis == expected_pack_axis, f"pack axis {y.pack_axis}, expected {expected_pack_axis}"

    assert y.qdata.shape == qdata.shape, f"qdata shape {tuple(y.qdata.shape)} != the op's {tuple(qdata.shape)}"
    assert torch.equal(y.qdata.view(torch.uint8), qdata.view(torch.uint8)), "qdata values differ from the op's"

    assert y.scale.shape == scale.shape, f"scale shape {tuple(y.scale.shape)} != the op's {tuple(scale.shape)}"

    # The op writes a group's scale for its ``j``-th 64-row block at
    # ``group_start // 64 + group_idx + j``. Rows reserved for group boundaries
    # but never written are slack and hold uninitialized bytes, so only the
    # meaningful rows are comparable.
    group_ends = offs.tolist()
    rows_per_scale_row = 2 * config.block_size
    meaningful_rows = [
        start // rows_per_scale_row + group_idx + block_idx
        for group_idx, (start, end) in enumerate(zip([0, *group_ends[:-1]], group_ends, strict=True))
        for block_idx in range(-(-(end - start) // rows_per_scale_row))
    ]
    assert torch.equal(y.scale[meaningful_rows].view(torch.uint8), scale[meaningful_rows].view(torch.uint8)), (
        "scale values differ from the op's on the meaningful rows"
    )


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([32, 64], device="npu", dtype=torch.int32)

    y = GroupedMXTensor.from_hp(x, offs, config)

    assert y.requires_grad, "from_hp dropped the input's requires_grad"
