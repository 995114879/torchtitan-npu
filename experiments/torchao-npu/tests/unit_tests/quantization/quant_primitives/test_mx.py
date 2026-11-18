# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torchao_npu.quantization.quant_primitives.mx import (
    _get_fp4_e2m1_pair_lut,
    mxfp4_dequantize,
)


def test_fp4_pair_lut_decodes_both_nibble_orders():
    low_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, True)
    high_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, False)

    assert low_first.shape == (256, 2)
    assert torch.equal(low_first[0x21], torch.tensor([0.5, 1.0]))
    assert torch.equal(high_first[0x21], torch.tensor([1.0, 0.5]))


def test_mxfp4_dequantize_unpacks_values_and_e8m0_scale():
    data = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scale = torch.full((1, 1, 2), 127, dtype=torch.uint8)

    result = mxfp4_dequantize(
        data,
        scale,
        axis=-1,
        block_size=32,
        output_shape=torch.Size((1, 32)),
        output_dtype=torch.float32,
    )

    expected = torch.tensor([0.5, 1.0] * 16).reshape(1, 32)
    assert torch.equal(result, expected)


def test_mxfp4_dequantize_rejects_partial_blocks():
    with pytest.raises(AssertionError, match="divisible by block_size"):
        mxfp4_dequantize(
            torch.zeros(1, 15, dtype=torch.uint8),
            torch.zeros(1, 1, 2, dtype=torch.uint8),
            axis=-1,
            block_size=32,
            output_shape=torch.Size((1, 30)),
            output_dtype=torch.float32,
        )
