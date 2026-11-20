# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full DeepSeek-V4 training with MXFP8 DeepEP on two Ascend 950 NPUs.

Uses the committed C4/tokenizer fixtures, TorchAO-NPU and CANN ElasticBuffer.
The model keeps its default router-score placement. This case checks training
completion and finite metrics, not numerical equivalence to high precision.
"""

__all__ = ["build_deepep_test_list"]

from tests.integration_tests import OverrideDefinitions


def build_deepep_test_list() -> list[OverrideDefinitions]:
    """Run the production model, quantization converters and trainer at EP2/FSDP2."""
    return [
        OverrideDefinitions(
            test_name="dsv4_deepep_mxfp8_ep2_fsdp2",
            test_descr="DeepSeek-V4 MXFP8 DeepEP forward/backward and optimizer updates on A5",
            env_vars={
                "MODULE": "tests.integration_tests.deepep_config",
                "CONFIG": "deepseek_v4_deepep_training",
            },
            override_args=[(
                "--override.imports",
                "torchtitan_npu.override.common.rope.workaround",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.steps=2",
                "--training.local-batch-size=1",
                "--training.global-batch-size=2",
                "--training.seq-len=256",
                "--parallelism.spmd-backend=spmd_types",
                "--parallelism.data-parallel-shard-degree=2",
                "--parallelism.data-parallel-replicate-degree=1",
                "--parallelism.expert-parallel-degree=2",
                "--parallelism.tensor-parallel-degree=1",
                "--parallelism.context-parallel-degree=1",
                "--parallelism.pipeline-parallel-degree=1",
                "--optimizer.name=AdamW",
                "--optimizer.lr=0.001",
                "--optimizer.weight-decay=0.0",
                "--lr-scheduler.warmup-steps=0",
                "--extension.quantization.enable-quantized-training",
                "--extension.quantization.recipe=all_block_fp8",
                "--extension.quantization.all-to-all-quant-format=mxfp8",
                "--extension.quantization.no-enable-fsdp-prequantize",
                "--checkpoint.no-enable",
                "--compile.no-enable",
                "--debug.seed=42",
            )],
            ngpu=2,
            use_golden=True,
            check_loss=False,
            check_grad_norm=True,
            expected_steps=((1, 2),),
            timeout=1800,
        ),
    ]
