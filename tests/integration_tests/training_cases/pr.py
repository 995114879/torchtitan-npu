# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cases introduced by the pr layer; inheritance lives in .github/ci/suites.json."""

from tests.integration_tests import OverrideDefinitions

from .common import ASSETS, COMMON, EAGER, V4_ENV, V41_ENV, case_defaults


def _script_default_env(env: dict[str, str]) -> dict[str, str]:
    """Drop CONFIG so each A5 launcher keeps its own ``*_deepep`` default."""
    return {key: value for key, value in env.items() if key != "CONFIG"}


# A5 nodes expose 8 NPUs. With CP2 the dense shard must shrink to 4 so that
# replicate(1) * shard(4) * cp(2) * tp(1) * pp(1) == 8. EP8 is independent of
# the dense DP axis; the sparse mesh becomes efsdp(1) * ep(8), and 8 divides
# both 16 and 24 experts.
CP2_EP8_ARGS = (
    "--parallelism.context-parallel-degree=2",
    "--parallelism.expert-parallel-degree=8",
    "--parallelism.data-parallel-shard-degree=4",
    "--parallelism.data-parallel-replicate-degree=1",
    "--dataloader.per-doc-alignment=256",
)

# CP, MBS and DP_SHARD stay at each launcher's own defaults unless a case
# exercises something else; only pr_01 overrides them for CP2.
V4_A5_ENV = _script_default_env(V4_ENV)

# The A5 V4.1 wrapper selects the 24-expert multimodal recipe and its own
# DeepEP config; keep the remaining V4.1 knobs from the shared environment.
V41_A5_ENV = {**_script_default_env(V41_ENV), "DATASET": "cc12m-test"}


def build_test_list() -> list[OverrideDefinitions]:
    common = case_defaults()
    return [
        OverrideDefinitions(
            test_name="dsv4_flash_a5_8p_cp2_ep8_eager",
            test_descr="pr_01: DSV4 Flash A5 1x8P CP2 EP8 block-FP8 eager",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh",
            env_vars={**V4_A5_ENV, "CP": "2", "DP_SHARD": "4"},
            override_args=[(*COMMON, *EAGER, *CP2_EP8_ARGS)],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a5_8p_cp1_ep8_compile",
            test_descr="pr_02: DSV4 Flash A5 1x8P CP1 EP8 block-FP8 compile",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh",
            env_vars={**V4_A5_ENV},
            override_args=[(*COMMON, "--compile.enable")],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv41_flash_a5_8p_cp1_ep8_eager",
            test_descr="pr_03: DSV4.1 Flash A5 1x8P CP1 EP8 multimodal eager",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh",
            env_vars={**V41_A5_ENV},
            override_args=[(*COMMON, *EAGER)],
            **{**common, "required_env": (*ASSETS, "DATASET_PATH")},
        ),
        OverrideDefinitions(
            test_name="dsv41_flash_a5_8p_cp1_ep8_compile",
            test_descr="pr_04: DSV4.1 Flash A5 1x8P CP1 EP8 multimodal compile",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh",
            env_vars={**V41_A5_ENV},
            override_args=[(*COMMON, "--compile.enable")],
            **{**common, "required_env": (*ASSETS, "DATASET_PATH")},
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a5_8p_cp1_ep8_graphtrainer",
            test_descr="pr_05: DSV4 Flash A5 1x8P CP1 EP8 GraphTrainer",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_graphtrainer.sh",
            env_vars={**V4_A5_ENV},
            override_args=[COMMON],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv41_flash_a5_8p_cp1_ep8_qat",
            test_descr="pr_06: DSV4.1 Flash A5 1x8P CP1 EP8 MXFP4 QAT compile",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_qat_4k_a5.sh",
            env_vars={**V41_A5_ENV},
            override_args=[(*COMMON, "--compile.enable")],
            **{**common, "required_env": (*ASSETS, "DATASET_PATH")},
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a5_8p_cp1_ep8_qat",
            test_descr="pr_07: DSV4 Flash A5 1x8P CP1 EP8 MXFP4 QAT compile",
            device="a5",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_qat_4k_a5.sh",
            env_vars={**V4_A5_ENV},
            override_args=[(*COMMON, "--compile.enable")],
            **common,
        ),
    ]
