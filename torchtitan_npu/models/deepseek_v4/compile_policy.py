# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default compile policy when using the upstream block wrapping order."""

from dataclasses import replace

from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torch_npu.device import is_a5

__all__ = ["compiles_under_cp", "skip_compile_under_cp"]


def compiles_under_cp(parallel_dims, *, a5: bool | None = None) -> bool:
    """Whether this run's blocks can each be one graph (see the module docstring).

    ``a5`` states the architecture; ``None`` asks the device.  The parameter exists so the
    decision can be pinned in a test without an NPU.
    """
    return (is_a5() if a5 is None else a5) or parallel_dims.cp <= 1


def skip_compile_under_cp(model, compile_config, parallel_dims, *, a5: bool | None = None):
    """``compile_config``, with compilation turned off when this run cannot support it.

    Logs once, with the reason, when it does: a skipped compile must never be silent.  Returns the
    config unchanged -- same object -- when compilation is possible or was not asked for.
    """
    if not compile_config.enable or compiles_under_cp(parallel_dims, a5=a5):
        return compile_config
    logger.warning(
        "Skipping torch.compile for %s: this arch22 context-parallel configuration runs eagerly.",
        type(model).__name__,
    )
    return replace(compile_config, enable=False)
