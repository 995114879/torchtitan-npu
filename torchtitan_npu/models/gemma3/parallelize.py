# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/models/llama3/parallelize.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 parallelization adapter for TorchTitan's public compile/FSDP APIs.

The TorchTitan v0.3.0 interfaces imported below own compilation and decoder
FSDP2 mechanics.  Gemma3-specific validation and the model contract stay in this
module so the plugin does not modify the upstream checkout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder

if TYPE_CHECKING:
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig

    from .model import Gemma3Model

_PARALLEL_DEGREES = (
    "dp_replicate",
    "dp_shard",
    "tp",
    "pp",
    "cp",
    "ep",
)


def _validate_single_rank_parallel_dims(parallel_dims: ParallelDims) -> None:
    if parallel_dims.world_size != 1:
        raise ValueError(f"Gemma3 without FSDP requires world_size=1; got world_size={parallel_dims.world_size}")

    for dimension in _PARALLEL_DEGREES:
        degree = getattr(parallel_dims, dimension)
        if degree != 1:
            raise ValueError(f"Gemma3 without FSDP requires {dimension}=1; got {dimension}={degree}")


def validate_fsdp_parallel_dims(parallel_dims: ParallelDims) -> None:
    if parallel_dims.world_size not in (1, 2) or parallel_dims.dp_shard != parallel_dims.world_size:
        raise ValueError(
            "Gemma3 mixed-precision FSDP2 currently supports only "
            "world_size=dp_shard in {1, 2}; larger world sizes and mixed parallel "
            "topologies are not supported yet "
            f"(got world_size={parallel_dims.world_size}, dp_shard={parallel_dims.dp_shard})"
        )
    for name in ("dp_replicate", "tp", "pp", "cp", "ep"):
        if getattr(parallel_dims, name, 1) != 1:
            raise ValueError(f"Gemma3 FSDP2 requires {name}=1")


def parallelize_gemma3(
    model: Gemma3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    enable_mixed_precision: bool = False,
) -> Gemma3Model:
    """Apply activation checkpointing and native mixed precision FSDP2."""
    if enable_mixed_precision:
        validate_fsdp_parallel_dims(parallel_dims)
        if parallelism.spmd_backend != "partial_dtensor" or training.enable_cpu_offload:
            raise ValueError("Gemma3 FSDP2 requires partial_dtensor and no CPU offload")
        if (training.dtype, training.mixed_precision_param, training.mixed_precision_reduce) != (
            "float32",
            "bfloat16",
            "float32",
        ):
            raise ValueError("Gemma3 requires FP32 storage, BF16 compute and FP32 reduction")
    else:
        _validate_single_rank_parallel_dims(parallel_dims)

    model_compile_enabled = bool(compile_config.enable and "model" in (compile_config.components or ()))
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    if enable_mixed_precision:
        mesh = parallel_dims.get_mesh("fsdp")
        if mesh.ndim != 1 or mesh.size() != parallel_dims.dp_shard:
            raise ValueError("Gemma3 FSDP2 requires a one-dimensional dp_shard mesh")
        apply_fsdp_to_decoder(
            model,
            mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )
    return model
