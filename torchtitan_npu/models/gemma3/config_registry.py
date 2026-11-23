# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma3 training recipes, data configuration, and learning-rate scheduling."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import CompileConfig, DebugConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import OptimizerConfig, ParallelismConfig, TrainingConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

from . import model_registry


def process_gsm8k(sample: Mapping[str, str]) -> list[dict[str, str]]:
    """Map a GSM8K row to TorchTitan's single-turn chat contract.

    ``ChatDataLoader`` owns message validation, tokenization, assistant-only
    labels, packing and checkpoint state. This adapter only maps the two
    columns exposed by the upstream ``openai/gsm8k`` dataset.
    """

    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _sft_dataloader() -> ChatDataLoader.Config:
    """Use TorchTitan's single-turn ChatDataLoader for GSM8K.

    The upstream ``openai/gsm8k`` dataset is loaded directly and the processor
    maps its ``question``/``answer`` columns to one chat turn.
    """

    kwargs: dict[str, Any] = {"name": "main", "split": "train"}
    return ChatDataLoader.Config(
        dataset_path="openai/gsm8k",
        load_dataset_kwargs=kwargs,
        sample_processor=process_gsm8k,
        infinite=True,
    )


def _gemma3_trainer_base(flavor: str, hf_assets_path: str) -> TrainerEx.Config:
    model_spec = model_registry(flavor)
    adamw = default_adamw(lr=1e-5, weight_decay=0.0)
    return TrainerEx.Config(
        hf_assets_path=hf_assets_path,
        model_spec=model_spec,
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        optimizer=OptimizerConfig(
            param_groups=adamw.param_groups,
            implementation="for-loop",
            optimizer_factory_kwargs_by_name=adamw.optimizer_factory_kwargs_by_name,
        ),
        lr_scheduler=LRSchedulersContainer.Config(),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16 if flavor == "debug" else 2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            spmd_backend="partial_dtensor",
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        profiler=CANNProfiler.Config(enable_profiling=False),
        debug=DebugConfig(seed=20260824, deterministic=True),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(enable=False, interval=10),
    )


def debug_gemma3_single_node() -> TrainerEx.Config:
    return _gemma3_trainer_base("debug", "./tests/assets/tokenizer")


def debug_gemma3_fsdp2() -> TrainerEx.Config:
    """Two-rank native mixed precision, preserving FP32 optimizer parameters."""
    base = debug_gemma3_single_node()
    assert base.model_spec is not None
    return replace(
        base,
        training=replace(base.training, dtype="float32", disable_cuda_graphs=True),
        parallelism=replace(base.parallelism, data_parallel_shard_degree=2),
        model_spec=replace(
            base.model_spec,
            parallelize_fn=partial(base.model_spec.parallelize_fn, enable_mixed_precision=True),
        ),
    )


def sft_gemma3_4b_fsdp2() -> TrainerEx.Config:
    """Two-rank single-turn GSM8K SFT from explicitly selected HF weights."""
    from torchtitan.distributed.activation_checkpoint import FullAC

    base = _gemma3_trainer_base("4B", "./.gemma3_gsm8k/models/gemma-3-4b-pt")
    assert base.model_spec is not None
    return replace(
        base,
        dataloader=_sft_dataloader(),
        tokenizer=HuggingFaceTokenizer.Config(),
        training=replace(base.training, dtype="float32", disable_cuda_graphs=True),
        parallelism=replace(base.parallelism, data_parallel_shard_degree=2),
        model_spec=replace(
            base.model_spec,
            parallelize_fn=partial(base.model_spec.parallelize_fn, enable_mixed_precision=True),
        ),
        activation_checkpoint=FullAC.Config(),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=10,
            last_save_model_only=False,
            initial_load_in_hf=True,
            initial_load_path=None,
        ),
    )
