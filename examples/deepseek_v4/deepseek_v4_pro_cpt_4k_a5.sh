#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run on each A5 node with NODE_IPS listing all participating addresses.
# Defaults: 64 nodes x 8 devices, FSDP=512, EP=128, GBS=4096.
# Append CLI arguments to override training defaults.
set -euo pipefail

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17330}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-2147480}"
export HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-2147480}"
# Allow long checkpoint loading and compilation without the HCCL watchdog.
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"

# Override this for the host topology reported by `npu-smi info -t topo`.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"
AC_MODE="${AC_MODE:-selective}"

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
NNODES=$(awk -F, '{print NF}' <<< "${NODE_IPS}")
WORLD_SIZE=$((NGPU * NNODES))

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
if [[ "${USE_GOLDEN:-0}" == "1" ]]; then
    CONFIG="${CONFIG:-deepseek_v4_pro}"
else
    CONFIG="${CONFIG:-deepseek_v4_pro_deepep}"
fi

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV4_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path
CKPT_INIT_LOAD_PATH="${CKPT_INIT_LOAD_PATH:-/path/to/init_load_ckpt}" # your model initial load ckpt path

# Parallelism
TP=1
PP=1
EP="${EP:-128}"
CP="${CP:-1}"
DP_SHARD="${DP_SHARD:-512}"
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
SEQ_LEN=4096
MBS=1
GBS="${GBS:-4096}"
STEPS=100

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.print-config
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
    --dataloader.per_doc_alignment $((128 * CP))
    --dataloader.dataset-path ${DATASET_PATH}
"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.context-parallel-load-balancer doc_chunk
    --parallelism.pipeline-parallel-degree ${PP}
"

# Compile
COMPILE_ARGS="
    --compile.enable
    --compile.components model
    --compile.backend inductor
"

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint
# `checkpoint.folder` is the output/resume root (`CKPT_SAVE_LOAD_PATH`). If it
# already contains a valid step-* checkpoint, upstream TorchTitan resumes from
# it and ignores `initial-load-path`; use a new/empty folder when cold-starting
# from `CKPT_INIT_LOAD_PATH`.
CHECKPOINT_ARGS="
    --checkpoint.enable
    --checkpoint.load-only
    --checkpoint.folder ${CKPT_SAVE_LOAD_PATH}
    --checkpoint.initial-load-path ${CKPT_INIT_LOAD_PATH}
    --checkpoint.initial-load-in-hf
"

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.save-traces-folder profiling_path
    --profiler.extension.no-enable-online-parse
    --profiler.extension.profiler-start 6
    --profiler.extension.profiler-end 7
    --profiler.extension.profile-ranks 0
"

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# Optimizer
OPTIMIZER_ARGS="
    --optimizer.name Muon
    --optimizer.lr 1.0e-5
    --optimizer.beta1 0.9
    --optimizer.beta2 0.95
    --optimizer.eps 1.0e-8
    --optimizer.weight_decay 1.0e-1
    --optimizer.muon_momentum 0.95
    --optimizer.muon_enable_nesterov
    --optimizer.muon_ns_steps 10
    --optimizer.muon_adjust_lr_fn match_rms_adamw
"
# Disable swap optimizer for better performance.
OPTIMIZER_OVERRIDES=""

if [[ "${USE_GOLDEN}" == "1" ]]; then
    DEFAULT_CLI_OVERRIDES=()
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rope.workaround
    )
else
    DEFAULT_CLI_OVERRIDES=(
        torchtitan_npu.override.common.rope.asc_partial
        torchtitan_npu.override.deepseek_v4.compressor.asc
        torchtitan_npu.override.common.swiglu_group.asc
        torchtitan_npu.override.common.swiglu_group.asc_shared_experts
        torchtitan_npu.override.common.token_dispatcher.asc_deepep
    )
    NPU_OPS_OVERRIDES=(
        # Attention / DSA
        torchtitan_npu.override.common.rms_norm.asc

        torchtitan_npu.override.deepseek_v4.lightning_indexer.asc

        'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"num_heads":128}'
        # MHC
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post
    )
fi

if [[ -z "${CLI_OVERRIDES:-}" ]]; then
    CLI_OVERRIDES="${DEFAULT_CLI_OVERRIDES[*]}"
fi

# Drop mm and bmm from the saved op list using update_save_ops override to avoid OOM
SELECTIVE_AC_SAVE_OP_EDITS=(
    -torch.ops.aten.bmm.default
    -torch.ops.aten.mm.default
)
if [[ "${AC_MODE}" == "selective" ]]; then
    SELECTIVE_AC_OVERRIDE="torchtitan_npu.override.common.activation_checkpoint.update_save_ops"
    if [[ ${#SELECTIVE_AC_SAVE_OP_EDITS[@]} -gt 0 ]]; then
        SELECTIVE_AC_SAVE_OP_EDITS_JOINED="$(IFS=,; printf '%s' "${SELECTIVE_AC_SAVE_OP_EDITS[*]}")"
        SELECTIVE_AC_OVERRIDE+="={\"edits\":\"${SELECTIVE_AC_SAVE_OP_EDITS_JOINED}\"}"
    fi
    CLI_OVERRIDES="${CLI_OVERRIDES:-} ${SELECTIVE_AC_OVERRIDE}"
    export CLI_OVERRIDES
fi

read -r -a CLI_OVERRIDES <<< "${CLI_OVERRIDES}"

QUANTIZATION_ARGS=()
if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    # FSDP shards dense dim0: only wq_b stays 64-aligned at DP_SHARD=512.
    # Routed experts use unsupported Shard(1) here (512 > 384 experts).
    QUANTIZATION_ARGS=(
        --extension.quantization.enable-quantized-training
        --extension.quantization.recipe all_block_fp8
        --extension.quantization.enable-fsdp-prequantize
        --extension.quantization.fsdp-prequantize-fqns .attention.wq_b
        --extension.quantization.li-quantization fp8
        --extension.quantization.kv-norm-quantization.format mxfp8
        --extension.quantization.kv-norm-quantization.fqns .attention.kv_norm,.attention.compressor.norm
        --extension.quantization.kv-norm-quantization.block-size 64
    )
fi

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NODE_IPS="${NODE_IPS}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train_multinodes.sh \
    $COMPILE_ARGS \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    "${QUANTIZATION_ARGS[@]}" \
    --debug.moe-force-load-balance \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "${CLI_OVERRIDES[@]}" \
    "$@" \
    activation-checkpoint:"${AC_MODE}"
