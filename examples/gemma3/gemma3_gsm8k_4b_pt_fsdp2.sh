#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run the Gemma3 4B text SFT recipe on one node with two NPU ranks.
# Prepare the Gemma3 model assets first; see examples/gemma3/gemma3_gsm8k_sft.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

WORKDIR="${GEMMA3_WORKDIR:-${REPO_ROOT}/.gemma3_gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/runs/gemma3-gsm8k-4b-pt}"

SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GBS="${GBS:--1}"
STEPS="${STEPS:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-$(( (STEPS * 3 + 99) / 100 ))}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-300}"

cd -- "${REPO_ROOT}"

MODULE="torchtitan_npu.models.gemma3" \
CONFIG="sft_gemma3_4b_fsdp2" \
NGPU=2 \
bash "${REPO_ROOT}/scripts/run_train.sh" \
    --dump-folder "${OUTPUT_DIR}" \
    --hf-assets-path "${WORKDIR}/models/gemma-3-4b-pt" \
    --training.local-batch-size "${MBS}" \
    --training.global-batch-size "${GBS}" \
    --training.seq-len "${SEQ_LEN}" \
    --training.steps "${STEPS}" \
    --lr-scheduler.warmup-steps "${WARMUP_STEPS}" \
    --checkpoint.interval "${CHECKPOINT_INTERVAL}" \
    "$@"
