#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.
set -euo pipefail
if [[ ${1:-} == --compile-worker ]]; then
    shift
    # ascendc reads these paths on import. torchrun supplies LOCAL_RANK to each worker.
    for key in TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_NPU_EXT_CACHE_DIR; do
        if [[ ${!key:-} != /* ]]; then
            echo "Small recovery requires an absolute cache root for $key" >&2
            exit 1
        fi
        export "${key}=${!key}/replica-${LAST_WORDS_TEST_REPLICA}-rank-${LOCAL_RANK}"
    done
    exec "$@"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON_BIN:-python3}" -m tests.integration_tests.last_words "$@"
