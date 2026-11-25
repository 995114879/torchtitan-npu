#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This source code is licensed under the BSD-style license found in LICENSE.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
patch="$root/torchtitan_npu/patches/torchft/torchft-0.2.0-lighthouse-preemption.patch"
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

if [[ -n "${TORCHFT_SDIST:-}" ]]; then
    archive=$(realpath "$TORCHFT_SDIST")
else
    archive="$work/torchft.tar.gz"
    curl -fsSL --retry 3 https://files.pythonhosted.org/packages/source/t/torchft/torchft-0.2.0.tar.gz -o "$archive"
fi
printf '%s  %s\n' 542e17910039735d6686517f98ed49686ba23fc4283b15cf1d984a251eb1546a "$archive" | sha256sum -c -
mkdir "$work/src"
tar -xzf "$archive" --strip-components=1 -C "$work/src"
cd "$work/src"
git init -q
git apply --check "$patch"
git apply "$patch"
cargo test --release --lib
