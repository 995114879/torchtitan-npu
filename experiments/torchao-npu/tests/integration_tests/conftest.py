# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from _support.npu import configure_npu_tests, reset_npu_state  # noqa: F401

# Nested conftests can load after sessionstart; check here before collection.
configure_npu_tests()
