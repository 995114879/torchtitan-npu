# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch
import torch_npu  # noqa: F401


def _torch_major_minor():
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _install_op_plugin_current_version_shim():
    """torch_npu >= 2.13 op_plugin bug: ``op_plugin/utils/Version.h`` computes
    ``VERSION_BETWEEN`` from a ``CURRENT_VERSION`` macro that only the
    torch_npu build system defines. The CANN JIT path (cann_ops_nn
    ``OpBuilder`` -> torch cpp_extension) never defines it, so
    ``NamedTensorCompat.h`` always includes ``ATen/NamedTensorUtils.h``, which
    torch >= 2.13 removed. Inject the macro (V2R<N> == torch 2.<N>, value
    N - 1) so the header takes its in-tree >= 2.13 fallback branch.
    """
    major, minor = _torch_major_minor()
    if major != 2 or minor < 13:
        return
    try:
        from cann_ops_nn.op_builder.builder import OpBuilder
    except ImportError:
        return
    if getattr(OpBuilder, "torchao_npu_shim", False):
        return
    current_version = (major - 2) * 10 + (minor - 1)
    orig_cxx_args = OpBuilder.cxx_args

    def cxx_args(self):
        args = list(orig_cxx_args(self))
        args.append(f"-DCURRENT_VERSION={current_version}")
        return args

    OpBuilder.cxx_args = cxx_args
    OpBuilder.torchao_npu_shim = True
