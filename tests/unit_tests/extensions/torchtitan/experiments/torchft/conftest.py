# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep broken TorchFT installations visible while allowing the package to be absent."""

from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec

collect_ignore_glob = []
if find_spec("torchft") is None:
    try:
        distribution("torchft")
    except PackageNotFoundError:
        collect_ignore_glob = ["test_*.py"]
