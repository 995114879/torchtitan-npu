# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep broken TorchFT installations visible while allowing the package to be absent."""

from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path

collect_ignore = []
if find_spec("torchft") is None:
    try:
        distribution("torchft")
    except PackageNotFoundError:
        # The control service runs without TorchFT.
        test_files = Path(__file__).parent.glob("test_*.py")
        collect_ignore = [path.name for path in test_files if path.stem != "test_last_words_service"]
