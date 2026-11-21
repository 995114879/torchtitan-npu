# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep known Triton launcher incompatibilities visible in every smoke entry point."""

from importlib import metadata
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

_CPP17_TRITON_RELEASES = frozenset({"3.2.1", "3.2.2", "3.2.2.post2", "3.6.0", "3.6.0.post2"})
_SMOKE_ROOT = Path(__file__).resolve().parent


def _triton_launcher_skip_reason():
    try:
        torch_version = metadata.version("torch")
        triton_version = metadata.version("triton-ascend")
        torch_release = Version(torch_version).release
        triton_release = Version(triton_version)
    except (metadata.PackageNotFoundError, InvalidVersion):
        return None

    # These published releases were inspected directly. Unknown releases and
    # custom local builds must run normally so a new regression is not hidden.
    if (
        torch_release < (2, 14)
        or triton_release.local is not None
        or triton_release.public not in _CPP17_TRITON_RELEASES
    ):
        return None

    return (
        f"Triton-Ascend {triton_version} uses C++17 launcher compilation, but Torch {torch_version} requires C++20 "
        "(https://github.com/triton-lang/triton-ascend/issues/1411); "
        "use --run-unsupported-triton when validating a compiler fix"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-unsupported-triton", default=False):
        return
    reason = _triton_launcher_skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(item.path).resolve().is_relative_to(_SMOKE_ROOT) and item.get_closest_marker("triton_ascend"):
            item.add_marker(skip)
