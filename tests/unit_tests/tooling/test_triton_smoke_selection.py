# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check pytest selection using CPU stand-ins, without executing NPU kernels."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling
_REPO = Path(__file__).resolve().parents[3]


def _run_smoke(tmp_path, arguments, *, torch_version="2.14.0+cpu", triton_version="3.2.1"):
    for relative in ("tests/conftest.py", "tests/smoke_tests/conftest.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / relative, destination)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    triton_ascend: Triton launcher case\n")
    (tmp_path / "tests/smoke_tests/test_selection.py").write_text(
        "import pytest\n@pytest.mark.triton_ascend\ndef test_triton():\n    pass\ndef test_other():\n    pass\n"
    )
    distributions = tmp_path / "distributions"
    for name, version in (("torch", torch_version), ("triton-ascend", triton_version)):
        info = distributions / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n")
    environment = dict(
        os.environ,
        PYTHONPATH=str(distributions),
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        OMP_NUM_THREADS="1",
    )
    environment.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("torch_version", "triton_version", "should_skip"),
    [
        pytest.param("2.14.0+cpu", "3.2.1", True, id="torch214"),
        pytest.param("2.14.1", "3.2.1", True, id="torch214patch"),
        pytest.param("2.15.0", "3.6.0.post2", True, id="torch215"),
        pytest.param("2.14.0.dev20260920+cpu", "3.2.1", True, id="torch214dev"),
        pytest.param("2.13.0", "3.2.1", False, id="oldertorch"),
        pytest.param("2.15.0", "3.6.1", False, id="unlistedtriton"),
        pytest.param("2.15.0", "3.2.1+patched", False, id="localbuild"),
    ],
)
def test_known_cpp17_releases_skip_on_torch214_and_newer(tmp_path, torch_version, triton_version, should_skip):
    result = _run_smoke(tmp_path, ["tests"], torch_version=torch_version, triton_version=triton_version)

    assert result.returncode == 0, result.stdout + result.stderr
    assert ("1 passed, 1 skipped" if should_skip else "2 passed") in result.stdout
    if should_skip:
        assert "uses C++17" in result.stdout and torch_version in result.stdout


@pytest.mark.parametrize("path", ["tests", "tests/smoke_tests"], ids=["parent", "smoke"])
def test_override_is_available_from_test_entrypoints(tmp_path, path):
    result = _run_smoke(tmp_path, [path, "-m", "triton_ascend", "--run-unsupported-triton"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout and "1 deselected" in result.stdout


def test_bare_pytest_keeps_default_skip_when_option_is_not_registered(tmp_path):
    result = _run_smoke(tmp_path, [])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 skipped" in result.stdout
