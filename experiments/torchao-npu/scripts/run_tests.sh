#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test entrypoint for CPU UT CI and manual NPU runs; see ../tests/README.md.

# Stop on setup errors, unset variables, and failures in pipelines ending with tee.
set -euo pipefail

# Handle script help before setup; arguments after a mode belong to pytest.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash experiments/torchao-npu/scripts/run_tests.sh [mode] [pytest args...]

Modes:
  cpu          unit_tests (default; included in UT CI)
  device       device_tests (requires NPU; manual)
  integration  integration_tests (requires NPU; manual)
  npu          device_tests, then integration_tests (requires NPU; manual)

Environment:
  PYTHON_BIN               Python executable (default: python3.12)
  CANN_ENV_PATH            Optional CANN environment script (default: unset)
  TORCHAO_TEST_REPORT_DIR   Report directory (default: <repo>/test_reports/torchao-npu)

Only this -h/--help entry skips setup. Mode arguments, including --collect-only,
run after dependency installation and environment validation.
See experiments/torchao-npu/tests/README.md for prerequisites and report handling.
HELP
    exit 0
fi

MODE="${1:-cpu}"
case "${MODE}" in
    cpu) SUITES=(unit_tests) ;;
    device) SUITES=(device_tests) ;;
    integration) SUITES=(integration_tests) ;;
    npu) SUITES=(device_tests integration_tests) ;;
    *) echo "Usage: bash experiments/torchao-npu/scripts/run_tests.sh {cpu|device|integration|npu}" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "${PROJECT_ROOT}"
# Record the final exit code even when dependency setup or device checks fail.
REPORT_DIR="${TORCHAO_TEST_REPORT_DIR:-${PROJECT_ROOT}/test_reports/torchao-npu}"
mkdir -p "${REPORT_DIR}"
REPORT_DIR="$(cd "${REPORT_DIR}" && pwd)"
exec {LOG_STDOUT_FD}>&1 {LOG_STDERR_FD}>&2
exec > >(tee "${REPORT_DIR}/run-${MODE}.log") 2>&1
LOG_TEE_PID=$!

_finish_test_run() {
    local result=$? log_result=0
    trap - EXIT
    # Close both pipe writers before waiting, so tee can drain and receive EOF.
    exec 1>&"${LOG_STDOUT_FD}" 2>&"${LOG_STDERR_FD}"
    exec {LOG_STDOUT_FD}>&- {LOG_STDERR_FD}>&-
    wait "${LOG_TEE_PID}" || log_result=$?
    if [[ ${result} -eq 0 && ${log_result} -ne 0 ]]; then
        result=${log_result}
    fi
    printf 'exit=%s\n' "${result}" >> "${REPORT_DIR}/status-${MODE}.txt"
    exit "${result}"
}
trap _finish_test_run EXIT
: > "${REPORT_DIR}/status-${MODE}.txt"

# Remove only reports owned by the selected suites, before setup can fail.
for suite in "${SUITES[@]}"; do
    rm -f -- "${REPORT_DIR}/${suite}.xml" "${REPORT_DIR}/${suite}.log"
done

# CI already sources CANN; manual runs can supply an environment-specific path.
if [[ -n "${CANN_ENV_PATH:-}" ]]; then
    # CANN environment scripts may read unset optional variables.
    set +u
    source "${CANN_ENV_PATH}"
    set -u
fi
# Load torch_npu explicitly where needed instead of through PyTorch autoload.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Temporary bootstrap while the CI images lack the experiment's pinned runtime
# and CPU-test dependencies. --no-deps preserves the image-owned torch/CANN stack.
# Read the TorchAO requirement from package metadata to avoid a second version pin.
TORCHAO_REQUIREMENT="$(
    "${PYTHON_BIN}" - "${EXPERIMENT_ROOT}" <<'PY'
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(sys.argv[1]).resolve()
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
matches = []
for dependency in project["dependencies"]:
    requirement = Requirement(dependency)
    if canonicalize_name(requirement.name) == "torchao":
        matches.append(requirement)
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one torchao dependency, found {matches}")
print(matches[0])
PY
)"
readonly TORCHAO_REQUIREMENT
"${PYTHON_BIN}" -m pip install --no-deps "${TORCHAO_REQUIREMENT}"
readonly SPMD_TYPES_REQUIREMENT="spmd_types==0.2.3"
if [[ "${MODE}" == "cpu" ]]; then
    # TorchTitan v0.3.0 pins this DeviceMesh-aware API, but its editable CI
    # install uses --no-deps and the current image carries an older release.
    "${PYTHON_BIN}" -m pip install --no-deps "${SPMD_TYPES_REQUIREMENT}"
fi
"${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation -e "${EXPERIMENT_ROOT}"
# Fail before collection if dependencies are incompatible or another installed
# torchao_npu package would be tested instead of this checkout.
"${PYTHON_BIN}" - "${EXPERIMENT_ROOT}" "${MODE}" "${SPMD_TYPES_REQUIREMENT}" <<'PY'
import importlib.metadata
import importlib.util
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

root = Path(sys.argv[1]).resolve()
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
requirements = [Requirement(dependency) for dependency in project["dependencies"]]
if sys.argv[2] == "cpu":
    requirements.append(Requirement(sys.argv[3]))
for requirement in requirements:
    installed = importlib.metadata.version(requirement.name)
    if installed not in requirement.specifier:
        raise RuntimeError(f"Expected {requirement}, found {installed}; check the test environment setup.")
spec = importlib.util.find_spec("torchao_npu")
expected = root / "torchao_npu" / "__init__.py"
if spec is None or Path(spec.origin).resolve() != expected:
    raise RuntimeError(f"Expected torchao_npu from {expected}, found {spec}")
print(f"Testing checkout: {spec.origin}", flush=True)
versions = ", ".join(
    f"{requirement.name}={importlib.metadata.version(requirement.name)}" for requirement in requirements
)
print(
    f"Test dependencies: {versions}",
    flush=True,
)

# Device modes require a usable NPU: test allocation, execution and synchronization
# in addition to availability. CPU mode skips this hardware probe.
if sys.argv[2] != "cpu":
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available; the NPU test environment must have a working device.")
    torch.ones(1, device="npu").sum().item()
    torch.npu.synchronize()
    print(f"NPU ready: torch={torch.__version__}, torch_npu={torch_npu.__version__}", flush=True)
PY

# Run each selected suite in its own pytest process. In npu mode, attempt both
# device and integration suites even if the first pytest process fails.
result=0
for suite in "${SUITES[@]}"; do
    suite_result=0
    # Anchor importlib names above both test trees. Explicit config and paths
    # make invocation from the repository or the experiment directory identical.
    PYTHONPATH="${EXPERIMENT_ROOT}/tests:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m pytest \
        -c "${EXPERIMENT_ROOT}/pyproject.toml" --rootdir="${PROJECT_ROOT}" \
        -v --tb=short "${EXPERIMENT_ROOT}/tests/${suite}" \
        --junitxml="${REPORT_DIR}/${suite}.xml" "$@" \
        2>&1 | tee "${REPORT_DIR}/${suite}.log" || suite_result=$?
    printf '%s=%s\n' "${suite}" "${suite_result}" >> "${REPORT_DIR}/status-${MODE}.txt"
    if [[ ${suite_result} -ne 0 ]]; then
        result=1
    fi
done
exit "${result}"
