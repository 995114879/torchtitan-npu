# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Four-NPU single emergency save and numerical cases; launches owned test jobs only."""

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.deepseek_v4 import GOLDEN_OVERRIDES, npu_overrides

_SERVICE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "torchtitan_npu/extensions/torchtitan/experiments/torchft/last_words/service.py"
    )
)


def start_child(command, **kwargs):
    return subprocess.Popen(command, start_new_session=True, **kwargs)


def stop_children(children):
    # The test runner acts as an external job manager; production never kills peers.
    for child in children:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


SCENARIOS = ("emergency_save", "numerical")
# Pre-fix reference for this regression and its LF-normalized Python source manifest.
BASELINE_COMMIT = "324ecaf27c70ed8c3cbe7944ed40ae584c5e74d3"
BASELINE_SOURCE_SHA256 = "faf10966e5cb810be9c51b162a4ed424cd92ee20751e35fccc696a452c9f712a"


def build_last_words_test_list():
    return [
        OverrideDefinitions(
            test_name=f"dsv4_torchft_last_words_{scenario}_2x2",
            ngpu=4,
            check_loss=False,
            required_env=("LAST_WORDS_BASELINE_REPO",) if scenario == "numerical" else (),
            test_descr=(
                "TorchFT single-writer emergency save without automatic relaunch"
                if scenario == "emergency_save"
                else "TorchFT shared-initial-state baseline/off/on numerical equality"
            ),
            train_script="tests/integration_tests/run_last_words.sh",
            train_args=(),
            override_args=[(f"--scenario={scenario}",)],
            timeout=7200 if os.environ.get("LAST_WORDS_COMPILE_BACKEND") else 2400,
        )
        for scenario in SCENARIOS
    ]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def verify_fault_injection(folder, phase, scenario, fault_step):
    path = folder / "evidence/injected-replica-1-rank-0.json"
    value = json.loads(path.read_text())
    if value["point"] != "boundary_process_exit" or value["step"] != fault_step:
        raise AssertionError(f"Wrong fault point or generation: {phase}/{scenario}, {value}")
    injections = [value]
    # Preserve each generation before the next phase overwrites per-rank files.
    (folder / "evidence" / f"injections-{phase}.json").write_text(json.dumps(injections))
    return injections


def run_phase(
    folder,
    *,
    phase,
    scenario,
    devices,
    fault_step=3,
    source=None,
    steps=6,
    interval=100,
    seq_len=128,
    recipe="reference",
    resume_path=None,
):
    recipe = os.environ.get("LAST_WORDS_RECIPE", recipe)
    if os.environ.get("LAST_WORDS_COMPILE_BACKEND"):
        recipe = os.environ.get("LAST_WORDS_COMPILE_RECIPE", recipe)
        seq_len = int(os.environ.get("LAST_WORDS_COMPILE_SEQ_LEN", seq_len))
    if recipe not in ("reference", "npu"):
        raise ValueError("Unknown last-words operator recipe")
    model_profile = os.environ.get("LAST_WORDS_MODEL_PROFILE", "")
    if model_profile not in ("", "small-v1"):
        raise ValueError("Unknown last-words model profile")
    keep_latest_k, required_free_bytes = 2, 96 * 1024**3
    model_module = (
        "tests.integration_tests.last_words_small_config" if model_profile else "torchtitan_npu.models.deepseek_v4"
    )
    model_config = "deepseek_v4_small_torchft" if model_profile else "deepseek_v4_flash_8p_torchft"
    run_id = f"st-{phase}-{uuid.uuid4().hex}"
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    lighthouse_log = (folder / f"lighthouse-{phase}.log").open("w")
    is_fault = phase == "fault"
    enabled = is_fault or phase in ("current_on", "resume")
    if os.environ.get("LAST_WORDS_COORDINATION_BACKEND", "service") != "service":
        raise ValueError("Only the service coordination backend is supported")
    lighthouse_command = ["torchft_lighthouse", "--bind", f"127.0.0.1:{port}", "--min_replicas", "2"]
    lighthouse = start_child(lighthouse_command, stdout=lighthouse_log, stderr=subprocess.STDOUT)
    children, logs = [], []
    service = None
    service_thread = None
    session_dir = folder / f"service-{phase}"
    train_script = Path(__file__).with_name("last_words_train_probe.py").resolve()
    started = time.monotonic()
    try:
        if shutil.disk_usage(folder).free < required_free_bytes:
            raise OSError("Insufficient checkpoint reserve for the NPU case")
        if enabled:
            service = _SERVICE["ServiceServer"](("127.0.0.1", 0))
            _SERVICE["prepare_session"](
                service,
                session_dir,
                address=f"http://127.0.0.1:{service.server_port}",
                run_id=run_id,
                num_replicas=2,
                world_size=2,
                heartbeat_seconds=30,
                startup_seconds=900,
                timeout_seconds=180,
            )
            service_thread = threading.Thread(target=service.serve_forever, daemon=True)
            service_thread.start()
        time.sleep(1)
        if lighthouse is not None and lighthouse.poll() is not None:
            raise RuntimeError("TorchFT Lighthouse failed to start")
        for replica in range(2):
            device_start, device_end = replica * 2, replica * 2 + 2
            env = dict(
                os.environ,
                TORCHFT_LIGHTHOUSE=f"http://127.0.0.1:{port}",
                ASCEND_RT_VISIBLE_DEVICES=",".join(devices[device_start:device_end]),
                LAST_WORDS_TEST_PHASE=phase,
                LAST_WORDS_TEST_REPLICA=str(replica),
                LAST_WORDS_TEST_SCENARIO=scenario,
                LAST_WORDS_TEST_FAULT_STEP=str(fault_step),
                LAST_WORDS_TEST_EVIDENCE=str(folder / "evidence"),
                OMP_NUM_THREADS="2",
                CPU_AFFINITY_CONF="0",
            )
            # The replicas are independent torchrun jobs on the same host.
            # Reserve a 16-device HCCL port block for each job.
            base_port = int(env.get("HCCL_IF_BASE_PORT", "61000")) + 16 * replica
            if not 1024 <= base_port <= 65535 - 15:
                raise ValueError("HCCL_IF_BASE_PORT must allow two consecutive 16-port blocks")
            env["HCCL_IF_BASE_PORT"] = str(base_port)
            if os.environ.get("LAST_WORDS_COMPILE_BACKEND") and model_profile:
                for key in ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_NPU_EXT_CACHE_DIR"):
                    env[key] = str(folder / "compile-cache" / run_id / key.lower())
            if enabled:
                env["LAST_WORDS_SESSION_FILE"] = str(session_dir / f"replica-{replica}.json")
            if source is not None:
                # Replace this repository while retaining externally supplied dependencies.
                current_root = Path(__file__).resolve().parents[2]
                dependencies = [
                    str(Path(path).resolve())
                    for path in env.get("PYTHONPATH", "").split(os.pathsep)
                    if path and Path(path).resolve() != current_root
                ]
                env["PYTHONPATH"] = os.pathsep.join([str(source.resolve()), *dependencies])
            log = (folder / f"{phase}-replica-{replica}.log").open("w")
            logs.append(log)
            compile_backend = os.environ.get("LAST_WORDS_COMPILE_BACKEND")
            worker = [str(train_script)]
            if compile_backend and env.get("LAST_WORDS_MODEL_PROFILE") == "small-v1":
                worker = [
                    "--no-python",
                    "bash",
                    str(train_script.with_name("run_last_words.sh")),
                    "--compile-worker",
                    sys.executable,
                    str(train_script),
                ]
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--rdzv-backend=c10d",
                "--rdzv-endpoint=localhost:0",
                "--max-restarts=0",
                *worker,
                f"--module={model_module}",
                f"--config={model_config}",
                f"--dump-folder={folder}",
                "--parallelism.data-parallel-shard-degree=2",
                "--parallelism.expert-parallel-degree=2",
                "--training.global-batch-size=8",
                f"--training.steps={steps}",
                f"--training.seq-len={seq_len}",
                "--metrics.log-freq=1",
                "--training.disable-cuda-graphs",
                "--dataloader.dataset-path=tests/assets/c4_test",
                "--hf-assets-path=tests/assets/deepseek_v3",
                f"--checkpoint.interval={interval}",
                "--checkpoint.no-last-save-model-only",
                f"--checkpoint.keep-latest-k={keep_latest_k}",
                f"--fault-tolerance.replica-id={replica}",
                *(GOLDEN_OVERRIDES if recipe == "reference" else npu_overrides(debug_model=bool(model_profile))),
            ]
            if compile_backend:
                compile_components = os.environ.get("LAST_WORDS_COMPILE_COMPONENTS", "model,loss")
                if compile_components not in ("model", "model,loss"):
                    raise ValueError("Invalid compile validation components")
                command += [
                    "--compile.enable",
                    f"--compile.components={compile_components}",
                    f"--compile.backend={compile_backend}",
                ]
            if resume_path is not None:
                command.append(f"--fault-tolerance.last-words.resume-path={resume_path}")
            if enabled:
                command += [
                    "--fault-tolerance.last-words.enable",
                    "--fault-tolerance.no-enable-elastic",
                    f"--fault-tolerance.last-words.run-id={run_id}",
                    "--fault-tolerance.last-words.heartbeat-timeout-seconds=30",
                    "--fault-tolerance.last-words.timeout-seconds=180",
                ]
            if scenario == "numerical":
                command += ["--debug.deterministic", "--debug.seed=42"]
            if scenario == "numerical" and phase != "init":
                env["LAST_WORDS_VERIFY_INPUTS"] = "1"
                command.append("--checkpoint.load-step=0")
            if enabled or scenario == "numerical":
                # Full state healing and cold NPU initialization can leave peers
                # more than the default 10s apart at their first HCCL ID lookup.
                command.append("--fault-tolerance.process-group-timeout-ms=60000")
            if compile_backend:
                command += [
                    "--fault-tolerance.process-group-timeout-ms=300000",
                    "--fault-tolerance.quorum-warmup-timeout-ms=600000",
                    "--fault-tolerance.quorum-timeout-seconds=300",
                    "--comm.init-timeout-seconds=900",
                    "--comm.train-timeout-seconds=600",
                ]
            children.append(start_child(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + max(3000 if os.environ.get("LAST_WORDS_COMPILE_BACKEND") else 660, steps * 30)
        candidate_root = folder / "checkpoint/last_words" / run_id
        while any(child.poll() is None for child in children):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Last-words {phase}/{scenario} phase timed out")
            if not is_fault and any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f"A replica failed during {phase}/{scenario}")
            time.sleep(0.5)
        if not is_fault:
            if any(child.returncode != 0 for child in children):
                raise RuntimeError(f"{phase} training did not finish")
            if enabled:
                result = json.loads((session_dir / "result.json").read_text())
                if not result["success"] or result["round"] != 0 or result["replica_id"] is not None:
                    raise AssertionError("Normal training requires explicit completion from every rank")
            return None
        verify_fault_injection(folder, phase, scenario, fault_step)
        result = json.loads((session_dir / "result.json").read_text())
        grant = json.loads((session_dir / "grant.json").read_text())
        if result["success"] is not True or any(
            record["run_id"] != run_id or record["replica_id"] != 0 or record["round"] != 1
            for record in (grant, result)
        ):
            raise AssertionError("Fault in replica 1 requires replica 0 to succeed as the only round-1 writer")
        candidates = list(candidate_root.glob("replica-*/manifest.json"))
        if len(candidates) != 1:
            raise AssertionError(f"Single-writer fault must publish exactly one complete candidate: {len(candidates)}")
        selected = candidates[0]
        # torchrun succeeds only when every worker exits with code zero.
        if children[0].returncode != 0:
            raise AssertionError("The healthy writer did not exit cleanly")
        writer_log = (folder / "fault-replica-0.log").read_text()
        for rank in (0, 1):
            if f"Writer rank {rank} completed last-words save at {selected.parent}" not in writer_log:
                raise AssertionError(f"Writer rank {rank} did not complete the final save barrier")
        saved = json.loads(selected.read_text())
        receipt = json.loads((selected.parent / "service-result.json").read_text())
        if (
            selected.parent.name != "replica-0"
            or saved["replica_id"] != 0
            or saved["run_id"] != run_id
            or any(
                receipt[key] != result[key]
                for key in ("run_id", "round", "replica_id", "coordinator_instance", "success")
            )
            or grant["coordinator_instance"] != result["coordinator_instance"]
        ):
            raise AssertionError("Emergency checkpoint writer or service identity differs from the grant and result")
        if saved["state"]["step"] != fault_step - 1:
            raise AssertionError("Checkpoint must preserve the last completed update before injection")
        return selected
    finally:
        if service_thread is not None and service_thread.is_alive():
            service.shutdown()
            service_thread.join(timeout=3)
        if service is not None:
            service.server_close()
        stop_children([child for child in [*children, lighthouse] if child.poll() is None])
        lighthouse_log.close()
        for log in logs:
            log.close()
        (folder / f"timing-{phase}.json").write_text(json.dumps({"seconds": time.monotonic() - started}))


def verified_baseline(folder):
    baseline = os.environ.get("LAST_WORDS_BASELINE_REPO")
    if not baseline or not (Path(baseline) / "torchtitan_npu").is_dir():
        raise ValueError(
            "Numerical case requires LAST_WORDS_BASELINE_REPO containing the original full source snapshot"
        )
    baseline_root = Path(baseline).resolve()
    if (baseline_root / ".git").exists():
        baseline_commit = subprocess.check_output(["git", "-C", baseline, "rev-parse", "HEAD"], text=True).strip()
        if baseline_commit != BASELINE_COMMIT:
            raise ValueError(f"Expected numerical baseline {BASELINE_COMMIT}, found {baseline_commit}")
        if subprocess.check_output(["git", "-C", baseline, "status", "--porcelain", "--untracked-files=no"], text=True):
            raise ValueError("Numerical baseline requires a clean tracked source checkout")
    hashes = {
        path.relative_to(baseline_root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in (baseline_root / "torchtitan_npu").rglob("*.py")
    }
    if hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() != BASELINE_SOURCE_SHA256:
        raise ValueError(f"Numerical baseline Python sources differ from {BASELINE_COMMIT}")
    evidence = {"repository": str(baseline_root), "commit": BASELINE_COMMIT, "python_sha256": hashes}
    (folder / "baseline-source.json").write_text(json.dumps(evidence, sort_keys=True))
    return baseline_root


def numerical_inputs(evidence, steps):
    # Global batch 8 across four ranks with local batch 1 gives two microbatches per step.
    expected = {
        f"input-step-{step}-microbatch-{microbatch}-replica-{replica}-rank-{rank}.json"
        for step in range(1, steps + 1)
        for microbatch in range(2)
        for replica in range(2)
        for rank in range(2)
    }
    found = {path.name for path in evidence.glob("input-step-*.json")}
    if found != expected:
        raise AssertionError(
            f"Input evidence in {evidence}: missing {sorted(expected - found)}, extra {sorted(found - expected)}"
        )
    inputs = {}
    for name in sorted(expected):
        digest = json.loads((evidence / name).read_text())["batch"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"Invalid batch SHA-256: {evidence / name}")
        inputs[name] = digest
    return inputs


def numerical_case(folder, devices):
    baseline_root = verified_baseline(folder)
    init = folder / "initial"
    run_phase(init, phase="init", scenario="numerical", devices=devices)
    values = []
    baseline_inputs = None
    steps = 6
    for phase in ("baseline_off", "current_off", "current_on"):
        target = folder / phase
        target.mkdir()
        shutil.copytree(init / "checkpoint", target / "checkpoint")
        run_phase(
            target,
            phase=phase,
            scenario="numerical",
            devices=devices,
            source=baseline_root if phase == "baseline_off" else None,
            steps=steps,
        )
        inputs = numerical_inputs(target / "evidence", steps)
        if baseline_inputs is None:
            baseline_inputs = inputs
        else:
            for name, digest in inputs.items():
                if digest != baseline_inputs[name]:
                    raise AssertionError(f"Training input differs from baseline_off: {phase}/{name}")
        initial = list((target / "evidence").glob(f"initial-loaded-{phase}-*.json"))
        if len(initial) != 4:
            raise AssertionError(f"Missing complete initial-state load evidence: {phase}")
        for replica in range(2):
            for rank in range(2):
                expected = {
                    "model": json.loads((init / "evidence" / f"initial-saved-replica-0-rank-{rank}.json").read_text())[
                        "model"
                    ],
                    "cursor": json.loads(
                        (init / "evidence" / f"initial-saved-replica-{replica}-rank-{rank}.json").read_text()
                    )["cursor"],
                    "rng": json.loads(
                        (init / "evidence" / f"initial-saved-replica-{replica}-rank-{rank}.json").read_text()
                    )["rng"],
                }
                actual = json.loads(
                    (target / "evidence" / f"initial-loaded-{phase}-replica-{replica}-rank-{rank}.json").read_text()
                )
                if actual != expected:
                    raise AssertionError(
                        "Complete initial model/optimizer/scheduler/progress, own cursor or RNG differs: "
                        f"{phase}, replica={replica}, rank={rank}"
                    )
        metrics = {}
        sources = list((target / "evidence").glob(f"source-{phase}-*.json"))
        expected_root = baseline_root if phase == "baseline_off" else Path(__file__).resolve().parents[2]
        feature = expected_root / "torchtitan_npu/extensions/torchtitan/experiments/torchft"
        checkpoint_dir = feature if phase == "baseline_off" else feature / "last_words"
        expected_sources = {"trainer": feature / "trainer.py", "checkpoint": checkpoint_dir / "checkpoint.py"}
        if len(sources) != 4:
            raise AssertionError(f"Missing production source evidence: {phase}")
        for path in sources:
            sources_loaded = {name: Path(source).resolve() for name, source in json.loads(path.read_text()).items()}
            if sources_loaded != expected_sources:
                raise AssertionError(f"Wrong production source loaded in {phase}: {path.read_text()}")
        for path in (target / "evidence").glob(f"metrics-{phase}-*.json"):
            value = json.loads(path.read_text())
            if not all(math.isfinite(item) for item in value.values()):
                raise AssertionError("Non-finite full-precision training metric")
            metrics[path.name.replace(phase, "phase")] = {key: item.hex() for key, item in value.items()}
        if len(metrics) != steps * 4:
            raise AssertionError(f"Missing full-precision per-rank metrics: {phase} has {len(metrics)} entries")
        values.append(metrics)
    if values[0] != values[1] or values[0] != values[2]:
        raise AssertionError("Original off, current off and current on loss/gradient norm are not bitwise equal")
    print("Verified full-precision loss/gradient norm equality from one initial checkpoint in all three phases")


def verify_resume(folder, manifest_path, fault_folder, steps=6):
    manifest = json.loads(manifest_path.read_text())
    saved_step = manifest["state"]["step"]
    writer = manifest["replica_id"]
    for replica in range(2):
        cursor_step = manifest["cursors"][str(replica)]["state"]["step"]
        for rank in range(2):
            suffix = f"replica-{replica}-rank-{rank}.json"
            loaded = json.loads((folder / "evidence" / f"loaded-resume-{suffix}").read_text())
            model = json.loads(
                (
                    fault_folder / "evidence" / f"completed-step-{saved_step}-replica-{writer}-rank-{rank}.json"
                ).read_text()
            )
            cursor = json.loads((fault_folder / "evidence" / f"completed-step-{cursor_step}-{suffix}").read_text())
            if loaded["model"] != model["model"] or loaded["cursor"] != cursor["cursor"]:
                raise AssertionError(
                    f"Resume did not restore complete model/optimizer/progress or own cursor: {suffix}"
                )
            if loaded["rng"] != cursor["rng"] or loaded["live_rng"] != cursor["rng"]:
                raise AssertionError(f"Resume did not restore own RNG: {suffix}")
            if loaded["progress"] != manifest["state"] or loaded["next_step"] != saved_step + 1:
                raise AssertionError(f"Resume started at the wrong completed update: {suffix}")
            for step in range(saved_step + 1, steps + 1):
                value = json.loads((folder / "evidence" / f"metrics-resume-step-{step}-{suffix}").read_text())
                if not all(math.isfinite(item) for item in value.values()):
                    raise AssertionError(f"Non-finite resumed metric: step={step}, {suffix}")
            completed = json.loads((folder / "evidence" / f"completed-step-{steps}-{suffix}").read_text())
            expected_updates = manifest["state"]["last_words_num_updates"] + steps - saved_step
            if (
                completed["progress"]["step"] != steps
                or completed["progress"]["last_words_num_updates"] != expected_updates
            ):
                raise AssertionError(f"Resume did not commit every requested update: {suffix}")
    (folder / "resume-verified.json").write_text(
        json.dumps(dict(passed=True, checkpoint=str(manifest_path), saved_step=saved_step, final_step=steps, ranks=4))
    )


def main():
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_folder", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="emergency_save")
    args = parser.parse_args()
    folder = Path(args.dump_folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(devices) != 4:
        raise ValueError("The last-words integration case requires exactly four assigned NPUs")
    if args.scenario == "numerical":
        numerical_case(folder, devices)
        return
    manifest = run_phase(folder, phase="fault", scenario=args.scenario, devices=devices)
    resumed = folder / "resume"
    run_phase(resumed, phase="resume", scenario=args.scenario, devices=devices, resume_path=manifest.parent)
    verify_resume(resumed, manifest, folder)
    print(
        f"Verified {args.scenario}: one 2x2 NPU candidate at {manifest}, externally loaded and trained through step 6"
    )


if __name__ == "__main__":
    main()
