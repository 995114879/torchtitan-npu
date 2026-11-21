# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract checks for the model repository and Lite Actions."""

import io
import json
import sys

import pytest

from tests.integration_tests.tools.lite_actions import entrypoint, runner
from tests.integration_tests.training_cases import build_test_list, dispatch_smoke

pytestmark = pytest.mark.tooling

SHA = "a" * 40
DISPATCH = "dispatch_123_1767225600000"
NAMES = ["worker_os_blue", "worker_os_yellow", "dispatch_smoke_8p", "dispatch_smoke_16p", "dispatch_smoke_8p"]


def _result_comments():
    return [
        {
            "user": {"login": "reporter"},
            "body": f"Lite-CI-V3 run123:1:{DISPATCH}:case_{order:03d}_{name}\n"
            + json.dumps(
                {
                    "run_id": "run123",
                    "attempt": 1,
                    "sha": SHA,
                    "dispatch_id": DISPATCH,
                    "case_id": f"case_{order:03d}_{name}",
                    "test_id": name,
                    "worker": "yellow" if name == "worker_os_yellow" else "blue",
                    "status": "PASS",
                    "exit_code": 0,
                    "cleanup_verified": True,
                    "last_20_lines": [],
                }
            ),
        }
        for order, name in enumerate(NAMES)
    ]


def _configure_waiter(monkeypatch):
    for name, value in {
        "ATOMGIT_REPOSITORY": "cann/torchtitan-npu",
        "ATOMGIT_RUN_ID": "run123",
        "ATOMGIT_RUN_ATTEMPT": "1",
        "ATOMGIT_SHA": SHA,
        "CI_RESULT_AUTHOR": "reporter",
        "CI_WAIT_TIMEOUT_SECONDS": "60",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["entrypoint", "wait-gitcode"])
    monkeypatch.setattr(entrypoint.time, "sleep", lambda _: None)


def _serve(monkeypatch, pages):
    pages = iter(pages)

    def urlopen(request, **_kwargs):
        if "/actions/runs/" in request.full_url:
            job = {"name": "Dispatcher Smoke", "last_dispatch_id": "dispatch_123", "start_time": 1767225600000}
            data = {"stages": [{"jobs": [job]}]}
            return io.BytesIO(json.dumps(data).encode())
        return io.BytesIO(next(pages))

    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", urlopen)


# 看护场景 1：模型仓冒烟与训练用例不重名，默认 profile 的重复 8P 具有独立实例并由入口正确解析。
def test_smoke_catalog_and_profile_contract():
    definitions = dispatch_smoke.resolve_smoke_profile()
    assert [case.test_name for case in definitions] == NAMES
    yellow = definitions[1]
    assert (yellow.ci_kind, yellow.nnodes, yellow.ngpu, yellow.ci_overrides) == (
        "worker_probe",
        0,
        0,
        {"worker": "yellow"},
    )
    assert not {case.test_name for case in dispatch_smoke.build_smoke_list()} & {
        case.test_name for case in build_test_list()
    }
    selected = [case.test_name for case in entrypoint.select(entrypoint.catalog(), suite="dispatch_smoke")]
    assert selected == NAMES
    assert list(entrypoint.expected_cases("dispatch_smoke")) == [
        f"case_{order:03d}_{name}" for order, name in enumerate(NAMES)
    ]


# 看护场景 1b：匿名读覆盖公开仓，私有 fork 必须用运行 token；两者不能互相替代。
def test_read_headers_use_run_token_only_when_present(monkeypatch):
    monkeypatch.delenv("CI_GITCODE_TOKEN", raising=False)
    assert "PRIVATE-TOKEN" not in entrypoint.read_headers()
    monkeypatch.setenv("CI_GITCODE_TOKEN", "run-token")
    headers = entrypoint.read_headers()
    assert headers["PRIVATE-TOKEN"] == "run-token"
    assert headers["Accept"] == "application/json"


# 看护场景 2：可信报告人为当前运行回传全部成功结果时，等待入口在短暂非 JSON 响应后仍能成功汇总。
def test_waiter_passes_when_all_expected_cases_succeed(monkeypatch):
    _configure_waiter(monkeypatch)
    monkeypatch.setenv("ATOMGIT_SHA", "b" * 40)
    monkeypatch.setenv("CI_SOURCE_SHA", SHA)
    _serve(monkeypatch, [b"<html>502</html>", json.dumps(_result_comments()).encode()])
    assert entrypoint.main() == 0


# GitCode runners may export ATOMGIT_RUN_ATTEMPT as an empty string.
def test_waiter_defaults_empty_run_attempt_to_one(monkeypatch, capsys):
    _configure_waiter(monkeypatch)
    monkeypatch.setenv("ATOMGIT_RUN_ATTEMPT", "")
    monkeypatch.setenv("CI_WAIT_TIMEOUT_SECONDS", "0")
    _serve(monkeypatch, [])
    assert entrypoint.main() == 1
    output = capsys.readouterr().out
    assert "Timed out waiting for Lite Actions smoke results" in output
    assert "invalid literal for int()" not in output


# 看护场景 3：忽略其他作者及提交的评论；失败、清理未确认、未知状态或缺例超时均不能成功。
def test_waiter_ignores_untrusted_comments_and_fails_on_case_failure_or_timeout(monkeypatch):
    _configure_waiter(monkeypatch)
    comments = _result_comments()
    other = dict(comments[0], user={"login": "someone_else"}, body=f"Lite-CI-V3 run123:1:{DISPATCH}:unexpected\n{{}}")
    marker, payload = comments[0]["body"].split("\n", 1)
    old = dict(comments[0], body=marker + "\n" + json.dumps({**json.loads(payload), "sha": "b" * 40}))
    previous_dispatch = dict(
        comments[0],
        body=marker.replace(DISPATCH, "previous_dispatch")
        + "\n"
        + json.dumps({**json.loads(payload), "dispatch_id": "previous_dispatch", "status": "FAIL"}),
    )

    def send(rows):
        _serve(monkeypatch, [json.dumps(rows).encode()])

    send([other, old, previous_dispatch, *comments])
    assert entrypoint.main() == 0
    # A Blue result cannot be accepted for the Yellow instance.
    bad = _result_comments()
    marker, payload = bad[1]["body"].split("\n", 1)
    bad[1]["body"] = marker + "\n" + json.dumps({**json.loads(payload), "worker": "blue"})
    send(bad)
    assert entrypoint.main() == 1
    for change in ({"status": "FAIL", "exit_code": 1}, {"cleanup_verified": False}, {"status": "UNKNOWN"}):
        broken = _result_comments()
        marker, payload = broken[-1]["body"].split("\n", 1)
        broken[-1]["body"] = marker + "\n" + json.dumps({**json.loads(payload), **change})
        send(broken)
        assert entrypoint.main() == 1
    monkeypatch.setenv("CI_WAIT_TIMEOUT_SECONDS", "0")
    send(comments[:-1])
    assert entrypoint.main() == 1


# 看护场景 4：双节点 16P 请求把节点编号、地址和端口租约传给 torchrun。
def test_distributed_runner_launches_torchrun_with_cluster_env(tmp_path, monkeypatch):
    case = next(case for case in dispatch_smoke.build_smoke_list() if case.test_name == "dispatch_smoke_16p")
    environment = {
        "NNODES": "2",
        "NGPU": "8",
        "NODE_RANK": "1",
        "NODE_IPS": "10.0.0.1,10.0.0.2",
        "MASTER_PORT": "16000",
        "HCCL_IF_BASE_PORT": "20000",
        "HCCL_HOST_SOCKET_PORT_RANGE": "24000-24063",
        "HCCL_NPU_SOCKET_PORT_RANGE": "28000-28063",
        "LITE_CI_JOB_ID": "worker-job",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    runner.run_distributed(case, phase="launch", output_dir=tmp_path / "smoke")
    command, options = calls[0]
    assert command[-2:] == ["-m", "tests.integration_tests.training_cases.dispatch_smoke"]
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("--node-rank") + 1] == "1"
    assert command[command.index("--master-addr") + 1] == "10.0.0.1"
    assert command[command.index("--master-port") + 1] == "16000"
    assert {key: options["env"][key] for key in environment if key.startswith("HCCL_")} == {
        key: value for key, value in environment.items() if key.startswith("HCCL_")
    }
