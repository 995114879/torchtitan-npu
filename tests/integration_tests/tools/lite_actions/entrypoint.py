#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lite Actions adapter: trusted case definitions and fixed SHA selectors."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from tests.integration_tests import OverrideDefinitions  # noqa: TC001 - retain runtime annotation resolution
from tests.integration_tests.tools.lite_actions.suites import case_ids
from tests.integration_tests.training_cases import build_test_list
from tests.integration_tests.training_cases.dispatch_smoke import (
    SMOKE_DEFAULTS,
    SMOKE_PROFILES,
    build_smoke_list,
    resolve_smoke_profile,
)

SAFE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")


def catalog() -> dict[str, OverrideDefinitions]:
    found = {}
    for test in [*build_test_list(), *build_smoke_list()]:
        if not SAFE.fullmatch(test.test_name) or test.test_name in found or test.disabled:
            raise ValueError("duplicate/invalid/disabled CI case")
        if test.ci_kind == "worker_probe":
            valid_topology = test.nnodes == test.ngpu == 0
        else:
            valid_topology = (
                isinstance(test.nnodes, int) and test.nnodes >= 1 and isinstance(test.ngpu, int) and test.ngpu >= 1
            )
        if not valid_topology:
            raise ValueError("invalid topology")
        found[test.test_name] = test
    if not found:
        raise ValueError("empty test catalog")
    return found


def select(catalogue: dict, *, test_id: str | None = None, suite: str | None = None):
    if (test_id is None) == (suite is None):
        raise ValueError("select exactly one ID or suite")
    if suite is not None:
        if suite in SMOKE_PROFILES:
            return [select(catalogue, test_id=name)[0] for name in SMOKE_PROFILES[suite]]
        return [select(catalogue, test_id=name)[0] for name in case_ids(suite)]
    if test_id not in catalogue:
        raise ValueError("unknown CI test")
    return [catalogue[test_id]]


def inspect_overrides(test: OverrideDefinitions) -> dict:
    if test.ci_kind == "worker_probe":
        return {"worker": SMOKE_DEFAULTS["worker"], **test.ci_overrides}
    if test.ci_kind == "npu_smoke":
        return {**SMOKE_DEFAULTS, **test.ci_overrides}
    return dict(test.ci_overrides)


def read_headers():
    """Anonymous reads cover public repositories; private forks need the run token."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "lite-actions-gitcode-waiter",
    }
    token = os.environ.get("CI_GITCODE_TOKEN")
    if token:
        headers["PRIVATE-TOKEN"] = token
    return headers


def read_comments(repo, sha):
    comments = []
    for page in range(1, 21):
        url = f"https://api.gitcode.com/api/v5/repos/{repo}/commits/{sha}/comments?per_page=100&page={page}"
        request = urllib.request.Request(url, headers=read_headers())
        with urllib.request.urlopen(request, timeout=25) as response:
            rows = json.load(response)
        if not isinstance(rows, list):
            raise CommentReadError("invalid comment response")
        comments.extend(rows)
        if len(rows) < 100:
            return comments
    raise CommentReadError("too many result comments")


class CommentReadError(Exception):
    """The platform response could not be read; retry on the next poll."""


def read_dispatch_id(repo, run_id):
    request = urllib.request.Request(
        f"https://api.gitcode.com/api/v8/repos/{repo}/actions/runs/{run_id}",
        headers=read_headers(),
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        data = json.load(response)
    jobs = (
        [
            job
            for stage in data.get("stages", [])
            for job in stage.get("jobs", [])
            if job.get("name") == "Dispatcher Smoke"
        ]
        if isinstance(data, dict)
        else []
    )
    dispatch_id = jobs[0].get("last_dispatch_id") if len(jobs) == 1 else None
    started = jobs[0].get("start_time") if len(jobs) == 1 else None
    if not isinstance(dispatch_id, str) or type(started) is not int or started <= 0:
        raise ValueError("missing GitCode dispatch identity")
    identity = f"{dispatch_id}_{started}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
        raise ValueError("invalid GitCode dispatch identity")
    return identity


def expected_cases(profile):
    return {
        f"case_{order:03d}_{case.test_name}": case.test_name
        for order, case in enumerate(resolve_smoke_profile(profile))
    }


def catalog_results(comments, cases, *, author, run_id, attempt, sha, dispatch_id):
    results = {}
    prefix = f"Lite-CI-V3 {run_id}:{attempt}:{dispatch_id}:"
    for comment in comments:
        body = comment.get("body", "")
        if comment.get("user", {}).get("login") != author or not body.startswith(prefix):
            continue
        marker, separator, payload = body.partition("\n")
        case_id = marker.removeprefix(prefix)
        try:
            result = json.loads(payload) if separator else None
        except json.JSONDecodeError as exc:
            raise ValueError("invalid case result") from exc
        if not isinstance(result, dict):
            raise ValueError("invalid case result")
        if result.get("sha") != sha:
            continue
        if case_id not in cases or any(
            result.get(key) != expected
            for key, expected in (
                ("run_id", run_id),
                ("attempt", attempt),
                ("dispatch_id", dispatch_id),
                ("case_id", case_id),
                ("test_id", cases[case_id]),
            )
        ):
            raise ValueError("case result identity mismatch")
        if cases[case_id] in {"worker_os_blue", "worker_os_yellow"} and result.get("worker") != cases[
            case_id
        ].removeprefix("worker_os_"):
            raise ValueError("OS probe worker identity mismatch")
        status = result.get("status")
        if not isinstance(status, str) or status not in {
            "PASS",
            "FAIL",
            "INFRA_ERROR",
            "RESOURCE_TIMEOUT",
            "TIMED_OUT",
            "CANCELLED",
            "NOT_RUN",
        }:
            raise ValueError(f"unknown case status: {status}")
        if not isinstance(result.get("last_20_lines"), list) or any(
            not isinstance(line, str) for line in result["last_20_lines"]
        ):
            raise ValueError("invalid case diagnostics")
        core = (
            "run_id",
            "attempt",
            "sha",
            "dispatch_id",
            "case_id",
            "test_id",
            "status",
            "exit_code",
            "cleanup_verified",
            "last_20_lines",
        )
        if case_id in results and any(results[case_id].get(key) != result.get(key) for key in core):
            raise ValueError("conflicting case results")
        results[case_id] = result
    return results


def wait_catalog():
    repo = os.environ["ATOMGIT_REPOSITORY"]
    run_id = os.environ["ATOMGIT_RUN_ID"]
    attempt = int(os.environ.get("ATOMGIT_RUN_ATTEMPT") or "1")
    sha = os.environ.get("CI_SOURCE_SHA") or os.environ["ATOMGIT_SHA"]
    author = os.environ["CI_RESULT_AUTHOR"]
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id)
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
        or not author
    ):
        raise ValueError("invalid run identity or missing result author")
    dispatch_id = read_dispatch_id(repo, run_id)
    cases = expected_cases(os.environ.get("CI_SMOKE_PROFILE", "dispatch_smoke"))
    deadline = time.monotonic() + int(os.environ.get("CI_WAIT_TIMEOUT_SECONDS", "10800"))
    displayed = set()
    while time.monotonic() < deadline:
        try:
            comments = read_comments(repo, sha)
        except (OSError, json.JSONDecodeError, CommentReadError) as exc:
            print(f"GitCode read retry: {type(exc).__name__}", flush=True)
        else:
            try:
                results = catalog_results(
                    comments, cases, author=author, run_id=run_id, attempt=attempt, sha=sha, dispatch_id=dispatch_id
                )
            except ValueError as exc:
                print(f"Smoke result error: {exc}", flush=True)
                return 1
            for case_id, result in results.items():
                if case_id not in displayed:
                    print(f"CASE {case_id}: {result['status']} rc={result.get('exit_code')}", flush=True)
                    for line in result["last_20_lines"]:
                        print(line, flush=True)
                    displayed.add(case_id)
                if (
                    result["status"] != "PASS"
                    or type(result.get("exit_code")) is not int
                    or result["exit_code"] != 0
                    or result.get("cleanup_verified") is not True
                ):
                    return 1
            if len(results) == len(cases):
                return 0
        print("Waiting for Lite Actions smoke results.", flush=True)
        time.sleep(10)
    print("Timed out waiting for Lite Actions smoke results", flush=True)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inspect", "launch", "verify", "wait-gitcode"))
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--test-id")
    sel.add_argument("--suite")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    if a.phase == "wait-gitcode":
        if a.test_id is not None or a.suite is not None or a.output_dir is not None:
            p.error("wait-gitcode does not accept a test ID, suite or output directory")
        try:
            return wait_catalog()
        except (KeyError, OSError, ValueError) as exc:
            print(f"Smoke request error: {exc}", flush=True)
            return 1
    if a.test_id is None and a.suite is None:
        p.error("one of --test-id or --suite is required")
    if a.phase != "inspect" and (a.test_id is None or a.output_dir is None):
        p.error("launch/verify require an explicit test ID and output directory")
    try:
        selected = select(catalog(), test_id=a.test_id, suite=a.suite)
    except ValueError as exc:
        p.error(str(exc))
    if a.phase == "inspect":
        print(
            json.dumps(
                [
                    {
                        "test_id": t.test_name,
                        "ngpu": t.ngpu,
                        "nnodes": t.nnodes,
                        "env_vars": dict(t.env_vars or {}),
                        "kind": t.ci_kind,
                        "ci_overrides": inspect_overrides(t),
                        "label": t.test_descr,
                        "timeout": t.timeout,
                        "device": t.device,
                        "required_env": list(t.required_env),
                        "phases": len(t.override_args),
                    }
                    for t in selected
                ],
                separators=(",", ":"),
            )
        )
        return
    test = selected[0]
    if test.ci_kind == "worker_probe":
        p.error("worker probes execute only in the trusted Lite Actions worker")
    if test.ci_kind not in {"train", "npu_smoke"}:
        p.error(f"unknown CI case kind: {test.ci_kind}")
    if test.ci_kind == "npu_smoke" and a.phase == "verify":
        # The Worker requires every node's launch to exit successfully first.
        return
    # Also support direct, manual invocation of the fixed model-owned entrypoint.
    # Remote SSH has already sourced the case's toolkit before Python starts.
    if test.ci_kind == "npu_smoke":
        for name, value in (test.env_vars or {}).items():
            os.environ.setdefault(name, value)
    else:
        os.environ.update(test.env_vars or {})
    from tests.integration_tests.tools.lite_actions.runner import (
        run_distributed,
        run_single,
        verify,
    )

    if test.nnodes == 1:
        if a.phase == "launch":
            run_single(test, output_dir=a.output_dir)
        else:
            verify(test, output_dir=a.output_dir)
    else:
        run_distributed(test, phase=a.phase, output_dir=a.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
