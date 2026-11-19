#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep the Training workflow's Lite Actions selectors aligned with the case catalog."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.integration_tests.tools.lite_actions import entrypoint
from tests.integration_tests.tools.lite_actions.suites import case_ids

pytestmark = pytest.mark.tooling

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def load_workflow(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name}: workflow root must be a mapping"
    return document


def event_section(workflow: dict):
    """Return the ``on:`` mapping.

    PyYAML parses the bare key ``on`` as the boolean ``True``; accept either
    spelling so the check does not depend on that quoting accident.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    return None


def test_workflows_are_parseable_yaml():
    files = workflow_files()
    assert files, "no workflows discovered under .github/workflows"
    for path in files:
        load_workflow(path)


def test_every_workflow_declares_permissions():
    for path in workflow_files():
        assert "permissions" in load_workflow(path), (
            f"{path.name}: missing top-level 'permissions'")


def test_manual_suite_choices_resolve_to_registered_cases():
    catalogue = entrypoint.catalog()
    for path in workflow_files():
        event = event_section(load_workflow(path)) or {}
        dispatch = event.get("workflow_dispatch") or {}
        suite = dispatch.get("inputs", {}).get("suite")
        if suite is None:
            continue
        assert suite["type"] == "choice"
        assert suite["default"] in suite["options"]
        for name in suite["options"]:
            assert entrypoint.select(catalogue, suite=name)


def test_every_catalog_case_is_reachable_from_nightly():
    """No orphan cases: everything defined by a layer is registered in the suite tree."""
    catalogue = entrypoint.catalog()
    assert set(catalogue) == set(case_ids("nightly"))


def test_every_case_points_at_an_existing_launcher():
    for test in entrypoint.catalog().values():
        script = REPO_ROOT / test.train_script
        assert script.is_file(), f"{test.test_name}: missing {test.train_script}"
        assert test.test_descr.strip(), f"{test.test_name}: empty description"


def test_a5_cases_use_the_launcher_deepep_default():
    """A5 launchers default to the ``*_deepep`` CONFIG plus matching CLI_OVERRIDES.

    Pinning a non-DeepEP CONFIG here would make the launcher's
    ``asc_deepep`` override miss its target and silently fall back to the
    reference ``AllToAllTokenDispatcher``.
    """
    for test in entrypoint.catalog().values():
        if test.device != "a5":
            continue
        assert "CONFIG" not in (test.env_vars or {}), (
            f"{test.test_name}: A5 cases must inherit the launcher's *_deepep CONFIG")


def launcher_chain(script: Path) -> list[Path]:
    """Follow ``exec bash .../<wrapper>.sh`` hops to the recipe that owns CONFIG."""
    chain = [script]
    while True:
        text = chain[-1].read_text(encoding="utf-8")
        match = re.search(
            r'exec\s+bash\s+"\$\{SCRIPT_DIR\}/([^"]+\.sh)"', text)
        if match is None:
            return chain
        chain.append(chain[-1].parent / match.group(1))


def test_a5_launchers_carry_deepep_config_and_site_assets():
    """Static guard for the two A5 launcher contracts the cases rely on.

    The case layer deliberately leaves ``CONFIG`` and ``HF_ASSETS_PATH`` to the
    launcher, so a launcher that pins a literal tokenizer path or drops the
    ``*_deepep`` default silently degrades the whole A5 matrix.
    """
    for test in entrypoint.catalog().values():
        if test.device != "a5":
            continue
        chain = launcher_chain(REPO_ROOT / test.train_script)
        defaults = [
            m.group(1)
            for path in chain
            for m in re.finditer(r'CONFIG:-(\S+?)\}', path.read_text(encoding="utf-8"))
        ]
        assert any("deepep" in value for value in defaults), (
            f"{test.test_name}: no *_deepep CONFIG default in "
            f"{[str(path.relative_to(REPO_ROOT)) for path in chain]}")
        overrides = "\n".join(path.read_text(encoding="utf-8") for path in chain)
        assert "token_dispatcher.asc_deepep" in overrides, (
            f"{test.test_name}: a *_deepep CONFIG without the matching "
            "token_dispatcher.asc_deepep override falls back to the reference dispatcher")
        for path in chain:
            text = path.read_text(encoding="utf-8")
            assert '--hf-assets-path "/' not in text, (
                f"{path.relative_to(REPO_ROOT)}: hardcoded --hf-assets-path overrides HF_ASSETS_PATH")
