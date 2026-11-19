# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/6180

"""Backport grouped override success logs to TorchTitan v0.3.0.

The configuration observability extension owns the runtime bindings.
Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from typing import TYPE_CHECKING

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.config.configurable import Configurable
    from torchtitan.config.override import OverrideConfig

_override = importlib.import_module("torchtitan.config.override")


def apply_overrides(
    override_config: OverrideConfig,
    config_root: Configurable.Config,
) -> list[str]:
    """Apply overrides using the v0.3.0 replacement and validation rules."""
    # Each entry is a bare target string or a (target, kwargs) tuple; a target is
    # a module.function path (see OverrideConfig.imports). Resolve each to
    # (module, function) -- importing its module, which triggers that module's
    # @override decorators -- and forward the kwargs to the override it names.
    entries = [(e, {}) if isinstance(e, str) else (e[0], e[1]) for e in override_config.imports]
    targets = [(*_override._resolve_target(target), kwargs) for target, kwargs in entries]

    resolved = _override._resolve_active(targets)
    claims = _override._collect_claims(resolved, config_root)
    _override._check_node_conflicts(claims)

    replacements: list[str] = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in claims:
        # ``**{}`` for a bare (no-kwargs) entry is just ``factory(cfg)``; a kwarg
        # the factory does not accept raises TypeError here, naming the factory.
        new_cfg = c.ov.factory(c.cfg, **c.kwargs)
        if isinstance(c.parent, list) and isinstance(c.attr, int):
            c.parent[c.attr] = new_cfg
        elif isinstance(c.attr, str):
            setattr(c.parent, c.attr, new_cfg)
        else:
            raise ValueError(f"Override '{c.ov.name}' claims root config '{c.fqn}', which cannot be replaced in place.")
        kwargs_note = " with " + ", ".join(f"{k}={v!r}" for k, v in c.kwargs.items()) if c.kwargs else ""
        transition = f"{type(c.cfg).__qualname__} -> {type(new_cfg).__qualname__}{kwargs_note}"
        replacements.append(f"[Override] {c.ov.name}: {c.fqn} {transition}")
        groups[c.ov.name, transition].append(c.fqn)

    if replacements:
        for (factory, transition), fqns in groups.items():
            logger.info(
                "[Override] %s: %s matched=%d fqns=%s",
                factory,
                transition,
                len(fqns),
                fqns,
            )
        logger.info(f"Applied {len(replacements)} override(s)")
    elif override_config.imports:
        # The user asked for overrides but nothing matched — likely a wrong
        # `target` class, a `fqns` glob that matches no node, or a module that
        # registered no override. Surface it rather than silently no-op.
        logger.warning(
            f"override.imports={override_config.imports} produced no "
            f"replacements. Check that the imported modules register overrides "
            f"whose `target` and `fqns` match nodes in the config tree."
        )

    return replacements
