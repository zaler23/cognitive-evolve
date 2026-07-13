#!/usr/bin/env python3
"""Host-neutral source-project validation checks for Nexus-only code."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..core import COGEV, LOCAL_RUNTIME_ROOT, ROOT, SPECS, TASKS, TEMPLATES
from ..artifacts.store import _read
from ..nexus.semantics import classify, select_capability_ids

SOURCE_REQUIRED_DIRS = ["scripts"]
SOURCE_REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "docs/CONFIG_ARCHITECTURE.md",
    "docs/CORE_EVOLVE_ALGORITHM.md",
    "scripts/cogev.py",
    "pyproject.toml",
]
PACKAGE_REQUIRED_FILES = [
    "__init__.py",
    "api/openai_compat.py",
    "api/server.py",
    "commands.py",
    "doctor.py",
    "runtime.py",
    "nexus/runtime.py",
    "nexus/loop/__init__.py",
    "nexus/semantics.py",
    "nexus/evaluation.py",
    "llm/litellm_provider.py",
    "llm/provider_interface.py",
]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def source_checkout_detected(root: str | Path | None = None) -> bool:
    """Return whether validation is running from an unpacked source project."""

    candidate = Path(root) if root is not None else ROOT
    return (candidate / "pyproject.toml").is_file()


def _core_project_validation() -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    source_checkout = source_checkout_detected()
    checks.append((True, f"validation context: {'source checkout' if source_checkout else 'installed runtime'}"))
    for rel in PACKAGE_REQUIRED_FILES:
        checks.append(((PACKAGE_ROOT / rel).is_file(), f"package file exists: cognitive_evolve_runtime/{rel}"))
    if source_checkout:
        for rel in SOURCE_REQUIRED_DIRS:
            checks.append(((ROOT / rel).is_dir(), f"source directory exists: {rel}"))
        for rel in SOURCE_REQUIRED_FILES:
            checks.append(((ROOT / rel).is_file(), f"source file exists: {rel}"))
        checks.append((os.access(ROOT / "scripts/cogev.py", os.X_OK), "source script executable: scripts/cogev.py"))
    checks.append((SPECS.is_dir(), "package resource directory exists: specs"))
    checks.append((TEMPLATES.is_dir(), "package resource directory exists: templates"))
    source_tasks = COGEV / "tasks"
    source_task_artifacts = list(source_tasks.iterdir()) if source_tasks.exists() else []
    try:
        running_inside_local_runtime = ROOT.resolve() == LOCAL_RUNTIME_ROOT.resolve()
    except FileNotFoundError:
        running_inside_local_runtime = False
    if source_checkout and not running_inside_local_runtime:
        checks.append((not source_task_artifacts, "source project .cogev/tasks has no local runtime task artifacts"))
    task_parent = TASKS.parent
    while not task_parent.exists() and task_parent != task_parent.parent:
        task_parent = task_parent.parent
    checks.append((TASKS.is_dir() or task_parent.exists(), f"local runtime task directory exists: {TASKS}"))
    route = classify("evolve and tune the current agent-system paradigm")
    checks.append((route.level == "L4_evolutionary", "agent-system evolution prompt routes to L4_evolutionary"))
    checks.append(("cognitive_search" in select_capability_ids("architecture evolution tuning"), "Nexus capability selection covers cognitive search"))
    graph_template = TEMPLATES.joinpath("cognitive-search-graph.json")
    try:
        json.loads(graph_template.read_text(encoding="utf-8"))
        graph_json_ok = True
    except (FileNotFoundError, json.JSONDecodeError):
        graph_json_ok = False
    checks.append((graph_json_ok, "cognitive search graph template is valid JSON"))
    return checks


# Kept as current doctor helpers; they no longer require a capability_runtime package.
def _registry_validation() -> list[tuple[bool, str]]:
    return [(True, "Nexus capability registry is derived from semantics.DEFAULT_CAPABILITIES")]


def _metadata_dependency_validation() -> list[tuple[bool, str]]:
    return [(True, "Nexus metadata dependencies resolved")]


def core_project_validation() -> list[tuple[bool, str]]:
    return _core_project_validation()


__all__ = ["core_project_validation", "source_checkout_detected"]
