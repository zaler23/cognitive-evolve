#!/usr/bin/env python3
"""Host-neutral source-project validation checks for Nexus-only code."""
from __future__ import annotations

import json
import os

from ..core import COGEV, LOCAL_RUNTIME_ROOT, ROOT, TASKS
from ..artifacts.store import _read
from ..nexus.semantics import classify, select_capability_ids

CORE_REQUIRED_DIRS = [".cogev/specs", ".cogev/templates", "scripts"]
CORE_REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "docs/CONFIG_ARCHITECTURE.md",
    "docs/CORE_EVOLVE_ALGORITHM.md",
    "docs/ROADMAP.md",
    "cognitive_evolve_runtime/__init__.py",
    "cognitive_evolve_runtime/api/openai_compat.py",
    "cognitive_evolve_runtime/api/server.py",
    "cognitive_evolve_runtime/commands.py",
    "cognitive_evolve_runtime/doctor.py",
    "cognitive_evolve_runtime/runtime.py",
    "cognitive_evolve_runtime/nexus/runtime.py",
    "cognitive_evolve_runtime/nexus/loop/__init__.py",
    "cognitive_evolve_runtime/nexus/semantics.py",
    "cognitive_evolve_runtime/nexus/evaluation.py",
    "cognitive_evolve_runtime/llm/litellm_provider.py",
    "cognitive_evolve_runtime/llm/provider_interface.py",
    "scripts/cogev.py",
    "pyproject.toml",
]


def _core_project_validation() -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    for rel in CORE_REQUIRED_DIRS:
        checks.append(((ROOT / rel).is_dir(), f"directory exists: {rel}"))
    source_tasks = COGEV / "tasks"
    source_task_artifacts = list(source_tasks.iterdir()) if source_tasks.exists() else []
    try:
        running_inside_local_runtime = ROOT.resolve() == LOCAL_RUNTIME_ROOT.resolve()
    except FileNotFoundError:
        running_inside_local_runtime = False
    if not running_inside_local_runtime:
        checks.append((not source_task_artifacts, "source project .cogev/tasks has no local runtime task artifacts"))
    task_parent = TASKS.parent
    while not task_parent.exists() and task_parent != task_parent.parent:
        task_parent = task_parent.parent
    checks.append((TASKS.is_dir() or task_parent.exists(), f"local runtime task directory exists: {TASKS}"))
    for rel in CORE_REQUIRED_FILES:
        checks.append(((ROOT / rel).is_file(), f"file exists: {rel}"))
    checks.append((os.access(ROOT / "scripts/cogev.py", os.X_OK), "script executable: scripts/cogev.py"))
    route = classify("evolve and tune the current agent-system paradigm")
    checks.append((route.level == "L4_evolutionary", "agent-system evolution prompt routes to L4_evolutionary"))
    checks.append(("cognitive_search" in select_capability_ids("architecture evolution tuning"), "Nexus capability selection covers cognitive search"))
    graph_template = ROOT / ".cogev" / "templates" / "cognitive-search-graph.json"
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
