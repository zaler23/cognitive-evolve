#!/usr/bin/env python3
"""Validation for standalone CognitiveEvolve runtime boundaries."""
from __future__ import annotations

import os

from ..core import LOCAL_RUNTIME_ROOT, ROOT, TASKS
from ..artifacts.store import _read

STANDALONE_REQUIRED_FILES = [
    "scripts/cogev.py",
    "scripts/cogev_api_smoke.py",
    "cognitive_evolve_runtime/api/openai_compat.py",
    "cognitive_evolve_runtime/api/server.py",
    "cognitive_evolve_runtime/api/config.py",
    "cognitive_evolve_runtime/llm/litellm_provider.py",
]

STANDALONE_EXECUTABLES = ["scripts/cogev.py"]
REMOVED_HOST_ADAPTER_PATHS = [
    ".host-private",
    "scripts/host_specific_smoke.sh",
    "adapters/host_specific_driver.py",
    "adapters/litellm_driver.py",
    "cognitive_evolve_runtime/validation/host_specific_adapter.py",
]


def _runtime_root_is_host_neutral() -> bool:
    normalized = str(LOCAL_RUNTIME_ROOT.expanduser())
    return "/.host-private/" not in normalized and not normalized.endswith("/.host-private")


def _task_root_is_initializable() -> bool:
    if TASKS.is_dir():
        return True
    parent = TASKS.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _standalone_runtime_validation() -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    checks.append((_runtime_root_is_host_neutral(), f"default runtime root is standalone: {LOCAL_RUNTIME_ROOT}"))
    checks.append((_task_root_is_initializable(), f"standalone task root is creatable or present: {TASKS}"))
    for rel in STANDALONE_REQUIRED_FILES:
        checks.append(((ROOT / rel).is_file(), f"standalone file exists: {rel}"))
    for rel in STANDALONE_EXECUTABLES:
        checks.append((os.access(ROOT / rel, os.X_OK), f"standalone executable: {rel}"))
    for rel in REMOVED_HOST_ADAPTER_PATHS:
        checks.append((not (ROOT / rel).exists(), f"removed host-specific dependency absent: {rel}"))
    env_example = _read(ROOT / ".env.example")
    checks.append(("COGEV_SERVER_API_KEY" in env_example, ".env.example defines frontend service key"))
    checks.append(("COGEV_LLM_API_KEY" in env_example, ".env.example defines upstream LLM key"))
    checks.append(("COGEV_API_TASK_ROOT" in env_example, ".env.example defines API task root"))
    makefile = _read(ROOT / "Makefile")
    checks.append(("runtime" in makefile and "pytest" in makefile, "Makefile is the canonical Python task runner"))
    checks.append((not (ROOT / "package.json").exists(), "package.json removed to avoid a parallel npm control plane"))
    return checks


def standalone_runtime_validation() -> list[tuple[bool, str]]:
    return _standalone_runtime_validation()
