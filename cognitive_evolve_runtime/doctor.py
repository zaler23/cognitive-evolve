#!/usr/bin/env python3
"""Source/runtime doctor orchestration.

Detailed checks live under ``cognitive_evolve_runtime.validation`` so doctor
stays a small coordinator instead of a monolithic validation script.
"""
from __future__ import annotations

import sys

from .artifacts.store import ok
from .llm import LLMConfigurationError
from .validation.project_health import ROOT, core_project_validation
from .validation.runtime_task import latest_runtime_task_validation
from .validation.source_runtime_sync import source_runtime_coverage_validation
from .validation.standalone_runtime import standalone_runtime_validation


def _append_validation(checks: list[bool], validation: list[tuple[bool, str]]) -> None:
    for condition, message in validation:
        checks.append(ok(condition, message))


def doctor(scope: str = "all") -> int:
    """Validate the source project, standalone runtime boundary, or both."""
    valid_scopes = {"core", "runtime", "task", "all"}
    if scope not in valid_scopes:
        print(f"Invalid doctor scope: {scope}", file=sys.stderr)
        return 2

    checks: list[bool] = []
    check_core = scope in {"core", "all"}
    check_runtime = scope in {"runtime", "all"}
    check_task = scope == "task"

    print("CognitiveEvolve doctor")
    print(f"project: {ROOT}")
    print(f"scope: {scope}")

    if check_core:
        try:
            _append_validation(checks, core_project_validation())
        except LLMConfigurationError as exc:
            checks.append(ok(False, f"LLM-first configuration required: {exc}"))
        _append_validation(checks, source_runtime_coverage_validation())

    if check_runtime:
        _append_validation(checks, standalone_runtime_validation())

    if check_task:
        _append_validation(checks, latest_runtime_task_validation())

    passed = sum(1 for c in checks if c)
    total = len(checks)
    print(f"summary: {passed}/{total} checks passed")
    return 0 if passed == total else 1
