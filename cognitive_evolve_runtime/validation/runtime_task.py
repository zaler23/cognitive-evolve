#!/usr/bin/env python3
"""Validation of the active CognitiveEvolve runtime task artifacts."""
from __future__ import annotations

import datetime as dt
import json

from ..artifacts.store import _load_json, latest_task_dir
from ..artifacts.task_files import _core_artifacts_have_templates
from .suite import SUITE_NAME, SUITE_VERSION, is_passing_suite_report, normalize_suite_report


LATEST_TASK_REQUIRED_FILES = [
    "task.yaml",
    "research-brief.md",
    "problem-contract.md",
    "decision-record.md",
    "validation-plan.md",
    "feedback.md",
    "working-memory.md",
    "trace.jsonl",
    "evaluations/checkmodel-report.md",
    "intake/enhanced-task-contract.md",
    "intake/enhanced-task-contract.json",
]


def _latest_runtime_task_validation() -> list[tuple[bool, str]]:
    """Validate the latest local runtime task used by core doctor checks."""
    checks: list[tuple[bool, str]] = []
    latest = latest_task_dir()
    if latest is None or not latest.exists():
        checks.append((True, "no local runtime task found; latest task checks skipped for clean source tree"))
        return checks
    checks.append((True, "latest local runtime task directory exists"))

    for rel in LATEST_TASK_REQUIRED_FILES:
        checks.append(((latest / rel).exists(), f"latest local runtime task file exists: {rel}"))
    checks.append((not _core_artifacts_have_templates(latest), "latest local runtime task artifacts are not untouched templates"))
    try:
        latest_eval = normalize_suite_report(_load_json(latest / "evaluations" / "native-eval-report.json"))
        latest_eval_task_ok = latest_eval.get("task") == latest.name
        latest_eval_suite_ok = latest_eval.get("suite") == SUITE_NAME
        latest_eval_suite_version_ok = latest_eval.get("suite_version") == SUITE_VERSION
        try:
            dt.datetime.fromisoformat(str(latest_eval.get("generated_at", "")))
            latest_eval_generated_at_ok = True
        except ValueError:
            latest_eval_generated_at_ok = False
        latest_eval_counts_ok = is_passing_suite_report(latest_eval)
        latest_eval_results = latest_eval.get("results", [])
        latest_eval_results_ok = (
            isinstance(latest_eval_results, list)
            and len(latest_eval_results) == latest_eval.get("total")
            and all(isinstance(item, dict) and item.get("passed") is True for item in latest_eval_results)
        )
        latest_eval_pass = (
            latest_eval.get("status") == "pass"
            and latest_eval_task_ok
            and latest_eval_suite_ok
            and latest_eval_suite_version_ok
            and latest_eval_generated_at_ok
            and latest_eval_counts_ok
            and latest_eval_results_ok
        )
    except (FileNotFoundError, json.JSONDecodeError):
        latest_eval_task_ok = False
        latest_eval_suite_ok = False
        latest_eval_suite_version_ok = False
        latest_eval_generated_at_ok = False
        latest_eval_counts_ok = False
        latest_eval_results_ok = False
        latest_eval_pass = False
    checks.append((latest_eval_task_ok, "latest native eval report targets latest task"))
    checks.append((latest_eval_suite_ok, "latest native eval report uses current suite name"))
    checks.append((latest_eval_suite_version_ok, "latest native eval report uses current suite version"))
    checks.append((latest_eval_generated_at_ok, "latest native eval report has parseable generated_at"))
    checks.append((latest_eval_counts_ok, "latest native eval report has complete pass count"))
    checks.append((latest_eval_results_ok, "latest native eval report marks every check passed"))
    checks.append((latest_eval_pass, "latest local runtime native eval passes"))
    try:
        latest_state = _load_json(latest / "runtime-state.json")
        latest_state_completed = latest_state.get("status") == "completed"
    except (FileNotFoundError, json.JSONDecodeError):
        latest_state_completed = False
    checks.append((latest_state_completed, "latest local runtime state is completed"))
    return checks


def latest_runtime_task_validation() -> list[tuple[bool, str]]:
    return _latest_runtime_task_validation()
