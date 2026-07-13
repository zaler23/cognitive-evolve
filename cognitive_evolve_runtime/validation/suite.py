#!/usr/bin/env python3
"""Canonical validation-suite report helpers."""
from __future__ import annotations

import datetime as dt
from typing import Any

from .result import aggregate_verification_results, verification_result_from_mapping

SUITE_NAME = "runtime-validation"
SUITE_VERSION = "1.0"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_suite_report(
    task: str,
    results: list[dict[str, Any]],
    generated_at: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    passed = sum(1 for item in results if item.get("passed") is True)
    total = len(results)
    canonical_objects = [verification_result_from_mapping(item, source="validation_suite") for item in results]
    canonical_results = [item.to_dict() for item in canonical_objects]
    canonical_summary = aggregate_verification_results(canonical_objects, source=SUITE_NAME).to_dict()
    report: dict[str, Any] = {
        "suite": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "task": task,
        "generated_at": generated_at or _now(),
        "status": "pass" if total > 0 and passed == total else "fail",
        "passed": passed,
        "total": total,
        "results": results,
        "verification_results": canonical_results,
        "verification_summary": canonical_summary,
    }
    if summary:
        supplemental = dict(summary)
        for key, value in supplemental.items():
            if key not in report:
                report[key] = value
        report["summary"] = supplemental
    return report


def normalize_suite_report(data: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(data or {})
    raw.setdefault("suite", SUITE_NAME)
    raw.setdefault("suite_version", SUITE_VERSION)
    return raw


def is_passing_suite_report(data: dict[str, Any] | None) -> bool:
    report = normalize_suite_report(data)
    results = report.get("results", [])
    total = report.get("total")
    passed = report.get("passed")
    counts_ok = isinstance(passed, int) and isinstance(total, int) and total > 0 and passed == total
    results_ok = isinstance(results, list) and len(results) == total and all(isinstance(item, dict) and item.get("passed") is True for item in results)
    return report.get("status") == "pass" and counts_ok and results_ok


__all__ = [
    "SUITE_NAME",
    "SUITE_VERSION",
    "build_suite_report",
    "normalize_suite_report",
    "is_passing_suite_report",
]
