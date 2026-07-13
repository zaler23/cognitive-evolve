from __future__ import annotations

from cognitive_evolve_runtime.validation.suite import (
    SUITE_NAME,
    SUITE_VERSION,
    build_suite_report,
    is_passing_suite_report,
    normalize_suite_report,
)


def test_suite_report_uses_canonical_suite_fields() -> None:
    report = build_suite_report(
        "task-a",
        [{"id": "check", "description": "check", "passed": True}],
        generated_at="2026-05-25T00:00:00+00:00",
    )

    assert report["suite"] == SUITE_NAME
    assert report["suite_version"] == SUITE_VERSION
    assert report["verification_summary"]["verdict"] == "pass"
    assert report["verification_results"][0]["source"] == "validation_suite"
    assert "name" not in report
    assert "version" not in report
    assert is_passing_suite_report(report)


def test_missing_suite_fields_normalize_to_current_defaults() -> None:
    normalized = normalize_suite_report(
        {
            "status": "pass",
            "passed": 1,
            "total": 1,
            "results": [{"id": "current", "passed": True}],
        }
    )

    assert normalized["suite"] == SUITE_NAME
    assert normalized["suite_version"] == SUITE_VERSION
    assert "compat_warnings" not in normalized
    assert is_passing_suite_report(normalized)


def test_empty_suite_report_is_not_reported_as_pass() -> None:
    report = build_suite_report("task-empty", [], generated_at="2026-05-25T00:00:00+00:00")

    assert report["status"] == "fail"
    assert report["passed"] == 0
    assert report["total"] == 0
    assert report["verification_summary"]["verdict"] == "inconclusive"
    assert not is_passing_suite_report(report)


def test_summary_cannot_override_authoritative_suite_fields() -> None:
    report = build_suite_report(
        "task-fail",
        [{"id": "check", "description": "check", "passed": False}],
        generated_at="2026-05-25T00:00:00+00:00",
        summary={"status": "pass", "passed": 1, "total": 1, "operator_note": "external summary"},
    )

    assert report["status"] == "fail"
    assert report["passed"] == 0
    assert report["total"] == 1
    assert report["summary"]["status"] == "pass"
    assert report["operator_note"] == "external summary"
    assert report["summary"]["operator_note"] == "external summary"
    assert not is_passing_suite_report(report)
