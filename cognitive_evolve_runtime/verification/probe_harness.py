"""Engine-owned JSON artifact assertion harness.

The command line accepts only two engine-created JSON files.  Candidate data
can select a template, JSON Pointer, operator, and JSON value; it cannot supply
code or a command to execute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


_MISSING = object()


def evaluate_cases(artifact: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        probe_id = str(case.get("probe_id") or "")
        path = str(case.get("path") or "")
        operator = str(case.get("operator") or "")
        expected = case.get("expected")
        actual = _resolve_pointer(artifact, path)
        try:
            survived = _assertion_survives(actual, operator, expected)
        except (TypeError, ValueError) as exc:
            results.append(
                {
                    "probe_id": probe_id,
                    "assertion_id": str(case.get("assertion_id") or ""),
                    "status": "unsupported",
                    "reason": f"operator_not_applicable:{type(exc).__name__}",
                }
            )
            continue
        result: dict[str, Any] = {
            "probe_id": probe_id,
            "assertion_id": str(case.get("assertion_id") or ""),
            "status": "survived" if survived else "counterexample",
            "path": path,
            "operator": operator,
        }
        if actual is not _MISSING:
            result["actual"] = actual
        results.append(result)
    return results


def _resolve_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("invalid_json_pointer")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return _MISSING
            if index < 0 or index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _assertion_survives(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return actual is not _MISSING
    if operator == "not_exists":
        return actual is _MISSING
    if actual is _MISSING:
        return False
    if operator == "equal":
        return actual == expected
    if operator == "not_equal":
        return actual != expected
    if operator == "contains":
        return expected in actual
    if operator == "not_contains":
        return expected not in actual
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "between":
        if not isinstance(expected, list) or len(expected) != 2:
            raise ValueError("between_expected_pair_required")
        return expected[0] <= actual <= expected[1]
    raise ValueError("unsupported_operator")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 2:
        raise SystemExit("usage: probe_harness.py ARTIFACT_JSON CASES_JSON")
    artifact = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    raw_cases = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise TypeError("cases payload must be a JSON array")
    print(json.dumps(evaluate_cases(artifact, [dict(item) for item in raw_cases if isinstance(item, dict)]), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_cases", "main"]
