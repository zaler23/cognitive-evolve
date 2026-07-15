"""Engine-owned JSON artifact assertion harness.

The command line accepts only two engine-created JSON files.  Candidate data
can select a template, JSON Pointer, operator, and JSON value; it cannot supply
code or a command to execute.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


_MISSING = object()


def evaluate_cases(artifact: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.get("relation_id"):
            results.append(_evaluate_metamorphic_case(artifact, case))
            continue
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


def _evaluate_metamorphic_case(artifact: Any, case: dict[str, Any]) -> dict[str, Any]:
    probe_id = str(case.get("probe_id") or "")
    assertion_id = str(case.get("assertion_id") or "")
    relation_id = str(case.get("relation_id") or "")
    mapping_path = str(case.get("mapping_path") or "")
    summary_path = str(case.get("summary_path") or "")
    mapping = _resolve_pointer(artifact, mapping_path)
    declared_summary = _resolve_pointer(artifact, summary_path)
    before_keys = list(mapping) if isinstance(mapping, dict) else []
    transformed = dict(reversed(list(mapping.items()))) if isinstance(mapping, dict) else {}
    numeric_values = (
        list(transformed.values())
        if transformed
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            for value in transformed.values()
        )
        else []
    )
    after_value = math.fsum(numeric_values) if numeric_values else None
    before_summary = {
        "source": "candidate_declared_summary",
        "value": declared_summary if declared_summary is not _MISSING else None,
    }
    after_summary = {
        "source": "engine_recomputed_after_reverse_object_keys",
        "value": after_value,
    }
    transformation = {
        "kind": "reverse_object_keys",
        "path": mapping_path,
        "before_key_order": before_keys,
        "after_key_order": list(transformed),
        "transformed_input_sha256": _json_sha256({"path": mapping_path, "items": list(transformed.items())}),
    }
    applicable = (
        relation_id == "dict_numeric_summary_permutation_invariance/v1"
        and bool(numeric_values)
        and not isinstance(declared_summary, bool)
        and isinstance(declared_summary, (int, float))
        and math.isfinite(declared_summary)
    )
    survived = bool(
        applicable
        and after_value is not None
        and math.isclose(float(declared_summary), after_value, rel_tol=1e-12, abs_tol=1e-12)
    )
    return {
        "probe_id": probe_id,
        "assertion_id": assertion_id,
        "status": "survived" if survived else "counterexample",
        "reason": "relation_satisfied" if survived else "metamorphic_relation_violated",
        "relation_id": relation_id,
        "input_transformation": transformation,
        "before_output_summary": before_summary,
        "before_output_sha256": _output_sha256(before_summary["value"]),
        "after_output_summary": after_summary,
        "after_output_sha256": _output_sha256(after_summary["value"]),
    }


def _output_sha256(value: Any) -> str:
    normalized = float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) else value
    return _json_sha256({"value": normalized})


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
