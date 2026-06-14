"""Material delta detection for evolution rounds."""
from __future__ import annotations

from typing import Any

from ..contracts import MATERIAL_DELTA_TYPES


def material_delta(previous: dict[str, Any] | list[dict[str, Any]] | None, current: dict[str, Any], *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    previous_records = previous if isinstance(previous, list) else ([previous] if isinstance(previous, dict) else [])
    prev_latest = previous_records[-1] if previous_records else {}
    deltas: list[str] = []

    if _new_values(prev_latest, current, "candidate_families") or current.get("new_candidate_family"):
        deltas.append("new_candidate_family")
    if _new_values(prev_latest, current, "search_axes") or current.get("new_search_axis"):
        deltas.append("new_search_axis")
    if _new_values(prev_latest, current, "mutation_operators") or current.get("new_mutation_operator"):
        deltas.append("new_mutation_operator")
    if current.get("new_verifier_result") or current.get("verification_results"):
        deltas.append("new_verifier_result")
    if current.get("new_external_evidence") or _has_entries(current.get("new_evidence")) or _has_status(current.get("nested_evidence_results"), "ok"):
        deltas.append("new_external_evidence")
    if current.get("new_computed_evidence") or _has_status(current.get("nested_evidence_results"), "computed"):
        deltas.append("new_computed_evidence")
    if _has_entries(current.get("failed_assumptions")):
        deltas.append("new_failed_assumption")
    if _has_entries(current.get("new_counterexamples")):
        deltas.append("new_counterexample")
    if current.get("contract_revision") or current.get("new_contract_revision"):
        deltas.append("new_contract_revision")
    if current.get("failure_memory") or current.get("new_failure_memory"):
        deltas.append("new_failure_memory")
    if _frontier_diversity_gain(prev_latest, current):
        deltas.append("frontier_diversity_gain")
    if _gate_gain(prev_latest, current):
        deltas.append("hard_gate_satisfaction_gain")

    valid = sorted(set(delta for delta in deltas if delta in MATERIAL_DELTA_TYPES))
    return {
        "material_delta": bool(valid),
        "delta_types": valid,
        "status": "material_delta" if valid else "no_material_delta",
        "contract_metric_ids": [item.get("id") for item in (contract or {}).get("progress_metrics", []) if isinstance(item, dict)],
    }


def _has_entries(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _has_status(value: Any, status_fragment: str) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, dict) and status_fragment in str(item.get("status") or item.get("source_type") or ""):
            return True
    return False


def _new_values(previous: dict[str, Any], current: dict[str, Any], key: str) -> bool:
    prev = set(map(str, previous.get(key, []) if isinstance(previous.get(key), list) else []))
    cur = set(map(str, current.get(key, []) if isinstance(current.get(key), list) else []))
    return bool(cur - prev)


def _frontier_diversity_gain(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    try:
        return float(current.get("frontier_diversity", 0) or 0) > float(previous.get("frontier_diversity", 0) or 0)
    except (TypeError, ValueError):
        return False


def _gate_gain(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    try:
        return float(current.get("hard_gate_satisfaction", 0) or 0) > float(previous.get("hard_gate_satisfaction", 0) or 0)
    except (TypeError, ValueError):
        return False


__all__ = ["material_delta"]
