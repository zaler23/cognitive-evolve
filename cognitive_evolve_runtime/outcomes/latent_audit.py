"""Run-level latent decision replay audit helpers.

This module is intentionally pure: callers provide the contract and runtime
objects to inspect, and the audit returns a stable dictionary without touching
files or mutating runtime state.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    LATENT_DECISION_TRACE_KEY,
    audit_latent_decision_replay,
)

TRACE_HASH_KEY = "latent_posterior_snapshot_hash"
TRACE_CURSOR_KEY = "latent_ledger_cursor"
TRACE_REF_KEY = "latent_decision_trace_ref"
TRACE_MODEL_KEY = "latent_update_model_version"


def collect_latent_decision_traces(
    *,
    contract: Any | None = None,
    population: Any | None = None,
    candidates: Any | None = None,
    generation_plan: Any | None = None,
    budget_history: Any | None = None,
    archives: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect latent decision traces from run-level in-memory structures.

    Returned entries have stable keys:
    ``source`` identifies where the trace was found, ``trace_ref`` is the trace
    ref from the trace or a deterministic fallback, and ``trace`` is a copy of
    the replay payload.
    """

    entries: list[dict[str, Any]] = []
    metadata = _contract_metadata(contract)
    if metadata:
        _collect_from_value(metadata, "contract.metadata", entries)

    for index, candidate in enumerate(_iter_candidates(population=population, candidates=candidates)):
        candidate_metadata = _candidate_metadata(candidate)
        if not candidate_metadata:
            continue
        label = _candidate_label(candidate, index)
        _collect_from_value(candidate_metadata, f"candidates[{label}].metadata", entries)

    if generation_plan is not None:
        _collect_from_value(generation_plan, "generation_plan", entries)

    if budget_history is not None:
        _collect_from_value(budget_history, "budget_history", entries)

    if archives is not None:
        _collect_from_value(archives, "archives", entries)

    return entries


def audit_latent_replay_bundle(
    contract: Any | None,
    *,
    population: Any | None = None,
    candidates: Any | None = None,
    generation_plan: Any | None = None,
    budget_history: Any | None = None,
    archives: Any | None = None,
) -> dict[str, Any]:
    """Replay-audit every latent decision trace found in a run bundle."""

    traces = collect_latent_decision_traces(
        contract=contract,
        population=population,
        candidates=candidates,
        generation_plan=generation_plan,
        budget_history=budget_history,
        archives=archives,
    )
    results: list[dict[str, Any]] = []
    for entry in traces:
        trace = coerce_dict(entry.get("trace"))
        raw_audit = audit_latent_decision_replay(contract, trace)
        expected_hash = str(trace.get(TRACE_HASH_KEY) or "")
        actual_hash = str(raw_audit.get("actual_snapshot_hash") or "")
        passed = bool(raw_audit.get("passed"))
        reason = str(raw_audit.get("reason") or "")

        if not expected_hash:
            passed = False
            reason = "missing_expected_snapshot_hash"
        elif not passed and not reason:
            reason = "snapshot_hash_mismatch" if actual_hash and actual_hash != expected_hash else "latent_decision_replay_failed"

        result = {
            "source": str(entry.get("source") or ""),
            "trace_ref": str(entry.get("trace_ref") or _trace_ref(trace)),
            "passed": passed,
            "reason": "" if passed else reason,
            "expected_snapshot_hash": expected_hash,
            "actual_snapshot_hash": actual_hash,
            "latent_ledger_cursor": _int(raw_audit.get(TRACE_CURSOR_KEY), default=_int(trace.get(TRACE_CURSOR_KEY))),
            "latent_update_model_version": str(raw_audit.get(TRACE_MODEL_KEY) or trace.get(TRACE_MODEL_KEY) or ""),
            "active_evidence_ids": list(raw_audit.get("active_evidence_ids") or []),
        }
        results.append(result)

    failures = [
        {
            "source": result["source"],
            "trace_ref": result["trace_ref"],
            "reason": result["reason"],
            "expected_snapshot_hash": result["expected_snapshot_hash"],
            "actual_snapshot_hash": result["actual_snapshot_hash"],
        }
        for result in results
        if not result["passed"]
    ]
    failed_count = len(failures)
    passed_count = len(results) - failed_count
    return {
        "passed": failed_count == 0,
        "total": len(results),
        "passed_count": passed_count,
        "failed": failed_count,
        "failed_count": failed_count,
        "trace_refs": [result["trace_ref"] for result in results],
        "results": results,
        "failures": failures,
        "failure_reasons": [failure["reason"] for failure in failures],
    }


def _collect_from_value(value: Any, source: str, entries: list[dict[str, Any]]) -> None:
    data = _as_dict(value)
    if data:
        direct_trace = _as_dict(data.get(LATENT_DECISION_TRACE_KEY))
        if direct_trace:
            _append_trace(entries, source=f"{source}.{LATENT_DECISION_TRACE_KEY}", trace=direct_trace)

        if LATENT_DECISION_TRACE_KEY not in data and _looks_like_trace(data):
            _append_trace(entries, source=source, trace=data)

        for key in sorted(data):
            if key == LATENT_DECISION_TRACE_KEY:
                continue
            _collect_from_value(data[key], f"{source}.{key}", entries)
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_from_value(item, f"{source}[{index}]", entries)


def _append_trace(entries: list[dict[str, Any]], *, source: str, trace: dict[str, Any]) -> None:
    entries.append(
        {
            "source": source,
            "trace_ref": _trace_ref(trace),
            "trace": dict(trace),
        }
    )


def _looks_like_trace(data: dict[str, Any]) -> bool:
    return any(key in data for key in (TRACE_HASH_KEY, TRACE_REF_KEY)) and any(
        key in data for key in (TRACE_CURSOR_KEY, TRACE_HASH_KEY, TRACE_MODEL_KEY)
    )


def _trace_ref(trace: dict[str, Any]) -> str:
    ref = str(trace.get(TRACE_REF_KEY) or "").strip()
    return ref or f"latent-decision:{stable_hash(trace)[:16]}"


def _contract_metadata(contract: Any | None) -> dict[str, Any]:
    if contract is None:
        return {}
    data = _as_dict(contract)
    if data:
        return coerce_dict(data.get("metadata"))
    return coerce_dict(getattr(contract, "metadata", {}))


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata:
        return metadata
    data = _as_dict(candidate)
    return coerce_dict(data.get("metadata")) or data


def _candidate_label(candidate: Any, index: int) -> str:
    candidate_id = str(getattr(candidate, "id", "") or _as_dict(candidate).get("id") or "").strip()
    return candidate_id or str(index)


def _iter_candidates(*, population: Any | None, candidates: Any | None) -> list[Any]:
    items: list[Any] = []
    for source in (population, candidates):
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            items.extend(source)
            continue
        data = _as_dict(source)
        nested = data.get("candidates") if data else getattr(source, "candidates", None)
        if isinstance(nested, (list, tuple)):
            items.extend(nested)
        elif nested is not None:
            items.append(nested)
        elif source is candidates:
            items.append(source)
    return items


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return coerce_dict(value.to_dict())
        except (TypeError, ValueError):
            return {}
    if is_dataclass(value):
        return coerce_dict(asdict(value))
    return {}


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "audit_latent_replay_bundle",
    "collect_latent_decision_traces",
]
