"""Run verification obligations through real verifier/cache boundaries."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, apply_evidence_record
from cognitive_evolve_runtime.llm.env import env_int
from cognitive_evolve_runtime.llm.governor import llm_governor
from cognitive_evolve_runtime.nexus._serde import stable_hash
from .cache import candidate_artifact_hash

_VERIFY_MAX_WORKERS_ENV = "COGEV_VERIFY_CONCURRENCY"
from .honesty_core import measure_verification_result
from .probe_executor import execute_probes
from .regime import compile_grounding_regime
from .replay_runner import build_replay_record
from .strength import measured_strength_from_result
from .types import VerificationResult


def run_obligations_for_population(candidates: list[Any], obligations: list[dict[str, Any]], *, cache: dict[str, dict[str, Any]] | None = None, max_checks: int = 8) -> list[dict[str, Any]]:
    cache = cache if cache is not None else {}
    cache_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    checks = 0
    max_workers = env_int(_VERIFY_MAX_WORKERS_ENV, llm_governor()._max_concurrent())

    for obligation in obligations or []:
        if checks >= max(0, int(max_checks or 0)):
            records.append({"changed": False, "reason": "obligation_budget_exhausted", "obligation": dict(obligation)})
            continue

        remaining = max(0, int(max_checks or 0)) - checks
        batch = candidates[:remaining]
        checks += len(batch)

        if max_workers <= 1 or len(batch) <= 1:
            batch_records = [_check_one(c, obligation, cache, cache_lock) for c in batch]
        else:
            futures_map: dict[Any, Any] = {}
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
                futures_map = {pool.submit(_check_one, c, obligation, cache, cache_lock): c for c in batch}
                batch_records = [f.result() for f in as_completed(futures_map)]

        records.extend(batch_records)
    return records


def _check_one(candidate: Any, obligation: dict[str, Any], cache: dict[str, dict[str, Any]], cache_lock: threading.Lock) -> dict[str, Any]:
    result = _check_obligation(candidate, obligation, cache=cache, cache_lock=cache_lock)
    changed = False
    if obligation.get("must_pass") and not result.passed:
        evidence = EvidenceRecord(
            candidate_id=str(getattr(candidate, "id", "")),
            source="verification_obligation_runner",
            stage="verification_obligation",
            score=float(result.score),
            confidence=0.8,
            final_blocked=True,
            parent_blocked=False,
            terminal_reject=False,
            repair_value=0.5,
            continuation_value=0.6,
            diagnostics=list(result.diagnostics),
            hints=["satisfy the must-pass verification obligation before final projection"],
            metadata={"obligation": dict(obligation), "verification_result": result.to_dict()},
        )
        apply_evidence_record(candidate, evidence)
        changed = True
    _append_verification_result(candidate, result)
    return {"changed": changed, "reason": "obligation_checked", "candidate_id": str(getattr(candidate, "id", "")), "obligation": dict(obligation), "verification_result": result.to_dict()}


def _check_obligation(candidate: Any, obligation: dict[str, Any], *, cache: dict[str, dict[str, Any]], cache_lock: threading.Lock | None = None) -> VerificationResult:
    oid = str(obligation.get("id") or "obligation")
    fingerprint = str(obligation.get("verifier_fingerprint") or "obligation:" + stable_hash(obligation)[:16])
    key = "obligation:" + stable_hash({"candidate": getattr(candidate, "id", ""), "artifact": getattr(candidate, "artifact", ""), "fingerprint": fingerprint})
    with (cache_lock or threading.Lock()):
        if key in cache and isinstance(cache[key], dict):
            entry = cache[key]
            if isinstance(entry.get("measured_result"), dict):
                return VerificationResult.from_dict(entry["measured_result"])
            entry["legacy_cache"] = True
            entry["diagnostics_only"] = True
    text = str(getattr(candidate, "artifact", "") or getattr(candidate, "concise_claim", "") or getattr(candidate, "core_mechanism", ""))
    matcher = str(obligation.get("diagnostic_matcher") or obligation.get("signature") or "")
    # Text obligations are low-strength unless backed by an executable/project oracle.
    replayable = bool(obligation.get("replayable") and obligation.get("oracle_kind") == "toolrunner")
    passed = not matcher or matcher.lower() not in text.lower()
    raw_result = VerificationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence_ref="evidence-" + stable_hash({"obligation": oid, "passed": passed})[:16],
        replayable=replayable,
        diagnostics=["obligation_passed" if passed else "obligation_regression_detected", f"obligation_id:{oid}"],
        metadata={
            "fingerprint": fingerprint,
            "verifier_fingerprint": fingerprint,
            "obligation_id": oid,
            "oracle_kind": str(obligation.get("oracle_kind") or ("toolrunner" if replayable else "diagnostic_matcher")),
            "replay_scope": "verifier_on_frozen_artifact" if replayable else "diagnostic_matcher_only",
            "diagnostics_only": True,
            "strength_contribution": obligation.get("strength_contribution", 0),
        },
    )
    artifact_sha = candidate_artifact_hash(candidate)
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint=fingerprint,
        artifact_hash=artifact_sha,
        raw_obligation=obligation,
        oracle_kind=str(raw_result.metadata.get("oracle_kind") or ""),
    )
    actual_probe_verdicts = execute_probes(raw_result, regime, candidate=candidate, raw_obligation=obligation)
    replay_record = build_replay_record(
        candidate,
        raw_result,
        verifier_fingerprint=fingerprint,
        cache_key=key,
        oracle_kind=str(raw_result.metadata.get("oracle_kind") or ""),
    )
    result = measure_verification_result(
        raw_result,
        regime,
        actual_probe_verdicts=actual_probe_verdicts,
        replay_record=replay_record,
    ).to_verification_result()
    entry = {
        "raw_result": raw_result.to_dict(),
        "measured_result": result.to_dict(),
        "honesty_measurements": result.metadata.get("honesty_measurements"),
        "obligation_id": oid,
        "verifier_fingerprint": fingerprint,
        "replay_record": replay_record,
        "actual_probe_verdicts": actual_probe_verdicts,
        "grounding_regime": regime.to_dict(),
    }
    with (cache_lock or threading.Lock()):
        cache[key] = entry
    return result


def _append_verification_result(candidate: Any, result: VerificationResult) -> None:
    if not hasattr(candidate, "verification_trace"):
        return
    trace = [dict(item) for item in getattr(candidate, "verification_trace", []) if isinstance(item, dict)]
    trace.append(result.to_dict())
    candidate.verification_trace = trace[-100:]
    if result.passed and result.replayable:
        current = getattr(candidate, "verification_result", {}) if isinstance(getattr(candidate, "verification_result", {}), dict) else {}
        current_strength = measured_strength_from_result(VerificationResult.from_dict(current)) if current else measured_strength_from_result(None)
        if measured_strength_from_result(result) >= current_strength:
            candidate.verification_result = result.to_dict()


__all__ = ["run_obligations_for_population"]
