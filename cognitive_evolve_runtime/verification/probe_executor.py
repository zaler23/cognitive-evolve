"""Execute engine-owned parameterized probes without executing candidate code."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.tools.runner import ToolRunner

from .honesty_core import GroundingRegime, ProbeCase
from .ladder import VerificationStrength
from .types import VerificationResult


_ARTIFACT_ASSERTION_TEMPLATE = "artifact_assertion/v1"
_NOT_JSON = object()


def execute_probes(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    *,
    candidate: Any = None,
    raw_obligation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute fixed JSON assertions and preserve every unexecuted case."""

    obligation = coerce_dict(raw_obligation)
    observations: dict[str, Any] = {}
    parameterized = [probe for probe in regime.probes if probe.template_id or probe.parameters]
    budget = max(0, int(regime.adversarial_budget or 0))
    runnable: list[ProbeCase] = []
    results_by_id: dict[str, dict[str, Any]] = {}
    for probe in parameterized:
        reason = str(coerce_dict(probe.parameters).get("unsupported_reason") or "")
        if probe.template_id != _ARTIFACT_ASSERTION_TEMPLATE:
            reason = reason or "unsupported_template"
        if reason:
            results_by_id[probe.probe_id] = _probe_result(probe, "unsupported", reason=reason)
        else:
            runnable.append(probe)

    artifact = _json_artifact(getattr(candidate, "artifact", candidate))
    if runnable and artifact is _NOT_JSON:
        for probe in runnable:
            results_by_id[probe.probe_id] = _probe_result(probe, "non_json", reason="materialized_artifact_not_json")
    elif runnable:
        selected = runnable[:budget]
        for probe in runnable[budget:]:
            results_by_id[probe.probe_id] = _probe_result(probe, "pending_budget", reason="adversarial_budget_exhausted")
        if selected:
            for item in _run_artifact_assertions(artifact, selected):
                results_by_id[str(item.get("probe_id") or "")] = item

    probe_results = [results_by_id[probe.probe_id] for probe in parameterized if probe.probe_id in results_by_id]
    survived = [item for item in probe_results if item.get("status") == "survived"]
    counterexamples = [item for item in probe_results if item.get("status") == "counterexample"]
    executed_count = len(survived) + len(counterexamples)
    for probe in parameterized:
        result = results_by_id.get(probe.probe_id)
        if not result or result.get("status") not in {"survived", "counterexample"}:
            continue
        flipped = result.get("status") == "counterexample"
        observations[probe.probe_id] = {
            "verdict_flipped": flipped,
            "matched_expected_flip": flipped == bool(probe.expected_verdict_flip),
            "engine_generated": True,
            "provenance": probe.provenance,
            "probe_content_sha256": _stable_probe_digest(probe.content),
        }
    observations.update(
        {
            "known_good_bad_distinguishable": _known_good_bad_distinguishable(
                raw_result,
                regime,
                candidate=candidate,
                obligation=obligation,
            ),
            "survived_count": len(survived),
            "counterexample_count": len(counterexamples),
            "executed_count": executed_count,
            "pending_count": len([item for item in probe_results if item.get("status") == "pending_budget"]),
            "unsupported_count": len([item for item in probe_results if item.get("status") in {"unsupported", "non_json"}]),
            "probe_survival_ratio": len(survived) / executed_count if executed_count else 0.0,
            "probe_results": probe_results,
            "probe_signature": regime.probe_signature,
            "engine_observation_schema": "probe_executor.v2",
        }
    )
    return observations


def result_with_probe_observations(raw_result: VerificationResult, observations: dict[str, Any]) -> VerificationResult:
    """Attach probe facts while preserving stronger independent verification."""

    probe_results = [dict(item) for item in observations.get("probe_results", []) if isinstance(item, dict)]
    if not probe_results:
        return raw_result
    counterexample_count = int(observations.get("counterexample_count") or 0)
    executed_count = int(observations.get("executed_count") or 0)
    metadata = dict(raw_result.metadata or {})
    metadata.update(
        {
            "parameterized_probe_provenance": "engine_template_model_parameters",
            "probe_results": probe_results,
            "probe_signature": str(observations.get("probe_signature") or ""),
            "probe_survival_ratio": float(observations.get("probe_survival_ratio") or 0.0),
            "probe_counterexample_count": counterexample_count,
            "probe_executed_count": executed_count,
            "probe_pending_count": int(observations.get("pending_count") or 0),
            "probe_unsupported_count": int(observations.get("unsupported_count") or 0),
        }
    )
    independently_grounded = raw_result.strength > VerificationStrength.NONE and not bool(metadata.get("diagnostics_only"))
    diagnostics = list(raw_result.diagnostics)
    if counterexample_count:
        metadata["probe_validation_status"] = "preliminary_failed"
        metadata["validation_status"] = "preliminary_failed"
        diagnostics.extend(
            "parameterized_probe_counterexample:" + str(item.get("assertion_id") or item.get("probe_id") or "")
            for item in probe_results
            if item.get("status") == "counterexample"
        )
        passed = False
        score = 0.0
    elif executed_count:
        metadata["probe_validation_status"] = "inconclusive"
        metadata["validation_status"] = (
            "preliminary_passed" if raw_result.passed else "preliminary_failed"
        ) if independently_grounded else "inconclusive"
        passed = raw_result.passed
        score = raw_result.score
    else:
        metadata["probe_validation_status"] = "not_run"
        metadata["validation_status"] = (
            "preliminary_passed" if raw_result.passed else "preliminary_failed"
        ) if independently_grounded else "not_run"
        passed = raw_result.passed
        score = raw_result.score
    return VerificationResult(
        passed=passed,
        score=score,
        strength=raw_result.strength,
        evidence_ref=raw_result.evidence_ref,
        replayable=raw_result.replayable,
        diagnostics=list(dict.fromkeys(str(item) for item in diagnostics if item)),
        metadata=metadata,
    )


def apply_probe_counterexample_evidence(
    candidate: Any,
    result: VerificationResult,
    *,
    round_index: int = 0,
) -> Any | None:
    """Route an executable counterexample into the existing evidence plane."""

    from cognitive_evolve_runtime.evaluators.challenge_memory import challenge_from_diagnostic
    from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, apply_evidence_record, evidence_records

    metadata = coerce_dict(result.metadata)
    counterexamples = [
        dict(item)
        for item in metadata.get("probe_results", [])
        if isinstance(item, dict) and item.get("status") == "counterexample"
    ]
    signature = str(metadata.get("probe_signature") or "")
    if not counterexamples:
        return None
    for existing in evidence_records(candidate):
        if existing.source == "engine_parameterized_probe" and str(existing.metadata.get("probe_signature") or "") == signature:
            return existing
    diagnostics = [
        "parameterized_probe_counterexample:"
        + str(item.get("assertion_id") or item.get("probe_id") or "")
        + ":"
        + str(item.get("path") or "")
        + ":"
        + str(item.get("operator") or "")
        for item in counterexamples
    ]
    executed = max(1, int(metadata.get("probe_executed_count") or 0))
    counterexample_ratio = len(counterexamples) / executed
    challenge_items = [
        challenge_from_diagnostic(
            candidate_id=str(getattr(candidate, "id", "")),
            source="engine_parameterized_probe",
            diagnostic=diagnostic,
            round_index=round_index,
            priority=counterexample_ratio,
        )
        for diagnostic in diagnostics
    ]
    record = EvidenceRecord(
        candidate_id=str(getattr(candidate, "id", "")),
        source="engine_parameterized_probe",
        stage="verification_probe",
        score=0.0,
        confidence=counterexample_ratio,
        final_blocked=True,
        parent_blocked=False,
        terminal_reject=False,
        repair_value=counterexample_ratio,
        continuation_value=counterexample_ratio,
        emitted_challenge_ids=[str(item.get("id") or "") for item in challenge_items if item.get("id")],
        diagnostics=diagnostics,
        hints=["repair the materialized artifact against the recorded parameterized counterexample"],
        metadata={
            "challenge_items": challenge_items,
            "probe_results": counterexamples,
            "probe_signature": signature,
            "provenance": "engine_template_model_parameters",
        },
    )
    apply_evidence_record(candidate, record)
    return record


def _run_artifact_assertions(artifact: Any, probes: list[ProbeCase]) -> list[dict[str, Any]]:
    cases = [
        {
            "probe_id": probe.probe_id,
            "assertion_id": str(probe.parameters.get("assertion_id") or ""),
            "path": str(probe.parameters.get("path") or ""),
            "operator": str(probe.parameters.get("operator") or ""),
            "expected": probe.parameters.get("expected"),
        }
        for probe in probes
    ]
    harness = Path(__file__).with_name("probe_harness.py")
    loader_path = os.environ.get("LD_LIBRARY_PATH")
    with tempfile.TemporaryDirectory(prefix="cogev-probe-") as raw_tmp:
        tmp = Path(raw_tmp)
        artifact_path = tmp / "artifact.json"
        cases_path = tmp / "cases.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        feedback = ToolRunner().run(
            [sys.executable, str(harness), str(artifact_path), str(cases_path)],
            cwd=tmp,
            env={"LD_LIBRARY_PATH": loader_path} if loader_path else None,
            enforce_resource_limits=False,
        )
    if feedback.status != "passed":
        reason = "probe_harness_" + str(feedback.status or "error")
        if feedback.diagnostics:
            reason += ":" + "; ".join(feedback.diagnostics)
        return [_probe_result(probe, "unsupported", reason=reason) for probe in probes]
    try:
        payload = json.loads(feedback.raw_output_ref)
    except (TypeError, json.JSONDecodeError):
        return [_probe_result(probe, "unsupported", reason="probe_harness_invalid_output") for probe in probes]
    return [dict(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _json_artifact(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return _NOT_JSON
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return _NOT_JSON


def _probe_result(probe: ProbeCase, status: str, *, reason: str) -> dict[str, Any]:
    return {
        "probe_id": probe.probe_id,
        "assertion_id": str(probe.parameters.get("assertion_id") or ""),
        "status": status,
        "reason": reason,
    }


def _known_good_bad_distinguishable(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    *,
    candidate: Any = None,
    obligation: dict[str, Any],
) -> bool:
    if not regime.probes:
        return False
    if _bool_hint(obligation, "known_bad_probe") or _bool_hint(obligation, "force_known_bad"):
        return False
    return False


def _bool_hint(mapping: dict[str, Any], key: str) -> bool:
    return bool(coerce_dict(mapping).get(key))


def _stable_probe_digest(content: str) -> str:
    return stable_hash({"probe_content": str(content or "")})[:24]


__all__ = [
    "apply_probe_counterexample_evidence",
    "execute_probes",
    "result_with_probe_observations",
]
