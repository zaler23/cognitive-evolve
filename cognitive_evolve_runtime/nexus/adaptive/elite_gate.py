"""Final evidence certificate for adaptive runs."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import evidence_final_blocked, evidence_state


def build_final_certificate(
    *,
    population: CandidatePopulation,
    synthesis: Any,
    closure_certificate: dict[str, Any],
    evaluator_required: bool,
) -> dict[str, Any]:
    candidate = _selected_candidate(population, synthesis)
    generic_passed = _generic_verifier_passed(candidate)
    evaluator_passed = _external_evaluator_passed(candidate) if evaluator_required else None
    blocking: list[str] = []
    if candidate is None:
        blocking.append("final_candidate_absent")
    if candidate is not None and not generic_passed:
        blocking.append("generic_verifier_not_passed")
    if evaluator_required and evaluator_passed is not True:
        blocking.append("external_evaluator_not_passed")
    closure_claimed = bool(closure_certificate.get("objective_solved"))
    objective_solved = bool(closure_claimed and generic_passed and (not evaluator_required or evaluator_passed is True) and not blocking)
    return {
        "version": "adaptive-final-certificate/v1",
        "objective_solved": objective_solved,
        "closure_objective_solved_before_gate": closure_claimed,
        "candidate_id": candidate.id if candidate is not None else "",
        "generic_verifier_passed": generic_passed,
        "external_evaluator_required": bool(evaluator_required),
        "external_evaluator_passed": evaluator_passed,
        "evidence_state": evidence_state(candidate) if candidate is not None else {},
        "robustness_score": _score(candidate, "robustness") if candidate is not None else None,
        "mdl": dict((candidate.metadata or {}).get("mdl") or {}) if candidate is not None and isinstance(candidate.metadata, dict) else {},
        "judge_quorum": {},
        "blocking_reasons": list(dict.fromkeys(blocking)),
    }


def apply_final_certificate_to_closure(closure_certificate: dict[str, Any], final_certificate: dict[str, Any]) -> dict[str, Any]:
    closure = dict(closure_certificate or {})
    closure["final_certificate"] = dict(final_certificate or {})
    if final_certificate and not final_certificate.get("objective_solved"):
        closure["objective_solved"] = False
        critical = list(closure.get("critical_failures") or [])
        critical.extend(str(item) for item in final_certificate.get("blocking_reasons", []) if item)
        if final_certificate.get("closure_objective_solved_before_gate"):
            critical.append("adaptive_final_certificate_gate_failed")
        closure["critical_failures"] = list(dict.fromkeys(critical))
    return closure


def _selected_candidate(population: CandidatePopulation, synthesis: Any) -> CandidateGenome | None:
    by_id = population.by_id()
    for attr in ("best_candidate_id", "reference_candidate_id"):
        candidate_id = str(getattr(synthesis, attr, "") or "")
        if candidate_id and candidate_id in by_id:
            return by_id[candidate_id]
    return None


def _generic_verifier_passed(candidate: CandidateGenome | None) -> bool:
    if candidate is None:
        return False
    result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
    if result.get("passed") is False:
        return False
    if result.get("final_eligible") is False:
        return False
    return bool(result.get("passed") is True or candidate.verification_trace or candidate.tool_results or candidate.evidence_refs or candidate.source_bindings)


def _external_evaluator_passed(candidate: CandidateGenome | None) -> bool:
    if candidate is None or not isinstance(candidate.metadata, dict):
        return False
    state = evidence_state(candidate)
    if state:
        return not evidence_final_blocked(candidate) and float(state.get("final_score") or 0.0) > 0.0
    evaluator = candidate.metadata.get("evaluator")
    return isinstance(evaluator, dict) and evaluator.get("status") == "passed" and evaluator.get("passed") is True


def _score(candidate: CandidateGenome, key: str) -> float | None:
    value = (candidate.multihead_scores or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["apply_final_certificate_to_closure", "build_final_certificate"]
