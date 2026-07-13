"""Final evidence certificate for adaptive runs."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import evidence_final_blocked, evidence_state, latest_evidence_record, select_preliminary_incumbent
from cognitive_evolve_runtime.nexus.stop_reasons import (
    CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
    is_solved_stop_reason,
    normalize_external_review_stop_reason,
)
from cognitive_evolve_runtime.validation.result import preliminary_passed_from_mapping


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
    if candidate is not None and generic_passed is False:
        blocking.append("preliminary_check_failed_advisory")
    elif candidate is not None and generic_passed is None and not evaluator_required:
        blocking.append("preliminary_check_not_run_advisory")
    if evaluator_required and evaluator_passed is False:
        blocking.append("preliminary_evaluator_failed_advisory")
    elif evaluator_required and evaluator_passed is None:
        blocking.append("preliminary_evaluator_not_run_advisory")
    legacy_claim = bool(closure_certificate.get("objective_solved"))
    stop_reason = closure_certificate.get("stop_reason")
    ready_for_external_review = bool(
        candidate is not None
        and (
            normalize_external_review_stop_reason(stop_reason) == CANDIDATE_READY_FOR_EXTERNAL_REVIEW
            or is_solved_stop_reason(stop_reason)
            or legacy_claim
        )
    )
    preliminary_passed = (
        False
        if generic_passed is False or (evaluator_required and evaluator_passed is False)
        else True
        if (evaluator_required and evaluator_passed is True) or (not evaluator_required and generic_passed is True)
        else None
    )
    return {
        "version": "adaptive-final-certificate/v1",
        "objective_solved": False,
        "ready_for_external_review": ready_for_external_review,
        "candidate_id": candidate.id if candidate is not None else "",
        "preliminary_validation_passed": preliminary_passed,
        "preliminary_evaluator_required": bool(evaluator_required),
        "preliminary_evaluator_passed": evaluator_passed,
        "validation_semantics": "preliminary_only_external_review_required",
        "evidence_state": evidence_state(candidate) if candidate is not None else {},
        "robustness_score": _score(candidate, "robustness") if candidate is not None else None,
        "mdl": dict((candidate.metadata or {}).get("mdl") or {}) if candidate is not None and isinstance(candidate.metadata, dict) else {},
        "judge_quorum": {},
        "blocking_reasons": list(dict.fromkeys(blocking)),
    }


def apply_final_certificate_to_closure(closure_certificate: dict[str, Any], final_certificate: dict[str, Any]) -> dict[str, Any]:
    closure = dict(closure_certificate or {})
    closure["objective_solved"] = False
    certificate = dict(final_certificate or {})
    certificate["objective_solved"] = False
    closure["final_certificate"] = certificate
    if final_certificate:
        closure["final_certificate_advisory"] = "nonblocking_answer_first"
    return closure


def _selected_candidate(population: CandidatePopulation, synthesis: Any) -> CandidateGenome | None:
    preliminary_incumbent = select_preliminary_incumbent(population.candidates)
    if preliminary_incumbent is not None:
        return preliminary_incumbent
    by_id = population.by_id()
    candidate_id = str(getattr(synthesis, "best_candidate_id", "") or "")
    return by_id.get(candidate_id) if candidate_id else None


def _generic_verifier_passed(candidate: CandidateGenome | None) -> bool | None:
    if candidate is None:
        return None
    result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
    return preliminary_passed_from_mapping(result) if result else None


def _external_evaluator_passed(candidate: CandidateGenome | None) -> bool | None:
    if candidate is None or not isinstance(candidate.metadata, dict):
        return None
    evaluator = candidate.metadata.get("evaluator")
    if isinstance(evaluator, dict):
        if isinstance(evaluator.get("passed"), bool):
            return bool(evaluator["passed"])
        if evaluator.get("status") in {"passed", "failed"}:
            return evaluator.get("status") == "passed"
    if latest_evidence_record(candidate) is None:
        return None
    state = evidence_state(candidate)
    return not evidence_final_blocked(candidate) and float(state.get("final_score") or 0.0) > 0.0


def _score(candidate: CandidateGenome, key: str) -> float | None:
    value = (candidate.multihead_scores or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



__all__ = ["apply_final_certificate_to_closure", "build_final_certificate"]
