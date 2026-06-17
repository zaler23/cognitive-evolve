"""Stop-decision engine for Nexus evolution loops.

Safety round limits are checkpoints, not proof of completion.  This engine keeps
model/verifier stop signals and self-observed convergence separate from the
outer loop's safety-cap handling.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusStopModelProtocol
from cognitive_evolve_runtime.nexus.stop_reasons import (
    CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
    DIMINISHING_RETURNS_CHECKPOINT,
    normalize_external_review_stop_reason,
)
from cognitive_evolve_runtime.outcomes.runtime_bridge import best_candidate_m5_certificate, latent_stop_allows_solved
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.strength import candidate_verification_strength



class StopDecisionEngine:
    """Evaluate completion without treating the safety cap as success."""

    def stop_reason_after_round(
        self,
        *,
        budget: Any,
        completed_round: int,
        diagnosis: SearchDiagnosis,
        best_answer_id: str,
        population: CandidatePopulation,
        model: NexusModelLike | None,
        contract: Any | None = None,
    ) -> str:
        policy = str(getattr(budget, "stop_policy", "") or "llm_after_minimum").strip().lower()
        if policy == "route_incomplete_single_diagnostic":
            return "route_incomplete_single_diagnostic"
        if completed_round < max(1, int(getattr(budget, "min_rounds_before_stop", 1) or 1)):
            return ""
        if policy == "max_rounds":
            return ""
        improvement_certificate = _candidate_certificate(population, best_answer_id)
        convergence_reason = self._self_observed_convergence(budget=budget, diagnosis=diagnosis, best_answer_id=best_answer_id)
        if convergence_reason:
            if normalize_external_review_stop_reason(convergence_reason):
                return convergence_reason
            if not latent_stop_allows_solved(contract=contract, synthesis_certificate=improvement_certificate):
                return "latent_problem_space_needs_continuation"
            if not _measured_strength_allows_solved(population, best_answer_id, contract):
                return "model_stop_needs_measured_verification"
            return convergence_reason
        if policy == "adaptive_until_solved":
            if isinstance(model, NexusStopModelProtocol):
                decision = self._model_stop_decision(model=model, budget=budget, diagnosis=diagnosis, best_answer_id=best_answer_id, population=population)
                if isinstance(decision, dict) and decision.get("stop"):
                    review_reason = _external_review_reason(decision)
                    if review_reason:
                        return review_reason
                    if decision.get("solved") is True:
                        if not latent_stop_allows_solved(contract=contract, synthesis_certificate=improvement_certificate):
                            return "latent_problem_space_needs_continuation"
                        if not _measured_strength_allows_solved(population, best_answer_id, contract):
                            return "model_stop_needs_measured_verification"
                        return str(decision.get("reason") or "objective_solved")
                    return "model_stop_unsolved_needs_continuation"
                if decision is True:
                    if not latent_stop_allows_solved(contract=contract, synthesis_certificate=improvement_certificate):
                        return "latent_problem_space_needs_continuation"
                    if not _measured_strength_allows_solved(population, best_answer_id, contract):
                        return "model_stop_needs_measured_verification"
                    return "objective_solved"
            return ""
        if policy == "convergence_or_max_rounds":
            if best_answer_id and not diagnosis.stagnation_detected:
                return "converged_after_minimum"
            return ""
        if policy == "llm_after_minimum" and isinstance(model, NexusStopModelProtocol):
            decision = self._model_stop_decision(model=model, budget=budget, diagnosis=diagnosis, best_answer_id=best_answer_id, population=population)
            if isinstance(decision, dict):
                if not decision.get("stop"):
                    return ""
                review_reason = _external_review_reason(decision)
                if review_reason:
                    return review_reason
                if decision.get("solved") is True:
                    if not latent_stop_allows_solved(contract=contract, synthesis_certificate=improvement_certificate):
                        return "latent_problem_space_needs_continuation"
                    if not _measured_strength_allows_solved(population, best_answer_id, contract):
                        return "model_stop_needs_measured_verification"
                    return str(decision.get("reason") or "objective_solved")
                return "model_stop_unsolved_needs_continuation"
            if bool(decision):
                if not latent_stop_allows_solved(contract=contract, synthesis_certificate=improvement_certificate):
                    return "latent_problem_space_needs_continuation"
                if not _measured_strength_allows_solved(population, best_answer_id, contract):
                    return "model_stop_needs_measured_verification"
                return "model_stop_after_minimum"
            return ""
        return ""

    def _self_observed_convergence(self, *, budget: Any, diagnosis: SearchDiagnosis, best_answer_id: str) -> str:
        diag_type = str(getattr(diagnosis, "stagnation_type", "") or "").lower()
        notes = str(getattr(diagnosis, "notes", "") or "").lower()
        if best_answer_id and any(token in diag_type or token in notes for token in ("diminishing_returns", "diminishing returns", "low marginal gain", "low expected gain")):
            return DIMINISHING_RETURNS_CHECKPOINT
        if best_answer_id and ("converged" in diag_type or "converged" in notes):
            return CANDIDATE_READY_FOR_EXTERNAL_REVIEW
        history = [item for item in getattr(budget, "history", []) or [] if isinstance(item, dict)]
        if not best_answer_id or len(history) < 2:
            return ""
        recent = history[-2:]
        recent_best = [str(((item.get("ranking") or {}).get("best_final_answer_id") if isinstance(item.get("ranking"), dict) else "") or "") for item in recent]
        recent_diag = [str(((item.get("diagnosis") or {}).get("stagnation_type") if isinstance(item.get("diagnosis"), dict) else "") or "") for item in recent]
        if recent_best and all(item == best_answer_id for item in recent_best) and recent_diag and all(item.lower() in {"none", "converged"} for item in recent_diag):
            return CANDIDATE_READY_FOR_EXTERNAL_REVIEW
        return ""

    def _model_stop_decision(self, *, model: NexusStopModelProtocol, budget: Any, diagnosis: SearchDiagnosis, best_answer_id: str, population: CandidatePopulation) -> dict[str, Any] | bool:
        try:
            return model.should_stop(
                budget=budget,
                diagnosis=diagnosis,
                best_answer_id=best_answer_id,
                population=population.candidates,
            )
        except MODEL_BOUNDARY_ERRORS as exc:
            if is_quota_error(exc):
                raise
            return False


def _external_review_reason(decision: dict[str, Any]) -> str:
    for key in ("stop_reason", "stop_kind", "reason", "status"):
        reason = normalize_external_review_stop_reason(decision.get(key))
        if reason:
            return reason
    if decision.get("stop") and decision.get("solved") is not True and decision.get("continuation_needed") is False:
        confidence = decision.get("confidence")
        try:
            if confidence is not None and float(confidence) >= 0.70:
                return CANDIDATE_READY_FOR_EXTERNAL_REVIEW
        except (TypeError, ValueError):
            return ""
    return ""


def _candidate_certificate(population: CandidatePopulation, candidate_id: str) -> Any | None:
    if not candidate_id:
        return None
    candidate = population.by_id().get(candidate_id)
    return best_candidate_m5_certificate(candidate)


def _measured_strength_allows_solved(population: CandidatePopulation, best_answer_id: str, contract: Any | None) -> bool:
    if not best_answer_id:
        return False
    candidate = population.by_id().get(best_answer_id)
    if candidate is None:
        return False
    threshold = _verification_threshold(contract)
    return candidate_verification_strength(candidate) >= threshold


def _verification_threshold(contract: Any | None) -> VerificationStrength:
    metadata = getattr(contract, "metadata", {}) if contract is not None else {}
    if isinstance(metadata, dict):
        return VerificationStrength.from_value(metadata.get("verification_threshold") or VerificationStrength.FORMAL)
    return VerificationStrength.FORMAL


__all__ = ["StopDecisionEngine"]
