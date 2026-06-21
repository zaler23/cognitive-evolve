"""Nexus evolution loop skeleton with deterministic fake-model support."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import utc_now
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round, amplify_population
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta, candidate_source_bindings
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol, NexusSeedModelProtocol, NexusStopModelProtocol
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.outcomes.latent_audit import audit_latent_replay_bundle
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    best_candidate_m5_certificate,
    improvement_certificate_from_any,
    ingest_latent_feedback,
    ingest_runtime_trial_feedback,
    latent_completion_override,
    latent_exploration_plan_for_contract,
    m5_certificate_summary,
    requires_verified_improvement,
)
from cognitive_evolve_runtime.nexus.reproduction import (
    dedupe_offspring_against_population,
    elite_gap_merge_offspring,
    parents_for_crossover,
    ranked_repair_fallback_parents,
    sync_repair_parent_attempts_to_dormant_archive,
    verify_offspring,
)
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack
from cognitive_evolve_runtime.theory import TheoryConfig, TheoryLayer, build_population_representation
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, positive_int
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater
from cognitive_evolve_runtime.nexus.stop_reasons import is_external_review_stop_reason, is_solved_stop_reason

from .budget import EvolutionBudget, EvolutionLoopResult

def _stop_reason_after_round(
    *,
    budget: EvolutionBudget,
    completed_round: int,
    diagnosis: SearchDiagnosis,
    best_answer_id: str,
    population: CandidatePopulation,
    model: NexusModelLike | None,
    contract: Any | None = None,
) -> str:
    return StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=completed_round,
        diagnosis=diagnosis,
        best_answer_id=best_answer_id,
        population=population,
        model=model,
        contract=contract,
    )


def _model_stop_decision(*, model: NexusStopModelProtocol, budget: EvolutionBudget, diagnosis: SearchDiagnosis, best_answer_id: str, population: CandidatePopulation) -> dict[str, Any] | bool:
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


def _model_boundary_interruption_policy(exc: Exception) -> tuple[str, str, list[str]]:
    if is_quota_error(exc):
        return "model_quota_pause_checkpointed", "QuotaPaused", ["pause_until_provider_quota_resets", "resume_from_checkpoint"]
    category = provider_error_category(exc)
    if category == "empty_assistant_content":
        return "model_empty_output_checkpointed", "ModelEmptyOutput", ["resume_from_checkpoint", "retry_with_smaller_prompt_or_larger_output_budget"]
    if category == "truncated_response":
        return "model_truncated_output_checkpointed", "ModelTruncatedOutput", ["resume_from_checkpoint", "increase_output_budget_or_reduce_prompt"]
    if category in {"rate_limit_429", "provider_5xx", "timeout", "network_or_transient"}:
        return f"model_{category}_checkpointed", "ProviderTransientError", ["resume_from_checkpoint", "retry_after_backoff"]
    if category == "response_json_or_contract_error":
        return "model_schema_repair_checkpointed", "ModelSchemaRepairNeeded", ["resume_from_checkpoint", "repair_model_output_contract"]
    return "model_boundary_error_checkpointed", "ModelSchemaQuotaOrTransport", ["resume_from_checkpoint", "return_failure_report"]


def _selected_improvement_certificate(population: CandidatePopulation, synthesis: SynthesizedResult) -> Any | None:
    by_id = population.by_id()
    for candidate_id in [str(getattr(synthesis, "best_candidate_id", "") or "")]:
        candidate = by_id.get(candidate_id)
        if candidate is not None:
            certificate = best_candidate_m5_certificate(candidate)
            if certificate is not None:
                return certificate
    verified: list[Any] = []
    fallback: list[Any] = []
    for candidate in population.candidates:
        certificate = best_candidate_m5_certificate(candidate)
        if certificate is None:
            continue
        if getattr(certificate, "verified", False):
            verified.append(certificate)
        else:
            fallback.append(certificate)
    return verified[0] if verified else (fallback[0] if fallback else None)


def _completion_status_for_budget(*, budget: EvolutionBudget, interrupted: bool, synthesis: SynthesizedResult) -> str:
    if interrupted:
        return "paused_quota" if str(budget.stop_reason or "").strip().lower() == "model_quota_pause_checkpointed" else "interrupted_checkpointed"
    reason = str(budget.stop_reason or "").strip().lower()
    synthesis_status = str(getattr(synthesis, "status", "") or "").strip().lower()
    if reason in {"needs_continuation", "model_stop_unsolved_needs_continuation"}:
        return "needs_continuation"
    if "failure" in synthesis_status:
        return "failed"
    return "completed"


def _closure_certificate(
    *,
    budget: EvolutionBudget,
    interrupted: bool,
    synthesis: SynthesizedResult,
    completion_status: str,
    contract: Any | None = None,
    improvement_certificate: Any | None = None,
    latent_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stop_reason = str(budget.stop_reason or "")
    synthesis_status = str(getattr(synthesis, "status", "") or "")
    terminal_status = str(completion_status or "")
    synthesis_failure = "failure" in synthesis_status.lower()
    certificate = improvement_certificate_from_any(improvement_certificate)
    improvement_summary = m5_certificate_summary(certificate)
    improvement_verified = bool(improvement_summary.get("improvement_verified"))
    latent_assessment = dict(latent_assessment or {})
    latent_converged = latent_assessment.get("converged")
    latent_blocks_solved = False
    answer_produced = bool(
        terminal_status not in {"needs_continuation", "interrupted_checkpointed", "paused_quota", "failed", "failed_verification"}
        and not interrupted
        and not synthesis_failure
        and str(getattr(synthesis, "final_answer", "") or "").strip()
    )
    objective_solved = False
    checks = [
        {
            "check": "not_interrupted",
            "passed": not interrupted,
            "detail": "interrupted runs cannot close the objective",
        },
        {
            "check": "solved_stop_reason",
            "passed": True,
            "detail": stop_reason,
        },
        {
            "check": "completion_status_solved",
            "passed": terminal_status in {"completed", "solved"},
            "detail": terminal_status,
        },
        {
            "check": "synthesis_non_failure",
            "passed": not synthesis_failure,
            "detail": synthesis_status,
        },
        {
            "check": "verified_improvement_certificate_required",
            "passed": True,
            "detail": {
                "requires_verified_improvement": False,
                "improvement_verified": improvement_verified,
                "answer_produced": answer_produced,
                "certificate_hash": improvement_summary.get("improvement_certificate_hash", ""),
                "effect": "advisory_only_nonblocking",
            },
        },
        {
            "check": "latent_problem_space_converged",
            "passed": not latent_blocks_solved,
            "detail": latent_assessment,
        },
    ]
    critical_failures: list[str] = []
    if interrupted:
        critical_failures.append("interrupted")
    if terminal_status in {"needs_continuation", "interrupted_checkpointed", "paused_quota", "failed", "failed_verification"}:
        critical_failures.append(terminal_status)
    if synthesis_failure:
        critical_failures.append(synthesis_status or "synthesis_failure")
    return {
        "version": "closure_certificate_v1",
        "issued_at_utc": utc_now(),
        "terminal_status": terminal_status,
        "stop_reason": stop_reason,
        "objective_solved": objective_solved,
        "answer_produced": answer_produced,
        "objective_solved_semantics": "not_claimed_without_user_or_external_verification",
        "synthesis_status": synthesis_status,
        "best_candidate_id": str(getattr(synthesis, "best_candidate_id", "") or ""),
        "continuation_available": bool(getattr(synthesis, "continuation_available", False)),
        **improvement_summary,
        "latent_convergence": latent_assessment,
        "checks": checks,
        "critical_failures": list(dict.fromkeys(critical_failures)),
    }


def _attach_latent_replay_audit_to_closure(synthesis: SynthesizedResult, audit: dict[str, Any]) -> None:
    """Bind run-level latent replay audit to the closure certificate."""

    if not isinstance(audit, dict):
        return
    certificate = dict(getattr(synthesis, "closure_certificate", {}) or {})
    summary = {
        "passed": bool(audit.get("passed")),
        "total": int(audit.get("total") or 0),
        "failed_count": int(audit.get("failed_count") or audit.get("failed") or 0),
        "trace_refs": list(audit.get("trace_refs") or []),
        "failure_reasons": list(audit.get("failure_reasons") or []),
    }
    certificate["latent_replay_audit"] = summary
    checks = [dict(item) for item in certificate.get("checks", []) if isinstance(item, dict)]
    checks.append(
        {
            "check": "latent_replay_audit_passed",
            "passed": summary["passed"] or summary["total"] == 0,
            "detail": summary,
        }
    )
    certificate["checks"] = checks
    if summary["total"] > 0 and not summary["passed"]:
        certificate["latent_replay_audit_advisory"] = "failed_nonblocking"
        synthesis.warnings.append("latent_replay_audit_failed_advisory_only")
    synthesis.closure_certificate = certificate


def _join_interruption_reference(primary: str, reference_answer: str) -> str:
    if not reference_answer:
        return primary
    if reference_answer.strip() == primary.strip():
        return primary
    return (
        f"{primary}\n\n---\n\n"
        "Local deterministic reference summary from the persisted population follows. "
        "It is answer material from the paused run:\n\n"
        f"{reference_answer}"
    )


def _is_solved_stop_reason(reason: str) -> bool:
    return is_solved_stop_reason(reason)


__all__ = ["_closure_certificate", "_completion_status_for_budget", "_is_solved_stop_reason", "_model_boundary_interruption_policy", "_selected_improvement_certificate", "_stop_reason_after_round"]
