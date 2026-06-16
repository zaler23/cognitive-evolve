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
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController, apply_final_certificate_to_closure, apply_research_final_gate_directives, build_final_certificate
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

from .budget import EvolutionBudget, EvolutionLoopResult
from .closure import _attach_latent_replay_audit_to_closure, _closure_certificate, _completion_status_for_budget, _join_interruption_reference, _model_boundary_interruption_policy, _selected_improvement_certificate, _stop_reason_after_round
from .round import EvolutionRound, RoundEvaluation
from .stage_helpers import _error_progress_event, _notify_observer, _raise_if_cancelled, _theory_config_from_policy

class EvolutionLoopController:
    """Lifecycle controller around the per-round stage pipeline."""

    def __init__(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        budget: EvolutionBudget,
        model: NexusModelLike | None = None,
        observer: Callable[[dict[str, Any]], None] | None = None,
        cancellation_callback: Callable[[], bool] | None = None,
        offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None = None,
        adaptive_config: dict[str, Any] | None = None,
        adaptive_state: dict[str, Any] | None = None,
    ) -> None:
        self.population = population
        self.archives = archives
        self.policy = policy
        self.contract = contract
        self.world = world
        self.budget = budget
        self.model = model
        self.observer = observer
        self.cancellation_callback = cancellation_callback
        self.offspring_verifier = offspring_verifier
        self.adaptive = AdaptiveRuntimeController.from_sources(
            explicit=adaptive_config,
            restored_state=adaptive_state,
            contract=contract,
            policy=policy,
            world=world,
        )
        self.round_pipeline = EvolutionRound(model=model, budget=budget, adaptive=self.adaptive)
        self.progress_events: list[dict[str, Any]] = []
        self.pipeline_events: list[dict[str, Any]] = [
            PipelineProgressEvent(
                stage="candidate_population",
                stage_index=1,
                stage_count=budget.round_limit,
                stage_progress=0.0,
                metadata={
                    "adaptive": budget.adaptive,
                    "progress_semantics": "open_ended_no_percent_complete" if budget.adaptive else "fixed_percent_complete",
                },
            ).to_dict()
        ]
        self.diagnosis = SearchDiagnosis()
        self.error: dict[str, Any] = {}
        self.interrupted = False

    def run(self) -> EvolutionLoopResult:
        while self.budget.remaining():
            current_round = self.budget.current_round + 1
            try:
                stop = self._run_round(current_round)
                if stop:
                    break
            except InterruptedError as exc:
                self._checkpoint_interruption(current_round, exc, stop_reason="cancelled", stagnation_type="Cancelled", actions=["resume_from_checkpoint"])
                break
            except MODEL_BOUNDARY_ERRORS as exc:
                stop_reason, stagnation_type, actions = _model_boundary_interruption_policy(exc)
                self._checkpoint_interruption(
                    current_round,
                    exc,
                    stop_reason=stop_reason,
                    stagnation_type=stagnation_type,
                    actions=actions,
                )
                if not self.budget.recover_model_errors:
                    raise
                break
        return self._finalize()

    def _run_round(self, planned_round: int) -> bool:
        _raise_if_cancelled(self.cancellation_callback)
        current_round = self.budget.step()
        _raise_if_cancelled(self.cancellation_callback)
        evaluation = self.round_pipeline.evaluate(
            current_round=current_round,
            population=self.population,
            archives=self.archives,
            policy=self.policy,
            contract=self.contract,
        )
        self.policy = evaluation.policy
        self.diagnosis = evaluation.diagnosis
        self._record_evaluation(current_round, evaluation)
        if evaluation.stop_reason:
            self.budget.stop_reason = evaluation.stop_reason
            return True
        if current_round >= self.budget.round_limit:
            self.budget.stop_reason = "adaptive_safety_checkpoint" if self.budget.adaptive else "max_rounds"
            return True
        _raise_if_cancelled(self.cancellation_callback)
        return self._reproduce(current_round, evaluation)

    def _record_evaluation(self, current_round: int, evaluation: RoundEvaluation) -> None:
        event = evaluation.progress_event
        self.progress_events.append(event)
        self.pipeline_events.append(evaluation.pipeline_event)
        latent_archive_feedback = ingest_latent_feedback(
            contract=self.contract,
            archive_observations=[
                {
                    "candidate_id": candidate.id,
                    "fate": candidate.current_fate,
                    "round": current_round,
                    "intent_id": (candidate.metadata or {}).get("latent_ranking", {}).get("candidate_id", ""),
                    "reason": ";".join(candidate.failure_lessons[:2]),
                }
                for candidate in self.population.candidates
            ],
        )
        self.budget.history.append(
            {
                "round": current_round,
                "ranking": evaluation.rankings.to_dict(),
                "diagnosis": self.diagnosis.to_dict(),
                "critiques": [critique.to_dict() for critique in evaluation.critiques],
                "verification": [item.to_dict() for item in evaluation.verification_results],
                "generation_plan": evaluation.generation_plan,
                "population_compaction": evaluation.population_compaction,
                "latent_archive_feedback": latent_archive_feedback,
                "stop_policy": self.budget.stop_policy,
                "stop_reason": evaluation.stop_reason,
                "progress_event": event,
            }
        )
        self._notify("post_ranking_critique", current_round, event)

    def _reproduce(self, current_round: int, evaluation: RoundEvaluation) -> bool:
        reproduction_stop, offspring_verification, reproduction_compaction = self.round_pipeline.reproduce(
            current_round=current_round,
            population=self.population,
            archives=self.archives,
            policy=self.policy,
            contract=self.contract,
            world=self.world,
            rankings=evaluation.rankings,
            diagnosis=self.diagnosis,
            critiques=evaluation.critiques,
            offspring_verifier=self.offspring_verifier,
            repair_parent_candidates=evaluation.repair_parent_candidates,
        )
        if self.round_pipeline.last_generation_plan:
            self.budget.history[-1]["generation_plan"] = dict(self.round_pipeline.last_generation_plan)
        if offspring_verification:
            self.budget.history[-1]["offspring_verification"] = offspring_verification
        if reproduction_compaction:
            self.budget.history[-1]["reproduction_compaction"] = reproduction_compaction
        if reproduction_stop:
            self.budget.stop_reason = reproduction_stop
            return True
        self._notify("post_mutation", current_round, evaluation.progress_event)
        return False

    def _checkpoint_interruption(self, current_round: int, exc: Exception, *, stop_reason: str, stagnation_type: str, actions: list[str]) -> None:
        self.error = {"type": exc.__class__.__name__, "message": str(exc), "round": current_round}
        self.interrupted = True
        self.budget.stop_reason = stop_reason
        self.diagnosis = SearchDiagnosis(
            stagnation_detected=True,
            stagnation_type=stagnation_type,
            recommended_actions=actions,
            notes=f"Nexus evolution interrupted and checkpointed: {exc}",
        )
        self.budget.history.append({"round": current_round, "error": self.error, "diagnosis": self.diagnosis.to_dict(), "stop_reason": self.budget.stop_reason})
        try:
            self._notify("error_checkpoint", current_round, _error_progress_event(self.progress_events[-1] if self.progress_events else {}, current_round), error=self.error)
        except Exception as checkpoint_exc:
            self.error["error_checkpoint_observer_error"] = f"{checkpoint_exc.__class__.__name__}: {checkpoint_exc}"
            self.budget.history[-1]["error_checkpoint_observer_error"] = self.error["error_checkpoint_observer_error"]

    def _finalize(self) -> EvolutionLoopResult:
        if not self.budget.stop_reason:
            self.budget.stop_reason = ("adaptive_safety_checkpoint" if self.budget.adaptive else "max_rounds") if self.budget.current_round >= self.budget.round_limit else "completed"
        synthesis_model = None if self.interrupted else self.model
        synthesis = synthesize_result(population=self.population, archives=self.archives, contract=self.contract, world=self.world, model=synthesis_model)
        improvement_certificate = _selected_improvement_certificate(self.population, synthesis)
        if improvement_certificate is not None:
            ingest_latent_feedback(contract=self.contract, certificates=[improvement_certificate])
        completion_status = _completion_status_for_budget(budget=self.budget, interrupted=self.interrupted, synthesis=synthesis)
        latent_override = latent_completion_override(
            contract=self.contract,
            completion_status=completion_status,
            synthesis=synthesis,
            improvement_certificate=improvement_certificate,
        )
        if latent_override.get("overridden"):
            completion_status = str(latent_override.get("completion_status") or completion_status)
            synthesis.status = "needs_continuation"
            synthesis.warnings.append(str(latent_override.get("reason") or "latent_problem_space_not_converged"))
            synthesis.failure_analysis = synthesis.failure_analysis or "Latent problem-space convergence is unresolved; continue exploration before claiming solved."
        self.budget.completion_status = completion_status
        if self.interrupted:
            if self.budget.stop_reason == "model_quota_pause_checkpointed":
                synthesis.status = "paused_quota"
                synthesis.final_answer = (
                    "Nexus evolution was interrupted before final convergence and paused on provider quota/rate exhaustion. "
                    f"A recoverable checkpoint and {len(self.population.candidates)} candidate genomes were persisted; "
                    "resume the run after quota resets instead of continuing to call the provider."
                )
                synthesis.warnings.append("model_quota_pause_checkpointed_no_more_provider_calls")
            else:
                synthesis.status = "interrupted_checkpointed"
                local_reference_answer = str(synthesis.final_answer or "").strip()
                interruption_note = (
                    "Nexus evolution was interrupted before final convergence. "
                    f"A recoverable checkpoint and {len(self.population.candidates)} candidate genomes were persisted; "
                    "resume the run after the model schema/provider quota/transport issue is resolved."
                )
                synthesis.final_answer = _join_interruption_reference(interruption_note, local_reference_answer)
                synthesis.warnings.append("model_schema_quota_or_transport_interruption_checkpointed_partial_population")
            synthesis.failure_analysis = synthesis.failure_analysis or self.error.get("message", "Nexus evolution interrupted.")
        elif completion_status == "needs_continuation":
            synthesis.status = "needs_continuation"
            synthesis.warnings.append("adaptive_safety_checkpoint_reached_without_objective_closure")
            synthesis.failure_analysis = synthesis.failure_analysis or (
                "The adaptive run reached its safety checkpoint before a model/verifier stop signal closed the objective. "
                "The persisted checkpoint is intended for continuation, not as proof of completion."
            )
        elif completion_status == "route_incomplete":
            synthesis.status = "route_incomplete"
            synthesis.warnings.append("route_incomplete_single_diagnostic_not_objective_solution")
        elif completion_status == "best_current_route":
            synthesis.status = "best_current_route"
            synthesis.warnings.append("best_current_route_not_objective_solution")
        synthesis.continuation_available = completion_status in {"needs_continuation", "interrupted_checkpointed", "paused_quota"}
        synthesis.completion_status = completion_status
        synthesis.closure_certificate = _closure_certificate(
            budget=self.budget,
            interrupted=self.interrupted,
            synthesis=synthesis,
            completion_status=completion_status,
            contract=self.contract,
            improvement_certificate=improvement_certificate,
            latent_assessment=latent_override.get("assessment") if isinstance(latent_override.get("assessment"), dict) else {},
        )
        final_certificate = build_final_certificate(
            population=self.population,
            synthesis=synthesis,
            closure_certificate=synthesis.closure_certificate,
            evaluator_required=self.adaptive.evaluator_enabled,
        ) if self.adaptive.enabled else {}
        self.adaptive.before_final_projection(candidates=self.population.candidates, final_certificate=final_certificate)
        if final_certificate:
            final_certificate = apply_research_final_gate_directives(final_certificate, self.adaptive.final_gate_directives())
        if final_certificate:
            synthesis.closure_certificate = apply_final_certificate_to_closure(synthesis.closure_certificate, final_certificate)
            self.adaptive.attach_final_certificate(final_certificate)
        latent_replay_audit = audit_latent_replay_bundle(
            self.contract,
            population=self.population,
            generation_plan=self.round_pipeline.last_generation_plan,
            budget_history=self.budget.history,
            archives=self.archives,
        )
        _attach_latent_replay_audit_to_closure(synthesis, latent_replay_audit)
        synthesis.objective_solved = bool(synthesis.closure_certificate.get("objective_solved"))
        final_progress_event = self.progress_events[-1] if self.progress_events else {}
        if self.interrupted:
            final_progress_event = _error_progress_event(final_progress_event, self.budget.current_round)
        self._notify("final_synthesis", self.budget.current_round, final_progress_event, error=self.error or None)
        return EvolutionLoopResult(
            population=self.population,
            archives=self.archives,
            policy=self.policy,
            diagnosis=self.diagnosis,
            synthesis=synthesis,
            progress_events=self.progress_events,
            pipeline_events=self.pipeline_events,
            budget_history=list(self.budget.history),
            elo=self.round_pipeline.elo.to_dict(),
            latent_replay_audit=latent_replay_audit,
            interrupted=self.interrupted,
            error=self.error,
            current_round=self.budget.current_round,
            max_rounds=self.budget.round_limit,
            stop_reason=self.budget.stop_reason,
            completion_status=completion_status,
            adaptive_state=self.adaptive.to_dict(),
        )

    def _notify(self, phase: str, round_index: int, progress_event: dict[str, Any], *, error: dict[str, Any] | None = None) -> None:
        _notify_observer(
            self.observer,
            phase=phase,
            round_index=round_index,
            population=self.population,
            archives=self.archives,
            policy=self.policy,
            diagnosis=self.diagnosis,
            progress_event=progress_event,
            budget_history=self.budget.history,
            error=error,
            adaptive_state=self.adaptive.to_dict(),
        )


def evolve_once(
    *,
    population: CandidatePopulation,
    archives: ArchiveManager,
    policy: EvolutionPolicy,
    contract: NexusObjectiveContract,
    world: Any,
    budget: EvolutionBudget,
    model: NexusModelLike | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
    offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None = None,
    adaptive_config: dict[str, Any] | None = None,
    adaptive_state: dict[str, Any] | None = None,
) -> EvolutionLoopResult:
    return EvolutionLoopController(
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
        world=world,
        budget=budget,
        model=model,
        observer=observer,
        cancellation_callback=cancellation_callback,
        offspring_verifier=offspring_verifier,
        adaptive_config=adaptive_config,
        adaptive_state=adaptive_state,
    ).run()


__all__ = ["EvolutionLoopController", "evolve_once"]
