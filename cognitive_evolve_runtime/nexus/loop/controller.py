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
from cognitive_evolve_runtime.nexus.display_selection import build_display_context, select_displayed_candidate
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta, candidate_source_bindings
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol, NexusSeedModelProtocol, NexusStopModelProtocol
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.grading import certificate_allows_verified_result
from cognitive_evolve_runtime.verification.types import GradedOutput, VerifiedResult, Direction, VerificationPlan
from cognitive_evolve_runtime.verification.strength import candidate_verification_strength, measured_strength_from_result, strongest_passed_replayable_result
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
        verification_plan: VerificationPlan | dict[str, Any] | None = None,
        fabric_state: dict[str, Any] | None = None,
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
        if verification_plan is not None:
            self.adaptive.set_verification_plan(verification_plan)
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
        self.fabric_state: dict[str, Any] = dict(fabric_state or {})

    def run(self) -> EvolutionLoopResult:
        while self.budget.remaining():
            current_round = self.budget.current_round + 1
            try:
                stop = self._run_scheduler_epoch(current_round)
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

    def _run_scheduler_epoch(self, planned_round: int) -> bool:
        from cognitive_evolve_runtime.fabric.executors import FabricExecutionContext
        from cognitive_evolve_runtime.fabric.config import FabricRuntimeConfig
        from cognitive_evolve_runtime.fabric.scheduler import EpochConfig, TaskGraphScheduler

        _raise_if_cancelled(self.cancellation_callback)
        fabric_config = FabricRuntimeConfig.from_runtime_context(policy=self.policy, contract=self.contract)
        graph = self._graph_for_scheduler_epoch(planned_round, fabric_config=fabric_config)
        context = FabricExecutionContext(
            population=self.population,
            archives=self.archives,
            policy=self.policy,
            contract=self.contract,
            world=self.world,
            budget=self.budget,
            model=self.model,
            observer=self.observer,
            adaptive=self.adaptive,
            offspring_verifier=self.offspring_verifier,
            cancellation_callback=self.cancellation_callback,
            record_evaluation=self._record_scheduler_evaluation,
            record_reproduction=self._record_reproduction_result,
            fabric_config=fabric_config,
            fabric_state=self.fabric_state,
            round_pipeline=self.round_pipeline,
            diagnosis=self.diagnosis,
        )
        result = TaskGraphScheduler(
            graph=graph,
            context=context,
            config=fabric_config,
            epoch_config=EpochConfig(barrier="full", raise_task_exceptions=True),
        ).run()
        self.fabric_state = context.fabric_state
        self.fabric_state["last_scheduler_result"] = result.to_dict()
        self.policy = context.policy
        self.diagnosis = context.diagnosis
        self.round_pipeline = context.pipeline()
        return bool(self.budget.stop_reason)

    def _graph_for_scheduler_epoch(self, planned_round: int, *, fabric_config: Any | None = None) -> Any:
        from cognitive_evolve_runtime.fabric.epoch_builder import build_round_parity_epoch_graph
        from cognitive_evolve_runtime.fabric.task_graph import TaskGraph

        raw_graph = self.fabric_state.get("graph")
        if isinstance(raw_graph, dict):
            try:
                graph = TaskGraph.from_dict(raw_graph)
                if not graph.is_drained():
                    return graph
            except Exception as exc:
                diagnostics = self.fabric_state.setdefault("diagnostics", [])
                if isinstance(diagnostics, list):
                    diagnostics.append({"type": "fabric_graph_restore_failed", "message": f"{exc.__class__.__name__}: {exc}"})
        include_preprocess = bool(
            fabric_config is not None
            and (fabric_config.preprocess.run_each_epoch or int(planned_round or 0) <= 1)
        )
        return build_round_parity_epoch_graph(round_index=planned_round, include_preprocess=include_preprocess)

    def _record_scheduler_evaluation(self, current_round: int, evaluation: RoundEvaluation, context: Any) -> None:
        self.policy = evaluation.policy
        self.diagnosis = evaluation.diagnosis
        self._record_evaluation(current_round, evaluation)

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
                "grounded_information_gain": dict(getattr(self.diagnosis, "grounded_information_gain", {}) or {}),
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

    def _record_reproduction_result(
        self,
        current_round: int,
        evaluation: RoundEvaluation,
        reproduction_stop: str,
        offspring_verification: list[Any],
        reproduction_compaction: dict[str, Any],
        context: Any,
    ) -> None:
        if self.round_pipeline.last_generation_plan:
            self.budget.history[-1]["generation_plan"] = dict(self.round_pipeline.last_generation_plan)
        if offspring_verification:
            self.budget.history[-1]["offspring_verification"] = offspring_verification
        if reproduction_compaction:
            self.budget.history[-1]["reproduction_compaction"] = reproduction_compaction
        if reproduction_stop:
            self.budget.stop_reason = reproduction_stop
            return
        self._notify("post_mutation", current_round, evaluation.progress_event)

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
            synthesis.warnings.append(str(latent_override.get("reason") or "latent_problem_space_not_converged") + ":advisory_only_nonblocking")
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
            synthesis.warnings.append("continuation_requested_without_answer_completion")
            synthesis.failure_analysis = synthesis.failure_analysis or (
                "The run requested continuation before answer-first completion. "
                "The persisted checkpoint is available for continuation; no project correctness claim is made."
            )
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
        latest_ranking = self.budget.history[-1].get("ranking") if self.budget.history and isinstance(self.budget.history[-1], dict) else {}
        synthesis.closure_certificate["display_context"] = build_display_context(
            candidates=self.population.candidates,
            ranking=latest_ranking,
            contract=self.contract,
            fallback_inputs={"best_candidate_id": synthesis.best_candidate_id},
        ).to_dict()
        graded_output = _graded_output_for_final_state(population=self.population, synthesis=synthesis, final_certificate=final_certificate, latent_replay_audit=latent_replay_audit, contract=self.contract)
        synthesis.closure_certificate["graded_output"] = graded_output.to_dict()
        if graded_output.mode != "verified_result":
            synthesis.closure_certificate["graded_output_advisory"] = "verification_result_not_required_for_answer_first_completion"
        synthesis.objective_solved = bool(synthesis.closure_certificate.get("objective_solved"))
        synthesis.answer_produced = bool(synthesis.closure_certificate.get("answer_produced"))
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
            graded_output=graded_output.to_dict(),
            fabric_state=dict(self.fabric_state),
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
            fabric_state=self.fabric_state,
        )



def _graded_output_for_final_state(*, population: CandidatePopulation, synthesis: SynthesizedResult, final_certificate: dict[str, Any], latent_replay_audit: dict[str, Any], contract: Any | None = None) -> GradedOutput:
    closure = dict(synthesis.closure_certificate or {})
    closure_solved = bool(closure.get("objective_solved"))
    threshold = _verification_threshold(contract)
    selected = _selected_final_candidate(population, synthesis=synthesis, final_certificate=final_certificate)
    strongest = strongest_passed_replayable_result(selected) if selected is not None else None
    strength = candidate_verification_strength(selected) if selected is not None else VerificationStrength.NONE
    replay_certificate = _replay_certificate_for_final_state(
        synthesis=synthesis,
        final_certificate=final_certificate,
        latent_replay_audit=latent_replay_audit,
        candidate=selected,
        verification_result=strongest,
    )
    if closure_solved and strongest is not None and strength >= threshold and strongest.replayable and certificate_allows_verified_result(replay_certificate, threshold):
        result = VerifiedResult(
            answer=synthesis.final_answer,
            replayable=True,
            evidence_ref=str(strongest.evidence_ref or replay_certificate.get("evidence_bundle_hash") or ""),
            verifier_fingerprint=str(strongest.metadata.get("verifier_fingerprint") or strongest.metadata.get("fingerprint") or replay_certificate.get("verifier_fingerprint") or ""),
        )
        return GradedOutput(mode="verified_result", verification_strength=strength, result=result, replay_certificate=replay_certificate)
    direction = Direction(
        core_insight=str(synthesis.final_answer or synthesis.failure_analysis or "continue search"),
        key_assumptions=[str(item) for item in synthesis.warnings[:5]] + (["closure gate passed but no FORMAL replayable verifier result was earned"] if closure_solved else []),
        falsification_test="Freeze the referenced candidate artifact and rerun the strongest available verifier; any counterexample rules out this direction.",
        lineage=[str(getattr(selected, "id", "") or synthesis.best_candidate_id or "")],
        why_non_obvious="Returned as a protected portfolio direction because the run did not reach FORMAL replayable verification.",
    )
    return GradedOutput(mode="graded_portfolio", verification_strength=strength, portfolio=[direction], ruled_out_map=[], replay_certificate=replay_certificate)


def _selected_final_candidate(population: CandidatePopulation, *, synthesis: SynthesizedResult, final_certificate: dict[str, Any]) -> CandidateGenome | None:
    candidate_id = str(final_certificate.get("candidate_id") or synthesis.best_candidate_id or "")
    by_id = {candidate.id: candidate for candidate in population.candidates}
    if candidate_id and candidate_id in by_id:
        return by_id[candidate_id]
    display_context = synthesis.closure_certificate.get("display_context") if isinstance(synthesis.closure_certificate, dict) else {}
    if isinstance(display_context, dict) and display_context:
        selection = select_displayed_candidate(display_context, candidates=population.candidates)
        if selection.candidate_id in by_id:
            return by_id[selection.candidate_id]
    return None


def _replay_certificate_for_final_state(*, synthesis: SynthesizedResult, final_certificate: dict[str, Any], latent_replay_audit: dict[str, Any], candidate: CandidateGenome | None = None, verification_result: Any | None = None) -> dict[str, Any]:
    from cognitive_evolve_runtime.nexus._serde import stable_hash
    from cognitive_evolve_runtime.verification.cache import candidate_artifact_hash

    frozen_hash = candidate_artifact_hash(candidate) if candidate is not None else "artifact-" + stable_hash({"answer": synthesis.final_answer, "candidate_id": synthesis.best_candidate_id})[:16]
    result_payload = verification_result.to_dict() if hasattr(verification_result, "to_dict") else {}
    metadata = dict(getattr(verification_result, "metadata", {}) or {}) if verification_result is not None else {}
    measured_strength = measured_strength_from_result(verification_result)
    honesty_measurements = metadata.get("honesty_measurements") if isinstance(metadata.get("honesty_measurements"), dict) else None
    evidence_hash = "evidence-" + stable_hash({"final_certificate": final_certificate, "latent_replay_audit": latent_replay_audit, "verification_result": result_payload})[:16]
    return {
        "scope": "verifier_on_frozen_artifact_only",
        "llm_generation_replayable": False,
        "candidate_id": str(getattr(candidate, "id", "") or ""),
        "frozen_artifact_hash": frozen_hash,
        "verifier_fingerprint": str(metadata.get("verifier_fingerprint") or metadata.get("fingerprint") or ""),
        "measured_strength": measured_strength.name,
        "measured_strength_value": int(measured_strength),
        "honesty_measurements": honesty_measurements,
        "verification_cache_key": str(metadata.get("cache_key") or ""),
        "tool_versions": {},
        "evidence_bundle_hash": evidence_hash,
        "replay_command": "cogev attack --resume <out-dir> --budget <compute>  # replays verifier state on frozen artifacts only",
        "verifier_seed": 0,
    }


def _verification_threshold(contract: Any | None) -> VerificationStrength:
    metadata = getattr(contract, "metadata", {}) if contract is not None else {}
    if isinstance(metadata, dict):
        return VerificationStrength.from_value(metadata.get("verification_threshold") or VerificationStrength.FORMAL)
    return VerificationStrength.FORMAL

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
    verification_plan: VerificationPlan | dict[str, Any] | None = None,
    fabric_state: dict[str, Any] | None = None,
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
        verification_plan=verification_plan,
        fabric_state=fabric_state,
    ).run()


__all__ = ["EvolutionLoopController", "evolve_once"]
