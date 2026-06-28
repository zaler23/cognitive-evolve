"""Nexus evolution loop skeleton with deterministic fake-model support."""
from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.archives.quality_diversity import candidate_bin_key
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner, ProgressiveEvaluator, apply_evidence_record, evidence_advisory_features
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import utc_now
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round, amplify_population
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.honesty_control import compute_honesty_control_signal
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.nextgen import ensure_nextgen_identity
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta, candidate_source_bindings
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol, NexusSeedModelProtocol, NexusStopModelProtocol
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings
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
from cognitive_evolve_runtime.verification.cache import check_with_cache
from cognitive_evolve_runtime.verification.factory import verifier_from_plan
from cognitive_evolve_runtime.verification.information_gain import population_information_gain_report
from cognitive_evolve_runtime.verification.obligation_runner import run_obligations_for_population
from cognitive_evolve_runtime.verification.strength import measured_strength_from_result
from cognitive_evolve_runtime.verification.types import VerificationPlan, VerificationResult
from cognitive_evolve_runtime.theory import TheoryConfig, TheoryLayer, build_population_representation
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, positive_int
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig, V23TheoryRuntimeConfig
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater

from .budget import EvolutionBudget
from .offspring import _best_auxiliary_id, _generate_offspring, _plan_mutations
from .policy_directives import _attach_policy_directives_to_plans, _critique_actions
from .stage_helpers import _eligibility_policy, _error_progress_event, _theory_config_from_policy

@dataclass
class RoundEvaluation:
    rankings: RelativeRankingResult
    policy: EvolutionPolicy
    diagnosis: SearchDiagnosis
    critiques: list[CandidateCritique]
    verification_results: list[Any]
    progress_event: dict[str, Any]
    pipeline_event: dict[str, Any]
    stop_reason: str
    population_compaction: dict[str, Any] = field(default_factory=dict)
    repair_parent_candidates: list[CandidateGenome] = field(default_factory=list)
    generation_plan: dict[str, Any] = field(default_factory=dict)


class EvolutionRound:
    """Single-round stage pipeline for rank → critique → diagnose → reproduce.

    ``evolve_once`` owns lifecycle concerns (budget, cancellation, checkpoint
    notification, final synthesis).  This class owns the testable round stages so
    adding a new stage no longer requires editing one giant try-block.
    """

    def __init__(self, *, model: NexusModelLike | None, budget: EvolutionBudget, adaptive: AdaptiveRuntimeController | None = None) -> None:
        self.model = model
        self.budget = budget
        self.adaptive = adaptive or AdaptiveRuntimeController.from_sources()
        self.rater = RelativeRater(model=model)
        self.elo = MultiHeadElo()
        self.diagnoser = SearchStateDiagnoser(model=model)
        self.updater = PolicyUpdater()
        self.selector = ParentSelector()
        self.mutation_planner = MutationPlanner()
        self.mutation_engine = MutationEngine()
        self.critique_engine = CritiqueEngine(model=model)
        self.evaluator_runner = ExternalEvaluatorRunner()
        self.stop_decider = StopDecisionEngine()
        self.theory_layer = TheoryLayer()
        self.last_generation_plan: dict[str, Any] = {}
        self.last_completed_stage_ops: list[str] = []

    def evaluate(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> RoundEvaluation:
        self.adaptive.begin_round(round_index=current_round)
        self._sync_model_context_controls()
        self.adaptive.observe_population(population=population, round_index=current_round)
        critiques, verification_results = self.critique_and_verify(
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
        )
        evaluator_config = dict(self.adaptive.config.evaluator or {})
        if self.adaptive.config.evidence:
            evaluator_config.setdefault("evidence", dict(self.adaptive.config.evidence))
        evaluator_spec = EvaluatorSpec.from_mapping(evaluator_config)
        evaluator_results = self.evaluator_runner.evaluate_population_if_configured(
            population.candidates,
            spec=evaluator_spec,
            round_index=current_round,
        )
        if not evaluator_results and self.adaptive.enabled:
            progressive = ProgressiveEvaluator()
            for candidate in population.candidates:
                apply_evidence_record(candidate, progressive.evaluate_result(candidate, None, spec=evaluator_spec, round_index=current_round))
        if self.adaptive.enabled:
            if evaluator_results:
                passed = len([item for item in evaluator_results if item.passed])
                evaluated = len(evaluator_results)
            else:
                evidence_items = [candidate.metadata.get("evidence_state") for candidate in population.candidates if isinstance(candidate.metadata, dict) and candidate.metadata.get("evidence_state")]
                passed = len([item for item in evidence_items if isinstance(item, dict) and not item.get("final_blocked")])
                evaluated = len(evidence_items)
            self.adaptive.record_evaluator_summary(
                round_index=current_round,
                evaluated=evaluated,
                passed=passed,
                failed=max(0, evaluated - passed),
                candidates=population.candidates,
            )
        self._apply_archive_directives(archives=archives, population=population)
        rankings = self.rank(population=population, archives=archives, policy=policy, contract=contract, current_round=current_round)
        self.adaptive.observe_population(population=population, round_index=current_round)
        plan = GenerationPlan.from_dict(self.last_generation_plan)
        completed_stage_ops = list(self.last_completed_stage_ops)
        repair_parent_candidates = list(population.candidates)
        assert_stage_ready(plan, "compact", completed_stage_ops)
        ranking_compaction = compact_live_population(
            population,
            archives,
            policy,
            branch_factor=self.budget.branch_factor,
            round_index=current_round,
        )
        completed_stage_ops.append("compact")
        assert_stage_ready(plan, "diagnose", completed_stage_ops)
        diagnosis, updated_policy = self.diagnose_and_update(population=population, archives=archives, policy=policy, contract=contract)
        completed_stage_ops.append("diagnose")
        best_aux = _best_auxiliary_id(population.candidates)
        best_answer = rankings.best_final_answer_id
        assert_stage_ready(plan, "stop_check", completed_stage_ops)
        stop_reason = self.stop_decider.stop_reason_after_round(
            budget=self.budget,
            completed_round=current_round,
            diagnosis=diagnosis,
            best_answer_id=best_answer,
            population=population,
            model=self.model,
            contract=contract,
        )
        completed_stage_ops.append("stop_check")
        generation_plan = dict(self.last_generation_plan)
        generation_plan["completed_stage_ops"] = list(completed_stage_ops)
        self.last_generation_plan = dict(generation_plan)
        self.last_completed_stage_ops = list(completed_stage_ops)
        progress_event = EvolutionProgressEvent(
            round=current_round,
            max_rounds=self.budget.round_limit,
            population_size=len(population.candidates),
            active_candidates=len([c for c in population.candidates if CandidateFate.normalize(c.current_fate) == CandidateFate.ACTIVE.value]),
            dormant_candidates=len(archives.dormant_archive.candidates),
            archive_elites=len(archives.answer_archive),
            tool_calls=sum(len(c.tool_results) for c in population.candidates),
            best_answer_candidate=best_answer,
            best_auxiliary_candidate=best_aux,
            search_diagnosis=diagnosis.stagnation_type,
            next_action=stop_reason or (diagnosis.recommended_actions[0] if diagnosis.recommended_actions else "continue"),
            metadata={
                "stop_policy": self.budget.stop_policy,
                "stop_reason": stop_reason,
                "adaptive": self.budget.adaptive,
                "round_safety_limit": self.budget.round_limit if self.budget.adaptive else 0,
                "completion_requires_stop_signal": self.budget.completion_requires_stop_signal,
                "progress_semantics": "open_ended_safety_checkpoint" if self.budget.adaptive else "fixed_round_budget",
                "current_round": current_round,
                "budget_current_round": self.budget.current_round,
                "round_limit": self.budget.round_limit,
                "generation_plan_id": generation_plan.get("plan_id", ""),
                "incubating_candidates": len([c for c in population.candidates if CandidateFate.normalize(c.current_fate) == CandidateFate.INCUBATING.value]),
                "population_vitality": vitality_snapshot(population.candidates, branch_factor=self.budget.branch_factor).to_dict(),
                "adaptive_features": dict(self.adaptive.state.enabled_features),
            },
        ).to_dict()
        stage_count = self.budget.round_limit
        pipeline_event = PipelineProgressEvent(
            stage="candidate_population",
            stage_index=min(current_round, stage_count),
            stage_count=stage_count,
            stage_progress=0.0 if self.budget.adaptive else current_round / max(1, stage_count),
            metadata={
                "adaptive": self.budget.adaptive,
                "progress_semantics": "open_ended_no_percent_complete" if self.budget.adaptive else "fixed_percent_complete",
                "current_round": current_round,
                "round_limit": self.budget.round_limit,
            },
        ).to_dict()
        return RoundEvaluation(
            rankings=rankings,
            policy=updated_policy,
            diagnosis=diagnosis,
            critiques=critiques,
            verification_results=verification_results,
            progress_event=progress_event,
            pipeline_event=pipeline_event,
            stop_reason=stop_reason,
            population_compaction=ranking_compaction.to_dict(),
            repair_parent_candidates=repair_parent_candidates,
            generation_plan=generation_plan,
        )

    def rank(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        current_round: int,
    ) -> RelativeRankingResult:
        self.last_generation_plan = {}
        self.last_completed_stage_ops = []
        rankings = self.rater.rank(candidates=population.candidates, contract=contract, policy=policy, archives=archives)
        self.elo.update_from_relative(rankings)
        self.elo.apply_to_candidates(population.candidates, axes=list(policy.fitness_axes or []))
        latent_ranking_summary = annotate_candidates_with_latent_signals(population.candidates, contract)
        assignments = archives.assign_by_policy(
            population.candidates,
            rankings,
            current_round=current_round,
            round_limit=self.budget.round_limit,
            branch_factor=self.budget.branch_factor,
            eligibility_policy=_eligibility_policy(policy),
        )
        generation_plan = build_generation_plan(
            round_index=current_round,
            candidates=population.candidates,
            fate_assignments=assignments,
            ranking=rankings,
            stage_graph=[
                {"op": "critique_and_verify"},
                {"op": "rank"},
                {"op": "archive_assign"},
                {"op": "generation_plan_validate"},
            {"op": "archive_update"},
            {"op": "compact"},
            {"op": "diagnose"},
            {"op": "stop_check"},
            {"op": "select_parents"},
            {"op": "plan_mutations"},
            {"op": "generate_offspring"},
            {"op": "verify_offspring"},
        ],
            source="runtime_rank_archive_transition",
        )
        if latent_ranking_summary:
            generation_plan.ranking_summary["latent_ranking"] = latent_ranking_summary
            object.__setattr__(generation_plan, "plan_id", expected_generation_plan_id(generation_plan))
        apply_generation_plan(generation_plan, population.candidates, archives)
        self.last_generation_plan = generation_plan.to_dict()
        self.last_completed_stage_ops = ["critique_and_verify", "rank", "archive_assign", "generation_plan_validate", "archive_update"]
        return rankings

    def critique_and_verify(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> tuple[list[CandidateCritique], list[Any]]:
        critiques = self.critique_engine.critique(
            candidates=population.candidates,
            round_index=current_round,
            contract=contract,
            policy=policy,
            archives=archives,
        )
        self.critique_engine.apply(candidates=population.candidates, critiques=critiques)
        for candidate in population.candidates:
            try:
                annotate_candidate_source_bindings(candidate)
            except Exception:
                if isinstance(candidate.metadata, dict):
                    candidate.metadata.setdefault("source_binding_manifest", {"binding_class": "no_binding", "admission_route": "repair_only", "diagnostics": ["source_binding_annotation_failed"]})
        verification_results: list[Any] = []
        verification_results.extend(self._run_synthesized_verifier(population.candidates, current_round=current_round))
        verification_results.extend(self._run_verification_obligations(population.candidates, current_round=current_round, policy=policy, contract=contract))
        ingest_latent_feedback(
            contract=contract,
            critiques=critiques,
            verifier_results=verification_results,
        )
        ingest_runtime_trial_feedback(contract=contract, candidates=population.candidates)
        return critiques, verification_results

    def _run_synthesized_verifier(self, candidates: list[CandidateGenome], *, current_round: int) -> list[Any]:
        plan_data = self.adaptive.verification_plan_dict()
        if not plan_data:
            return []
        plan = VerificationPlan.from_dict(plan_data)
        verifier = verifier_from_plan(plan)
        if verifier is None:
            return []
        cache = self.adaptive.verification_cache()
        results: list[Any] = []
        viable = [
            candidate
            for candidate in candidates
            if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
        ]
        max_checks = max(1, min(len(viable), self._branch_limit() if self.budget.adaptive else max(self._branch_limit(), 4)))
        for candidate in viable[:max_checks]:
            result, cache_key, cache_hit = check_with_cache(candidate, verifier, cache)
            trace_item = result.to_dict()
            trace_item.setdefault("metadata", {})
            if isinstance(trace_item["metadata"], dict):
                trace_item["metadata"].update({"round_index": current_round, "cache_key": cache_key, "cache_hit": cache_hit})
            candidate.verification_trace = [*candidate.verification_trace, trace_item][-100:]
            if result.passed and result.replayable:
                current = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
                current_strength = measured_strength_from_result(VerificationResult.from_dict(current)) if current else measured_strength_from_result(None)
                if measured_strength_from_result(result) >= current_strength:
                    candidate.verification_result = trace_item
            results.append(result)
        self.adaptive.update_verification_cache(cache)
        return results

    def _run_verification_obligations(self, candidates: list[CandidateGenome], *, current_round: int, policy: EvolutionPolicy, contract: NexusObjectiveContract) -> list[Any]:
        obligations = self.adaptive.verification_obligation_features()
        if not obligations:
            return []
        cache = self.adaptive.verification_cache()
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(candidates))
        records = run_obligations_for_population(candidates, obligations, cache=cache, max_checks=max(1, self._branch_limit()), policy=policy, v23_config=v23_config)
        self.adaptive.update_verification_cache(cache)
        for record in records:
            obligation = record.get("obligation") if isinstance(record.get("obligation"), dict) else {}
            self.adaptive.record_effect_application(
                channel="verification_obligations",
                item=obligation,
                changed=bool(record.get("changed")),
                consumer="verification.obligation_runner",
                reason=str(record.get("reason") or ""),
                result=record,
                consume=bool(record.get("changed")),
            )
        return records

    def diagnose_and_update(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> tuple[SearchDiagnosis, EvolutionPolicy]:
        gain_report = population_information_gain_report(population.candidates, self.budget.history)
        policy.metadata.setdefault("engine_grounded_information_gain", gain_report)
        policy.metadata["engine_grounded_information_gain"] = gain_report
        diagnosis = self.diagnoser.diagnose(population=population.candidates, archives=archives, history=self.budget.history, contract=contract, policy=policy)
        diagnosis.grounded_information_gain = gain_report
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        signal = compute_honesty_control_signal(
            candidates=population.candidates,
            config=v23_config.honesty_control,
            history=self.adaptive.state.honesty_error_history,
        )
        diagnosis.metadata["honesty_control"] = signal.to_dict()
        diagnosis.metadata["v23_theory_config_hash"] = v23_config.config_hash
        if v23_config.diagnostics:
            diagnosis.metadata["v23_theory_config_diagnostics"] = list(v23_config.diagnostics)
        self.adaptive.record_honesty_control_signal(signal, history_limit=v23_config.honesty_control.history_limit)
        return diagnosis, self.updater.update(policy, diagnosis, model=self.model, archives=archives)

    def reproduce(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        critiques: list[CandidateCritique],
        offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None,
        repair_parent_candidates: list[CandidateGenome] | None = None,
        provided_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        plan = GenerationPlan.from_dict(self.last_generation_plan) if self.last_generation_plan else None
        completed_stage_ops = list(self.last_completed_stage_ops or self.last_generation_plan.get("completed_stage_ops") or [])
        if plan is not None:
            assert_stage_ready(plan, "select_parents", completed_stage_ops)
        parents = self._select_reproduction_parents(
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            rankings=rankings,
            diagnosis=diagnosis,
            repair_parent_candidates=repair_parent_candidates,
        )
        if plan is not None:
            completed_stage_ops.append("select_parents")
            self.last_generation_plan["parent_ids"] = [parent.id for parent in parents]
            self._refresh_generation_plan_id()
            self._record_generation_stage_progress(completed_stage_ops)
        if not parents:
            return "no_parents_available", [], {}
        actions = action_palette_for_round(
            current_round,
            diagnosis.recommended_actions if diagnosis.stagnation_detected else _critique_actions(critiques),
        )
        latent_exploration_plan = latent_exploration_plan_for_contract(contract, limit=self._branch_limit())
        latent_actions = [str(item) for item in latent_exploration_plan.get("mutation_actions", []) if item]
        if latent_actions:
            actions = list(dict.fromkeys(latent_actions + actions))
        if plan is not None:
            assert_stage_ready(plan, "plan_mutations", completed_stage_ops)
        self._sync_model_context_controls()
        plans = _plan_mutations(model=self.model, mutation_planner=self.mutation_planner, parents=parents, actions=actions, archives=archives, diagnosis=diagnosis, policy=policy, provided_context=provided_context)
        plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(plans, contract, exploration=latent_exploration_plan)
        plans = self._apply_search_pressure_to_plans(plans, parents=parents)
        plans = self._apply_candidate_transforms_to_plans(plans, parents=parents)
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        plans = self._apply_ca_crossover_to_plans(plans, parents=parents, population=population.candidates, config=v23_config.ca_crossover)
        if plan is not None:
            completed_stage_ops.append("plan_mutations")
            self.last_generation_plan["mutation_objectives"] = list(actions)
            self.last_generation_plan["mutation_plan_count"] = len(plans)
            if latent_exploration_plan:
                self.last_generation_plan["latent_exploration_planning"] = latent_exploration_plan
            self._refresh_generation_plan_id()
            self._record_generation_stage_progress(completed_stage_ops)
            assert_stage_ready(plan, "generate_offspring", completed_stage_ops)
        offspring = self._build_reproduction_offspring(
            current_round=current_round,
            parents=parents,
            plans=plans,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            rankings=rankings,
            diagnosis=diagnosis,
            provided_context=provided_context,
        )
        offspring = dedupe_offspring_against_population(offspring, population)
        activation_map = _cell_activation_map(parents=parents, plans=plans, offspring=offspring)
        if activation_map:
            self.adaptive.record_cell_activation_map(activation_map, round_index=current_round, history_limit=v23_config.ca_crossover.activation_history_limit)
        canonical_metrics = _canonical_family_metrics([*population.candidates, *offspring])
        if canonical_metrics:
            self.adaptive.record_canonical_family_metrics(canonical_metrics, round_index=current_round)
        if plan is not None:
            completed_stage_ops.append("generate_offspring")
            self.last_generation_plan["offspring_ids"] = [candidate.id for candidate in offspring]
            self.last_generation_plan["cell_activation_map"] = activation_map
            self._record_generation_stage_progress(completed_stage_ops)
        if not offspring:
            return "no_new_unique_offspring", [], {}
        for candidate in offspring:
            candidate.metadata.setdefault("created_in_round", current_round)
        if plan is not None:
            assert_stage_ready(plan, "verify_offspring", completed_stage_ops)
        return self._verify_and_integrate_offspring(
            offspring=offspring,
            offspring_verifier=offspring_verifier,
            population=population,
            archives=archives,
            policy=policy,
            current_round=current_round,
            generation_plan=plan,
            completed_stage_ops=completed_stage_ops,
        )

    def _record_generation_stage_progress(self, completed_stage_ops: list[str]) -> None:
        self.last_completed_stage_ops = list(completed_stage_ops)
        if self.last_generation_plan:
            self.last_generation_plan["completed_stage_ops"] = list(completed_stage_ops)

    def _refresh_generation_plan_id(self) -> None:
        if not self.last_generation_plan:
            return
        plan = GenerationPlan.from_dict(self.last_generation_plan)
        self.last_generation_plan["plan_id"] = expected_generation_plan_id(plan)

    def _branch_limit(self) -> int:
        return max(2, int(self.budget.branch_factor or 2))

    def _sync_model_context_controls(self) -> None:
        metadata = getattr(self.model, "metadata", None)
        if not isinstance(metadata, dict):
            return
        transforms = self.adaptive.context_transform_features()
        plan = self.adaptive.verification_plan_dict()
        obligations = self.adaptive.verification_obligation_features()
        if not transforms and not plan and not obligations:
            return
        protect_refs: list[str] = []
        drop_refs: list[str] = []
        view_hash_parts: list[str] = []
        known_protect = {"problem_spec", "verification_plan", "honesty_invariant", "contract", "world", "policy", "prompt_contract"}
        known_drop = {"drop:history", "drop:archive_elites", "drop:failure_lessons"}
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            protect_refs.extend(str(item) for item in transform.get("protect_refs", []) if item) if isinstance(transform.get("protect_refs"), list) else None
            drop_refs.extend(str(item) for item in transform.get("drop_refs", []) if item) if isinstance(transform.get("drop_refs"), list) else None
            if transform.get("view_hash"):
                view_hash_parts.append(str(transform.get("view_hash")))
        controls = {
            "protect_refs": list(dict.fromkeys([item for item in protect_refs if item in known_protect])),
            "drop_refs": list(dict.fromkeys([item for item in drop_refs if item in known_drop])),
            "view_hash": ":".join(view_hash_parts),
            "verification_plan": plan,
        }
        regime = [_prompt_verification_regime_item(item) for item in obligations if isinstance(item, dict)]
        regime = [item for item in regime if item]
        if regime:
            controls["verification_regime"] = regime
        before = dict(metadata.get("prompt_context_controls") or {}) if isinstance(metadata.get("prompt_context_controls"), dict) else {}
        metadata["prompt_context_controls"] = controls
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            mapped = bool(set(transform.get("protect_refs", []) or []) & known_protect or set(transform.get("drop_refs", []) or []) & known_drop)
            self.adaptive.record_effect_application(
                channel="context_transforms",
                item=transform,
                changed=mapped and before != controls,
                consumer="StructuredModelAdapterCore.prompt_context_controls",
                reason="context_transform_controls_injected" if mapped else "context_transform_unknown_refs",
                result={"controls": controls},
                consume=False,
            )

    def _apply_archive_directives(self, *, archives: ArchiveManager, population: CandidatePopulation) -> None:
        if not self.adaptive.enabled:
            return
        directives = self.adaptive.archive_directive_features()
        if not directives:
            return
        for directive in directives:
            if self.adaptive.effect_consumed("archive_directives", directive):
                continue
            records = archives.apply_archive_directives([directive], population.candidates)
            result = records[0] if records else {"changed": False, "reason": "archive_directive_no_result"}
            changed = bool(result.get("changed"))
            self.adaptive.record_effect_application(
                channel="archive_directives",
                item=directive,
                changed=changed,
                consumer="ArchiveManager.apply_archive_directives",
                reason=str(result.get("reason") or ""),
                result=result,
                consume=changed,
            )

    def _apply_budget_directives_to_parents(self, parents: list[CandidateGenome], *, population: CandidatePopulation, limit: int) -> list[CandidateGenome]:
        if not parents or not self.adaptive.enabled:
            return parents
        directives = self.adaptive.budget_directive_features()
        if not directives:
            return parents
        active_mode = str(getattr(self.adaptive.config.research, "mode", "observe") or "observe").lower() == "active"
        viable_by_id = {candidate.id: candidate for candidate in population.candidates if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}}
        candidates = sorted(
            [dict(item) for item in directives if isinstance(item, dict) and not self.adaptive.effect_consumed("budget_directives", item)],
            key=lambda item: (float(item.get("roi_estimate") or 0.0), float(item.get("weight") or 0.0), str(item.get("target") or "")),
            reverse=True,
        )
        selected = list(parents)[: max(0, int(limit or len(parents)))]
        selected_ids = {candidate.id for candidate in selected}
        for directive in candidates[:1]:
            target_id = str(directive.get("target") or "")
            target = viable_by_id.get(target_id)
            if target is None:
                self.adaptive.record_effect_application(channel="budget_directives", item=directive, changed=False, consumer="EvolutionRound._apply_budget_directives_to_parents", reason="budget_target_not_viable", consume=False)
                continue
            if not active_mode:
                self.adaptive.record_effect_application(channel="budget_directives", item=directive, changed=False, consumer="EvolutionRound._apply_budget_directives_to_parents", reason="budget_directive_advisory_only", consume=False)
                continue
            before = [candidate.id for candidate in selected]
            if target.id not in selected_ids:
                if len(selected) < max(1, int(limit or len(selected))):
                    selected.append(target)
                elif selected:
                    selected[-1] = target
                else:
                    selected = [target]
            after = [candidate.id for candidate in selected]
            changed = before != after
            self.adaptive.record_effect_application(
                channel="budget_directives",
                item=directive,
                changed=changed,
                consumer="EvolutionRound._apply_budget_directives_to_parents",
                reason="budget_parent_slot_reserved" if changed else "budget_target_already_selected",
                result={"before_parent_ids": before, "after_parent_ids": after},
                consume=changed,
            )
            break
        return selected[: max(1, int(limit or len(selected)))]

    def _select_reproduction_parents(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        repair_parent_candidates: list[CandidateGenome] | None,
    ) -> list[CandidateGenome]:
        limit = self._branch_limit()
        advisory_features = self._combined_advisory_features(policy=policy, candidates=population.candidates, current_round=current_round)
        parents = self.selector.select(population.candidates, archives, limit=limit, eligibility_policy=_eligibility_policy(policy), advisory_features=advisory_features)
        if parents:
            return self._apply_budget_directives_to_parents(parents, population=population, limit=limit)
        parents = ranked_repair_fallback_parents(population.candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
        if parents:
            return parents
        if repair_parent_candidates:
            parents = ranked_repair_fallback_parents(repair_parent_candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
            if parents:
                return parents
        parents = recover_repairable_dormant_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents:
            return parents
        parents = recover_failure_archive_repair_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents or population.candidates:
            return parents
        return emergency_activation_reseed(
            contract=contract,
            world=world,
            policy=policy,
            limit=max(1, min(2, limit)),
            current_round=current_round,
        )

    def _combined_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for source in (
            self._theory_advisory_features(policy=policy, candidates=candidates, current_round=current_round),
            evidence_advisory_features(candidates),
            self.adaptive.selection_advisory_features(candidates=candidates, policy=policy),
        ):
            for candidate_id, feature in (source or {}).items():
                current = dict(combined.get(candidate_id) or {})
                data = dict(feature) if isinstance(feature, dict) else {
                    "rank_prior": getattr(feature, "rank_prior", 0.0),
                    "plan_value": getattr(feature, "plan_value", 0.0),
                    "diversity": getattr(feature, "diversity", 0.0),
                    "risk": getattr(feature, "risk", 0.0),
                }
                for key in ("rank_prior", "plan_value", "diversity"):
                    current[key] = max(float(current.get(key, 0.0) or 0.0), float(data.get(key, 0.0) or 0.0))
                current["risk"] = max(float(current.get("risk", 0.0) or 0.0), float(data.get("risk", 0.0) or 0.0))
                combined[str(candidate_id)] = current
        return combined

    def _theory_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        config = _theory_config_from_policy(policy)
        if not config.enabled:
            return {}
        representation = build_population_representation(candidates, cycle_id=f"round:{current_round}")
        return self.theory_layer.advisory_features_for_population(representation, config=config)

    def _apply_candidate_transforms_to_plans(self, plans: list[MutationPlan], *, parents: list[CandidateGenome]) -> list[MutationPlan]:
        if not plans or not parents or not self.adaptive.enabled:
            return plans
        transforms = self.adaptive.candidate_transform_features()
        if not transforms:
            return plans
        by_parent = {parent.id: parent for parent in parents}
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for transform in transforms:
            if not isinstance(transform, dict) or self.adaptive.effect_consumed("candidate_transforms", transform):
                continue
            by_candidate.setdefault(str(transform.get("candidate_id") or ""), []).append(transform)
        out: list[MutationPlan] = []
        applied_keys: set[str] = set()
        for index, plan in enumerate(plans):
            parent = None
            for parent_id in plan.parent_ids:
                parent = by_parent.get(parent_id)
                if parent is not None:
                    break
            if parent is None:
                parent = parents[index % len(parents)]
            selected = by_candidate.get(parent.id, [])
            if not selected:
                out.append(plan)
                continue
            metadata = dict(plan.metadata or {})
            applied: list[dict[str, Any]] = []
            instruction = plan.instruction
            for transform in selected:
                transform_key = effect_key("candidate_transforms", transform)
                if transform_key in applied_keys or self.adaptive.effect_consumed("candidate_transforms", transform, key=transform_key):
                    continue
                kind = str(transform.get("kind") or "")
                payload = transform.get("payload") if isinstance(transform.get("payload"), dict) else {}
                if kind == "collapse_params" and not payload.get("parameter_slots"):
                    self.adaptive.record_effect_application(channel="candidate_transforms", item=transform, changed=False, consumer="EvolutionRound._apply_candidate_transforms_to_plans", reason="collapse_params_missing_parameter_slots", consume=False)
                    continue
                applied.append(dict(transform))
                applied_keys.add(transform_key)
                if kind == "collapse_params":
                    instruction = (instruction.rstrip() + "\n\nApply the explicit parameter_slots assignment from candidate_transform.collapse_params; remove parameter_space after freezing concrete values.").strip()
            if applied:
                metadata["candidate_transforms"] = applied
                out.append(MutationPlan.from_dict({**plan.to_dict(), "instruction": instruction, "metadata": metadata}))
                for transform in applied:
                    self.adaptive.record_effect_application(channel="candidate_transforms", item=transform, changed=True, consumer="EvolutionRound._apply_candidate_transforms_to_plans", reason="candidate_transform_attached_to_mutation_plan", result={"parent_id": parent.id}, consume=True)
            else:
                out.append(plan)
        return out

    def _apply_search_pressure_to_plans(self, plans: list[MutationPlan], *, parents: list[CandidateGenome]) -> list[MutationPlan]:
        if not plans or not parents or not self.adaptive.enabled:
            return plans
        out: list[MutationPlan] = []
        by_id = {parent.id: parent for parent in parents}
        for index, plan in enumerate(plans):
            parent = None
            for parent_id in plan.parent_ids:
                parent = by_id.get(parent_id)
                if parent is not None:
                    break
            if parent is None:
                parent = parents[index % len(parents)]
            pressure = self.adaptive.compile_search_pressure(parent_id=parent.id, scope="candidate", parent=parent, candidates=parents)
            if pressure is None or not _search_pressure_has_effect(pressure):
                out.append(plan)
                continue
            metadata = dict(plan.metadata or {})
            metadata["search_pressure"] = pressure.to_dict()
            metadata["search_pressure_id"] = pressure.id
            if pressure.target_challenge_ids:
                metadata["target_challenge_ids"] = list(pressure.target_challenge_ids)
            metadata["artifact_policy"] = dict(pressure.artifact_requirements or {})
            instruction = plan.instruction
            if pressure.mutation_instruction and pressure.mutation_instruction not in instruction:
                instruction = (instruction.rstrip() + "\n\n" + pressure.mutation_instruction).strip()
            out.append(MutationPlan.from_dict({**plan.to_dict(), "instruction": instruction, "metadata": metadata}))
        return out

    def _apply_ca_crossover_to_plans(
        self,
        plans: list[MutationPlan],
        *,
        parents: list[CandidateGenome],
        population: list[CandidateGenome],
        config: CACrossoverConfig,
    ) -> list[MutationPlan]:
        if not plans or not parents:
            return plans
        by_id = {candidate.id: candidate for candidate in [*population, *parents]}
        out: list[MutationPlan] = []
        for index, plan in enumerate(plans):
            if str(plan.operator) != MutationOperator.CROSSOVER:
                out.append(plan)
                continue
            pivot = _parent_for_plan_id(plan, parents, index)
            if pivot is None:
                out.append(plan)
                continue
            partner = None
            for parent_id in plan.parent_ids:
                candidate = by_id.get(parent_id)
                if candidate is not None and candidate.id != pivot.id:
                    partner = candidate
                    break
            if partner is None:
                partner = neighborhood_crossover_partner(pivot, list(by_id.values()), config)
            if partner is None:
                out.append(plan)
                continue
            metadata = dict(plan.metadata or {})
            metadata["ca_crossover"] = {
                "pivot_id": pivot.id,
                "partner_id": partner.id,
                "selection": "descriptor_neighborhood_or_configured_global_donor",
            }
            parent_ids = list(dict.fromkeys([pivot.id, partner.id]))
            out.append(MutationPlan.from_dict({**plan.to_dict(), "parent_ids": parent_ids, "metadata": metadata}))
        return out

    def _build_reproduction_offspring(
        self,
        *,
        current_round: int,
        parents: list[CandidateGenome],
        plans: list[MutationPlan],
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        provided_context: dict[str, Any] | None = None,
    ) -> list[CandidateGenome]:
        sync_repair_parent_attempts_to_dormant_archive(archives, parents)
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        offspring = _generate_offspring(
            model=self.model,
            mutation_engine=self.mutation_engine,
            parents=parents,
            plans=plans,
            world=world,
            contract=contract,
            policy=policy,
            candidate_pool=population.candidates,
            ca_config=v23_config.ca_crossover,
            provided_context=provided_context,
        )
        for child in offspring:
            metadata = child.metadata if isinstance(child.metadata, dict) else {}
            targets = [str(item) for item in metadata.get("target_challenge_ids", []) if item] if isinstance(metadata.get("target_challenge_ids"), list) else []
            if targets:
                self.adaptive.record_generated_targets(candidate_id=child.id, challenge_ids=targets, pressure_id=str(metadata.get("search_pressure_id") or ""), round_index=current_round)
        if len(parents) >= 2 and rankings.crossover_pairs:
            first, second = parents_for_crossover(parents, rankings.crossover_pairs[0])
            ranked_child = crossover(first, second)
            ranked_child.metadata["ca_crossover"] = {
                "parent_ids": [first.id, second.id],
                "selection": "ranking_pair",
                "operator": MutationOperator.CROSSOVER,
            }
            offspring.append(ranked_child)
        offspring.extend(elite_gap_merge_offspring(population.candidates, archives=archives, policy=policy, branch_factor=self.budget.branch_factor))
        reactivated = archives.reactivate_dormant() if "reactivate_dormant" in diagnosis.recommended_actions else None
        if reactivated:
            reactivated.metadata["reactivated_in_round"] = current_round
            reactivated.metadata.setdefault("created_in_round", current_round)
            offspring.append(reactivated)
        return offspring

    def _verify_and_integrate_offspring(
        self,
        *,
        offspring: list[CandidateGenome],
        offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        current_round: int,
        generation_plan: GenerationPlan | None = None,
        completed_stage_ops: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        offspring_verification = verify_offspring(offspring, offspring_verifier)
        reproduction_archive_updates: list[dict[str, Any]] = []
        if offspring_verification:
            failed_ids = {item.get("candidate_id") for item in offspring_verification if item.get("passed") is False}
            failed = [
                candidate
                for candidate in offspring
                if candidate.id in failed_ids and CandidateFate.normalize(candidate.current_fate) == CandidateFate.FAILED.value
            ]
            if failed:
                failed_assignments = archives.update({candidate.id: CandidateFate.FAILED for candidate in failed}, candidates=failed)
                reproduction_archive_updates.extend(
                    {
                        "candidate_id": assignment.candidate_id,
                        "fate": assignment.fate,
                        "source": "verify_offspring",
                    }
                    for assignment in failed_assignments
                )
        population.integrate(offspring)
        reproduction_compaction = compact_live_population(
            population,
            archives,
            policy,
            branch_factor=self.budget.branch_factor,
            round_index=current_round,
        )
        if generation_plan is not None:
            completed = list(completed_stage_ops or [])
            completed.append("verify_offspring")
            self.last_generation_plan["offspring_verification_count"] = len(offspring_verification)
            self.last_generation_plan["reproduction_archive_updates"] = reproduction_archive_updates
            self._record_generation_stage_progress(completed)
        return "", offspring_verification, reproduction_compaction.to_dict()


def _search_pressure_has_effect(pressure: Any) -> bool:
    return bool(
        getattr(pressure, "target_challenge_ids", None)
        or getattr(pressure, "avoid_challenge_ids", None)
        or getattr(pressure, "artifact_requirements", None)
        or getattr(pressure, "success_criteria", None)
        or str(getattr(pressure, "mutation_instruction", "") or "").strip()
    )


def _parent_for_plan_id(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _canonical_family_metrics(candidates: list[CandidateGenome]) -> dict[str, Any]:
    if not candidates:
        return {}
    family_counts: dict[str, int] = {}
    bin_keys: set[str] = set()
    novelty_ratios: list[float] = []
    migration_samples = 0
    migration_changed = 0
    for candidate in candidates:
        meta = ensure_nextgen_identity(candidate)
        family = str(meta.get("canonical_mechanism_family_id") or candidate.id)
        family_counts[family] = family_counts.get(family, 0) + 1
        bin_keys.add(candidate_bin_key(candidate))
        novelty_terms = {str(item) for item in [*candidate.novelty_descriptors, *candidate.niche_memberships] if str(item or "").strip()}
        novelty_ratios.append(len(novelty_terms) / max(1, family_counts[family]))
        migration = meta.get("canonical_mechanism_migration") if isinstance(meta.get("canonical_mechanism_migration"), dict) else {}
        if migration:
            migration_samples += 1
            if str(migration.get("from_canonical_mechanism_family_id") or "") != str(migration.get("to_canonical_mechanism_family_id") or ""):
                migration_changed += 1
    total = sum(family_counts.values())
    probabilities = [count / max(1, total) for count in family_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probabilities if p > 0.0)
    top_count = max(family_counts.values()) if family_counts else 0
    ratios = sorted(novelty_ratios)
    return {
        "population_count": total,
        "canonical_family_entropy": round(entropy, 6),
        "max_canonical_family_share": round(top_count / max(1, total), 6),
        "top_canonical_family_count": top_count,
        "distinct_canonical_family_count": len(family_counts),
        "canonical_family_count_to_population_count": round(len(family_counts) / max(1, total), 6),
        "candidate_bin_count": len(bin_keys),
        "candidate_bin_count_to_canonical_family_count": round(len(bin_keys) / max(1, len(family_counts)), 6),
        "same_declared_changed_canonical_share": round(migration_changed / max(1, migration_samples), 6),
        "same_declared_changed_canonical_sample_count": migration_samples,
        "novelty_to_canonical_family_ratio_p50": round(_percentile(ratios, 0.50), 6),
        "novelty_to_canonical_family_ratio_p95": round(_percentile(ratios, 0.95), 6),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(float(quantile or 0.0) * len(values)) - 1))
    return float(values[index])


def _cell_activation_map(*, parents: list[CandidateGenome], plans: list[MutationPlan], offspring: list[CandidateGenome]) -> dict[str, Any]:
    activation: dict[str, dict[str, Any]] = {}

    def _entry(cell: str) -> dict[str, Any]:
        return activation.setdefault(cell, {"parent_ids": [], "offspring_ids": [], "operators": []})

    for parent in parents:
        cell = candidate_bin_key(parent)
        entry = _entry(cell)
        if parent.id not in entry["parent_ids"]:
            entry["parent_ids"].append(parent.id)
    for plan in plans:
        operator = str(plan.operator or "")
        parent_ids = [str(item) for item in plan.parent_ids if item]
        for parent in parents:
            if parent_ids and parent.id not in parent_ids:
                continue
            entry = _entry(candidate_bin_key(parent))
            if operator and operator not in entry["operators"]:
                entry["operators"].append(operator)
    for child in offspring:
        cell = candidate_bin_key(child)
        entry = _entry(cell)
        if child.id not in entry["offspring_ids"]:
            entry["offspring_ids"].append(child.id)
        for operator in getattr(child, "mutation_history", []) or []:
            op = str(operator or "")
            if op and op not in entry["operators"]:
                entry["operators"].append(op)
    return {cell: entry for cell, entry in activation.items() if entry.get("parent_ids") or entry.get("offspring_ids")}


def _prompt_verification_regime_item(obligation: dict[str, Any]) -> dict[str, Any]:
    """Return a model-facing obligation view without certification shortcuts."""

    allowed = {
        "id",
        "origin",
        "must_pass",
        "exogeneity_probe",
        "variety_probe",
        "falsification_budget",
        "replay_record",
    }
    out = {key: obligation.get(key) for key in allowed if key in obligation}
    for forbidden in ("strength_contribution", "replayable", "strength", "strength_value", "measured_strength"):
        out.pop(forbidden, None)
    return out


__all__ = ["EvolutionRound", "RoundEvaluation"]
