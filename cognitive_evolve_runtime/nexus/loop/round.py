"""Single-round ranking, diagnosis, allocation, and reproduction pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.archives.quality_diversity import candidate_bin_key
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner, ProgressiveEvaluator, apply_evidence_record, evidence_advisory_features
from cognitive_evolve_runtime.evaluators.evidence import select_preliminary_incumbent
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.honesty_control import compute_honesty_control_signal
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.nextgen import ensure_nextgen_identity, structurally_blocked
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.prompt_view import archive_prompt_view
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_outcome_signature, candidate_phenotype_signature
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import ProductiveBranchAllocation, allocate_productive_branches
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    ingest_latent_feedback,
    ingest_runtime_trial_feedback,
    latent_exploration_plan_for_contract,
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
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.strength import measured_strength_from_result
from cognitive_evolve_runtime.verification.types import VerificationPlan, VerificationResult
from cognitive_evolve_runtime.theory import TheoryLayer, build_population_representation
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig, V23TheoryRuntimeConfig
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater

from .budget import EvolutionBudget
from .offspring import _best_auxiliary_id, _generate_offspring, _plan_mutations
from .policy_directives import _attach_policy_directives_to_plans, _critique_actions
from .stage_helpers import _eligibility_policy, _theory_config_from_policy

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
        self.last_offspring_harvest_outcome: dict[str, Any] = {}

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
        evaluator_config = dict(self.adaptive.config.evaluator or {})
        if self.adaptive.config.evidence:
            evaluator_config.setdefault("evidence", dict(self.adaptive.config.evidence))
        evaluator_spec = EvaluatorSpec.from_mapping(evaluator_config)
        evaluator_led = self._evaluator_led(evaluator_spec)
        evaluator_results = self.evaluator_runner.evaluate_population_if_configured(
            population.candidates,
            spec=evaluator_spec,
            round_index=current_round,
        )
        if not evaluator_results and self.adaptive.enabled:
            progressive = ProgressiveEvaluator()
            for candidate in population.candidates:
                apply_evidence_record(candidate, progressive.evaluate_result(candidate, None, spec=evaluator_spec, round_index=current_round))
        critiques, verification_results = self.critique_and_verify(
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            use_model_critique=not evaluator_led,
        )
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
            model=None if evaluator_led else self.model,
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
        preliminary_incumbent = select_preliminary_incumbent(population.candidates)
        if preliminary_incumbent is not None:
            rankings = RelativeRater(model=None).rank(
                candidates=population.candidates,
                contract=contract,
                policy=policy,
                archives=archives,
            )
            rankings.best_final_answer_id = preliminary_incumbent.id
            rankings.raw_notes = (
                rankings.raw_notes + "; " if rankings.raw_notes else ""
            ) + "preliminary_evaluator_is_selection_authority"
        else:
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
        use_model_critique: bool = True,
    ) -> tuple[list[CandidateCritique], list[Any]]:
        critiques: list[CandidateCritique] = []
        if use_model_critique:
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
                annotate_candidate_source_bindings(candidate, project_root=getattr(archives, "project_root", None) or None)
            except Exception:
                if isinstance(candidate.metadata, dict):
                    candidate.metadata.setdefault("source_binding_manifest", {"binding_class": "no_binding", "admission_route": "repair_only", "diagnostics": ["source_binding_annotation_failed"]})
        verification_results: list[Any] = []
        verification_results.extend(self._run_synthesized_verifier(population.candidates, current_round=current_round))
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
            candidate.verification_trace = [*candidate.verification_trace, trace_item]
            measured_strength = measured_strength_from_result(result)
            if result.passed and result.replayable and measured_strength > VerificationStrength.NONE:
                current = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
                current_strength = measured_strength_from_result(VerificationResult.from_dict(current)) if current else measured_strength_from_result(None)
                if measured_strength >= current_strength:
                    candidate.verification_result = trace_item
            results.append(result)
        self.adaptive.update_verification_cache(cache)
        return results

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
        control_model = None if self._evaluator_led() else self.model
        diagnoser = SearchStateDiagnoser(model=control_model) if control_model is None else self.diagnoser
        diagnosis = diagnoser.diagnose(population=population.candidates, archives=archives, history=self.budget.history, contract=contract, policy=policy)
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
        return diagnosis, self.updater.update(policy, diagnosis, model=control_model, archives=archives)

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
        context_provider: Callable[[list[CandidateGenome], str], dict[str, Any] | None] | None = None,
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
        if not parents:
            return "no_parents_available", [], {}
        evaluator_spec = EvaluatorSpec.from_mapping(dict(self.adaptive.config.evaluator or {}))
        metric_directions = {item.name: item.direction for item in evaluator_spec.metrics}
        branch_allocation = allocate_productive_branches(
            parents=parents,
            candidates=_branch_credit_candidates(
                population=population,
                archives=archives,
                repair_parent_candidates=repair_parent_candidates,
            ),
            budget_history=self.budget.history,
            metric_directions=metric_directions,
            total_slots=self._branch_limit(),
        )
        parent_by_id = {parent.id: parent for parent in parents}
        allocated_parent_ids = list(dict.fromkeys(slot.parent_id for slot in branch_allocation.slots))
        parents = [parent_by_id[parent_id] for parent_id in allocated_parent_ids]
        if plan is not None:
            completed_stage_ops.append("select_parents")
            self.last_generation_plan["parent_ids"] = [parent.id for parent in parents]
            self.last_generation_plan["productive_branch_allocation"] = branch_allocation.to_dict()
            self._refresh_generation_plan_id()
            self._record_generation_stage_progress(completed_stage_ops)
        generation_policy = EvolutionPolicy.from_dict(policy.to_dict())
        generation_policy.metadata["productive_branch_allocation"] = branch_allocation.to_dict()
        generation_policy.metadata["requested_candidate_count"] = len(branch_allocation.slots)
        model_backed_path = self.model is not None
        evaluator_led = self._evaluator_led()
        actions = action_palette_for_round(
            current_round,
            diagnosis.recommended_actions if diagnosis.stagnation_detected else _critique_actions(critiques),
        )
        latent_exploration_plan = latent_exploration_plan_for_contract(contract, limit=self._branch_limit())
        latent_actions = [str(item) for item in latent_exploration_plan.get("mutation_actions", []) if item]
        if latent_actions:
            actions = list(dict.fromkeys(latent_actions + actions))
        round_context = provided_context
        if context_provider is not None and not model_backed_path:
            round_context = context_provider(parents, "; ".join(actions)) or provided_context
        if plan is not None:
            assert_stage_ready(plan, "plan_mutations", completed_stage_ops)
        self._sync_model_context_controls()
        if model_backed_path:
            direct_plan, latent_exploration_plan = self._direct_model_plan(
                parents=parents,
                current_round=current_round,
                branch_allocation=branch_allocation,
                actions=actions,
                diagnosis=diagnosis,
                contract=contract,
                archives=archives,
                population=population.candidates,
                latent_exploration_plan=latent_exploration_plan,
                policy=generation_policy,
                evaluator_led=evaluator_led,
            )
            plans = [direct_plan]
            if context_provider is not None:
                slot_instructions = [
                    str((slot.get("directive") or {}).get("instruction") or "")
                    for slot in direct_plan.metadata.get("branch_slots", [])
                    if isinstance(slot, dict) and isinstance(slot.get("directive"), dict)
                ]
                context_instruction = "\n\n".join(
                    item for item in [direct_plan.instruction, *slot_instructions] if item
                )
                round_context = context_provider(parents, context_instruction) or provided_context
        else:
            plans = _plan_mutations(
                model=None,
                mutation_planner=self.mutation_planner,
                parents=parents,
                actions=actions,
                archives=archives,
                diagnosis=diagnosis,
                policy=generation_policy,
                provided_context=round_context,
                target_count=len(branch_allocation.slots),
            )
            plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(plans, contract, exploration=latent_exploration_plan)
            plans = self._apply_search_pressure_to_plans(plans, parents=parents)
        unallocated_plans: list[dict[str, Any]] = []
        if not model_backed_path:
            plans = _attach_branch_allocation_to_plans(
                plans,
                branch_allocation,
                include_manifest=False,
                rejected_out=unallocated_plans,
            )
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        if not model_backed_path:
            plans = self._apply_ca_crossover_to_plans(plans, parents=parents, population=population.candidates, config=v23_config.ca_crossover)
        if plan is not None:
            completed_stage_ops.append("plan_mutations")
            self.last_generation_plan["mutation_objectives"] = list(actions)
            self.last_generation_plan["mutation_plan_count"] = len(plans)
            if unallocated_plans:
                self.last_generation_plan["unallocated_mutation_plans"] = unallocated_plans
            if model_backed_path:
                self.last_generation_plan["mutation_plan_source"] = "runtime_lineage_envelope"
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
            policy=generation_policy,
            contract=contract,
            world=world,
            rankings=rankings,
            diagnosis=diagnosis,
            provided_context=round_context,
            requested_offspring_count=len(branch_allocation.slots),
        )
        duplicate_exhausted = self.last_offspring_harvest_outcome.get("status") == "duplicate_exhausted"
        model_abstained = self.model is not None and not offspring and not duplicate_exhausted
        duplicate_offspring: list[dict[str, Any]] = []
        offspring = dedupe_offspring_against_population(offspring, population, rejected_out=duplicate_offspring)
        activation_map = _cell_activation_map(parents=parents, plans=plans, offspring=offspring)
        if activation_map:
            self.adaptive.record_cell_activation_map(activation_map, round_index=current_round, history_limit=v23_config.ca_crossover.activation_history_limit)
        canonical_metrics = _canonical_family_metrics([*population.candidates, *offspring])
        if canonical_metrics:
            self.adaptive.record_canonical_family_metrics(canonical_metrics, round_index=current_round)
        if plan is not None:
            completed_stage_ops.append("generate_offspring")
            self.last_generation_plan["offspring_ids"] = [candidate.id for candidate in offspring]
            self.last_generation_plan["offspring_harvest"] = dict(self.last_offspring_harvest_outcome)
            self.last_generation_plan["duplicate_offspring"] = duplicate_offspring
            self.last_generation_plan["cell_activation_map"] = activation_map
            self._record_generation_stage_progress(completed_stage_ops)
        if not offspring:
            if duplicate_exhausted:
                return "no_new_unique_offspring", [], {}
            if model_abstained:
                incumbent = _accepted_preliminary_incumbent(population.candidates)
                if evaluator_led and incumbent is not None:
                    if plan is not None:
                        self.last_generation_plan["offspring_outcome"] = "model_abstained_empty_batch"
                    return "", [], {}
                raise ModelResponseSchemaError(
                    "nexus_generate_offspring returned an empty batch without an accepted preliminary incumbent"
                )
            return "no_new_unique_offspring", [], {}
        for candidate in offspring:
            candidate.metadata["created_in_round"] = current_round
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

    def _evaluator_led(self, evaluator_spec: EvaluatorSpec | None = None) -> bool:
        spec = evaluator_spec or EvaluatorSpec.from_mapping(dict(self.adaptive.config.evaluator or {}))
        return bool(self.adaptive.evaluator_enabled and spec.enabled)

    def _direct_model_plan(
        self,
        *,
        parents: list[CandidateGenome],
        current_round: int,
        branch_allocation: ProductiveBranchAllocation,
        actions: list[str],
        diagnosis: SearchDiagnosis,
        contract: NexusObjectiveContract,
        archives: ArchiveManager,
        population: list[CandidateGenome],
        latent_exploration_plan: dict[str, Any],
        policy: EvolutionPolicy,
        evaluator_led: bool,
    ) -> tuple[MutationPlan, dict[str, Any]]:
        parent_ids = list(dict.fromkeys(parent.id for parent in parents if parent.id))
        plan_id = "runtime-direct-" + stable_hash(
            {
                "round": current_round,
                "parent_ids": parent_ids,
                "slot_ids": [slot.slot_id for slot in branch_allocation.slots],
            }
        )[:16]
        action_palette = list(dict.fromkeys(str(action) for action in actions if str(action).strip()))
        slot_plans = [
            MutationPlan(
                operator="ModelDirected",
                parent_ids=[slot.parent_id],
                instruction=(
                    f"Branch intent: {slot.intent}. "
                    + (
                        f"Optional semantic direction: {action_palette[index % len(action_palette)]}. "
                        if action_palette
                        else ""
                    )
                    + "Choose the concrete mutation strategy that best advances the objective; the hint is not a fixed operator."
                ),
            )
            for index, slot in enumerate(branch_allocation.slots)
        ]
        slot_plans = _attach_policy_directives_to_plans(slot_plans, policy, parents=parents)
        slot_plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(
            slot_plans,
            contract,
            exploration=latent_exploration_plan,
        )
        slot_plans = self._apply_search_pressure_to_plans(slot_plans, parents=parents)
        branch_slots: list[dict[str, Any]] = []
        for index, (slot, slot_plan) in enumerate(zip(branch_allocation.slots, slot_plans)):
            plan_metadata = dict(slot_plan.metadata or {})
            runtime_keys = (
                "search_pressure",
                "search_pressure_id",
                "target_challenge_ids",
                "artifact_policy",
                "latent_exploration_action",
                "latent_decision_trace",
                "problem_model_discrimination_action",
                "problem_model_decision_trace",
                "problem_model_snapshot_hash",
                "problem_model_ledger_cursor",
            )
            directive = {
                "instruction": slot_plan.instruction,
                "action_hint": action_palette[index % len(action_palette)] if action_palette else "",
            }
            for key in runtime_keys:
                if key in plan_metadata:
                    directive[key] = plan_metadata[key]
            policy_directives = {
                key: value
                for key, value in plan_metadata.items()
                if key not in runtime_keys
            }
            if policy_directives:
                directive["policy_directives"] = policy_directives
            branch_slots.append({**slot.to_dict(), "directive": directive})
        diagnosis_context = {
            "stagnation_detected": diagnosis.stagnation_detected,
            "stagnation_type": diagnosis.stagnation_type,
            "over_explored_families": list(diagnosis.over_explored_families),
            "under_explored_families": list(diagnosis.under_explored_families),
            "recommended_actions": list(diagnosis.recommended_actions),
            "notes": diagnosis.notes,
            "grounded_information_gain": dict(diagnosis.grounded_information_gain),
        }
        return MutationPlan(
            operator="ModelDirected",
            parent_ids=parent_ids,
            instruction=(
                "Directly evolve actual task artifacts from the allocated primary parents. "
                "For every branch slot return exactly one materially changed artifact, copy slot_id into metadata.branch_slot_id, "
                "and put slot.parent_id first in truthful parent_ids. You may use other selected parents as secondary sources. "
                "Actions and slot directives are semantic search guidance, not fixed operators; choose the mutation, transfer, "
                "representation, probe, or crossover strategy yourself. Return artifacts, not plans, commentary, or unchanged copies."
            ),
            metadata={
                "plan_id": plan_id,
                "plan_source": "runtime_lineage_envelope",
                "completion_mode": "complete_task_artifact_only" if evaluator_led else "concrete_progress_allowed",
                "semantic_mutation_owner": "model",
                "requested_candidate_count": len(branch_allocation.slots),
                "branch_slots": branch_slots,
                "action_palette": action_palette,
                "search_diagnosis": diagnosis_context,
                "latent_exploration": dict(latent_exploration_plan),
                "archive_search_memory": archive_prompt_view(archives, population=population),
            },
        ), latent_exploration_plan

    def _sync_model_context_controls(self) -> None:
        metadata = getattr(self.model, "metadata", None)
        if not isinstance(metadata, dict):
            return
        plan = self.adaptive.verification_plan_dict()
        if not plan:
            return
        metadata["prompt_context_controls"] = {"verification_plan": plan}

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
        preliminary_incumbent = select_preliminary_incumbent(population.candidates)

        def _with_incumbent(selected: list[CandidateGenome]) -> list[CandidateGenome]:
            if preliminary_incumbent is None or structurally_blocked(preliminary_incumbent):
                return selected[:limit]
            ordered = [preliminary_incumbent, *selected]
            return list({candidate.id: candidate for candidate in ordered}.values())[:limit]

        advisory_features = self._combined_advisory_features(policy=policy, candidates=population.candidates, current_round=current_round)
        parents = self.selector.select(population.candidates, archives, limit=limit, eligibility_policy=_eligibility_policy(policy), advisory_features=advisory_features)
        if parents:
            return _with_incumbent(parents)
        parents = ranked_repair_fallback_parents(population.candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
        if parents:
            return _with_incumbent(parents)
        if repair_parent_candidates:
            parents = ranked_repair_fallback_parents(repair_parent_candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
            if parents:
                return _with_incumbent(parents)
        parents = recover_repairable_dormant_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents:
            return _with_incumbent(parents)
        parents = recover_failure_archive_repair_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents or population.candidates:
            return _with_incumbent(parents)
        return _with_incumbent(emergency_activation_reseed(
            contract=contract,
            world=world,
            policy=policy,
            limit=max(1, min(2, limit)),
            current_round=current_round,
        ))

    def _combined_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for source in (
            self._theory_advisory_features(policy=policy, candidates=candidates, current_round=current_round),
            evidence_advisory_features(candidates),
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
        requested_offspring_count: int | None = None,
    ) -> list[CandidateGenome]:
        sync_repair_parent_attempts_to_dormant_archive(archives, parents)
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        harvest_outcome: dict[str, Any] = {}
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
            target_size=requested_offspring_count,
            harvest_outcome=harvest_outcome,
        )
        self.last_offspring_harvest_outcome = harvest_outcome
        if self.model is None:
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


def _accepted_preliminary_incumbent(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    incumbent = select_preliminary_incumbent(candidates)
    if incumbent is None:
        return None
    metadata = incumbent.metadata if isinstance(incumbent.metadata, dict) else {}
    evaluator = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
    status = str(evaluator.get("status") or "").strip().lower()
    if evaluator.get("passed") is True or status in {"passed", "pass", "ok", "success"}:
        return incumbent
    return None


def _parent_for_plan_id(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _attach_branch_allocation_to_plans(
    plans: list[MutationPlan],
    allocation: ProductiveBranchAllocation,
    *,
    include_manifest: bool,
    rejected_out: list[dict[str, Any]] | None = None,
) -> list[MutationPlan]:
    if not plans or not allocation.slots:
        return plans
    unused = list(allocation.slots)
    out: list[MutationPlan] = []
    for plan in plans:
        slot = next((item for item in unused if item.parent_id in plan.parent_ids), None)
        if slot is None:
            if rejected_out is not None:
                rejected_out.append(
                    {
                        "reason": "parent_slot_quota_exceeded",
                        "plan_id": str((plan.metadata or {}).get("plan_id") or ""),
                        "parent_ids": list(plan.parent_ids),
                    }
                )
            continue
        metadata = dict(plan.metadata or {})
        if not include_manifest or len(plans) > 1 or len(allocation.slots) == 1:
            unused.remove(slot)
            metadata.update(
                {
                    "branch_slot_id": slot.slot_id,
                    "branch_arm_id": slot.arm_id,
                    "branch_slot_parent_id": slot.parent_id,
                    "branch_intent": slot.intent,
                    "branch_slot_binding_status": "planned",
                }
            )
        if include_manifest and not out:
            metadata["branch_slots"] = [item.to_dict() for item in allocation.slots]
        out.append(MutationPlan.from_dict({**plan.to_dict(), "metadata": metadata}))
    return out


def _branch_credit_candidates(
    *,
    population: CandidatePopulation,
    archives: ArchiveManager,
    repair_parent_candidates: list[CandidateGenome] | None,
) -> list[CandidateGenome]:
    candidates = [*population.candidates, *(repair_parent_candidates or [])]
    direct_stores = (
        archives.answer_archive,
        archives.mechanism_archive,
        archives.novelty_archive,
        archives.project_patch_archive,
        archives.rarity_archive.candidates,
        archives.auxiliary_archive.candidates,
        archives.dormant_archive.candidates,
        archives.latent_pareto_archive.candidates,
    )
    for store in direct_stores:
        candidates.extend(candidate_from_dict(data) for data in store.values() if isinstance(data, dict) and data.get("id"))
    for store in (archives.quality_diversity.elites_by_niche, archives.quality_diversity.cell_elites):
        candidates.extend(
            candidate_from_dict(item["candidate"])
            for item in store.values()
            if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
        )
    return candidates


def _canonical_family_metrics(candidates: list[CandidateGenome]) -> dict[str, Any]:
    if not candidates:
        return {}
    family_counts: dict[str, int] = {}
    bin_keys: set[str] = set()
    descriptive_families: set[str] = set()
    descriptive_bins: set[str] = set()
    seen_phenotypes: set[str] = set()
    outcome_signatures: set[str] = set()
    novelty_term_counts: list[tuple[str, int]] = []
    migration_samples = 0
    migration_changed = 0
    for candidate in candidates:
        meta = ensure_nextgen_identity(candidate)
        family = str(meta.get("canonical_mechanism_family_id") or candidate.id)
        bin_key = candidate_bin_key(candidate)
        descriptive_families.add(family)
        descriptive_bins.add(bin_key)
        phenotype = candidate_phenotype_signature(candidate)
        duplicate_phenotype = bool(phenotype and phenotype in seen_phenotypes)
        if phenotype:
            seen_phenotypes.add(phenotype)
        migration = meta.get("canonical_mechanism_migration") if isinstance(meta.get("canonical_mechanism_migration"), dict) else {}
        if migration:
            migration_samples += 1
            if str(migration.get("from_canonical_mechanism_family_id") or "") != str(migration.get("to_canonical_mechanism_family_id") or ""):
                migration_changed += 1
        if duplicate_phenotype:
            continue
        outcome = candidate_outcome_signature(candidate)
        if outcome:
            outcome_signatures.add(outcome)
        behavior_key = outcome or phenotype or family
        family_counts[behavior_key] = family_counts.get(behavior_key, 0) + 1
        bin_keys.add(outcome or phenotype or bin_key)
        novelty_terms = {str(item) for item in [*candidate.novelty_descriptors, *candidate.niche_memberships] if str(item or "").strip()}
        novelty_term_counts.append((behavior_key, len(novelty_terms)))
    population_count = len(candidates)
    phenotype_count = sum(family_counts.values())
    probabilities = [count / max(1, phenotype_count) for count in family_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probabilities if p > 0.0)
    top_count = max(family_counts.values()) if family_counts else 0
    # Divide by the final family size so the ratio does not depend on iteration order.
    ratios = sorted(count / max(1, family_counts[family]) for family, count in novelty_term_counts)
    return {
        "population_count": population_count,
        "phenotype_count": phenotype_count,
        "exact_clone_excess": max(0, population_count - phenotype_count),
        "evaluator_outcome_signature_count": len(outcome_signatures),
        "canonical_family_entropy": round(entropy, 6),
        "max_canonical_family_share": round(top_count / max(1, phenotype_count), 6),
        "top_canonical_family_count": top_count,
        "distinct_canonical_family_count": len(family_counts),
        "canonical_family_count_to_population_count": round(len(family_counts) / max(1, population_count), 6),
        "candidate_bin_count": len(bin_keys),
        "candidate_bin_count_to_canonical_family_count": round(len(bin_keys) / max(1, len(family_counts)), 6),
        "descriptive_canonical_family_count": len(descriptive_families),
        "descriptive_candidate_bin_count": len(descriptive_bins),
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
