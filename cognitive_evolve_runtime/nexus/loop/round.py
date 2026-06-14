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

    def __init__(self, *, model: NexusModelLike | None, budget: EvolutionBudget) -> None:
        self.model = model
        self.budget = budget
        self.rater = RelativeRater(model=model)
        self.elo = MultiHeadElo()
        self.diagnoser = SearchStateDiagnoser(model=model)
        self.updater = PolicyUpdater()
        self.selector = ParentSelector()
        self.mutation_planner = MutationPlanner()
        self.mutation_engine = MutationEngine()
        self.critique_engine = CritiqueEngine(model=model)
        self.verifier_stack = NexusVerifierStack()
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
        critiques, verification_results = self.critique_and_verify(
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
        )
        rankings = self.rank(population=population, archives=archives, policy=policy, contract=contract, current_round=current_round)
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
        blocking_obligation_ids = [
            str(item)
            for item in (policy.metadata or {}).get("blocked_or_overexplored_obligations", [])
            if item
        ]
        verification_results = self.verifier_stack.verify_population(
            population.candidates,
            contract=contract,
            blocking_obligation_ids=blocking_obligation_ids,
            current_round=current_round,
            round_limit=self.budget.round_limit,
        )
        ingest_latent_feedback(
            contract=contract,
            critiques=critiques,
            verifier_results=verification_results,
        )
        ingest_runtime_trial_feedback(contract=contract, candidates=population.candidates)
        return critiques, verification_results

    def diagnose_and_update(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> tuple[SearchDiagnosis, EvolutionPolicy]:
        diagnosis = self.diagnoser.diagnose(population=population.candidates, archives=archives, history=self.budget.history, contract=contract, policy=policy)
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
        plans = _plan_mutations(model=self.model, mutation_planner=self.mutation_planner, parents=parents, actions=actions, archives=archives, diagnosis=diagnosis, policy=policy)
        plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(plans, contract, exploration=latent_exploration_plan)
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
        )
        offspring = dedupe_offspring_against_population(offspring, population)
        if plan is not None:
            completed_stage_ops.append("generate_offspring")
            self.last_generation_plan["offspring_ids"] = [candidate.id for candidate in offspring]
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
        advisory_features = self._theory_advisory_features(policy=policy, candidates=population.candidates, current_round=current_round)
        parents = self.selector.select(population.candidates, archives, limit=limit, eligibility_policy=_eligibility_policy(policy), advisory_features=advisory_features)
        if parents:
            return parents
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

    def _theory_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        config = _theory_config_from_policy(policy)
        if not config.enabled:
            return {}
        representation = build_population_representation(candidates, cycle_id=f"round:{current_round}")
        return self.theory_layer.advisory_features_for_population(representation, config=config)

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
    ) -> list[CandidateGenome]:
        sync_repair_parent_attempts_to_dormant_archive(archives, parents)
        offspring = _generate_offspring(model=self.model, mutation_engine=self.mutation_engine, parents=parents, plans=plans, world=world, contract=contract, policy=policy)
        if len(parents) >= 2 and rankings.crossover_pairs:
            first, second = parents_for_crossover(parents, rankings.crossover_pairs[0])
            offspring.append(crossover(first, second))
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


__all__ = ["EvolutionRound", "RoundEvaluation"]
