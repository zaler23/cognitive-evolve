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

from .policy_directives import _attach_policy_directives_to_plans
from .repair_guidance import _failure_micro_guidance_for_parent, _repair_operator_for_requirement, _repair_requirement_for_parent, _repair_seed_for_parent, _source_integration_points_for_parent

def _plan_mutations(*, model: NexusModelLike | None, mutation_planner: MutationPlanner, parents: list[CandidateGenome], actions: list[str], archives: ArchiveManager, diagnosis: SearchDiagnosis, policy: EvolutionPolicy) -> list[MutationPlan]:
    fallback = mutation_planner.plan_from_actions(parents, actions, rarity_seeds=archives.rarity_archive.rare_seeds(limit=max(2, len(parents))))
    fallback = _attach_policy_directives_to_plans(fallback, policy, parents=parents)
    if isinstance(model, NexusMutationPlannerModelProtocol):
        try:
            raw = model.plan_mutations(parents=parents, actions=actions, archives=archives, diagnosis=diagnosis, policy=policy)
        except MODEL_BOUNDARY_ERRORS as exc:
            if is_quota_error(exc):
                raise
            return fallback
        model_plans = [item if isinstance(item, MutationPlan) else MutationPlan.from_dict(item) for item in raw if isinstance(item, (MutationPlan, dict))]
        if model_plans:
            merged = _attach_policy_directives_to_plans(list(model_plans), policy, parents=parents)
            if len(merged) < len(parents):
                merged.extend(fallback[len(merged):])
            return merged
    return fallback


def _parent_for_plan(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _generate_offspring(*, model: NexusModelLike | None, mutation_engine: MutationEngine, parents: list[CandidateGenome], plans: list[MutationPlan], world: Any, contract: NexusObjectiveContract, policy: EvolutionPolicy) -> list[CandidateGenome]:
    fallback = [mutation_engine.mutate(parent, plan) for parent, plan in zip(parents, plans)]
    if isinstance(model, NexusOffspringModelProtocol):
        try:
            raw = model.generate_offspring(plans=plans, parents=parents, world=world, contract=contract, policy=policy)
        except MODEL_BOUNDARY_ERRORS as exc:
            if is_quota_error(exc):
                raise
            for candidate in fallback:
                candidate.failure_lessons.append(f"model offspring generation failed; deterministic mutation fallback used: {exc}")
                candidate.metadata["model_offspring_degraded"] = True
            return fallback
        model_offspring = [item if isinstance(item, CandidateGenome) else candidate_from_dict(item) for item in raw if isinstance(item, (CandidateGenome, dict))]
        if model_offspring:
            _merge_plan_metadata_into_model_offspring(model_offspring, plans)
            known = {candidate.id for candidate in model_offspring}
            model_offspring.extend(candidate for candidate in fallback if candidate.id not in known)
            return model_offspring
    return fallback


def _merge_plan_metadata_into_model_offspring(offspring: list[CandidateGenome], plans: list[MutationPlan]) -> None:
    if not plans:
        return
    for index, candidate in enumerate(offspring):
        plan = plans[index % len(plans)]
        if not isinstance(candidate.metadata, dict):
            candidate.metadata = {}
        for key, value in (plan.metadata or {}).items():
            candidate.metadata.setdefault(key, value)


def _positive_int(value: Any) -> int | None:
    return positive_int(value)


def _best_auxiliary_id(candidates: list[CandidateGenome]) -> str:
    auxiliary = [c for c in candidates if c.current_fate == CandidateFate.AUXILIARY or c.multihead_scores.get("auxiliary_value", 0.0) > 0]
    if not auxiliary:
        return ""
    return max(auxiliary, key=lambda c: c.multihead_scores.get("auxiliary_value", 0.0)).id


__all__ = ["_best_auxiliary_id", "_generate_offspring", "_merge_plan_metadata_into_model_offspring", "_plan_mutations"]
