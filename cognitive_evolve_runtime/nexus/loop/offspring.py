"""Nexus evolution loop skeleton with deterministic fake-model support."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import coerce_str_list, utc_now
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
from cognitive_evolve_runtime.theory import TheoryConfig, TheoryLayer, build_population_representation
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, call_with_optional_context, positive_int
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy, dedupe_plans, plan_signature
from cognitive_evolve_runtime.nexus.search_kernel.skill_library import search_skill_payload
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater

from .policy_directives import _attach_policy_directives_to_plans
from .repair_guidance import _failure_micro_guidance_for_parent, _repair_operator_for_requirement, _repair_requirement_for_parent, _repair_seed_for_parent, _source_integration_points_for_parent

def _plan_mutations(*, model: NexusModelLike | None, mutation_planner: MutationPlanner, parents: list[CandidateGenome], actions: list[str], archives: ArchiveManager, diagnosis: SearchDiagnosis, policy: EvolutionPolicy, provided_context: dict[str, Any] | None = None) -> list[MutationPlan]:
    fallback = mutation_planner.plan_from_actions(parents, actions, rarity_seeds=archives.rarity_archive.rare_seeds(limit=max(2, len(parents))))
    fallback = _attach_policy_directives_to_plans(fallback, policy, parents=parents)
    target = max(1, len(parents))
    if isinstance(model, NexusMutationPlannerModelProtocol):
        accepted: list[MutationPlan] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()
        low_gain_streak = 0
        for batch_index in range(_mutation_plan_batch_limit(target)):
            try:
                raw = call_with_optional_context(
                    model.plan_mutations,
                    parents=parents,
                    actions=actions,
                    archives=archives,
                    diagnosis=diagnosis,
                    policy=_policy_for_generation_batch(policy, batch_index=batch_index, accepted_signatures=list(seen), rejected=rejected, kind="mutation_plan"),
                    provided_context=provided_context,
                )
            except MODEL_BOUNDARY_ERRORS as exc:
                if is_quota_error(exc):
                    raise
                return fallback
            model_plans = [item if isinstance(item, MutationPlan) else MutationPlan.from_dict(item) for item in raw if isinstance(item, (MutationPlan, dict))]
            model_plans = _attach_policy_directives_to_plans(list(model_plans), policy, parents=parents)
            batch_new = 0
            for plan in model_plans:
                sig = plan_signature(plan)
                plan.metadata.setdefault("search_kernel_plan_signature", sig)
                plan.metadata["search_kernel_batch"] = batch_index
                if sig in seen:
                    rejected.append({"batch": batch_index, "reason": "duplicate_plan_signature", "signature": sig, "operator": plan.operator})
                    continue
                seen.add(sig)
                accepted.append(plan)
                batch_new += 1
            if batch_new <= 0:
                low_gain_streak += 1
            else:
                low_gain_streak = 0
            if len(accepted) >= target and batch_index + 1 >= _mutation_plan_min_batches(target):
                break
            if low_gain_streak >= _mutation_plan_low_gain_patience(target):
                break
        if accepted:
            merged = list(accepted)
            if len(merged) < target:
                fallback_deduped, _ = dedupe_plans(fallback)
                known = {plan.metadata.get("search_kernel_plan_signature") or plan_signature(plan) for plan in merged}
                for plan in fallback_deduped:
                    sig = plan.metadata.get("search_kernel_plan_signature") or plan_signature(plan)
                    if sig not in known:
                        merged.append(plan)
                        known.add(sig)
                    if len(merged) >= target:
                        break
            for plan in merged:
                plan.metadata.setdefault("search_kernel_plan_harvest", {"accepted": len(accepted), "rejected": rejected[-20:]})
            return merged[:target]
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


def _generate_offspring(
    *,
    model: NexusModelLike | None,
    mutation_engine: MutationEngine,
    parents: list[CandidateGenome],
    plans: list[MutationPlan],
    world: Any,
    contract: NexusObjectiveContract,
    policy: EvolutionPolicy,
    candidate_pool: list[CandidateGenome] | None = None,
    ca_config: CACrossoverConfig | None = None,
    provided_context: dict[str, Any] | None = None,
) -> list[CandidateGenome]:
    fallback = _deterministic_fallback_offspring(
        mutation_engine=mutation_engine,
        parents=parents,
        plans=plans,
        candidate_pool=candidate_pool or parents,
        ca_config=ca_config,
    )
    target = max(1, len(fallback))
    if isinstance(model, NexusOffspringModelProtocol):
        harvester = CandidateHarvester(
            deduper=CandidateDeduper(),
            policy=HarvestPolicy(
                target_size=target,
                max_batches=_offspring_batch_limit(target),
                min_batches=_offspring_min_batches(target),
                low_gain_patience=_offspring_low_gain_patience(target),
                relevance_floor=0.15,
                stage="offspring",
            ),
        )

        def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
            raw = call_with_optional_context(
                model.generate_offspring,
                plans=plans,
                parents=parents,
                world=world,
                contract=contract,
                policy=_policy_for_generation_batch(policy, batch_index=batch_index, accepted_signatures=[c.metadata.get("dedupe_signature", "") for c in accepted], rejected=rejected, kind="offspring"),
                provided_context=provided_context,
            )
            model_offspring = [item if isinstance(item, CandidateGenome) else candidate_from_dict(item) for item in raw if isinstance(item, (CandidateGenome, dict))]
            if model_offspring:
                _merge_plan_metadata_into_model_offspring(model_offspring, plans, parents)
            return model_offspring

        result = harvester.harvest(
            request_batch=_request,
            context={"contract": contract, "policy": policy, "world": world},
            recoverable_errors=MODEL_BOUNDARY_ERRORS,
        )
        if result.fatal_model_error is not None:
            if isinstance(result.fatal_model_error, MODEL_BOUNDARY_ERRORS) and is_quota_error(result.fatal_model_error):
                raise result.fatal_model_error
            if result.accepted:
                known = {candidate.id for candidate in result.accepted}
                offspring = list(result.accepted)
                offspring.extend(candidate for candidate in fallback if candidate.id not in known)
                for candidate in offspring:
                    candidate.metadata.setdefault("offspring_harvest", result.to_dict())
                    candidate.metadata.setdefault("partial_model_offspring_error", f"{result.fatal_model_error.__class__.__name__}: {result.fatal_model_error}")
                return offspring
            for candidate in fallback:
                candidate.failure_lessons.append(f"model offspring generation failed; deterministic mutation fallback used: {result.fatal_model_error}")
                candidate.metadata["model_offspring_degraded"] = True
            return fallback
        if result.accepted:
            known = {candidate.id for candidate in result.accepted}
            offspring = list(result.accepted)
            offspring.extend(candidate for candidate in fallback if candidate.id not in known)
            for candidate in offspring:
                candidate.metadata.setdefault("offspring_harvest", result.to_dict())
                if result.recoverable_batch_errors:
                    candidate.metadata.setdefault(
                        "partial_model_offspring_error",
                        "; ".join(f"{item.get('error_type')}: {item.get('error')}" for item in result.recoverable_batch_errors[:3]),
                    )
            return offspring
    return fallback


def _deterministic_fallback_offspring(
    *,
    mutation_engine: MutationEngine,
    parents: list[CandidateGenome],
    plans: list[MutationPlan],
    candidate_pool: list[CandidateGenome],
    ca_config: CACrossoverConfig | None,
) -> list[CandidateGenome]:
    offspring: list[CandidateGenome] = []
    for index, plan in enumerate(plans):
        parent = _parent_for_plan(plan, parents, index)
        if parent is None:
            continue
        if str(plan.operator) == MutationOperator.CROSSOVER:
            child = _deterministic_crossover_child(parent=parent, plan=plan, parents=parents, candidate_pool=candidate_pool, ca_config=ca_config)
            if child is not None:
                offspring.append(child)
                continue
        offspring.append(mutation_engine.mutate(parent, plan))
    return offspring


def _deterministic_crossover_child(
    *,
    parent: CandidateGenome,
    plan: MutationPlan,
    parents: list[CandidateGenome],
    candidate_pool: list[CandidateGenome],
    ca_config: CACrossoverConfig | None,
) -> CandidateGenome | None:
    by_id = {candidate.id: candidate for candidate in [*candidate_pool, *parents]}
    partner: CandidateGenome | None = None
    for parent_id in plan.parent_ids:
        candidate = by_id.get(parent_id)
        if candidate is not None and candidate.id != parent.id:
            partner = candidate
            break
    if partner is None:
        partner = neighborhood_crossover_partner(parent, list(by_id.values()), ca_config)
    if partner is None or partner.id == parent.id:
        return None
    child = crossover(parent, partner, instruction=plan.instruction or "descriptor-neighborhood crossover")
    if not isinstance(child.metadata, dict):
        child.metadata = {}
    child.metadata.update(dict(plan.metadata or {}))
    child.metadata["ca_crossover"] = {
        "parent_ids": [parent.id, partner.id],
        "selection": "descriptor_neighborhood_or_configured_global_donor",
        "operator": MutationOperator.CROSSOVER,
    }
    return child



def _policy_for_generation_batch(policy: EvolutionPolicy, *, batch_index: int, accepted_signatures: list[str], rejected: list[dict[str, Any]], kind: str) -> EvolutionPolicy:
    data = policy.to_dict()
    metadata = dict(data.get("metadata") or {})
    metadata.update(
        {
            f"{kind}_batch_index": batch_index,
            f"accepted_{kind}_signatures": [sig for sig in accepted_signatures[-16:] if sig],
            f"rejected_{kind}_count": len(rejected),
            f"{kind}_instruction": "Produce alternatives that land in new descriptor cells and avoid accepted signatures; do not merely paraphrase.",
            "search_kernel_skills": search_skill_payload(limit=4),
        }
    )
    data["metadata"] = metadata
    return EvolutionPolicy.from_dict(data)


def _mutation_plan_batch_limit(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_MUTATION_PLAN_BATCH_LIMIT", maximum=16)
    if configured:
        return configured
    return max(2, min(5, int(target or 1)))


def _mutation_plan_min_batches(target: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_MUTATION_PLAN_MIN_BATCHES"))
    if configured:
        return max(1, min(_mutation_plan_batch_limit(target), configured))
    return 2 if _mutation_plan_batch_limit(target) >= 2 else 1


def _mutation_plan_low_gain_patience(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_MUTATION_PLAN_LOW_GAIN_PATIENCE", maximum=8)
    if configured:
        return configured
    return 2 if target <= 4 else 3


def _offspring_batch_limit(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", maximum=16)
    if configured:
        return configured
    return max(2, min(5, int(target or 1)))


def _offspring_min_batches(target: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_OFFSPRING_MIN_BATCHES"))
    if configured:
        return max(1, min(_offspring_batch_limit(target), configured))
    return 2 if _offspring_batch_limit(target) >= 2 else 1


def _offspring_low_gain_patience(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_OFFSPRING_LOW_GAIN_PATIENCE", maximum=8)
    if configured:
        return configured
    return 2 if target <= 4 else 3


def _bounded_env_int(name: str, *, maximum: int) -> int | None:
    configured = _positive_int(os.environ.get(name))
    if configured:
        return min(maximum, configured)
    return None

def _merge_plan_metadata_into_model_offspring(offspring: list[CandidateGenome], plans: list[MutationPlan], parents: list[CandidateGenome] | None = None) -> None:
    if not plans:
        return
    parents = parents or []
    parent_by_id = {parent.id: parent for parent in parents}
    plan_by_id = {str((plan.metadata or {}).get("plan_id") or (plan.metadata or {}).get("id") or ""): plan for plan in plans if isinstance(plan.metadata, dict)}
    for candidate in offspring:
        plan = _plan_for_model_offspring(candidate, plans=plans, plan_by_id=plan_by_id)
        if not isinstance(candidate.metadata, dict):
            candidate.metadata = {}
        if plan is None:
            candidate.metadata.setdefault("plan_lineage_unresolved", True)
            continue
        for key, value in (plan.metadata or {}).items():
            candidate.metadata.setdefault(key, value)
        _merge_edge_lineage_fields(candidate, plan.metadata or {})
        lineage_parent_ids = _offspring_parent_ids(candidate, plan)
        if not candidate.parent_ids and lineage_parent_ids:
            candidate.parent_ids = list(lineage_parent_ids)
        _merge_parent_edge_lineage(candidate, [parent_by_id[item] for item in lineage_parent_ids if item in parent_by_id])


def _plan_for_model_offspring(candidate: CandidateGenome, *, plans: list[MutationPlan], plan_by_id: dict[str, MutationPlan]) -> MutationPlan | None:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for parent_ids in (
        set(coerce_str_list(getattr(candidate, "parent_ids", []))),
        set(coerce_str_list(metadata.get("parent_ids")) + coerce_str_list(metadata.get("parent_id"))),
    ):
        if not parent_ids:
            continue
        matches = [plan for plan in plans if parent_ids.intersection(set(str(item) for item in plan.parent_ids))]
        if len(matches) == 1:
            return matches[0]
    plan_id = str(metadata.get("plan_id") or metadata.get("mutation_plan_id") or "").strip()
    if plan_id and plan_id in plan_by_id:
        return plan_by_id[plan_id]
    if len(plans) == 1:
        return plans[0]
    return None


def _offspring_parent_ids(candidate: CandidateGenome, plan: MutationPlan) -> list[str]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    ids = coerce_str_list(getattr(candidate, "parent_ids", [])) or coerce_str_list(metadata.get("parent_ids")) or coerce_str_list(metadata.get("parent_id")) or coerce_str_list(plan.parent_ids)
    return list(dict.fromkeys(str(item) for item in ids if str(item).strip()))


def _merge_parent_edge_lineage(candidate: CandidateGenome, parents: list[CandidateGenome]) -> None:
    if not parents:
        return
    for parent in parents:
        _merge_edge_lineage_fields(candidate, parent)


def _merge_edge_lineage_fields(candidate: CandidateGenome, source: Any) -> None:
    for attr in ("edge_knowledge_seeds", "inherited_genes", "novelty_descriptors", "niche_memberships"):
        current = list(getattr(candidate, attr, []) or [])
        merged = list(current)
        values = source.get(attr) if isinstance(source, dict) else getattr(source, attr, [])
        for item in coerce_str_list(values):
            if item not in merged:
                merged.append(item)
        setattr(candidate, attr, merged)


def _positive_int(value: Any) -> int | None:
    return positive_int(value)


def _best_auxiliary_id(candidates: list[CandidateGenome]) -> str:
    auxiliary = [c for c in candidates if c.current_fate == CandidateFate.AUXILIARY or c.multihead_scores.get("auxiliary_value", 0.0) > 0]
    if not auxiliary:
        return ""
    return max(auxiliary, key=lambda c: c.multihead_scores.get("auxiliary_value", 0.0)).id


__all__ = ["_best_auxiliary_id", "_generate_offspring", "_merge_plan_metadata_into_model_offspring", "_plan_mutations"]
