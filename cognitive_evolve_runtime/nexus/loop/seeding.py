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
from cognitive_evolve_runtime.nexus.factor_resurrection import resurrect_factor_trace
from cognitive_evolve_runtime.nexus.minimal_core import apply_seed_active_frontier, run_core_ablation
from cognitive_evolve_runtime.nexus.seed_coverage import SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY, assess_seed_coverage, seed_reservoir_sidecar_payload
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy
from cognitive_evolve_runtime.nexus.search_kernel.skill_library import search_skill_payload
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
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, positive_int as _positive_int
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater

TEXT_SEED_TYPES = [
    "Direct Solver Seed",
    "Known Pattern Seed",
    "Edge Knowledge Seed",
    "Analogy Seed",
    "Inversion Seed",
    "Decomposition Seed",
    "Tool-Grounded Seed",
    "Wildcard Seed",
]

PROJECT_SEED_TYPES = [
    "Minimal Patch Seed",
    "Architecture Refactor Seed",
    "Test-First Seed",
    "Compatibility-Preserving Seed",
    "Internal Forgotten Pattern Seed",
]

def seed_population(
    *,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    model: NexusModelLike | None = None,
    min_population_size: int | None = None,
) -> CandidatePopulation:
    model_error: Exception | None = None
    model_candidates: list[CandidateGenome] = []
    rejected_model_seeds: list[dict[str, Any]] = []
    target_size = _seed_target_size(policy=policy, world=world, requested_minimum=min_population_size)
    if isinstance(model, NexusSeedModelProtocol):
        model_candidates, rejected_model_seeds, model_error = _generate_model_seed_batches(
            model=model,
            contract=contract,
            world=world,
            policy=policy,
            target_size=target_size,
        )
    candidates: list[CandidateGenome] = list(model_candidates)
    population = amplify_population(
        population=CandidatePopulation(candidates),
        contract=contract,
        world=world,
        policy=policy,
        minimum_size=target_size,
    )
    if model_error is not None:
        accepted_model_ids = {candidate.id for candidate in model_candidates}
        for candidate in population.candidates:
            if accepted_model_ids and candidate.id in accepted_model_ids:
                continue
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata.setdefault("model_seed_error", f"{model_error.__class__.__name__}: {model_error}")
            candidate.failure_lessons.append("model seed generation failed before completion; resume from checkpoint when provider quota recovers")
    if rejected_model_seeds:
        for candidate in population.candidates:
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata.setdefault("model_seed_rejections", list(rejected_model_seeds[:10]))
    for candidate in population.candidates:
        candidate.metadata.setdefault("created_in_round", 0)
    return population


def _seed_target_size(*, policy: EvolutionPolicy, world: Any, requested_minimum: int | None) -> int:
    if requested_minimum and requested_minimum > 0:
        return int(requested_minimum)
    configured = _positive_int((policy.metadata or {}).get("initial_candidate_count"))
    if configured:
        return configured
    niche_count = len({str(item).strip().lower() for item in policy.candidate_niches if str(item).strip()})
    template_count = len(PROJECT_SEED_TYPES if getattr(world, "kind", "text") == "project" else TEXT_SEED_TYPES)
    return max(1, niche_count or template_count)


def _generate_model_seed_batches(
    *,
    model: NexusSeedModelProtocol,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    target_size: int,
) -> tuple[list[CandidateGenome], list[dict[str, Any]], Exception | None]:
    deduper = CandidateDeduper()
    harvester = CandidateHarvester(
        deduper=deduper,
        policy=HarvestPolicy(
            target_size=target_size,
            max_batches=_seed_safety_batch_limit(policy=policy),
            min_batches=_seed_min_batches(policy=policy),
            low_gain_patience=_seed_low_novelty_patience(policy=policy),
            relevance_floor=0.20,
            stage="seed",
            fanout_workers=_seed_fanout_workers(policy=policy, target_size=target_size),
            stop_at_target=False,
            exhaust_on_no_new=True,
            reservoir_mode=True,
        ),
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        raw = model.seed_population(contract=contract, world=world, policy=_policy_for_seed_batch(policy, batch_index=batch_index, accepted=accepted, rejected=rejected))
        batch = _coerce_seed_batch(raw)
        priority = _seed_family_priority(policy, accepted)
        for candidate in batch:
            candidate.metadata.setdefault("exploration_source", "nexus_model_seed_batch")
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata["model_seed_batch"] = batch_index
            candidate.metadata.setdefault("seed_family_priority_trace", {"batch_index": batch_index, "source": priority.get("source"), "requested_families": [item.get("id") or item.get("name") for item in list(priority.get("families") or [])[:4]]})
        return batch

    result = harvester.harvest(
        request_batch=_request,
        context={"contract": contract, "policy": policy, "world": world},
        recoverable_errors=MODEL_BOUNDARY_ERRORS,
    )
    coverage = assess_seed_coverage(
        result.accepted,
        reservoir=result.reservoir,
        rejected=result.rejected,
        harvest_summary=result.to_dict(),
        contract=contract,
        policy=policy,
    )
    if isinstance(policy.metadata, dict):
        frontier = apply_seed_active_frontier(result.accepted, limit=_seed_active_frontier_limit(policy=policy))
        ablation = run_core_ablation(result.accepted, policy=policy)
        factors = resurrect_factor_trace([*result.accepted, *result.reservoir], limit=16)
        policy.metadata["seed_harvest"] = result.to_dict()
        policy.metadata["seed_coverage"] = coverage
        policy.metadata["seed_active_frontier"] = frontier
        policy.metadata["minimal_core_ablation"] = ablation
        policy.metadata["factor_resurrection_summary"] = {"factor_count": len(factors), "factors": factors[:8], "policy": "advisory_seed_pool_factor_trace"}
        if result.reservoir:
            policy.metadata[SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY] = seed_reservoir_sidecar_payload(result.reservoir)
        policy.metadata["algorithm_efficiency"] = {
            "seed_batches": result.batches,
            "accepted_per_batch": round(len(result.accepted) / max(1, result.batches), 4),
            "reservoir_count": len(result.reservoir),
            "partial_failure_count": len(result.failed_batch_ids),
            "active_frontier_size": len(frontier.get("selected_ids") or []),
            "dormant_seed_reserve_count": int(frontier.get("dormant_count") or 0),
            "policy": "measure_only_no_capability_tradeoff",
        }
        policy.metadata["model_parallel_efficiency"] = {
            "seed_fanout_workers": _seed_fanout_workers(policy=policy, target_size=target_size),
            "max_batches": _seed_safety_batch_limit(policy=policy),
            "policy": "parallelism_observed_not_seed_breadth_reduced",
        }
    for candidate in result.accepted:
        candidate.metadata.setdefault("seed_harvest", _candidate_seed_harvest_trace(result, candidate))
        candidate.metadata.setdefault("seed_coverage", _candidate_seed_coverage_trace(coverage))
        candidate.metadata.setdefault("minimal_core_ablation_profile", ablation.get("recommendation", "advisory"))
        if result.reservoir:
            candidate.metadata.setdefault(
                "seed_reservoir",
                {
                    "mode": "soft_reject_retention",
                    "candidate_ids": [item.id for item in result.reservoir[-100:]],
                    "count": len(result.reservoir),
                    "checkpoint_policy": "coverage_summary_plus_candidate_ids",
                },
            )
    return result.accepted, result.rejected, result.fatal_model_error


def _candidate_seed_harvest_trace(result: Any, candidate: CandidateGenome) -> dict[str, Any]:
    return {
        "schema": "seed_harvest_candidate_trace.v1",
        "stage": str(getattr(result, "stage", "") or "seed"),
        "candidate_id": candidate.id,
        "batch": int((candidate.metadata or {}).get("model_seed_batch") or (candidate.metadata or {}).get("search_kernel_batch") or 0),
        "batches": int(getattr(result, "batches", 0) or 0),
        "accepted_count": len(getattr(result, "accepted", []) or []),
        "rejected_count": len(getattr(result, "rejected", []) or []),
        "reservoir_count": len(getattr(result, "reservoir", []) or []),
        "stopped_reason": str(getattr(result, "stopped_reason", "") or ""),
        "failed_batch_ids": list(getattr(result, "failed_batch_ids", []) or []),
        "partial_failure_count": len(getattr(result, "failed_batch_ids", []) or []),
        "fatal_model_error": f"{result.fatal_model_error.__class__.__name__}: {result.fatal_model_error}" if getattr(result, "fatal_model_error", None) else "",
        "policy": "per_candidate_compact_trace_full_harvest_in_policy_metadata",
    }


def _candidate_seed_coverage_trace(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "seed_coverage_candidate_trace.v1",
        "status": coverage.get("status") or coverage.get("coverage_status") or "",
        "coverage_status": coverage.get("coverage_status") or coverage.get("status") or "",
        "candidate_count": coverage.get("candidate_count"),
        "family_count": coverage.get("family_count"),
        "singleton_family_count": coverage.get("singleton_family_count"),
        "top1_family_share": coverage.get("top1_family_share"),
        "top3_family_share": coverage.get("top3_family_share"),
        "fingerprint": coverage.get("fingerprint"),
        "needs_more_seed": coverage.get("needs_more_seed"),
        "needs_target_perturb": coverage.get("needs_target_perturb"),
        "policy": "compact_candidate_trace_full_coverage_in_policy_metadata",
    }


def _seed_active_frontier_limit(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(metadata.get("seed_active_frontier_size") or metadata.get("active_frontier_size") or metadata.get("seed_active_evaluation_budget"))
    if configured:
        return configured
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_ACTIVE_FRONTIER_SIZE"))
    if configured:
        return configured
    return 64


def _coerce_seed_batch(raw: Any) -> list[CandidateGenome]:
    if isinstance(raw, CandidatePopulation):
        return list(raw.candidates)
    if isinstance(raw, list):
        return [item if isinstance(item, CandidateGenome) else candidate_from_dict(item) for item in raw if isinstance(item, (CandidateGenome, dict))]
    return []


def _seed_family_priority(policy: EvolutionPolicy, accepted: list[CandidateGenome]) -> dict[str, Any]:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    plan = metadata.get("search_space_plan") if isinstance(metadata.get("search_space_plan"), dict) else {}
    raw_families = plan.get("candidate_families") or plan.get("exploration_planes") or plan.get("planes") or []
    families = [dict(item) for item in raw_families if isinstance(item, dict) and (item.get("id") or item.get("name"))]
    source = str(plan.get("source") or metadata.get("seed.family_priority_source") or metadata.get("seed_family_priority_source") or "model_authored_search_space")
    counts: dict[str, int] = {}
    for candidate in accepted:
        candidate_metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
        search_space = candidate_metadata.get("search_space") if isinstance(candidate_metadata, dict) else {}
        if isinstance(search_space, dict):
            family_id = str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()
            if family_id:
                counts[family_id] = counts.get(family_id, 0) + 1
    prioritized = []
    for family in families:
        family_id = str(family.get("id") or family.get("name") or "").strip()
        if not family_id:
            continue
        item = dict(family)
        item["accepted_count"] = counts.get(family_id, 0)
        item["priority_reason"] = "undercovered_model_authored_family" if counts.get(family_id, 0) == 0 else "covered_family_soft_followup"
        prioritized.append(item)
    prioritized.sort(key=lambda item: (int(item.get("accepted_count") or 0), str(item.get("id") or "")))
    if not prioritized:
        source = "objective_placeholder"
    return {"source": source, "families": prioritized[:8], "coverage": counts}


def _policy_for_seed_batch(policy: EvolutionPolicy, *, batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> EvolutionPolicy:
    data = policy.to_dict()
    metadata = dict(data.get("metadata") or {})
    family_priority = _seed_family_priority(policy, accepted)
    metadata.update(
        {
            "seed_batch_index": batch_index,
            "accepted_seed_signatures": [candidate.metadata.get("dedupe_signature") for candidate in accepted[-12:] if candidate.metadata.get("dedupe_signature")],
            "rejected_seed_count": len(rejected),
            "seed_instruction": "Generate candidates that differ semantically from accepted_seed_signatures and prioritize undercovered seed_family_priority entries when present; do not rephrase the same mechanism.",
            "seed_family_priority": list(family_priority.get("families") or []),
            "seed_family_priority_source": str(family_priority.get("source") or "objective_placeholder"),
            "seed_family_coverage_snapshot": dict(family_priority.get("coverage") or {}),
            "search_kernel_skills": search_skill_payload(limit=4),
        }
    )
    data["metadata"] = metadata
    return EvolutionPolicy.from_dict(data)


SEED_BATCH_DEFAULT_MAX = 8
_UNBOUNDED_SEED_LIMIT_VALUES = {"0", "none", "no_limit", "unbounded", "until_exhausted"}


def _seed_safety_batch_limit(*, policy: EvolutionPolicy) -> int | None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    raw_configured = (
        metadata.get("seed_safety_max_batches")
        or metadata.get("seed_harvest_safety_max_batches")
        or metadata.get("seed_max_batches")
    )
    if _seed_limit_is_unbounded(raw_configured):
        return None
    configured = _positive_int(
        raw_configured
    )
    if configured:
        return configured
    raw_env = os.environ.get("COGEV_NEXUS_SEED_BATCH_LIMIT")
    if _seed_limit_is_unbounded(raw_env):
        return None
    configured = _positive_int(raw_env)
    if configured:
        return configured
    return SEED_BATCH_DEFAULT_MAX


def _seed_limit_is_unbounded(value: Any) -> bool:
    return str(value or "").strip().lower() in _UNBOUNDED_SEED_LIMIT_VALUES


def _seed_min_batches(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(metadata.get("seed_min_batches") or metadata.get("seed_min_batches_before_exhaustion"))
    if configured:
        max_batches = _seed_safety_batch_limit(policy=policy)
        return max(1, configured if max_batches is None else min(max_batches, configured))
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_MIN_BATCHES"))
    if configured:
        max_batches = _seed_safety_batch_limit(policy=policy)
        return max(1, configured if max_batches is None else min(max_batches, configured))
    return 1

def _seed_low_novelty_patience(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(
        metadata.get("seed_no_new_patience")
        or metadata.get("seed_low_novelty_patience")
        or metadata.get("seed_exhaustion_patience")
    )
    if configured:
        return configured
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE"))
    if configured:
        return configured
    return 1


def _seed_fanout_workers(*, policy: EvolutionPolicy, target_size: int) -> int | None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(
        metadata.get("seed_fanout_concurrency")
        or metadata.get("seed_batch_concurrency")
        or metadata.get("seed_harvest_fanout_workers")
    )
    if configured:
        return configured
    # No seed-specific override: follow the shared model fanout governor.
    # Concurrent seed prompts intentionally share a previous-window snapshot of
    # accepted signatures; the post-fanout harvester remains the serial
    # dedupe/merge authority for deterministic acceptance.
    return None


__all__ = ["TEXT_SEED_TYPES", "PROJECT_SEED_TYPES", "seed_population"]
