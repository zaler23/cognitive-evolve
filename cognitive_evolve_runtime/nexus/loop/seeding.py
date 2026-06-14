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
        for candidate in population.candidates:
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
    accepted: list[CandidateGenome] = []
    rejected: list[dict[str, Any]] = []
    deduper = CandidateDeduper()
    max_batches = _seed_batch_limit(target_size)
    low_novelty_streak = 0
    model_error: Exception | None = None
    for batch_index in range(max_batches):
        try:
            raw = model.seed_population(contract=contract, world=world, policy=_policy_for_seed_batch(policy, batch_index=batch_index, accepted=accepted, rejected=rejected))
        except MODEL_BOUNDARY_ERRORS as exc:
            model_error = exc
            break
        batch = _coerce_seed_batch(raw)
        before = len(accepted)
        for candidate in batch:
            candidate.metadata.setdefault("exploration_source", "nexus_model_seed_batch")
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata["model_seed_batch"] = batch_index
            if deduper.add(candidate):
                accepted.append(candidate)
            else:
                rejected.append({"batch": batch_index, "reason": "duplicate_semantic_signature", "candidate_id": candidate.id, "signature": candidate.metadata.get("dedupe_signature")})
        novelty_gain = len(accepted) - before
        if len(accepted) >= target_size:
            break
        if novelty_gain <= 0:
            low_novelty_streak += 1
        else:
            low_novelty_streak = 0
        if low_novelty_streak >= _seed_low_novelty_patience(target_size):
            break
    return accepted, rejected, model_error


def _coerce_seed_batch(raw: Any) -> list[CandidateGenome]:
    if isinstance(raw, CandidatePopulation):
        return list(raw.candidates)
    if isinstance(raw, list):
        return [item if isinstance(item, CandidateGenome) else candidate_from_dict(item) for item in raw if isinstance(item, (CandidateGenome, dict))]
    return []


def _policy_for_seed_batch(policy: EvolutionPolicy, *, batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> EvolutionPolicy:
    data = policy.to_dict()
    metadata = dict(data.get("metadata") or {})
    metadata.update(
        {
            "seed_batch_index": batch_index,
            "accepted_seed_signatures": [candidate.metadata.get("dedupe_signature") for candidate in accepted[-12:] if candidate.metadata.get("dedupe_signature")],
            "rejected_seed_count": len(rejected),
            "seed_instruction": "Generate candidates that differ semantically from accepted_seed_signatures; do not rephrase the same mechanism.",
        }
    )
    data["metadata"] = metadata
    return EvolutionPolicy.from_dict(data)


def _seed_batch_limit(target_size: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_BATCH_LIMIT"))
    if configured:
        return configured
    return max(2, min(8, int(target_size or 1)))


def _seed_low_novelty_patience(target_size: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE"))
    if configured:
        return configured
    return 2 if target_size <= 4 else 3


__all__ = ["TEXT_SEED_TYPES", "PROJECT_SEED_TYPES", "seed_population"]
