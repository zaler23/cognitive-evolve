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

def _error_progress_event(previous_event: dict[str, Any], current_round: int) -> dict[str, Any]:
    """Return progress metadata that is safe to store inside an error checkpoint."""

    event = dict(previous_event or {})
    previous_round = event.get("round")
    event["round"] = int(current_round or 0)
    event.setdefault("type", "evolution_progress")
    metadata = dict(event.get("metadata") or {})
    try:
        previous_round_int = int(previous_round or 0)
    except (TypeError, ValueError):
        previous_round_int = -1
    if previous_round is not None and previous_round_int != int(current_round or 0):
        metadata["previous_progress_round"] = previous_round
        metadata["error_checkpoint_round"] = int(current_round or 0)
        metadata["round_reconciled_for_error_checkpoint"] = True
    event["metadata"] = metadata
    return event


def _eligibility_policy(policy: EvolutionPolicy | None) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("eligibility_policy") or metadata.get("stage_policy")
    return dict(raw) if isinstance(raw, dict) else {}


def _theory_config_from_policy(policy: EvolutionPolicy | None) -> TheoryConfig:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    if not isinstance(metadata, dict):
        return TheoryConfig()
    raw = metadata.get("theory") or metadata.get("theory_config") or {}
    return TheoryConfig.from_mapping(raw if isinstance(raw, dict) else {})


def _raise_if_cancelled(cancellation_callback: Callable[[], bool] | None) -> None:
    if cancellation_callback is not None and cancellation_callback():
        raise InterruptedError("nexus evolution cancellation requested")


def _notify_observer(
    observer: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    round_index: int,
    population: CandidatePopulation,
    archives: ArchiveManager,
    policy: EvolutionPolicy,
    diagnosis: SearchDiagnosis,
    progress_event: dict[str, Any],
    budget_history: list[dict[str, Any]],
    error: dict[str, Any] | None = None,
    adaptive_state: dict[str, Any] | None = None,
    fabric_state: dict[str, Any] | None = None,
) -> None:
    if observer is None:
        return
    observer(
        {
            "phase": phase,
            "round": round_index,
            "population": population,
            "archives": archives,
            "policy": policy,
            "diagnosis": diagnosis,
            "progress_event": progress_event,
            "budget_history": list(budget_history),
            "error": error,
            "adaptive_state": dict(adaptive_state or {}),
            "fabric": dict(fabric_state or {}),
        }
    )


__all__ = ["_eligibility_policy", "_error_progress_event", "_notify_observer", "_raise_if_cancelled", "_theory_config_from_policy"]
