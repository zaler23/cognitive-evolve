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

@dataclass
class EvolutionBudget:
    max_rounds: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    branch_factor: int = 0
    initial_candidate_count: int = 0
    recover_model_errors: bool = True
    stop_policy: str = "llm_after_minimum"
    min_rounds_before_stop: int = 1
    stop_reason: str = ""
    adaptive: bool = False
    round_safety_limit: int = 0
    completion_requires_stop_signal: bool = False
    completion_status: str = "running"

    def remaining(self) -> bool:
        return self.current_round < self.round_limit

    @property
    def round_limit(self) -> int:
        if self.adaptive:
            return max(1, int(self.round_safety_limit or self.max_rounds or 1))
        return max(1, int(self.max_rounds or 1))

    def step(self) -> int:
        self.current_round += 1
        return self.current_round

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionLoopResult:
    population: CandidatePopulation
    archives: ArchiveManager
    policy: EvolutionPolicy
    diagnosis: SearchDiagnosis
    synthesis: SynthesizedResult
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    pipeline_events: list[dict[str, Any]] = field(default_factory=list)
    budget_history: list[dict[str, Any]] = field(default_factory=list)
    elo: dict[str, Any] = field(default_factory=dict)
    latent_replay_audit: dict[str, Any] = field(default_factory=dict)
    interrupted: bool = False
    error: dict[str, Any] = field(default_factory=dict)
    current_round: int = 0
    max_rounds: int = 0
    stop_reason: str = ""
    completion_status: str = "running"
    adaptive_state: dict[str, Any] = field(default_factory=dict)
    graded_output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population.to_dict(),
            "archives": self.archives.to_dict(),
            "policy": self.policy.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "progress_events": self.progress_events,
            "pipeline_events": self.pipeline_events,
            "budget_history": self.budget_history,
            "elo": self.elo,
            "latent_replay_audit": self.latent_replay_audit,
            "interrupted": self.interrupted,
            "error": self.error,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "stop_reason": self.stop_reason,
            "completion_status": self.completion_status,
            "adaptive_state": self.adaptive_state,
            "graded_output": self.graded_output,
        }


__all__ = ["EvolutionBudget", "EvolutionLoopResult"]
