"""Budget-derived controls for Nexus evolution rounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.persistence.checkpoint_profile import (
    apply_checkpoint_profile_to_archives,
    apply_checkpoint_profile_to_history,
    apply_checkpoint_profile_to_population,
    checkpoint_profile_from_env,
)

SEARCH_PHASES = frozenset({"explore", "exit_sweep"})


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
    search_phase: str = "explore"

    def __post_init__(self) -> None:
        self.search_phase = str(self.search_phase or "explore").strip().lower()
        if self.search_phase not in SEARCH_PHASES:
            raise ValueError(f"search_phase must be one of: {', '.join(sorted(SEARCH_PHASES))}")

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
    fabric_state: dict[str, Any] = field(default_factory=dict)
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    search_phase: str = "explore"

    def to_dict(self) -> dict[str, Any]:
        profile = checkpoint_profile_from_env()
        population_payload = apply_checkpoint_profile_to_population(self.population.to_dict(), profile)
        archive_payload = apply_checkpoint_profile_to_archives(self.archives.to_dict(), population_payload, profile)
        budget_history_payload = apply_checkpoint_profile_to_history(self.budget_history, profile)
        policy_metadata = self.policy.metadata if isinstance(getattr(self.policy, "metadata", None), dict) else {}
        search_kernel_summary = {
            key: policy_metadata[key]
            for key in (
                "seed_coverage",
                "target_perturb_seed_judgment",
                "minimal_core_ablation",
                "seed_active_frontier",
                "algorithm_efficiency",
                "model_parallel_efficiency",
                "seed_reservoir_ref",
            )
            if key in policy_metadata
        }
        return {
            "population": population_payload,
            "archives": archive_payload,
            "policy": self.policy.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "progress_events": self.progress_events,
            "pipeline_events": self.pipeline_events,
            "budget_history": budget_history_payload,
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
            "fabric_state": self.fabric_state,
            "cost_ledger": self.cost_ledger,
            "search_phase": self.search_phase,
            "search_kernel_summary": search_kernel_summary,
        }


__all__ = ["EvolutionBudget", "EvolutionLoopResult", "SEARCH_PHASES"]
