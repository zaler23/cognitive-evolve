"""Shared state records for one Nexus evolution round."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.critique import CandidateCritique
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


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


__all__ = ["RoundEvaluation"]
