"""Pure checkpoint assembly helpers shared by Nexus resume entrypoints."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.budget_factory import resume_evolution_budget
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.persistence.checkpoint import NexusCheckpoint


def restore_population(restored: dict[str, Any]) -> CandidatePopulation:
    population = restored["population"]
    if not isinstance(population, CandidatePopulation):
        raise TypeError("checkpoint population was not hydrated")
    return population


def restore_archives(restored: dict[str, Any]) -> ArchiveManager:
    archives = restored["archives"]
    if not isinstance(archives, ArchiveManager):
        raise TypeError("checkpoint archives were not hydrated")
    return archives


def restore_mode_specific(restored: dict[str, Any], checkpoint: NexusCheckpoint) -> str:
    return str(restored.get("mode") or checkpoint.mode or "text")


def restore_budget(
    restored: dict[str, Any],
    checkpoint: NexusCheckpoint,
    *,
    max_rounds: int | None,
) -> EvolutionBudget:
    budget = resume_evolution_budget(
        checkpoint_round=checkpoint.round,
        checkpoint_max_rounds=checkpoint.max_rounds,
        budget_data=dict(checkpoint.budget or {}),
        max_rounds=max_rounds,
    )
    budget.history = [dict(item) for item in restored.get("budget_history", []) if isinstance(item, dict)]
    return budget


__all__ = ["restore_archives", "restore_budget", "restore_mode_specific", "restore_population"]
