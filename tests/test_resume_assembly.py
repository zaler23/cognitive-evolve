from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.durable.resume_assembly import restore_archives, restore_budget, restore_mode_specific, restore_population
from cognitive_evolve_runtime.persistence.checkpoint import NexusCheckpoint


def test_resume_assembly_restores_one_checkpoint_authority() -> None:
    checkpoint = NexusCheckpoint(
        round=3,
        max_rounds=5,
        population={"candidates": []},
        archives={},
        mode="project",
        budget={"adaptive": False, "branch_factor": 4},
    )
    population = CandidatePopulation()
    archives = ArchiveManager()
    restored = {
        "population": population,
        "archives": archives,
        "budget_history": [{"round": 1}],
    }

    assert restore_population(restored) is population
    assert restore_archives(restored) is archives
    assert restore_mode_specific(restored, checkpoint) == "project"
    budget = restore_budget(restored, checkpoint, max_rounds=8)
    assert budget.current_round == 3
    assert budget.max_rounds == 8
    assert budget.history == [{"round": 1}]
