from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome


def test_culled_edge_candidate_is_kept_as_failure_lesson_not_reproduction_seed() -> None:
    candidate = CandidateGenome(
        id="edge",
        edge_knowledge_seeds=["forgotten construction"],
        current_fate=CandidateFate.CULLED,
        multihead_scores={"objective_alignment": 0.05, "rarity": 0.95},
    )

    archives = ArchiveManager()
    archives.update([candidate])

    assert "edge" not in archives.rarity_archive.candidates
    assert "forgotten construction" not in archives.rarity_archive.rare_seeds()
    assert "edge" in archives.failure_archive.records
