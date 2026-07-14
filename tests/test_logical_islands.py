from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.search_kernel.islands import allocate_logical_islands, derive_island_count


def _candidate(index: int) -> CandidateGenome:
    return CandidateGenome(
        id=f"candidate-{index}",
        lineage=[f"root-{index}", f"candidate-{index}"],
        artifact={"value": index},
        current_fate=CandidateFate.ACTIVE.value,
    )


def test_auto_islands_never_exceed_slots_and_cover_every_nonempty_island() -> None:
    candidates = [_candidate(index) for index in range(9)]
    allocation = allocate_logical_islands(parents=candidates, candidates=candidates, total_slots=5)

    assert derive_island_count(total_slots=5, lineage_count=9) == 3
    assert set(allocation.slot_islands.values()) == {0, 1, 2}
    assert len(allocation.branches.slots) == 5
    assert len(set(allocation.candidate_islands.values())) == 3


def test_island_migration_only_borrows_parent_references() -> None:
    candidates = [_candidate(index) for index in range(4)]
    original_fates = {candidate.id: candidate.current_fate for candidate in candidates}
    allocation = allocate_logical_islands(
        parents=candidates,
        candidates=candidates,
        total_slots=4,
        config={"count": 2, "migration_interval": 2},
        current_round=2,
    )

    assert allocation.borrowed_parent_ids
    assert {candidate.id: candidate.current_fate for candidate in candidates} == original_fates
