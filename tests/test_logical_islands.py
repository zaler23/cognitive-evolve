from __future__ import annotations

import pytest

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


def test_island_allocators_share_one_global_family_density_snapshot() -> None:
    dense_parent = CandidateGenome(
        id="dense-root",
        lineage=["dense-root"],
        artifact={"value": "dense-root"},
        metadata={"search_space": {"family_id": "dense"}},
    )
    sparse_parent = CandidateGenome(
        id="sparse-root",
        lineage=["sparse-root"],
        artifact={"value": "sparse-root"},
        metadata={"search_space": {"family_id": "sparse"}},
    )
    explored_dense = [
        CandidateGenome(
            id=f"dense-{index}",
            lineage=["dense-root", f"dense-{index}"],
            artifact={"value": index},
            current_fate=CandidateFate.FAILED.value if index == 0 else CandidateFate.CULLED.value,
            metadata={"search_space": {"family_id": "dense"}},
        )
        for index in range(2)
    ]
    budget_history = [
        {
            "round": 0,
            "generation_plan": {
                "productive_branch_allocation": {
                    "slots": [
                        {"slot_id": "dense-0", "arm_id": "dense-root"},
                        {"slot_id": "sparse-0", "arm_id": "sparse-root"},
                    ]
                }
            },
        }
    ]

    allocation = allocate_logical_islands(
        parents=[dense_parent, sparse_parent],
        candidates=[dense_parent, sparse_parent, *explored_dense],
        total_slots=2,
        budget_history=budget_history,
        config={"count": 2},
    )

    assert allocation.branches.observed_family_counts == {"dense": 3, "sparse": 1}
    sparse_slots = [slot for slot in allocation.branches.slots if slot.arm_id == "sparse-root"]
    assert len(sparse_slots) == 1
    assert sparse_slots[0].coverage_bonus == pytest.approx(2 / 3, abs=1e-6)
