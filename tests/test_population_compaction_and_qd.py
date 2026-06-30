from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.quality_diversity import candidate_bin_key
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_control import compact_live_population


def _candidate(candidate_id: str, *, niche: str = "same", score: float = 0.1, fate: str = CandidateFate.ACTIVE.value, rarity: float = 0.0) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=2,
        current_fate=fate,
        core_mechanism=niche,
        niche_memberships=[niche],
        concise_claim=f"{niche}-{candidate_id}",
        multihead_scores={
            "objective_alignment": score,
            "answer_likelihood": score,
            "verifiability": score,
            "rarity": rarity,
            "novelty": rarity,
        },
    )


def test_nonstructural_terminal_candidates_reopen_as_dormant_reserve() -> None:
    failed = _candidate("failed", fate=CandidateFate.FAILED.value)
    live = _candidate("live")
    population = CandidatePopulation([failed, live])
    archives = ArchiveManager()

    result = compact_live_population(
        population,
        archives,
        EvolutionPolicy(metadata={"live_bin_capacity": 4}),
        branch_factor=2,
        round_index=3,
    )

    assert [candidate.id for candidate in population.candidates] == ["failed", "live"]
    assert failed.current_fate == CandidateFate.DORMANT.value
    assert result.removed_terminal_ids == []
    assert "failed" not in archives.terminal_tombstones
    reloaded = ArchiveManager.from_dict(archives.to_dict())
    assert "failed" not in reloaded.terminal_tombstones


def test_quality_diversity_compacts_per_bin_without_fixed_global_cap() -> None:
    same_bin = [_candidate(f"a{i}", niche="alpha", score=0.1 + i * 0.01) for i in range(5)]
    rare = _candidate("a-rare", niche="alpha", score=0.05, rarity=0.95)
    other_bin = [_candidate(f"b{i}", niche="beta", score=0.1 + i * 0.01) for i in range(3)]
    population = CandidatePopulation(same_bin + [rare] + other_bin)
    archives = ArchiveManager()

    result = compact_live_population(
        population,
        archives,
        EvolutionPolicy(metadata={"live_bin_capacity": 2, "quality_diversity_rare_reserve_per_bin": 1}),
        branch_factor=1,
        round_index=4,
    )

    live_ids = {candidate.id for candidate in population.candidates}
    assert "a-rare" in live_ids
    assert result.compacted_clone_ids
    bin_counts: dict[str, int] = {}
    for candidate in population.candidates:
        key = candidate_bin_key(candidate)
        bin_counts[key] = bin_counts.get(key, 0) + 1
    assert max(count for key, count in bin_counts.items() if key.startswith("alpha")) <= 3
    assert max(count for key, count in bin_counts.items() if key.startswith("beta")) <= 3
    assert len(population.candidates) > 2  # bins, not one fixed global cap
    assert set(result.compacted_clone_ids).issubset(set(archives.dormant_archive.candidates))


def test_candidate_bin_key_splits_applied_from_not_applied() -> None:
    # #7 regression: two candidates identical in canonical family and src/patch/target
    # path buckets but differing only in patch application status must land in distinct
    # QD bins.  If the bin key ever drops the applied/not_applied descriptor token again
    # (e.g. by slicing descriptor[1:4] instead of [1:5]), both collapse into one cell
    # and this assertion fails.
    applied = _candidate("applied-1")
    not_applied = _candidate("notapplied-1")
    for candidate in (applied, not_applied):
        candidate.metadata["nextgen"] = {"canonical_mechanism_family_id": "fam"}
    applied.metadata["patch_result"] = {"status": "applied"}
    not_applied.metadata["patch_result"] = {"status": "rejected"}

    assert candidate_bin_key(applied) != candidate_bin_key(not_applied)
