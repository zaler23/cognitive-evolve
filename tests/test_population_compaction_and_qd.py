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


def test_terminal_candidates_leave_live_population_but_keep_tombstones() -> None:
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

    assert [candidate.id for candidate in population.candidates] == ["live"]
    assert result.removed_terminal_ids == ["failed"]
    assert "failed" in archives.terminal_tombstones
    assert "failed" in archives.failure_archive.records
    reloaded = ArchiveManager.from_dict(archives.to_dict())
    assert "failed" in reloaded.terminal_tombstones


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
    assert len([candidate for candidate in population.candidates if candidate_bin_key(candidate).startswith("alpha|")]) <= 3
    assert len([candidate for candidate in population.candidates if candidate_bin_key(candidate).startswith("beta|")]) <= 3
    assert len(population.candidates) > 2  # bins, not one fixed global cap
    assert set(result.compacted_clone_ids).issubset(set(archives.terminal_tombstones))

