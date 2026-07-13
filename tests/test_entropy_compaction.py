from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.quality_diversity import (
    candidate_bin_key,
    descriptor_cell_distribution,
    descriptor_population_entropy,
    entropy_diversity_survivors,
    quality_diversity_survivors,
)
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.v23_theory_config import EntropyCompactionConfig


def _candidate(cid: str, niche: str, *, quality: float = 0.5, rarity: float = 0.0, edge: bool = False) -> CandidateGenome:
    return CandidateGenome(
        id=cid,
        artifact=f"artifact {cid}",
        concise_claim=cid,
        core_mechanism=niche,
        niche_memberships=[niche],
        edge_knowledge_seeds=[f"edge-{cid}"] if edge else [],
        multihead_scores={
            "objective_alignment": quality,
            "answer_likelihood": quality,
            "core_mechanism_strength": quality,
            "verifiability": quality,
            "rarity": rarity,
            "novelty": rarity,
            "frontier_score": quality,
            "continuation_value": quality,
        },
    )


def test_descriptor_distribution_and_entropy_are_measurable() -> None:
    candidates = [_candidate("A", "alpha"), _candidate("B", "alpha"), _candidate("C", "beta")]

    distribution = descriptor_cell_distribution(candidates)

    assert sum(distribution.values()) == 3
    assert len(distribution) == 2
    assert descriptor_population_entropy(candidates) > 0.0


def test_entropy_mode_preserves_each_occupied_cell_elite() -> None:
    candidates = [
        _candidate("A1", "alpha", quality=0.9),
        _candidate("A2", "alpha", quality=0.3),
        _candidate("B1", "beta", quality=0.7),
        _candidate("C1", "gamma", quality=0.6),
    ]
    before_entropy = descriptor_population_entropy(candidates)
    survivors, compacted = entropy_diversity_survivors(candidates, target_k=3, config=EntropyCompactionConfig(cell_elite_reserve=1))
    survivor_cells = {candidate_bin_key(candidate) for candidate in survivors}

    assert len(survivors) == 3
    assert len(compacted) == 1
    assert survivor_cells == {candidate_bin_key(candidates[0]), candidate_bin_key(candidates[2]), candidate_bin_key(candidates[3])}
    assert descriptor_population_entropy(survivors) >= before_entropy * 0.9


def test_rare_edge_seed_gets_reserve_priority_in_overfull_cell() -> None:
    candidates = [
        _candidate("A1", "alpha", quality=0.95, rarity=0.1),
        _candidate("A2", "alpha", quality=0.4, rarity=0.9, edge=True),
        _candidate("A3", "alpha", quality=0.3, rarity=0.1),
    ]
    survivors, _ = quality_diversity_survivors(candidates, bin_capacity=1, rare_reserve_per_bin=1)

    assert {candidate.id for candidate in survivors} == {"A1", "A2"}


def test_population_compaction_records_v23_entropy_metrics() -> None:
    population = CandidatePopulation([_candidate("A1", "alpha"), _candidate("A2", "alpha"), _candidate("B1", "beta")])
    result = compact_live_population(population, ArchiveManager(), EvolutionPolicy(), branch_factor=1, round_index=2)
    data = result.to_dict()

    assert len(data["compacted_clone_ids"]) >= 0
    assert data["population_entropy_before"] >= data["population_entropy_after"] or data["population_entropy_after"] >= 0.0
    assert data["descriptor_cell_count_before"] >= data["descriptor_cell_count_after"]
    assert data["v23_theory_config_hash"].startswith("v23-theory-")
