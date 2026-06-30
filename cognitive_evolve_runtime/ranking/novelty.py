from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.search_kernel.descriptor_cells import behavior_descriptor


def novelty_distance(a: CandidateGenome, b: CandidateGenome) -> float:
    a_set = _descriptor_set(a)
    b_set = _descriptor_set(b)
    if not a_set and not b_set:
        return 0.0
    return 1.0 - (len(a_set & b_set) / max(1, len(a_set | b_set)))


def population_novelty(candidate: CandidateGenome, population: list[CandidateGenome]) -> float:
    others = [item for item in population if item.id != candidate.id]
    if not others:
        return 1.0 if _descriptor_set(candidate) else 0.0
    return sum(novelty_distance(candidate, other) for other in others) / len(others)


def _descriptor_set(candidate: CandidateGenome) -> set[str]:
    return set(candidate.novelty_descriptors + candidate.niche_memberships + candidate.edge_knowledge_seeds + list(behavior_descriptor(candidate)))


__all__ = ["novelty_distance", "population_novelty"]
