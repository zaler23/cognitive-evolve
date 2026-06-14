from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome


def novelty_distance(a: CandidateGenome, b: CandidateGenome) -> float:
    a_set = set(a.novelty_descriptors + a.niche_memberships + a.edge_knowledge_seeds)
    b_set = set(b.novelty_descriptors + b.niche_memberships + b.edge_knowledge_seeds)
    if not a_set and not b_set:
        return 0.0
    return 1.0 - (len(a_set & b_set) / max(1, len(a_set | b_set)))


def population_novelty(candidate: CandidateGenome, population: list[CandidateGenome]) -> float:
    others = [item for item in population if item.id != candidate.id]
    if not others:
        return 1.0 if candidate.novelty_descriptors or candidate.edge_knowledge_seeds else 0.0
    return sum(novelty_distance(candidate, other) for other in others) / len(others)


__all__ = ["novelty_distance", "population_novelty"]
