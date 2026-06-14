"""Lineage utilities for Nexus candidate genomes."""
from __future__ import annotations

from collections import Counter

from .genome import CandidateGenome


def lineage_family(candidate: CandidateGenome) -> str:
    if candidate.lineage:
        return candidate.lineage[0]
    if candidate.parent_ids:
        return candidate.parent_ids[0]
    return candidate.id


def lineage_counts(candidates: list[CandidateGenome]) -> dict[str, int]:
    return dict(Counter(lineage_family(candidate) for candidate in candidates))


def saturated_lineages(candidates: list[CandidateGenome], *, threshold: int = 3) -> list[str]:
    return [family for family, count in lineage_counts(candidates).items() if count >= threshold]


__all__ = ["lineage_family", "lineage_counts", "saturated_lineages"]
