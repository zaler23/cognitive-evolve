"""Archive/population geometry advisories for M6.4.

This module operates on immutable candidate representations only.  It does not
read live archive manager state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .representations import CandidateRepresentation, PopulationRepresentation
from .signals import TheorySignal


@dataclass(frozen=True)
class GeometrySummary:
    cycle_id: str
    candidate_count: int
    descriptor_count: int
    mean_pairwise_distance: float
    coverage: float
    transport_spread: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def descriptor(candidate: CandidateRepresentation) -> tuple[str, ...]:
    values = [candidate.artifact_type, *candidate.novelty_descriptors, *candidate.niche_memberships]
    if candidate.source_binding_count:
        values.append("source_bound")
    if candidate.evidence_ref_count:
        values.append("evidence_bound")
    return tuple(sorted({str(item).strip().lower() for item in values if str(item).strip()}))


def summarize_population_geometry(population: PopulationRepresentation) -> GeometrySummary:
    descriptors = {candidate.candidate_id: descriptor(candidate) for candidate in population.candidates}
    distances: list[float] = []
    candidates = list(population.candidates)
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            distances.append(_jaccard_distance(descriptors[left.candidate_id], descriptors[right.candidate_id]))
    unique_descriptors = {value for values in descriptors.values() for value in values}
    coverage = len(unique_descriptors) / max(1, len(candidates) * 4)
    mean_distance = sum(distances) / len(distances) if distances else 0.0
    return GeometrySummary(
        cycle_id=population.cycle_id,
        candidate_count=len(candidates),
        descriptor_count=len(unique_descriptors),
        mean_pairwise_distance=mean_distance,
        coverage=max(0.0, min(1.0, coverage)),
        transport_spread=mean_distance,
    )


def geometry_advisory_signals(population: PopulationRepresentation) -> tuple[TheorySignal, ...]:
    summary = summarize_population_geometry(population)
    signals: list[TheorySignal] = []
    for candidate in population.candidates:
        own_descriptor = descriptor(candidate)
        rarity = 1.0 if own_descriptor else 0.0
        if own_descriptor:
            same = sum(1 for other in population.candidates if descriptor(other) == own_descriptor)
            rarity = 1.0 / max(1, same)
        signals.append(
            TheorySignal(
                source="geometry",
                kind="diversity",
                target_type="candidate",
                cycle_id=population.cycle_id,
                target_id=candidate.candidate_id,
                value=max(0.0, min(1.0, rarity)),
                confidence=0.5,
                provenance=("geometry:descriptor_coverage",),
                meta={"descriptor_size": len(own_descriptor), "population_coverage": summary.coverage},
            )
        )
    return tuple(signals)


def _jaccard_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(1, len(a | b)))


__all__ = ["GeometrySummary", "descriptor", "geometry_advisory_signals", "summarize_population_geometry"]
