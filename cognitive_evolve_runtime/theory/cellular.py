"""Cellular-field advisory layer for M6.5.

Cells consume already-aggregated sidecar features and immutable candidate
representations.  Local pressure is advisory and cannot become proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .geometry import descriptor
from .representations import PopulationRepresentation
from .signals import AdvisoryRankingFeatures, TheorySignal


@dataclass(frozen=True)
class SearchCell:
    cell_id: str
    candidate_ids: tuple[str, ...]
    local_pressure: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"candidate_ids": list(self.candidate_ids)}


def build_search_cells(population: PopulationRepresentation, features: Mapping[str, AdvisoryRankingFeatures] | None = None) -> tuple[SearchCell, ...]:
    features = features or {}
    groups: dict[str, list[str]] = {}
    pressure: dict[str, float] = {}
    for candidate in population.candidates:
        key = "+".join(descriptor(candidate)[:4]) or "untyped"
        groups.setdefault(key, []).append(candidate.candidate_id)
        feature = features.get(candidate.candidate_id)
        if feature is not None:
            pressure[key] = pressure.get(key, 0.0) + feature.rank_prior + feature.plan_value + feature.diversity - feature.risk
    cells: list[SearchCell] = []
    for key, ids in sorted(groups.items()):
        cells.append(SearchCell(cell_id=key, candidate_ids=tuple(sorted(ids)), local_pressure=pressure.get(key, 0.0) / max(1, len(ids))))
    return tuple(cells)


def cellular_advisory_signals(population: PopulationRepresentation, features: Mapping[str, AdvisoryRankingFeatures] | None = None) -> tuple[TheorySignal, ...]:
    signals: list[TheorySignal] = []
    for cell in build_search_cells(population, features):
        signals.append(
            TheorySignal(
                source="cellular",
                kind="plan_value",
                target_type="lineage",
                cycle_id=population.cycle_id,
                target_id=cell.cell_id,
                value=max(-1.0, min(1.0, cell.local_pressure)),
                confidence=0.4,
                provenance=("cellular:local_advisory_pressure",),
                meta={"candidate_count": len(cell.candidate_ids)},
            )
        )
    return tuple(signals)


__all__ = ["SearchCell", "build_search_cells", "cellular_advisory_signals"]
