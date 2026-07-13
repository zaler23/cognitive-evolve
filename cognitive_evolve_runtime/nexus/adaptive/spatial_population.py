"""2D spatial population observe-mode support."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict


@dataclass(frozen=True)
class SpatialCoord:
    x: int
    y: int

    def key(self) -> str:
        return f"{self.x},{self.y}"

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass
class SpatialCell:
    coord: SpatialCoord
    resident_id: str | None = None
    incubating_ids: list[str] = field(default_factory=list)
    auxiliary_ids: list[str] = field(default_factory=list)
    niche_key: str = ""
    lineage_key: str = ""
    local_energy: float = 0.0
    last_updated_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["coord"] = self.coord.to_dict()
        return data


@dataclass
class SpatialRegion:
    region_id: str
    x0: int
    y0: int
    width: int
    height: int
    active_count: int = 0
    elite_count: int = 0
    failed_count: int = 0
    avg_local_energy: float = 0.0
    niche_entropy: float = 0.0
    lineage_entropy: float = 0.0
    roi: float = 0.0
    backpressure: bool = False
    stagnation_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialPopulationState:
    width: int
    height: int
    region_size: int = 3
    neighborhood: str = "moore"
    toroidal: bool = True
    cells: dict[str, SpatialCell] = field(default_factory=dict)
    regions: dict[str, SpatialRegion] = field(default_factory=dict)
    candidate_to_coord: dict[str, SpatialCoord] = field(default_factory=dict)
    candidate_selection_counts: dict[str, int] = field(default_factory=dict)
    region_selection_counts: dict[str, int] = field(default_factory=dict)
    round_index: int = 0
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "region_size": self.region_size,
            "neighborhood": self.neighborhood,
            "toroidal": self.toroidal,
            "cells": {key: cell.to_dict() for key, cell in self.cells.items()},
            "regions": {key: region.to_dict() for key, region in self.regions.items()},
            "candidate_to_coord": {key: coord.to_dict() for key, coord in self.candidate_to_coord.items()},
            "candidate_selection_counts": dict(self.candidate_selection_counts),
            "region_selection_counts": dict(self.region_selection_counts),
            "round_index": self.round_index,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpatialPopulationState | None":
        if not isinstance(data, dict) or not data:
            return None
        state = cls(
            width=max(1, _int(data.get("width"), 1)),
            height=max(1, _int(data.get("height"), 1)),
            region_size=max(1, _int(data.get("region_size"), 3)),
            neighborhood=str(data.get("neighborhood") or "moore"),
            toroidal=bool(data.get("toroidal", True)),
            round_index=_int(data.get("round_index"), 0),
            seed=_int(data.get("seed"), 42),
            metadata=coerce_dict(data.get("metadata")),
        )
        for key, raw in coerce_dict(data.get("cells")).items():
            coord_raw = coerce_dict(raw.get("coord") if isinstance(raw, dict) else {})
            state.cells[str(key)] = SpatialCell(
                coord=SpatialCoord(_int(coord_raw.get("x"), 0), _int(coord_raw.get("y"), 0)),
                resident_id=raw.get("resident_id") if isinstance(raw, dict) and raw.get("resident_id") is not None else None,
                incubating_ids=[str(item) for item in raw.get("incubating_ids", []) if item] if isinstance(raw, dict) else [],
                auxiliary_ids=[str(item) for item in raw.get("auxiliary_ids", []) if item] if isinstance(raw, dict) else [],
                niche_key=str(raw.get("niche_key") or "") if isinstance(raw, dict) else "",
                lineage_key=str(raw.get("lineage_key") or "") if isinstance(raw, dict) else "",
                local_energy=_float(raw.get("local_energy"), 0.0) if isinstance(raw, dict) else 0.0,
                last_updated_round=_int(raw.get("last_updated_round"), 0) if isinstance(raw, dict) else 0,
                metadata=coerce_dict(raw.get("metadata")) if isinstance(raw, dict) else {},
            )
        for cid, coord_raw in coerce_dict(data.get("candidate_to_coord")).items():
            coord = coerce_dict(coord_raw)
            state.candidate_to_coord[str(cid)] = SpatialCoord(_int(coord.get("x"), 0), _int(coord.get("y"), 0))
        return state


def build_or_update_spatial_state(
    candidates: list[CandidateGenome],
    *,
    existing: SpatialPopulationState | None,
    round_index: int,
    width: int = 0,
    height: int = 0,
    region_size: int = 3,
    neighborhood: str = "moore",
    toroidal: bool = True,
) -> SpatialPopulationState:
    live = [candidate for candidate in candidates if CandidateFate.normalize(candidate.current_fate) not in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}]
    size = max(1, len(live))
    computed_width = max(1, int(width or math.ceil(math.sqrt(size))))
    computed_height = max(1, int(height or math.ceil(size / computed_width)))
    if existing is None or existing.width * existing.height < size or existing.width != computed_width or existing.height != computed_height:
        state = SpatialPopulationState(width=computed_width, height=computed_height, region_size=max(1, region_size), neighborhood=neighborhood, toroidal=toroidal)
    else:
        state = existing
        state.region_size = max(1, region_size)
        state.neighborhood = neighborhood
        state.toroidal = toroidal
    state.round_index = round_index
    state.cells = {coord_key(x, y): SpatialCell(coord=SpatialCoord(x, y), last_updated_round=round_index) for y in range(state.height) for x in range(state.width)}
    state.candidate_to_coord = {}
    ordered = sorted(live, key=lambda c: (_fate_rank(c), _niche_key(c), _lineage_key(c), c.id))
    for index, candidate in enumerate(ordered):
        coord = SpatialCoord(index % state.width, index // state.width)
        cell = state.cells[coord.key()]
        fate = CandidateFate.normalize(candidate.current_fate)
        if fate in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value} and cell.resident_id is None:
            cell.resident_id = candidate.id
        elif fate == CandidateFate.INCUBATING.value:
            cell.incubating_ids.append(candidate.id)
        else:
            cell.auxiliary_ids.append(candidate.id)
        cell.niche_key = _niche_key(candidate)
        cell.lineage_key = _lineage_key(candidate)
        cell.local_energy = compute_local_energy(candidate)
        cell.last_updated_round = round_index
        state.candidate_to_coord[candidate.id] = coord
        region_id = region_id_for(coord, state.region_size)
        candidate.metadata["spatial"] = {
            "x": coord.x,
            "y": coord.y,
            "region_id": region_id,
            "niche_key": cell.niche_key,
            "lineage_key": cell.lineage_key,
            "local_energy": cell.local_energy,
            "mode": "observe",
        }
    state.regions = _regions(state)
    state.metadata["candidate_count"] = len(live)
    state.metadata["mode"] = "observe"
    return state


def compute_local_energy(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores or {}
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    objective = float(scores.get("objective_score", scores.get("answer_likelihood", scores.get("objective_alignment", 0.0))) or 0.0)
    verifiability = float(scores.get("verifiability", 0.0) or 0.0)
    robustness = float(scores.get("robustness", 0.0) or 0.0)
    novelty = float(scores.get("novelty", scores.get("rarity", 0.0)) or 0.0)
    mdl_penalty = _float(coerce_dict(metadata.get("mdl")).get("penalty"), 0.0)
    burn_penalty = _float(coerce_dict(metadata.get("budget_account")).get("burn_penalty"), 0.0)
    failure_penalty = min(0.30, 0.04 * len(candidate.failure_lessons or []))
    return max(0.0, min(1.0, 0.40 * objective + 0.25 * verifiability + 0.15 * robustness + 0.15 * novelty - 0.15 * mdl_penalty - 0.10 * burn_penalty - failure_penalty))


def coord_key(x: int, y: int) -> str:
    return f"{x},{y}"


def region_id_for(coord: SpatialCoord, region_size: int) -> str:
    return f"R{coord.x // max(1, region_size)}-{coord.y // max(1, region_size)}"


def _regions(state: SpatialPopulationState) -> dict[str, SpatialRegion]:
    regions: dict[str, SpatialRegion] = {}
    energies: dict[str, list[float]] = {}
    niches: dict[str, dict[str, int]] = {}
    lineages: dict[str, dict[str, int]] = {}
    for cell in state.cells.values():
        rid = region_id_for(cell.coord, state.region_size)
        region = regions.setdefault(rid, SpatialRegion(region_id=rid, x0=(cell.coord.x // state.region_size) * state.region_size, y0=(cell.coord.y // state.region_size) * state.region_size, width=state.region_size, height=state.region_size))
        if cell.resident_id:
            region.active_count += 1
        energies.setdefault(rid, []).append(cell.local_energy)
        niches.setdefault(rid, {})[cell.niche_key or "unknown"] = niches.setdefault(rid, {}).get(cell.niche_key or "unknown", 0) + 1
        lineages.setdefault(rid, {})[cell.lineage_key or "unknown"] = lineages.setdefault(rid, {}).get(cell.lineage_key or "unknown", 0) + 1
    for rid, region in regions.items():
        vals = energies.get(rid, [])
        region.avg_local_energy = sum(vals) / max(1, len(vals))
        region.niche_entropy = _entropy(niches.get(rid, {}))
        region.lineage_entropy = _entropy(lineages.get(rid, {}))
    return regions


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p, 2)
    return value


def _fate_rank(candidate: CandidateGenome) -> int:
    fate = CandidateFate.normalize(candidate.current_fate)
    return {CandidateFate.ELITE.value: 0, CandidateFate.ACTIVE.value: 1, CandidateFate.INCUBATING.value: 2, CandidateFate.AUXILIARY.value: 3}.get(fate, 4)


def _niche_key(candidate: CandidateGenome) -> str:
    return str((candidate.niche_memberships or [candidate.artifact_type or "default"])[0] or "default")[:80]


def _lineage_key(candidate: CandidateGenome) -> str:
    return str((candidate.lineage or candidate.parent_ids or [candidate.id])[0] or candidate.id)[:80]


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "SpatialCoord",
    "SpatialCell",
    "SpatialRegion",
    "SpatialPopulationState",
    "build_or_update_spatial_state",
    "compute_local_energy",
    "region_id_for",
]
