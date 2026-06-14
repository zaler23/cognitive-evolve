"""Immutable JSON-safe snapshots consumed by the theory layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


def _stable_str(value: Any, limit: int = 2000) -> str:
    return str(value or "")[:limit]


def _score_items(value: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        return ()
    items: list[tuple[str, float]] = []
    for key, raw in value.items():
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in {float("inf"), float("-inf")}:
            items.append((str(key), parsed))
    return tuple(sorted(items))


def _str_tuple(value: Any, limit: int = 32) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item)[:500] for item in list(value)[:limit] if str(item))
    text = str(value or "")
    return (text[:500],) if text else ()


@dataclass(frozen=True)
class CandidateRepresentation:
    candidate_id: str
    generation: int = 0
    fate: str = ""
    artifact_type: str = ""
    concise_claim: str = ""
    core_mechanism: str = ""
    scores: tuple[tuple[str, float], ...] = ()
    novelty_descriptors: tuple[str, ...] = ()
    niche_memberships: tuple[str, ...] = ()
    missing_parts: tuple[str, ...] = ()
    uncertainty_notes: tuple[str, ...] = ()
    source_binding_count: int = 0
    evidence_ref_count: int = 0

    @classmethod
    def from_candidate(cls, candidate: Any) -> "CandidateRepresentation":
        candidate_id = str(getattr(candidate, "id", "") or "")
        if not candidate_id:
            raise ValueError("candidate representation requires stable CandidateGenome.id")
        return cls(
            candidate_id=candidate_id,
            generation=int(getattr(candidate, "generation", 0) or 0),
            fate=_stable_str(getattr(candidate, "current_fate", ""), 80),
            artifact_type=_stable_str(getattr(candidate, "artifact_type", ""), 80),
            concise_claim=_stable_str(getattr(candidate, "concise_claim", ""), 1000),
            core_mechanism=_stable_str(getattr(candidate, "core_mechanism", ""), 1000),
            scores=_score_items(getattr(candidate, "multihead_scores", {})),
            novelty_descriptors=_str_tuple(getattr(candidate, "novelty_descriptors", ())),
            niche_memberships=_str_tuple(getattr(candidate, "niche_memberships", ())),
            missing_parts=_str_tuple(getattr(candidate, "missing_parts", ())),
            uncertainty_notes=_str_tuple(getattr(candidate, "uncertainty_notes", ())),
            source_binding_count=len(list(getattr(candidate, "source_bindings", []) or [])),
            evidence_ref_count=len(list(getattr(candidate, "evidence_refs", []) or [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"scores": list(self.scores)}


@dataclass(frozen=True)
class PopulationRepresentation:
    cycle_id: str
    candidates: tuple[CandidateRepresentation, ...]

    @classmethod
    def from_candidates(cls, candidates: Iterable[Any], *, cycle_id: str) -> "PopulationRepresentation":
        reps = tuple(CandidateRepresentation.from_candidate(candidate) for candidate in candidates)
        return cls(cycle_id=str(cycle_id or "cycle:unknown"), candidates=tuple(sorted(reps, key=lambda item: item.candidate_id)))

    def to_dict(self) -> dict[str, Any]:
        return {"cycle_id": self.cycle_id, "candidates": [candidate.to_dict() for candidate in self.candidates]}


@dataclass(frozen=True)
class CompletedEventSnapshot:
    cycle_id: str
    event_type: str
    target_id: str
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    metrics: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, cycle_id: str) -> "CompletedEventSnapshot":
        return cls(
            cycle_id=str(cycle_id or value.get("cycle_id") or "cycle:unknown"),
            event_type=str(value.get("event_type") or value.get("type") or "completed_event"),
            target_id=str(value.get("target_id") or value.get("candidate_id") or value.get("id") or "population"),
            diagnostics=_str_tuple(value.get("diagnostics", ())),
            metrics=_score_items(value.get("metrics") or value.get("scores") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"metrics": list(self.metrics)}


def build_population_representation(candidates: Iterable[Any], *, cycle_id: str) -> PopulationRepresentation:
    return PopulationRepresentation.from_candidates(candidates, cycle_id=cycle_id)


__all__ = [
    "CandidateRepresentation",
    "CompletedEventSnapshot",
    "PopulationRepresentation",
    "build_population_representation",
]
