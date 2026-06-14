"""Pipeline and evolution progress event schemas."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineProgressEvent:
    stage: str
    stage_index: int
    stage_count: int
    stage_progress: float
    type: str = "pipeline_progress"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage_progress"] = max(0.0, min(1.0, float(self.stage_progress)))
        return data


@dataclass(frozen=True)
class EvolutionProgressEvent:
    round: int
    max_rounds: int
    population_size: int
    active_candidates: int
    dormant_candidates: int
    archive_elites: int
    tool_calls: int
    best_answer_candidate: str = ""
    best_auxiliary_candidate: str = ""
    search_diagnosis: str = ""
    next_action: str = "continue"
    type: str = "evolution_progress"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["PipelineProgressEvent", "EvolutionProgressEvent"]
