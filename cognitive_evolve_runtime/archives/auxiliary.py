from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, candidate_from_dict


@dataclass
class AuxiliaryArchive:
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, candidate: CandidateGenome) -> None:
        self.candidates[candidate.id] = candidate.to_dict()

    def core_extraction_targets(self) -> list[CandidateGenome]:
        return [candidate_from_dict(data) for data in self.candidates.values()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuxiliaryArchive":
        return cls(candidates=dict(data.get("candidates") or {}))
