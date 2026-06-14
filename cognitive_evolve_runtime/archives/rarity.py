from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome


@dataclass
class RarityArchive:
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)

    def add(self, candidate: CandidateGenome) -> None:
        if candidate.edge_knowledge_seeds or candidate.multihead_scores.get("rarity", 0.0) > 0:
            self.candidates[candidate.id] = candidate.to_dict()
            for seed in candidate.edge_knowledge_seeds:
                if seed not in self.seeds:
                    self.seeds.append(seed)

    def rare_seeds(self, limit: int | None = None) -> list[str]:
        return self.seeds[:limit] if limit is not None else list(self.seeds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RarityArchive":
        return cls(candidates=dict(data.get("candidates") or {}), seeds=[str(item) for item in data.get("seeds", [])])
