from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, candidate_from_dict


@dataclass
class DormantArchive:
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    reactivation_conditions: dict[str, str] = field(default_factory=dict)

    def add(self, candidate: CandidateGenome, *, condition: str = "reactivate_when_complementary_parent_or_rare_seed_needed") -> None:
        data = candidate.to_dict()
        data["current_fate"] = CandidateFate.DORMANT.value
        self.candidates[candidate.id] = data
        self.reactivation_conditions[candidate.id] = condition

    def reactivate(self, candidate_id: str | None = None) -> CandidateGenome | None:
        key = candidate_id or (next(iter(self.candidates)) if self.candidates else None)
        if not key or key not in self.candidates:
            return None
        candidate = candidate_from_dict(self.candidates.pop(key))
        self.reactivation_conditions.pop(key, None)
        candidate.mark_fate(CandidateFate.ACTIVE.value)
        candidate.mutation_history.append("DormantReactivation")
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DormantArchive":
        return cls(candidates=dict(data.get("candidates") or {}), reactivation_conditions=dict(data.get("reactivation_conditions") or {}))
