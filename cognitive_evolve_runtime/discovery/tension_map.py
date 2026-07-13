"""Tension map wrapper around ChallengeMemory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory, ChallengeMemoryItem


@dataclass
class TensionMap:
    memory: ChallengeMemory = field(default_factory=ChallengeMemory)
    ruled_out: list[dict[str, Any]] = field(default_factory=list)
    frontier_descriptors: list[Any] = field(default_factory=list)

    @property
    def open_tensions(self) -> list[dict[str, Any]]:
        out = []
        for raw in self.memory.items.values():
            item = ChallengeMemoryItem.from_dict(raw)
            if item is not None and not item.resolved_by_candidate_ids:
                out.append(item.to_dict())
        return out

    def mark_ruled_out(self, *, candidate_id: str, descriptor: Any, evidence_ref: str) -> None:
        key = {"candidate_id": str(candidate_id), "descriptor": descriptor, "evidence_ref": str(evidence_ref)}
        if key not in self.ruled_out:
            self.ruled_out.append(key)

    def is_ruled_out(self, descriptor: Any) -> bool:
        return any(item.get("descriptor") == descriptor for item in self.ruled_out)

    def to_dict(self) -> dict[str, Any]:
        return {"challenge_memory": self.memory.to_dict(), "open_tensions": self.open_tensions, "ruled_out": list(self.ruled_out), "frontier_descriptors": list(self.frontier_descriptors)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TensionMap":
        raw = data or {}
        memory = ChallengeMemory.from_dict(raw.get("challenge_memory") or raw.get("memory") or {})
        return cls(memory=memory, ruled_out=[dict(item) for item in raw.get("ruled_out", []) if isinstance(item, dict)], frontier_descriptors=list(raw.get("frontier_descriptors", [])))


__all__ = ["TensionMap"]
