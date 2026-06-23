"""Problem reformulation helpers that increase verifiability without changing success criteria."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import stable_hash


@dataclass(frozen=True)
class ReformulationRecord:
    reformulation_id: str
    original_kind: str
    reformulated_prompt: str
    success_criterion_preserved: bool = True
    subproblems: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reformulation_id": self.reformulation_id,
            "original_kind": self.original_kind,
            "reformulated_prompt": self.reformulated_prompt,
            "success_criterion_preserved": self.success_criterion_preserved,
            "subproblems": list(self.subproblems),
        }


def reformulate_for_verification(problem: str) -> list[ReformulationRecord]:
    text = str(problem or "").strip()
    if not text:
        return []
    record = ReformulationRecord(
        reformulation_id="reformulation-" + stable_hash({"problem": text, "kind": "falsifiable"})[:16],
        original_kind="open_discourse",
        reformulated_prompt="State falsifiable predictions, explicit failure cases, and a minimal executable or empirical check for: " + text[:500],
        subproblems=[{"kind": "falsification_test", "prompt": "Find a counterexample or failing fixture for the proposed answer."}],
    )
    return [record]


__all__ = ["ReformulationRecord", "reformulate_for_verification"]
