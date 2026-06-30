"""External evaluator results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import utc_now
from cognitive_evolve_runtime.tools.feedback import ToolFeedback


@dataclass
class EvaluatorResult:
    candidate_id: str
    status: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_feedback(self) -> ToolFeedback:
        return ToolFeedback(
            tool_id="external_evaluator",
            status=self.status,
            diagnostics=list(self.diagnostics),
            verified_fragments=["external_evaluator_passed"] if self.passed else [],
            failed_fragments=[] if self.passed else ["external_evaluator_failed"],
            cost=dict(self.cost),
            confidence=1.0 if self.passed else 0.8,
            raw_output_ref=f"external_evaluator:{self.status}",
        )


__all__ = ["EvaluatorResult"]
