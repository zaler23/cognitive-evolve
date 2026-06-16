"""Final/frontier BFT quorum directive extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class BFTQuorumExtension:
    extension_id = "bft_quorum"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.reports: list[dict[str, Any]] = []

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        report = {"scope": str(self.config.get("scope") or "final_only"), "objective_solved_authority": False, "status": "directive_only"}
        self.reports.append(report)
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, final_gate_directives=[{"kind": "bft_quorum_report", **report}], metrics={"bft_disagreement_rate": 0.0, "bft_report_count": len(self.reports)})

    def snapshot(self) -> dict[str, Any]:
        return {"reports": self.reports[-50:]} if self.reports else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.reports = [dict(item) for item in ((state or {}).get("reports") or []) if isinstance(item, dict)]


__all__ = ["BFTQuorumExtension"]
