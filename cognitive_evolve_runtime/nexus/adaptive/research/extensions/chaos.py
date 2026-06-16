"""Deterministic final/frontier chaos advisory extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for

from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ChaosExtension:
    extension_id = "chaos"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.reports: list[dict[str, Any]] = []

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        seed = int(self.config.get("seed", 42) or 42)
        champion = max(ctx.candidates, key=lambda c: float((c.multihead_scores or {}).get("frontier_score", 0.0)), default=None)
        if champion is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        report = {"candidate_id": champion.id, "seed": seed, "scope": "champion_final_only", "replayable": True, "status": "not_executed_without_explicit_profile"}
        self.reports.append(report)
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, final_gate_directives=[{"kind": "chaos_replay_required_if_configured", **report}], metrics={"chaos_report_count": len(self.reports)})

    def snapshot(self) -> dict[str, Any]:
        return {"reports": self.reports[-50:]} if self.reports else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.reports = [dict(item) for item in ((state or {}).get("reports") or []) if isinstance(item, dict)]


__all__ = ["ChaosExtension"]
