"""MDL compression advisory extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import CandidateTransform

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import SearchPressure
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class MDLCompressionExtension:
    extension_id = "mdl_compression"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.complexity: dict[str, float] = {}

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        for candidate in ctx.candidates:
            self.complexity[candidate.id] = _description_length(candidate)
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics={"mdl_scored_candidates": len(self.complexity)})

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        if not self.complexity:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        max_len = max(self.complexity.values() or [1.0])
        advisory = {cid: {"plan_value": bounded_score(1.0 - length / max(1.0, max_len)), "rank_prior": 0.0, "diversity": 0.0, "risk": 0.0} for cid, length in self.complexity.items()}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        if ctx.parent is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        length = self.complexity.get(ctx.parent.id, _description_length(ctx.parent))
        threshold = float(self.config.get("compression_threshold", 3000) or 3000)
        if length <= threshold:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction="Compress this candidate without broadening behavior: remove dead scaffolding, deduplicate logic, and preserve evaluator-passing behavior.", metadata={"source_extension": self.extension_id, "description_length": length})
        transform = CandidateTransform(candidate_id=ctx.parent.id, kind="compress", payload={"description_length": length, "threshold": threshold}, preserve_score_within=float(self.config.get("score_epsilon", 0.02) or 0.02))
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure], candidate_transforms=[transform], metrics={"mdl_compression_pressure_count": 1})

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"complexity": self.complexity} if self.complexity else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.complexity = {str(k): float(v) for k, v in ((state or {}).get("complexity") or {}).items() if isinstance(v, (int, float))}


def _description_length(candidate: Any) -> float:
    text = str(getattr(candidate, "artifact", "") or "")
    return float(len(text) + 20 * len(getattr(candidate, "source_bindings", []) or []) + 10 * len(getattr(candidate, "tool_results", []) or []))


__all__ = ["MDLCompressionExtension"]
