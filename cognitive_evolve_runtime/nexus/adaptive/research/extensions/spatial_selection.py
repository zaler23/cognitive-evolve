"""Spatial selection advisory extension using the existing spatial state model."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import evidence_state
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal
from cognitive_evolve_runtime.nexus.adaptive.spatial_population import SpatialPopulationState


class SpatialSelectionExtension:
    extension_id = "spatial_selection"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.metrics: dict[str, Any] = {}

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        state = SpatialPopulationState.from_dict(getattr(ctx.adaptive_state, "spatial", None) if ctx.adaptive_state is not None else None)
        self.metrics = {"spatial_state_model_count": 1, "spatial_extension_uses_existing_state": True, "spatial_candidate_count": len(state.candidate_to_coord) if state else 0}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics=self.metrics)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        state = SpatialPopulationState.from_dict(getattr(ctx.adaptive_state, "spatial", None) if ctx.adaptive_state is not None else None)
        if state is None:
            return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics={"spatial_extension_uses_existing_state": True})
        region_counts: dict[str, int] = {}
        for candidate in ctx.candidates:
            spatial = candidate.metadata.get("spatial", {}) if isinstance(candidate.metadata, dict) else {}
            region_id = str(spatial.get("region_id") or "") if isinstance(spatial, dict) else ""
            if region_id:
                region_counts[region_id] = region_counts.get(region_id, 0) + 1
        advisory: dict[str, dict[str, float]] = {}
        for candidate in ctx.candidates:
            energy = _local_energy(candidate)
            spatial = candidate.metadata.get("spatial", {}) if isinstance(candidate.metadata, dict) else {}
            region_id = str(spatial.get("region_id") or "") if isinstance(spatial, dict) else ""
            rarity = 1.0 / max(1, region_counts.get(region_id, 1)) if region_id else 0.0
            advisory[candidate.id] = {"rank_prior": energy, "diversity": bounded_score(rarity), "plan_value": bounded_score(0.5 * energy + 0.2 * rarity), "risk": 0.0}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory, metrics={"spatial_advisory_candidate_count": len(advisory), "spatial_extension_uses_existing_state": True})

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics)} if self.metrics else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.metrics = dict((state or {}).get("metrics") or {})


def _local_energy(candidate: Any) -> float:
    scores = getattr(candidate, "multihead_scores", {}) or {}
    state = evidence_state(candidate)
    objective = scores.get("objective_score", scores.get("frontier_score", 0.0))
    return bounded_score(0.35 * bounded_score(objective) + 0.25 * bounded_score(state.get("search_score", 0.0)) + 0.20 * bounded_score(scores.get("verifiability", 0.0)) + 0.20 * bounded_score(scores.get("novelty", 0.0)) - 0.20 * bounded_score(state.get("terminal_reject", 0.0)))


__all__ = ["SpatialSelectionExtension"]
