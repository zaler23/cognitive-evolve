"""Cost ledger and backpressure extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import SearchPressure, evidence_records, evidence_state
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class BudgetBackpressureExtension:
    extension_id = "budget_backpressure"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.ledger: dict[str, dict[str, Any]] = {}
        self.backpressure = False

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        for candidate in ctx.candidates:
            records = evidence_records(candidate)
            if not records:
                continue
            cost = sum(_cost_value(record.cost) for record in records)
            state = evidence_state(candidate)
            improvement = bounded_score(state.get("search_score", 0.0))
            self.ledger[candidate.id] = {"spent_cost": cost, "search_score": improvement, "roi": bounded_score(improvement / max(1.0, cost)), "targeted": len(state.get("target_challenge_ids") or [])}
        pending = sum(1 for item in self.ledger.values() if item.get("targeted", 0) and item.get("search_score", 0.0) < 0.5)
        threshold = _int(self.config.get("pending_threshold"), 8)
        self.backpressure = pending > threshold
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics={"budget_ledger_candidate_count": len(self.ledger), "backpressure_active": self.backpressure, "pending_targeted_low_score": pending})

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        advisory = {}
        for candidate in ctx.candidates:
            item = self.ledger.get(candidate.id)
            if not item:
                continue
            advisory[candidate.id] = {"plan_value": bounded_score(item.get("roi", 0.0)), "risk": 0.2 if self.backpressure and item.get("roi", 0.0) < 0.1 else 0.0, "rank_prior": 0.0, "diversity": 0.0}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        if not self.backpressure or ctx.parent is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction="Backpressure is active: prefer cheap repair, schema cleanup, evaluator-drain, and avoid broad speculative branching this round.", metadata={"source_extension": self.extension_id, "backpressure": True})
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure], warnings=["budget_backpressure_active_generation_pressure_reduced"])

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"ledger": self.ledger, "backpressure": self.backpressure} if self.ledger or self.backpressure else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.ledger = {str(k): dict(v) for k, v in ((state or {}).get("ledger") or {}).items() if isinstance(v, dict)}
        self.backpressure = bool((state or {}).get("backpressure"))


def _cost_value(cost: dict[str, Any]) -> float:
    for key in ("tokens", "seconds", "tool_seconds", "estimated"):
        try:
            value = float((cost or {}).get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 1.0


__all__ = ["BudgetBackpressureExtension"]


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
