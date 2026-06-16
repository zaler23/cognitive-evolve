"""Online concept effect accounting; no rerun benchmark is performed."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConceptEffectStats:
    concept_id: str
    effect_count: int = 0
    decision_changed_count: int = 0
    token_cost: float = 0.0
    sandbox_seconds: float = 0.0
    frontier_gain: float = 0.0
    novelty_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        total_cost = max(1.0, self.token_cost + self.sandbox_seconds)
        data["decision_changed_per_cost"] = self.decision_changed_count / total_cost
        return data


@dataclass
class ConceptEffectReport:
    stats: dict[str, ConceptEffectStats] = field(default_factory=dict)

    def record_trace_entry(self, entry: dict[str, Any]) -> None:
        concept_id = str(entry.get("concept_id") or "unknown")
        stats = self.stats.setdefault(concept_id, ConceptEffectStats(concept_id=concept_id))
        produced = entry.get("produced_effects") if isinstance(entry.get("produced_effects"), dict) else {}
        if entry.get("decision_changed"):
            # Only actual application traces count as effective concept effects.
            stats.effect_count += sum(1 for value in produced.values() if value) or (1 if produced else 0)
            stats.decision_changed_count += 1
        cost = entry.get("cost") if isinstance(entry.get("cost"), dict) else {}
        stats.token_cost += _float(cost.get("tokens"), 0.0)
        stats.sandbox_seconds += _float(cost.get("sandbox_seconds") or cost.get("tool_seconds") or cost.get("seconds"), 0.0)
        stats.frontier_gain += _float(produced.get("frontier_gain"), 0.0)
        stats.novelty_gain += _float(produced.get("novelty_gain"), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"concepts": {key: value.to_dict() for key, value in sorted(self.stats.items())}}

    @classmethod
    def from_trace_entries(cls, entries: list[dict[str, Any]] | None) -> "ConceptEffectReport":
        report = cls()
        for entry in entries or []:
            if isinstance(entry, dict):
                report.record_trace_entry(entry)
        return report


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ConceptEffectReport", "ConceptEffectStats"]
