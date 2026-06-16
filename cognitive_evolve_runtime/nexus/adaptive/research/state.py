"""State envelope for adaptive research extensions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now
from cognitive_evolve_runtime.nexus.adaptive.state import sanitize_adaptive_event

RESEARCH_STATE_VERSION = "adaptive-research-extensions/v2"


@dataclass
class ResearchRegistryState:
    version: str = RESEARCH_STATE_VERSION
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_search_pressures: list[dict[str, Any]] = field(default_factory=list)
    final_gate_directives: list[dict[str, Any]] = field(default_factory=list)
    verification_obligations: list[dict[str, Any]] = field(default_factory=list)
    archive_directives: list[dict[str, Any]] = field(default_factory=list)
    budget_directives: list[dict[str, Any]] = field(default_factory=list)
    context_transforms: list[dict[str, Any]] = field(default_factory=list)
    candidate_transforms: list[dict[str, Any]] = field(default_factory=list)
    contract_delta_proposals: list[dict[str, Any]] = field(default_factory=list)
    trace_entries: list[dict[str, Any]] = field(default_factory=list)
    concept_effect_report: dict[str, Any] = field(default_factory=dict)
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    tension_map: dict[str, Any] = field(default_factory=dict)
    verification_plan: dict[str, Any] = field(default_factory=dict)
    graded_output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    target_tracking: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchRegistryState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            version=str(data.get("version") or RESEARCH_STATE_VERSION),
            extensions={str(k): coerce_dict(v) for k, v in coerce_dict(data.get("extensions")).items()},
            pending_search_pressures=[dict(item) for item in data.get("pending_search_pressures", []) if isinstance(item, dict)],
            final_gate_directives=[dict(item) for item in data.get("final_gate_directives", []) if isinstance(item, dict)],
            verification_obligations=[dict(item) for item in data.get("verification_obligations", []) if isinstance(item, dict)],
            archive_directives=[dict(item) for item in data.get("archive_directives", []) if isinstance(item, dict)],
            budget_directives=[dict(item) for item in data.get("budget_directives", []) if isinstance(item, dict)],
            context_transforms=[dict(item) for item in data.get("context_transforms", []) if isinstance(item, dict)],
            candidate_transforms=[dict(item) for item in data.get("candidate_transforms", []) if isinstance(item, dict)],
            contract_delta_proposals=[dict(item) for item in data.get("contract_delta_proposals", []) if isinstance(item, dict)],
            trace_entries=[dict(item) for item in data.get("trace_entries", []) if isinstance(item, dict)],
            concept_effect_report=coerce_dict(data.get("concept_effect_report")),
            cost_ledger=coerce_dict(data.get("cost_ledger")),
            tension_map=coerce_dict(data.get("tension_map")),
            verification_plan=coerce_dict(data.get("verification_plan")),
            graded_output=coerce_dict(data.get("graded_output")),
            metrics=coerce_dict(data.get("metrics")),
            warnings=[str(item) for item in data.get("warnings", []) if item],
            events=[dict(item) for item in data.get("events", []) if isinstance(item, dict)],
            target_tracking=coerce_dict(data.get("target_tracking")),
            updated_at=str(data.get("updated_at") or utc_now()),
        )

    def record_event(self, event: dict[str, Any]) -> None:
        safe = sanitize_adaptive_event(event)
        safe.setdefault("at", utc_now())
        self.events.append(safe)
        self.events = self.events[-200:]
        self.updated_at = utc_now()


__all__ = ["RESEARCH_STATE_VERSION", "ResearchRegistryState"]
