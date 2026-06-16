"""State envelope for adaptive research extensions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now
from cognitive_evolve_runtime.nexus.adaptive.state import sanitize_adaptive_event

RESEARCH_STATE_VERSION = "adaptive-research-extensions/v1"


@dataclass
class ResearchRegistryState:
    version: str = RESEARCH_STATE_VERSION
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_search_pressures: list[dict[str, Any]] = field(default_factory=list)
    final_gate_directives: list[dict[str, Any]] = field(default_factory=list)
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
