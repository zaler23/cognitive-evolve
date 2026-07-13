"""Isolated advisory telemetry namespace for M6.2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .signals import TheorySignal

THEORY_TELEMETRY_NAMESPACE = "theory_advisory"


@dataclass
class TheoryTelemetry:
    namespace: str = THEORY_TELEMETRY_NAMESPACE
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, cycle_id: str, producer: str, signals: tuple[TheorySignal, ...], diagnostics: tuple[str, ...] = ()) -> None:
        self.records.append(
            {
                "namespace": self.namespace,
                "cycle_id": str(cycle_id),
                "producer": str(producer),
                "signals": [signal.to_dict() for signal in signals],
                "diagnostics": [str(item) for item in diagnostics if str(item)],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "records": list(self.records)}


__all__ = ["THEORY_TELEMETRY_NAMESPACE", "TheoryTelemetry"]
