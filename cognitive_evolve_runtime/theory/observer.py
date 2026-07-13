"""Observer diagnostics over completed immutable snapshots."""
from __future__ import annotations

from .representations import CompletedEventSnapshot
from .signals import TheorySignal


def observe_completed_events(events: tuple[CompletedEventSnapshot, ...]) -> tuple[TheorySignal, ...]:
    signals: list[TheorySignal] = []
    for event in events:
        diagnostic_mass = min(1.0, len(event.diagnostics) / 5.0)
        if diagnostic_mass <= 0.0:
            continue
        signals.append(
            TheorySignal(
                source="observer",
                kind="diagnostic",
                target_type="outcome",
                cycle_id=event.cycle_id,
                target_id=event.target_id,
                value=diagnostic_mass,
                confidence=0.5,
                provenance=("observer:completed_event_snapshot",),
                meta={"event_type": event.event_type, "diagnostic_count": len(event.diagnostics)},
            )
        )
    return tuple(signals)


__all__ = ["observe_completed_events"]
