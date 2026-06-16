"""Telemetry helpers for adaptive evidence-layer events."""
from __future__ import annotations

from typing import Any


def adaptive_event(event_type: str, **payload: Any) -> dict[str, Any]:
    event = {"type": str(event_type)}
    event.update(payload)
    return event


__all__ = ["adaptive_event"]
