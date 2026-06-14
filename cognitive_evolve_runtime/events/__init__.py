"""Lightweight synchronous runtime event bus.

This is deliberately dependency-free: it gives engine components a stable way
to publish coarse lifecycle signals without turning the whole runtime into a
framework or exposing private reasoning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, DefaultDict, Generic, TypeVar
from collections import defaultdict

from ..artifacts.store import _now


@dataclass(frozen=True)
class RuntimeEvent:
    """Base event type for safe runtime lifecycle events."""

    event_type: str
    timestamp: str = field(default_factory=_now)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressRecorded(RuntimeEvent):
    event_type: str = "progress_recorded"


@dataclass(frozen=True)
class CandidateGenerated(RuntimeEvent):
    event_type: str = "candidate_generated"


@dataclass(frozen=True)
class EvidenceCollected(RuntimeEvent):
    event_type: str = "evidence_collected"


EventT = TypeVar("EventT", bound=RuntimeEvent)
Subscriber = Callable[[RuntimeEvent], None]


class EventBus:
    """Simple in-process pub/sub bus with best-effort subscriber isolation."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, list[Subscriber]] = defaultdict(list)
        self._all_subscribers: list[Subscriber] = []

    def subscribe(self, event_type: str | type[RuntimeEvent], handler: Subscriber) -> None:
        key = event_type if isinstance(event_type, str) else getattr(event_type, "event_type", event_type.__name__)
        self._subscribers[str(key)].append(handler)

    def subscribe_all(self, handler: Subscriber) -> None:
        self._all_subscribers.append(handler)

    def publish(self, event: RuntimeEvent) -> RuntimeEvent:
        handlers = list(self._all_subscribers) + list(self._subscribers.get(event.event_type, []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Event observers are diagnostic; core runtime must not fail because
                # a telemetry subscriber crashed.
                continue
        return event

    def clear(self) -> None:
        self._subscribers.clear()
        self._all_subscribers.clear()


GLOBAL_EVENT_BUS = EventBus()

from .progress import PipelineProgressEvent, EvolutionProgressEvent


__all__ = [
    "RuntimeEvent",
    "ProgressRecorded",
    "CandidateGenerated",
    "EvidenceCollected",
    "EventBus",
    "GLOBAL_EVENT_BUS",
    "PipelineProgressEvent",
    "EvolutionProgressEvent",
]
