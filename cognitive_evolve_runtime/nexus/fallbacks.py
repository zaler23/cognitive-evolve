"""Structured fallback audit events for Nexus runtime.

Fallbacks are valid runtime behavior, but they must be visible in durable run
artifacts even when CLI or API logging handlers are not configured.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
import logging
from typing import Any, Iterator

LOGGER = logging.getLogger("cognitive_evolve_runtime.nexus.fallbacks")
_FALLBACK_EVENTS: ContextVar[list[dict[str, str]] | None] = ContextVar("cogev_nexus_fallback_events", default=None)


def _safe_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    for marker in ("/Users/", "file://", "Bearer ", "api_key", "API_KEY", "password", "secret"):
        if marker in text:
            text = text.replace(marker, "[redacted]")
    return text[:limit]


def start_fallback_capture() -> Token[list[dict[str, str]] | None]:
    return _FALLBACK_EVENTS.set([])


def finish_fallback_capture(token: Token[list[dict[str, str]] | None]) -> list[dict[str, str]]:
    events = list(_FALLBACK_EVENTS.get() or [])
    _FALLBACK_EVENTS.reset(token)
    return events


@contextmanager
def capture_fallback_events() -> Iterator[list[dict[str, str]]]:
    token = start_fallback_capture()
    events = _FALLBACK_EVENTS.get()
    try:
        yield events if events is not None else []
    finally:
        _FALLBACK_EVENTS.reset(token)


def record_fallback(*, stage: str, reason: str, detail: str = "", target: Any | None = None) -> dict[str, str]:
    event = {
        "type": "nexus_fallback",
        "event": "nexus_fallback",
        "stage": _safe_text(stage or "unknown", limit=120),
        "reason": _safe_text(reason or "unknown", limit=160),
    }
    if detail:
        event["detail"] = _safe_text(detail, limit=500)
    if target is not None:
        event["target"] = _safe_text(target, limit=200)
    captured = _FALLBACK_EVENTS.get()
    if captured is not None:
        captured.append(dict(event))
        del captured[:-200]
    LOGGER.warning("nexus_fallback", extra={"nexus_fallback": event})
    return event


__all__ = ["capture_fallback_events", "finish_fallback_capture", "record_fallback", "start_fallback_capture"]
