from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

EVENTS: list[dict[str, Any]] = []


@dataclass
class LLMSession:
    """Request-local LLM telemetry, budget state, and optional durable journal."""

    events: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    run_id: str | None = None
    journal_dir: str | None = None
    call_ledger_path: str | None = None

    def record(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)

    def clear(self) -> None:
        with self.lock:
            self.events.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)

    def total_estimated_cost_usd(self) -> float:
        return round(sum(float(event.get("estimated_cost_usd") or 0.0) for event in self.snapshot()), 6)


_DEFAULT_SESSION = LLMSession(EVENTS)
_CURRENT_SESSION: ContextVar[LLMSession | None] = ContextVar("cogev_llm_session", default=None)
_LAST_RETRY_HISTORY: ContextVar[list[dict[str, Any]]] = ContextVar("cogev_llm_retry_history", default=[])


def current_llm_session() -> LLMSession:
    return _CURRENT_SESSION.get() or _DEFAULT_SESSION


@contextmanager
def llm_session(session: LLMSession | None = None) -> Iterator[LLMSession]:
    scoped = session or LLMSession()
    token = _CURRENT_SESSION.set(scoped)
    try:
        yield scoped
    finally:
        _CURRENT_SESSION.reset(token)


def reset_llm_events() -> None:
    current_llm_session().clear()
