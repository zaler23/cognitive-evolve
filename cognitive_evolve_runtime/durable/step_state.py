"""Durable step state contracts for resumable execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StepStateName = Literal[
    "created",
    "ready",
    "inflight",
    "committed",
    "retryable_failed",
    "terminal_failed",
    "cancelled",
    "resumed",
]

TERMINAL_STATES = {"committed", "terminal_failed", "cancelled"}
RETRYABLE_STATES = {"created", "ready", "retryable_failed", "resumed"}
ALL_STEP_STATES: tuple[str, ...] = (
    "created",
    "ready",
    "inflight",
    "committed",
    "retryable_failed",
    "terminal_failed",
    "cancelled",
    "resumed",
)

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {"ready", "inflight", "cancelled", "terminal_failed"},
    "ready": {"inflight", "cancelled", "terminal_failed"},
    "inflight": {"committed", "retryable_failed", "terminal_failed", "cancelled"},
    "retryable_failed": {"ready", "inflight", "resumed", "terminal_failed", "cancelled"},
    "resumed": {"ready", "inflight", "terminal_failed", "cancelled"},
    "committed": set(),
    "terminal_failed": set(),
    "cancelled": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepStatus:
    run_id: str
    round_id: str
    step_id: str
    step_name: str
    state: str = "created"
    attempt: int = 0
    input_hash: str = ""
    output_hash: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    reason: str = ""
    error: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: str, *, reason: str = "", error: dict[str, Any] | None = None, **metadata: Any) -> "StepStatus":
        validate_transition(self.state, new_state)
        self.state = new_state
        self.reason = reason or self.reason
        if error is not None:
            self.error = dict(error)
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = utc_now()
        return self

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepStatus":
        return cls(
            run_id=str(data.get("run_id") or "run"),
            round_id=str(data.get("round_id") or "round"),
            step_id=str(data.get("step_id") or "step"),
            step_name=str(data.get("step_name") or data.get("step_id") or "step"),
            state=str(data.get("state") or "created"),
            attempt=int(data.get("attempt") or 0),
            input_hash=str(data.get("input_hash") or ""),
            output_hash=str(data.get("output_hash") or ""),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            reason=str(data.get("reason") or ""),
            error=dict(data.get("error") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


def validate_state(state: str) -> str:
    if state not in ALL_STEP_STATES:
        raise ValueError(f"invalid durable step state: {state!r}")
    return state


def validate_transition(old: str, new: str) -> None:
    validate_state(old)
    validate_state(new)
    if old == new:
        return
    if old in TERMINAL_STATES:
        raise ValueError(f"cannot transition terminal durable step from {old!r} to {new!r}")
    if new not in _ALLOWED_TRANSITIONS.get(old, set()):
        raise ValueError(f"invalid durable step transition from {old!r} to {new!r}")


__all__ = [
    "StepStateName",
    "StepStatus",
    "ALL_STEP_STATES",
    "TERMINAL_STATES",
    "RETRYABLE_STATES",
    "validate_state",
    "validate_transition",
    "utc_now",
]
