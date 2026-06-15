"""Serializable adaptive evidence-layer runtime state.

This module is deliberately a narrow state envelope.  Adaptive mechanisms may
add feature-specific summaries, but checkpoint/resume and final persistence only
need to understand this one object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now

ADAPTIVE_STATE_VERSION = "adaptive-runtime-state/v1"


@dataclass
class AdaptiveRuntimeState:
    version: str = ADAPTIVE_STATE_VERSION
    round_index: int = 0
    enabled_features: dict[str, bool] = field(default_factory=dict)
    spatial: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    immune: dict[str, Any] | None = None
    pattern_memory: dict[str, Any] | None = None
    mdl: dict[str, Any] | None = None
    evaluator: dict[str, Any] | None = None
    challenge_bank: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    final_certificate: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "AdaptiveRuntimeState" | None) -> "AdaptiveRuntimeState":
        if isinstance(data, AdaptiveRuntimeState):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            version=str(data.get("version") or ADAPTIVE_STATE_VERSION),
            round_index=_int(data.get("round_index"), default=0),
            enabled_features={str(k): bool(v) for k, v in coerce_dict(data.get("enabled_features")).items()},
            spatial=_optional_dict(data.get("spatial")),
            budget=_optional_dict(data.get("budget")),
            immune=_optional_dict(data.get("immune")),
            pattern_memory=_optional_dict(data.get("pattern_memory")),
            mdl=_optional_dict(data.get("mdl")),
            evaluator=_optional_dict(data.get("evaluator")),
            challenge_bank=_optional_dict(data.get("challenge_bank")),
            metrics=coerce_dict(data.get("metrics")),
            warnings=[str(item) for item in data.get("warnings", []) if item],
            events=[dict(item) for item in data.get("events", []) if isinstance(item, dict)],
            final_certificate=coerce_dict(data.get("final_certificate")),
            updated_at=str(data.get("updated_at") or utc_now()),
        )

    def record_event(self, event: dict[str, Any]) -> None:
        payload = sanitize_adaptive_event(event)
        payload.setdefault("at", utc_now())
        self.events.append(payload)
        self.events = self.events[-200:]
        self.updated_at = utc_now()


def sanitize_adaptive_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep adaptive events safe for public artifacts.

    Events intentionally avoid full prompts, API keys, and absolute local paths.
    """

    safe: dict[str, Any] = {}
    for key, value in dict(event or {}).items():
        key_s = str(key)
        if any(token in key_s.lower() for token in ("key", "secret", "token", "password", "prompt")):
            continue
        if isinstance(value, str):
            safe[key_s] = _safe_string(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key_s] = value
        elif isinstance(value, list):
            safe[key_s] = [_safe_string(str(item)) if isinstance(item, str) else item for item in value[:20]]
        elif isinstance(value, dict):
            safe[key_s] = {str(k): _safe_string(str(v)) for k, v in list(value.items())[:20] if not any(t in str(k).lower() for t in ("key", "secret", "token", "password", "prompt"))}
        else:
            safe[key_s] = _safe_string(str(value))
    return safe


def _safe_string(value: str) -> str:
    text = str(value or "")
    for prefix in ("/Users/", "/Volumes/", "/private/", "/tmp/", "/var/", "/opt/"):
        if prefix in text:
            return "[redacted-path]"
    return text[:500]


def _optional_dict(value: Any) -> dict[str, Any] | None:
    data = coerce_dict(value)
    return data or None


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ADAPTIVE_STATE_VERSION", "AdaptiveRuntimeState", "sanitize_adaptive_event"]
