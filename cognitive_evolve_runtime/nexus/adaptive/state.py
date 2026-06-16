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
    config: dict[str, Any] = field(default_factory=dict)
    spatial: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    immune: dict[str, Any] | None = None
    pattern_memory: dict[str, Any] | None = None
    mdl: dict[str, Any] | None = None
    evaluator: dict[str, Any] | None = None
    challenge_memory: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    final_certificate: dict[str, Any] = field(default_factory=dict)
    research_extensions: dict[str, Any] = field(default_factory=dict)
    research_metrics: dict[str, Any] = field(default_factory=dict)
    research_warnings: list[str] = field(default_factory=list)
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
            config=coerce_dict(data.get("config")),
            spatial=_optional_dict(data.get("spatial")),
            budget=_optional_dict(data.get("budget")),
            immune=_optional_dict(data.get("immune")),
            pattern_memory=_optional_dict(data.get("pattern_memory")),
            mdl=_optional_dict(data.get("mdl")),
            evaluator=_optional_dict(data.get("evaluator")),
            challenge_memory=_optional_dict(data.get("challenge_memory") or data.get("challenge_bank")),
            metrics=coerce_dict(data.get("metrics")),
            warnings=[str(item) for item in data.get("warnings", []) if item],
            events=[dict(item) for item in data.get("events", []) if isinstance(item, dict)],
            final_certificate=coerce_dict(data.get("final_certificate")),
            research_extensions=coerce_dict(data.get("research_extensions")),
            research_metrics=coerce_dict(data.get("research_metrics")),
            research_warnings=[str(item) for item in data.get("research_warnings", []) if item],
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
        safe[key_s] = _sanitize_value(value)
    return safe


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated-depth]"
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_sanitize_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:20]:
            key_s = str(key)
            if any(token in key_s.lower() for token in ("key", "secret", "token", "password", "prompt")):
                continue
            out[key_s] = _sanitize_value(item, depth=depth + 1)
        return out
    return _safe_string(str(value))


def _safe_string(value: str) -> str:
    text = str(value or "")
    for prefix in (("/" + "Users/"), "/Volumes/", "/private/", "/tmp/", "/var/", "/opt/"):
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
