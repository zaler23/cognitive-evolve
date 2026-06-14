"""External evaluator specifications."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict


@dataclass(frozen=True)
class EvaluatorMetricSpec:
    name: str
    direction: str = "maximize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvaluatorMetricSpec":
        direction = str(data.get("direction") or "maximize").strip().lower()
        if direction not in {"maximize", "minimize", "pass"}:
            direction = "maximize"
        return cls(name=str(data.get("name") or "score"), direction=direction)


@dataclass(frozen=True)
class EvaluatorSpec:
    enabled: bool = False
    command: str = ""
    timeout_seconds: float = 30.0
    deterministic: bool = True
    metrics: tuple[EvaluatorMetricSpec, ...] = field(default_factory=tuple)
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Do not persist cwd because it may be an absolute local path.
        return {
            "enabled": self.enabled,
            "command_configured": bool(self.command),
            "timeout_seconds": self.timeout_seconds,
            "deterministic": self.deterministic,
            "metrics": [item.to_dict() for item in self.metrics],
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EvaluatorSpec":
        data = coerce_dict(data)
        command = str(data.get("command") or "").strip()
        enabled = _bool(data.get("enabled"), default=bool(command))
        metrics_raw = data.get("metrics", [])
        metrics = tuple(
            EvaluatorMetricSpec.from_mapping(item)
            for item in metrics_raw
            if isinstance(item, dict) and item.get("name")
        )
        if not metrics:
            metrics = (EvaluatorMetricSpec("correctness", "pass"),)
        return cls(
            enabled=enabled and bool(command),
            command=command,
            timeout_seconds=_float(data.get("timeout_seconds"), 30.0),
            deterministic=_bool(data.get("deterministic"), default=True),
            metrics=metrics,
            cwd=str(data.get("cwd") or ""),
        )

    def cwd_path(self) -> Path:
        return Path(self.cwd).expanduser() if self.cwd else Path.cwd()


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["EvaluatorMetricSpec", "EvaluatorSpec"]
