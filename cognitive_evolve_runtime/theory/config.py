"""Configuration for the advisory-only theory layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", ""}:
            return False
    return default


def _float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else default


@dataclass(frozen=True)
class TheoryConfig:
    """Disabled-by-default configuration for M6.2 advisory producers."""

    enabled: bool = False
    mdl_enabled: bool = False
    boed_enabled: bool = False
    observer_enabled: bool = False
    geometry_enabled: bool = False
    causal_enabled: bool = False
    cellular_enabled: bool = False
    bandit_enabled: bool = False
    stability_enabled: bool = False
    mdl_weight: float = 0.0
    boed_weight: float = 0.0
    observer_weight: float = 0.0
    geometry_weight: float = 0.0
    causal_weight: float = 0.0
    cellular_weight: float = 0.0
    bandit_weight: float = 0.0
    stability_weight: float = 0.0
    per_producer_timeout_seconds: float = 0.05
    total_timeout_seconds: float = 0.15
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    cache_bound: int = 256
    telemetry_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TheoryConfig":
        data = dict(value or {})
        producers = dict(data.get("producers") or {}) if isinstance(data.get("producers"), Mapping) else {}
        weights = dict(data.get("weights") or {}) if isinstance(data.get("weights"), Mapping) else {}
        timeouts = dict(data.get("timeouts") or {}) if isinstance(data.get("timeouts"), Mapping) else {}
        enabled = _bool(data.get("enabled"), default=False)
        mdl_enabled = enabled and _bool(data.get("mdl_enabled", producers.get("mdl")), default=False)
        boed_enabled = enabled and _bool(data.get("boed_enabled", producers.get("boed")), default=False)
        observer_enabled = enabled and _bool(data.get("observer_enabled", producers.get("observer")), default=False)
        geometry_enabled = enabled and _bool(data.get("geometry_enabled", producers.get("geometry")), default=False)
        causal_enabled = enabled and _bool(data.get("causal_enabled", producers.get("causal")), default=False)
        cellular_enabled = enabled and _bool(data.get("cellular_enabled", producers.get("cellular")), default=False)
        bandit_enabled = enabled and _bool(data.get("bandit_enabled", producers.get("bandit")), default=False)
        stability_enabled = enabled and _bool(data.get("stability_enabled", producers.get("stability")), default=False)
        return cls(
            enabled=enabled,
            mdl_enabled=mdl_enabled,
            boed_enabled=boed_enabled,
            observer_enabled=observer_enabled,
            geometry_enabled=geometry_enabled,
            causal_enabled=causal_enabled,
            cellular_enabled=cellular_enabled,
            bandit_enabled=bandit_enabled,
            stability_enabled=stability_enabled,
            mdl_weight=_float(data.get("mdl_weight", weights.get("mdl")), default=0.0),
            boed_weight=_float(data.get("boed_weight", weights.get("boed")), default=0.0),
            observer_weight=_float(data.get("observer_weight", weights.get("observer")), default=0.0),
            geometry_weight=_float(data.get("geometry_weight", weights.get("geometry")), default=0.0),
            causal_weight=_float(data.get("causal_weight", weights.get("causal")), default=0.0),
            cellular_weight=_float(data.get("cellular_weight", weights.get("cellular")), default=0.0),
            bandit_weight=_float(data.get("bandit_weight", weights.get("bandit")), default=0.0),
            stability_weight=_float(data.get("stability_weight", weights.get("stability")), default=0.0),
            per_producer_timeout_seconds=max(0.0, _float(data.get("per_producer_timeout_seconds", timeouts.get("per_producer")), default=0.05)),
            total_timeout_seconds=max(0.0, _float(data.get("total_timeout_seconds", timeouts.get("total")), default=0.15)),
            clamp_min=_float(data.get("clamp_min"), default=-1.0),
            clamp_max=_float(data.get("clamp_max"), default=1.0),
            cache_bound=max(0, int(data.get("cache_bound", 256) or 0)),
            telemetry_enabled=enabled and _bool(data.get("telemetry_enabled"), default=False),
            metadata={str(k): v for k, v in data.items() if k not in {"enabled", "producers", "weights", "timeouts"}},
        )

    def weight_for(self, source: str) -> float:
        if source == "mdl":
            return self.mdl_weight
        if source == "boed":
            return self.boed_weight
        if source == "observer":
            return self.observer_weight
        if source == "geometry":
            return self.geometry_weight
        if source == "causal":
            return self.causal_weight
        if source == "cellular":
            return self.cellular_weight
        if source == "bandit":
            return self.bandit_weight
        if source == "stability":
            return self.stability_weight
        return 0.0


__all__ = ["TheoryConfig"]
