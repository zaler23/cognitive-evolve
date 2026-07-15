"""Configuration discovery for the adaptive evidence layer."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict


@dataclass(frozen=True)
class SpatialAdaptiveConfig:
    enabled: bool = False
    width: int = 0
    height: int = 0
    region_size: int = 3
    neighborhood: str = "moore"
    toroidal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SpatialAdaptiveConfig":
        data = coerce_dict(data)
        return cls(
            enabled=_bool(data.get("enabled"), default=False),
            width=_int(data.get("width"), default=0),
            height=_int(data.get("height"), default=0),
            region_size=max(1, _int(data.get("region_size"), default=3)),
            neighborhood=str(data.get("neighborhood") or "moore").strip().lower() or "moore",
            toroidal=_bool(data.get("toroidal"), default=True),
        )


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = False
    evaluator: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    spatial: SpatialAdaptiveConfig = field(default_factory=SpatialAdaptiveConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spatial"] = self.spatial.to_dict()
        return data

    @classmethod
    def from_sources(
        cls,
        *,
        explicit: dict[str, Any] | None = None,
        contract: Any | None = None,
        policy: Any | None = None,
        world: Any | None = None,
    ) -> "AdaptiveConfig":
        merged: dict[str, Any] = {}
        for source in (_metadata(contract), _criterion_evaluator_config(contract), _metadata(policy), _metadata(world), coerce_dict(explicit)):
            merged = _deep_merge(merged, coerce_dict(source.get("adaptive") if "adaptive" in source else source))
        env_config = _env_config()
        if env_config:
            merged = _deep_merge(merged, env_config)
        evaluator = coerce_dict(merged.get("evaluator"))
        if "command" in coerce_dict(merged.get("external_evaluator")):
            evaluator = _deep_merge(evaluator, coerce_dict(merged.get("external_evaluator")))
        evaluator_enabled = _bool(evaluator.get("enabled"), default=bool(str(evaluator.get("command") or "").strip()))
        if evaluator_enabled:
            evaluator["enabled"] = True
        spatial = SpatialAdaptiveConfig.from_mapping(merged.get("spatial"))
        enabled = _bool(merged.get("enabled"), default=evaluator_enabled or spatial.enabled)
        return cls(
            enabled=enabled,
            evaluator=evaluator,
            evidence=coerce_dict(merged.get("evidence")),
            spatial=spatial,
        )

    @property
    def enabled_features(self) -> dict[str, bool]:
        evaluator_enabled = _bool(self.evaluator.get("enabled"), default=bool(str(self.evaluator.get("command") or "").strip()))
        return {
            "adaptive": self.enabled,
            "evidence_control_plane": self.enabled,
            "external_evaluator": self.enabled and evaluator_enabled,
            "spatial_observe": self.enabled and self.spatial.enabled,
        }


def _metadata(source: Any | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return coerce_dict(source.get("metadata") if "metadata" in source else source)
    metadata = getattr(source, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    if hasattr(source, "to_dict"):
        return coerce_dict(source.to_dict().get("metadata"))
    return {}


def _criterion_evaluator_config(contract: Any | None) -> dict[str, Any]:
    bindings = getattr(contract, "evaluators", None)
    if not isinstance(bindings, list):
        return {}
    metrics: list[dict[str, Any]] = []
    for binding in bindings:
        data = binding.to_dict() if hasattr(binding, "to_dict") else asdict(binding) if hasattr(binding, "__dataclass_fields__") else coerce_dict(binding)
        if data.get("kind") != "criterion_metric":
            continue
        metrics.append({
            "name": str(data.get("metric") or ""),
            "direction": str(data.get("direction") or ""),
            "value_type": str(data.get("value_type") or ""),
            "source_span": coerce_dict(data.get("source_span")),
        })
    return {"evaluator": {"metrics": metrics}} if metrics else {}


def _env_config() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if os.environ.get("COGEV_ADAPTIVE_ENABLED"):
        out["enabled"] = os.environ.get("COGEV_ADAPTIVE_ENABLED")
    if os.environ.get("COGEV_EXTERNAL_EVALUATOR_COMMAND"):
        out.setdefault("evaluator", {})["enabled"] = True
        out["evaluator"]["command"] = os.environ.get("COGEV_EXTERNAL_EVALUATOR_COMMAND")
    if os.environ.get("COGEV_EXTERNAL_EVALUATOR_TIMEOUT"):
        out.setdefault("evaluator", {})["timeout_seconds"] = os.environ.get("COGEV_EXTERNAL_EVALUATOR_TIMEOUT")
    if os.environ.get("COGEV_MACHINE_ARTIFACT_REQUIRED"):
        out.setdefault("evidence", {})["machine_artifact_required"] = os.environ.get("COGEV_MACHINE_ARTIFACT_REQUIRED")
    if os.environ.get("COGEV_ARTIFACT_TYPE"):
        out.setdefault("evidence", {})["artifact_type"] = os.environ.get("COGEV_ARTIFACT_TYPE")
    return out


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["AdaptiveConfig", "SpatialAdaptiveConfig"]
