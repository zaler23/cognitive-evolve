"""Validated advisory-only signal schemas for M6.2."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

TheorySignalSource = Literal["mdl", "boed", "observer", "causal", "geometry", "cellular", "bandit", "stability"]
TheorySignalKind = Literal["rank_prior", "plan_value", "risk", "diversity", "diagnostic"]
TheoryTargetType = Literal["candidate", "lineage", "population", "outcome", "plan"]

_FORBIDDEN_KEYS = {
    "pass",
    "fail",
    "passed",
    "failed",
    "promote",
    "promotion",
    "certified",
    "verdict",
    "gate",
    "proof",
    "certificate",
    "certref",
    "certificateid",
    "gateresult",
    "promotiondecision",
    "accepted",
    "rejected",
}
_ALLOWED_SOURCES = {"mdl", "boed", "observer", "causal", "geometry", "cellular", "bandit", "stability"}
_ALLOWED_KINDS = {"rank_prior", "plan_value", "risk", "diversity", "diagnostic"}
_ALLOWED_TARGETS = {"candidate", "lineage", "population", "outcome", "plan"}


@dataclass(frozen=True)
class TheorySignal:
    source: TheorySignalSource
    kind: TheorySignalKind
    target_type: TheoryTargetType
    cycle_id: str
    target_id: str
    value: float
    confidence: float = 1.0
    interval: tuple[float, float] | None = None
    provenance: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)
    advisory_only: bool = True

    def __post_init__(self) -> None:
        if self.source not in _ALLOWED_SOURCES:
            raise ValueError(f"unknown theory signal source: {self.source}")
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"unknown theory signal kind: {self.kind}")
        if self.target_type not in _ALLOWED_TARGETS:
            raise ValueError(f"unknown theory target type: {self.target_type}")
        if not str(self.cycle_id).strip() or not str(self.target_id).strip():
            raise ValueError("theory signal requires non-empty cycle_id and target_id")
        if self.advisory_only is not True:
            raise ValueError("theory signals must be advisory_only=True")
        value = _finite_float(self.value, "value")
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        interval = None
        if self.interval is not None:
            if len(self.interval) != 2:
                raise ValueError("interval must contain two finite floats")
            lo = _finite_float(self.interval[0], "interval.low")
            hi = _finite_float(self.interval[1], "interval.high")
            if lo > hi:
                raise ValueError("interval low must be <= high")
            interval = (lo, hi)
        meta = _freeze_json_safe(self.meta)
        _scan_forbidden_keys(meta)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "provenance", tuple(str(item) for item in self.provenance if str(item)))
        object.__setattr__(self, "meta", meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "target_type": self.target_type,
            "cycle_id": self.cycle_id,
            "target_id": self.target_id,
            "value": self.value,
            "confidence": self.confidence,
            "interval": list(self.interval) if self.interval is not None else None,
            "provenance": list(self.provenance),
            "meta": _thaw(self.meta),
            "advisory_only": True,
        }


@dataclass(frozen=True)
class AdvisoryRankingFeatures:
    candidate_id: str
    rank_prior: float = 0.0
    plan_value: float = 0.0
    risk: float = 0.0
    diversity: float = 0.0
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id is required")
        for field_name in ("rank_prior", "plan_value", "risk", "diversity"):
            object.__setattr__(self, field_name, _finite_float(getattr(self, field_name), field_name))
        object.__setattr__(self, "provenance", tuple(str(item) for item in self.provenance if str(item)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"provenance": list(self.provenance)}


def validate_theory_signal_json_safe(signal: TheorySignal) -> None:
    json.dumps(signal.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)


def forbidden_key_paths(payload: Any) -> tuple[str, ...]:
    paths: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if _normalize_key(key_text) in _FORBIDDEN_KEYS:
                    paths.append(child_path)
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")
    return tuple(paths)


def _scan_forbidden_keys(payload: Any) -> None:
    paths = forbidden_key_paths(payload)
    if paths:
        raise ValueError("theory metadata contains forbidden proof/gate/verdict keys: " + ", ".join(paths[:8]))


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _freeze_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json_safe(child) for key, child in value.items()}
        json.dumps(_thaw(frozen), ensure_ascii=False, sort_keys=True, allow_nan=False)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        frozen_tuple = tuple(_freeze_json_safe(item) for item in value)
        json.dumps(_thaw(frozen_tuple), ensure_ascii=False, sort_keys=True, allow_nan=False)
        return frozen_tuple
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("meta must be JSON-safe and finite")
        return value
    raise ValueError(f"meta contains non-JSON-safe value: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "AdvisoryRankingFeatures",
    "TheorySignal",
    "forbidden_key_paths",
    "validate_theory_signal_json_safe",
]
