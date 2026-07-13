"""v2.3 honesty PI control signals for high-ceiling search pressure.

The signal computed here is advisory only.  It can change search pressure and
resource allocation, but it never grants solved status or verification strength.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.nexus.v23_theory_config import HonestyControlConfig
from cognitive_evolve_runtime.verification.strength import candidate_verification_results

HONESTY_DIMENSIONS = ("exogeneity", "variety", "falsification", "replay")


@dataclass(frozen=True)
class HonestyControlSignal:
    signal_id: str
    created_at: str
    sample_count: int
    honesty_vector: dict[str, float] = field(default_factory=dict)
    error_vector: dict[str, float] = field(default_factory=dict)
    proportional_term: dict[str, float] = field(default_factory=dict)
    integral_term: dict[str, float] = field(default_factory=dict)
    pressure: dict[str, float] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def neutral(cls, *, diagnostics: list[str] | None = None) -> "HonestyControlSignal":
        payload = {
            "sample_count": 0,
            "honesty_vector": {dimension: 1.0 for dimension in HONESTY_DIMENSIONS},
            "error_vector": {dimension: 0.0 for dimension in HONESTY_DIMENSIONS},
            "proportional_term": {dimension: 0.0 for dimension in HONESTY_DIMENSIONS},
            "integral_term": {dimension: 0.0 for dimension in HONESTY_DIMENSIONS},
            "pressure": _pressure_from_error({dimension: 0.0 for dimension in HONESTY_DIMENSIONS}),
            "diagnostics": list(diagnostics or []),
        }
        return cls(signal_id=_signal_id(payload), created_at=utc_now(), **payload)


def compute_honesty_control_signal(
    *,
    candidates: list[Any],
    config: HonestyControlConfig | None = None,
    history: list[dict[str, Any]] | None = None,
) -> HonestyControlSignal:
    cfg = config or HonestyControlConfig()
    samples = _honesty_samples(candidates)
    if not samples:
        return HonestyControlSignal.neutral(diagnostics=["honesty_control_neutral:no_honesty_measurements"])
    vector: dict[str, float] = {}
    for dimension in HONESTY_DIMENSIONS:
        key = f"{dimension}_score"
        values = [bounded_score(sample.get(key)) for sample in samples]
        vector[dimension] = bounded_score(sum(values) / max(1, len(values)))
    error = {dimension: bounded_score(1.0 - vector.get(dimension, 1.0)) for dimension in HONESTY_DIMENSIONS}
    history_errors = _history_errors(history or [])
    window = max(1, int(cfg.window or 1))
    integral_window = (history_errors + [error])[-window:]
    integral_mean = {
        dimension: bounded_score(sum(item.get(dimension, 0.0) for item in integral_window) / max(1, len(integral_window)))
        for dimension in HONESTY_DIMENSIONS
    }
    proportional = {
        dimension: _clamp(float(cfg.proportional_gain.get(dimension, 0.0)) * error[dimension], cfg.clamp_abs)
        for dimension in HONESTY_DIMENSIONS
    }
    integral = {
        dimension: _clamp(float(cfg.integral_gain.get(dimension, 0.0)) * integral_mean[dimension], cfg.clamp_abs)
        for dimension in HONESTY_DIMENSIONS
    }
    controlled_error = {
        dimension: _clamp(proportional[dimension] + integral[dimension], cfg.clamp_abs)
        for dimension in HONESTY_DIMENSIONS
    }
    pressure = _pressure_from_error(controlled_error)
    payload = {
        "sample_count": len(samples),
        "honesty_vector": vector,
        "error_vector": error,
        "proportional_term": proportional,
        "integral_term": integral,
        "pressure": pressure,
        "diagnostics": [],
    }
    return HonestyControlSignal(signal_id=_signal_id(payload), created_at=utc_now(), **payload)


def _honesty_samples(candidates: list[Any]) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for candidate in candidates or []:
        for result in candidate_verification_results(candidate):
            measurements = coerce_dict(result.metadata.get("honesty_measurements"))
            if not measurements:
                continue
            sample = {
                f"{dimension}_score": bounded_score(measurements.get(f"{dimension}_score"))
                for dimension in HONESTY_DIMENSIONS
            }
            samples.append(sample)
    return samples


def _history_errors(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in history:
        data = coerce_dict(item)
        vector = coerce_dict(data.get("error_vector")) or data
        if not vector:
            continue
        out.append({dimension: bounded_score(vector.get(dimension)) for dimension in HONESTY_DIMENSIONS})
    return out


def _pressure_from_error(error: dict[str, float]) -> dict[str, float]:
    exogeneity = bounded_score(error.get("exogeneity"))
    variety = bounded_score(error.get("variety"))
    falsification = bounded_score(error.get("falsification"))
    replay = bounded_score(error.get("replay"))
    return {
        "adversarial_budget_pressure": bounded_score(max(exogeneity, falsification)),
        "rarity_budget_pressure": bounded_score(variety),
        "edge_seed_pressure": bounded_score(variety),
        "frontier_exploration_pressure": bounded_score(max(variety, exogeneity)),
        "replay_verifier_pressure": replay,
        "verification_pressure": bounded_score(max(falsification, replay)),
    }


def _clamp(value: float, clamp_abs: float) -> float:
    limit = abs(float(clamp_abs or 0.0))
    if limit <= 0.0:
        return 0.0
    return max(-limit, min(limit, float(value)))


def _signal_id(payload: dict[str, Any]) -> str:
    return "honesty-control-" + stable_hash(payload)[:16]


__all__ = ["HONESTY_DIMENSIONS", "HonestyControlSignal", "compute_honesty_control_signal"]
