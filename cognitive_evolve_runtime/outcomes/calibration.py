"""Calibration tables and statistical solve blocking for M6."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now


CALIBRATION_VERSION = "m6-calibration/v1"


@dataclass(frozen=True)
class CalibrationPolicy:
    min_total_count: int = 30
    min_count_per_required_bin: int = 3
    max_ece: float = 0.12
    max_mce: float = 0.25
    max_brier_score: float = 0.22
    min_lower_confidence_coverage: float = 0.80
    required_bins: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def policy_hash(self) -> str:
        return "calp:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class CalibrationEvent:
    event_id: str
    prediction: float
    realization: float
    lower_confidence_bound: float = 0.0
    evidence_ref: str = ""
    trusted: bool = True
    observed_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", str(self.event_id or stable_hash(self.to_dict())))
        object.__setattr__(self, "prediction", max(0.0, min(1.0, _float(self.prediction))))
        object.__setattr__(self, "realization", max(0.0, min(1.0, _float(self.realization))))
        object.__setattr__(self, "lower_confidence_bound", max(0.0, min(1.0, _float(self.lower_confidence_bound))))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or ""))
        object.__setattr__(self, "trusted", bool(self.trusted))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def event_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("observed_at_utc", None)
        return "cale:" + stable_hash(payload)


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int = 0
    predicted_sum: float = 0.0
    realized_sum: float = 0.0
    squared_error_sum: float = 0.0
    lower_confidence_covered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def mean_prediction(self) -> float:
        return self.predicted_sum / self.count if self.count else 0.0

    def mean_realization(self) -> float:
        return self.realized_sum / self.count if self.count else 0.0

    def calibration_error(self) -> float:
        return abs(self.mean_prediction() - self.mean_realization())


@dataclass(frozen=True)
class CalibrationSnapshot:
    key: str
    bins: tuple[CalibrationBin, ...]
    total_count: int
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    lower_confidence_coverage: float
    event_hashes: tuple[str, ...]
    rejected_events: tuple[dict[str, Any], ...] = ()
    materialized_at_utc: str = field(default_factory=utc_now)
    version: str = CALIBRATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = self.stable_payload()
        payload["materialized_at_utc"] = self.materialized_at_utc
        payload["snapshot_hash"] = self.snapshot_hash()
        return payload

    def stable_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "key": self.key,
            "bins": [item.to_dict() for item in self.bins],
            "total_count": self.total_count,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "lower_confidence_coverage": self.lower_confidence_coverage,
            "event_hashes": list(self.event_hashes),
            "rejected_events": [dict(item) for item in self.rejected_events],
        }

    def snapshot_hash(self) -> str:
        return "cals:" + stable_hash(self.stable_payload())


def materialize_calibration_snapshot(
    events: list[CalibrationEvent | dict[str, Any]],
    *,
    key: str = "default",
    bin_count: int = 10,
) -> CalibrationSnapshot:
    bin_count = max(1, int(bin_count or 10))
    bins = [CalibrationBin(index / bin_count, (index + 1) / bin_count) for index in range(bin_count)]
    rejected: list[dict[str, Any]] = []
    accepted: list[CalibrationEvent] = []
    seen: set[str] = set()
    for raw in events:
        event = _event_from_any(raw)
        if event is None:
            rejected.append({"reason": "malformed_calibration_event"})
            continue
        if not event.trusted:
            rejected.append({"reason": "untrusted_calibration_event", "event_hash": event.event_hash()})
            continue
        if not event.evidence_ref:
            rejected.append({"reason": "missing_replayable_evidence_ref", "event_hash": event.event_hash()})
            continue
        if event.event_id in seen:
            rejected.append({"reason": "duplicate_calibration_event", "event_hash": event.event_hash()})
            continue
        seen.add(event.event_id)
        accepted.append(event)
        index = min(bin_count - 1, max(0, int(event.prediction * bin_count)))
        current = bins[index]
        covered = 1 if event.lower_confidence_bound <= event.realization else 0
        bins[index] = CalibrationBin(
            lower=current.lower,
            upper=current.upper,
            count=current.count + 1,
            predicted_sum=current.predicted_sum + event.prediction,
            realized_sum=current.realized_sum + event.realization,
            squared_error_sum=current.squared_error_sum + (event.prediction - event.realization) ** 2,
            lower_confidence_covered=current.lower_confidence_covered + covered,
        )
    total = sum(item.count for item in bins)
    brier = sum(item.squared_error_sum for item in bins) / total if total else 0.0
    ece = sum(item.count / total * item.calibration_error() for item in bins if total and item.count) if total else 1.0
    mce = max((item.calibration_error() for item in bins if item.count), default=1.0)
    coverage = sum(item.lower_confidence_covered for item in bins) / total if total else 0.0
    return CalibrationSnapshot(
        key=str(key or "default"),
        bins=tuple(bins),
        total_count=total,
        brier_score=brier,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        lower_confidence_coverage=coverage,
        event_hashes=tuple(item.event_hash() for item in accepted),
        rejected_events=tuple(rejected),
    )


def calibration_block_reasons(snapshot: CalibrationSnapshot | None, policy: CalibrationPolicy | None = None, *, candidate_confidence: float | None = None) -> tuple[str, ...]:
    policy = policy or CalibrationPolicy()
    if snapshot is None:
        return ("missing_calibration_snapshot",)
    reasons: list[str] = []
    if snapshot.total_count < policy.min_total_count:
        reasons.append("calibration_total_count_below_minimum")
    required = policy.required_bins or tuple(index for index, item in enumerate(snapshot.bins) if item.count > 0)
    for index in required:
        if 0 <= index < len(snapshot.bins) and snapshot.bins[index].count < policy.min_count_per_required_bin:
            reasons.append("calibration_required_bin_underpopulated")
            break
    if snapshot.expected_calibration_error > policy.max_ece:
        reasons.append("calibration_ece_above_limit")
    if snapshot.max_calibration_error > policy.max_mce:
        reasons.append("calibration_mce_above_limit")
    if snapshot.brier_score > policy.max_brier_score:
        reasons.append("calibration_brier_above_limit")
    if snapshot.lower_confidence_coverage < policy.min_lower_confidence_coverage:
        reasons.append("calibration_lower_confidence_coverage_below_limit")
    if candidate_confidence is not None:
        confidence = max(0.0, min(1.0, float(candidate_confidence)))
        covered = any(item.count > 0 and item.lower <= confidence <= item.upper for item in snapshot.bins)
        if not covered:
            reasons.append("candidate_confidence_outside_calibrated_support")
    return tuple(dict.fromkeys(reasons))


def calibration_allows_solve(snapshot: CalibrationSnapshot | None, policy: CalibrationPolicy | None = None, *, candidate_confidence: float | None = None) -> bool:
    return not calibration_block_reasons(snapshot, policy, candidate_confidence=candidate_confidence)


def _event_from_any(raw: CalibrationEvent | dict[str, Any] | None) -> CalibrationEvent | None:
    if isinstance(raw, CalibrationEvent):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    try:
        return CalibrationEvent(
            event_id=str(data.get("event_id") or ""),
            prediction=float(data.get("prediction")),
            realization=float(data.get("realization")),
            lower_confidence_bound=float(data.get("lower_confidence_bound") or 0.0),
            evidence_ref=str(data.get("evidence_ref") or ""),
            trusted=bool(data.get("trusted", True)),
            observed_at_utc=str(data.get("observed_at_utc") or utc_now()),
        )
    except Exception:
        return None


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


__all__ = [
    "CALIBRATION_VERSION",
    "CalibrationBin",
    "CalibrationEvent",
    "CalibrationPolicy",
    "CalibrationSnapshot",
    "calibration_allows_solve",
    "calibration_block_reasons",
    "materialize_calibration_snapshot",
]
