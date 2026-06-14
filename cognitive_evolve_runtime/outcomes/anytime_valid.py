"""Anytime-valid e-process certificates for M6 closure gates."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now


ANYTIME_VALID_VERSION = "anytime-valid-eprocess/v1"
DEFAULT_BETTING_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)


@dataclass(frozen=True)
class EProcessConfig:
    metric_id: str
    direction: str = "higher"
    null_margin: float = 0.0
    value_min: float = 0.0
    value_max: float = 1.0
    alpha: float = 0.05
    min_trials: int = 1
    max_single_trial_weight: float = 1.0
    betting_fractions: tuple[float, ...] = DEFAULT_BETTING_FRACTIONS
    version: str = ANYTIME_VALID_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_id", str(self.metric_id or "metric"))
        direction = str(self.direction or "higher").lower()
        object.__setattr__(self, "direction", "lower" if direction in {"lower", "minimize"} else "higher")
        if self.value_max <= self.value_min:
            object.__setattr__(self, "value_max", float(self.value_min) + 1.0)
        object.__setattr__(self, "alpha", max(1e-9, min(1.0, float(self.alpha))))
        object.__setattr__(self, "min_trials", max(1, int(self.min_trials or 1)))
        object.__setattr__(self, "max_single_trial_weight", max(1e-9, float(self.max_single_trial_weight)))
        fractions = tuple(float(item) for item in self.betting_fractions if 0.0 < float(item) < 1.0)
        object.__setattr__(self, "betting_fractions", fractions or DEFAULT_BETTING_FRACTIONS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return "epc:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class EProcessObservation:
    trial_id: str
    sequence_no: int
    candidate_id: str
    baseline_id: str
    metric_id: str
    candidate_value: float
    baseline_value: float
    weight: float = 1.0
    evidence_ref: str = ""
    trusted: bool = True
    observed_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", str(self.trial_id or f"trial:{self.sequence_no}"))
        object.__setattr__(self, "sequence_no", max(1, int(self.sequence_no or 1)))
        object.__setattr__(self, "candidate_id", str(self.candidate_id or "candidate"))
        object.__setattr__(self, "baseline_id", str(self.baseline_id or "baseline"))
        object.__setattr__(self, "metric_id", str(self.metric_id or "metric"))
        object.__setattr__(self, "candidate_value", _finite_float(self.candidate_value))
        object.__setattr__(self, "baseline_value", _finite_float(self.baseline_value))
        object.__setattr__(self, "weight", max(0.0, _finite_float(self.weight, default=1.0)))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or ""))
        object.__setattr__(self, "trusted", bool(self.trusted))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def observation_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("observed_at_utc", None)
        return "epo:" + stable_hash(payload)


@dataclass(frozen=True)
class EProcessState:
    config_hash: str
    observation_hashes: tuple[str, ...] = ()
    count: int = 0
    log_component_values: tuple[float, ...] = ()
    log_e_value: float = 0.0
    max_log_e_value: float = 0.0
    crossed: bool = False
    crossing_sequence_no: int = 0
    rejected_observations: tuple[dict[str, Any], ...] = ()
    version: str = ANYTIME_VALID_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def state_hash(self) -> str:
        return "eps:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class AnytimeValidSolveCertificate:
    scope: str
    candidate_id: str
    baseline_id: str
    problem_model_snapshot_hash: str
    e_process_state: EProcessState
    threshold_log_e_value: float
    calibration_snapshot_hash: str = ""
    falsification_bundle_hash: str = ""
    structural_replay_bundle_hash: str = ""
    trusted_evidence_refs: tuple[str, ...] = ()
    verified: bool = False
    version: str = "anytime-valid-solve-certificate/v1"

    def __post_init__(self) -> None:
        verified = bool(
            self.e_process_state.crossed
            and self.e_process_state.log_e_value >= self.threshold_log_e_value
            and self.calibration_snapshot_hash
            and self.falsification_bundle_hash
            and self.structural_replay_bundle_hash
        )
        object.__setattr__(self, "verified", verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "problem_model_snapshot_hash": self.problem_model_snapshot_hash,
            "e_process_state": self.e_process_state.to_dict(),
            "threshold_log_e_value": self.threshold_log_e_value,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "falsification_bundle_hash": self.falsification_bundle_hash,
            "structural_replay_bundle_hash": self.structural_replay_bundle_hash,
            "trusted_evidence_refs": list(self.trusted_evidence_refs),
            "verified": self.verified,
            "certificate_hash": self.certificate_hash(),
        }

    def stable_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "problem_model_snapshot_hash": self.problem_model_snapshot_hash,
            "e_process_state": self.e_process_state.to_dict(),
            "threshold_log_e_value": self.threshold_log_e_value,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "falsification_bundle_hash": self.falsification_bundle_hash,
            "structural_replay_bundle_hash": self.structural_replay_bundle_hash,
            "trusted_evidence_refs": list(self.trusted_evidence_refs),
            "verified": self.verified,
        }

    def certificate_hash(self) -> str:
        return "avc:" + stable_hash(self.stable_payload())


def run_e_process(config: EProcessConfig, observations: list[EProcessObservation | dict[str, Any]]) -> EProcessState:
    parsed = [_observation_from_any(item) for item in observations]
    rejected: list[dict[str, Any]] = []
    accepted: list[EProcessObservation] = []
    seen_trials: set[str] = set()
    seen_sequences: set[int] = set()
    for item in parsed:
        if item is None:
            rejected.append({"reason": "malformed_observation"})
            continue
        reason = _reject_reason(config, item, seen_trials=seen_trials, seen_sequences=seen_sequences)
        if reason:
            rejected.append({"reason": reason, "observation_hash": item.observation_hash(), "trial_id": item.trial_id})
            continue
        seen_trials.add(item.trial_id)
        seen_sequences.add(item.sequence_no)
        accepted.append(item)
    accepted.sort(key=lambda item: (item.sequence_no, item.trial_id))
    log_components = [0.0 for _ in config.betting_fractions]
    max_log = 0.0
    crossed = False
    crossing = 0
    for item in accepted:
        x = normalized_improvement(config, item)
        weight = min(config.max_single_trial_weight, max(0.0, item.weight))
        for index, lam in enumerate(config.betting_fractions):
            increment = max(1e-12, 1.0 + lam * weight * x)
            log_components[index] += math.log(increment)
        log_e = _log_mean_exp(log_components)
        max_log = max(max_log, log_e)
        if not crossed and len([obs for obs in accepted if obs.sequence_no <= item.sequence_no]) >= config.min_trials and log_e >= math.log(1.0 / config.alpha):
            crossed = True
            crossing = item.sequence_no
    log_e_value = _log_mean_exp(log_components) if accepted else 0.0
    return EProcessState(
        config_hash=config.config_hash(),
        observation_hashes=tuple(item.observation_hash() for item in accepted),
        count=len(accepted),
        log_component_values=tuple(log_components),
        log_e_value=log_e_value,
        max_log_e_value=max(max_log, log_e_value),
        crossed=crossed,
        crossing_sequence_no=crossing,
        rejected_observations=tuple(rejected),
    )


def normalized_improvement(config: EProcessConfig, observation: EProcessObservation) -> float:
    raw = observation.candidate_value - observation.baseline_value
    if config.direction == "lower":
        raw = -raw
    scale = max(1e-12, config.value_max - config.value_min)
    return max(-1.0, min(1.0, (raw - config.null_margin) / scale))


def build_anytime_valid_certificate(
    *,
    scope: str,
    candidate_id: str,
    baseline_id: str,
    problem_model_snapshot_hash: str,
    config: EProcessConfig,
    observations: list[EProcessObservation | dict[str, Any]],
    calibration_snapshot_hash: str = "",
    falsification_bundle_hash: str = "",
    structural_replay_bundle_hash: str = "",
) -> AnytimeValidSolveCertificate:
    state = run_e_process(config, observations)
    trusted_refs = tuple(item.evidence_ref for item in (_observation_from_any(raw) for raw in observations) if item is not None and item.trusted and item.evidence_ref)
    return AnytimeValidSolveCertificate(
        scope=scope,
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        problem_model_snapshot_hash=problem_model_snapshot_hash,
        e_process_state=state,
        threshold_log_e_value=math.log(1.0 / config.alpha),
        calibration_snapshot_hash=calibration_snapshot_hash,
        falsification_bundle_hash=falsification_bundle_hash,
        structural_replay_bundle_hash=structural_replay_bundle_hash,
        trusted_evidence_refs=tuple(dict.fromkeys(trusted_refs)),
    )


def verify_anytime_valid_certificate(certificate: AnytimeValidSolveCertificate | dict[str, Any]) -> bool:
    cert = certificate if isinstance(certificate, AnytimeValidSolveCertificate) else _certificate_from_any(certificate)
    return bool(cert and cert.verified and cert.e_process_state.state_hash())


def _reject_reason(config: EProcessConfig, observation: EProcessObservation, *, seen_trials: set[str], seen_sequences: set[int]) -> str:
    if not observation.trusted:
        return "untrusted_observation"
    if observation.metric_id != config.metric_id:
        return "metric_mismatch"
    if observation.trial_id in seen_trials:
        return "duplicate_trial_id"
    if observation.sequence_no in seen_sequences:
        return "duplicate_sequence_no"
    if observation.weight <= 0 or observation.weight > config.max_single_trial_weight:
        return "invalid_observation_weight"
    if not observation.evidence_ref:
        return "missing_replayable_evidence_ref"
    return ""


def _observation_from_any(raw: EProcessObservation | dict[str, Any] | None) -> EProcessObservation | None:
    if isinstance(raw, EProcessObservation):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    try:
        return EProcessObservation(
            trial_id=str(data.get("trial_id") or ""),
            sequence_no=int(data.get("sequence_no") or 1),
            candidate_id=str(data.get("candidate_id") or "candidate"),
            baseline_id=str(data.get("baseline_id") or "baseline"),
            metric_id=str(data.get("metric_id") or "metric"),
            candidate_value=float(data.get("candidate_value")),
            baseline_value=float(data.get("baseline_value")),
            weight=float(data.get("weight") or 1.0),
            evidence_ref=str(data.get("evidence_ref") or ""),
            trusted=bool(data.get("trusted", True)),
            observed_at_utc=str(data.get("observed_at_utc") or utc_now()),
        )
    except Exception:
        return None


def _certificate_from_any(raw: dict[str, Any]) -> AnytimeValidSolveCertificate | None:
    data = coerce_dict(raw)
    state_data = coerce_dict(data.get("e_process_state"))
    if not state_data:
        return None
    state = EProcessState(
        config_hash=str(state_data.get("config_hash") or ""),
        observation_hashes=tuple(str(item) for item in state_data.get("observation_hashes", [])),
        count=int(state_data.get("count") or 0),
        log_component_values=tuple(float(item) for item in state_data.get("log_component_values", [])),
        log_e_value=float(state_data.get("log_e_value") or 0.0),
        max_log_e_value=float(state_data.get("max_log_e_value") or 0.0),
        crossed=bool(state_data.get("crossed")),
        crossing_sequence_no=int(state_data.get("crossing_sequence_no") or 0),
        rejected_observations=tuple(coerce_dict(item) for item in state_data.get("rejected_observations", [])),
    )
    return AnytimeValidSolveCertificate(
        scope=str(data.get("scope") or ""),
        candidate_id=str(data.get("candidate_id") or ""),
        baseline_id=str(data.get("baseline_id") or ""),
        problem_model_snapshot_hash=str(data.get("problem_model_snapshot_hash") or ""),
        e_process_state=state,
        threshold_log_e_value=float(data.get("threshold_log_e_value") or 0.0),
        calibration_snapshot_hash=str(data.get("calibration_snapshot_hash") or ""),
        falsification_bundle_hash=str(data.get("falsification_bundle_hash") or ""),
        structural_replay_bundle_hash=str(data.get("structural_replay_bundle_hash") or ""),
        trusted_evidence_refs=tuple(str(item) for item in data.get("trusted_evidence_refs", [])),
    )


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _log_mean_exp(values: list[float]) -> float:
    if not values:
        return 0.0
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values) / len(values))


__all__ = [
    "ANYTIME_VALID_VERSION",
    "AnytimeValidSolveCertificate",
    "EProcessConfig",
    "EProcessObservation",
    "EProcessState",
    "build_anytime_valid_certificate",
    "normalized_improvement",
    "run_e_process",
    "verify_anytime_valid_certificate",
]
