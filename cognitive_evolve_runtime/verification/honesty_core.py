"""Engine-owned honesty measurements for verification results.

The verifier may produce raw pass/fail evidence, but certification strength is
only earned after the engine measures grounding, variety, falsification, and
replayability.  Model-emitted fields are deliberately ignored here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict

from .ladder import VerificationStrength
from .types import VerificationResult


@dataclass(frozen=True)
class ProbeCase:
    probe_id: str
    content: str
    provenance: str
    expected_verdict_flip: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingRegime:
    regime_id: str
    probes: list[ProbeCase] = field(default_factory=list)
    adversarial_budget: int = 0
    isolation_enforced: bool = False
    replay_artifact_hash: str = ""
    verifier_fingerprint: str = ""
    oracle_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["probes"] = [probe.to_dict() for probe in self.probes]
        return data


@dataclass(frozen=True)
class HonestyMeasurements:
    exogeneity_score: float
    variety_score: float
    falsification_score: float
    replay_score: float
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasuredVerificationResult:
    raw_result: VerificationResult
    measurements: HonestyMeasurements
    measured_strength: VerificationStrength
    grounding_regime_id: str
    replay_record: dict[str, Any]

    def to_verification_result(self) -> VerificationResult:
        """Return a backwards-compatible result carrying measured fields."""

        metadata = dict(self.raw_result.metadata or {})
        metadata.update(
            {
                "measured_strength": self.measured_strength.name,
                "measured_strength_value": int(self.measured_strength),
                "honesty_measurements": self.measurements.to_dict(),
                "grounding_regime_id": self.grounding_regime_id,
                "replay_record": dict(self.replay_record or {}),
                "diagnostics_only": False,
                "legacy": False,
            }
        )
        diagnostics = list(self.raw_result.diagnostics)
        diagnostics.extend(self.measurements.diagnostics)
        return VerificationResult(
            passed=bool(self.raw_result.passed),
            score=float(self.raw_result.score or 0.0),
            strength=self.measured_strength,
            evidence_ref=self.raw_result.evidence_ref,
            replayable=bool(self.raw_result.replayable and self.measurements.replay_score >= 1.0),
            diagnostics=list(dict.fromkeys(str(item) for item in diagnostics if item)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_result": self.raw_result.to_dict(),
            "measurements": self.measurements.to_dict(),
            "measured_strength": self.measured_strength.name,
            "measured_strength_value": int(self.measured_strength),
            "grounding_regime_id": self.grounding_regime_id,
            "replay_record": dict(self.replay_record or {}),
        }


def measure_honesty(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    actual_probe_verdicts: dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
) -> HonestyMeasurements:
    """Measure verifier honesty from engine observations only.

    ``actual_probe_verdicts`` is an engine-side observation map.  Model output
    fields such as ``isolated`` or ``falsification_rounds`` are ignored unless
    the engine has copied them into this map before calling this function.
    """

    observed = coerce_dict(actual_probe_verdicts)
    replay = coerce_dict(replay_record)
    diagnostics: list[str] = []
    trusted_provenances = {"archive", "user", "tool", "engine_counterexample", "engine"}
    trusted_probes = [probe for probe in regime.probes if str(probe.provenance) in trusted_provenances]
    if regime.probes and len(trusted_probes) != len(regime.probes):
        diagnostics.append("untrusted_probe_provenance_ignored")
    if trusted_probes:
        hits = 0
        for probe in trusted_probes:
            if probe.probe_id not in observed:
                diagnostics.append(f"probe_not_observed:{probe.probe_id}")
                continue
            value = observed.get(probe.probe_id)
            if isinstance(value, dict):
                flipped = bool(value.get("verdict_flipped", value.get("matched_expected_flip", False)))
            else:
                flipped = bool(value)
            if flipped == bool(probe.expected_verdict_flip):
                hits += 1
        exogeneity_score = hits / max(1, len(trusted_probes))
    else:
        exogeneity_score = 0.0
        diagnostics.append("no_engine_injected_probe")

    known_good_bad = bool(observed.get("known_good_bad_distinguishable"))
    variety_score = 1.0 if regime.isolation_enforced and known_good_bad else 0.0
    if not regime.isolation_enforced:
        diagnostics.append("isolation_not_engine_enforced")
    if not known_good_bad:
        diagnostics.append("known_good_bad_not_distinguished")

    budget = max(0, int(regime.adversarial_budget or 0))
    if budget <= 0:
        falsification_score = 0.0
        diagnostics.append("no_engine_adversarial_budget")
    else:
        survived_count = _int(observed.get("survived_count"), 0)
        falsification_score = max(0.0, min(1.0, survived_count / max(1, budget)))

    artifact_match = str(replay.get("frozen_artifact_hash") or replay.get("artifact_sha256") or "") == str(regime.replay_artifact_hash or "")
    fingerprint_match = str(replay.get("verifier_fingerprint") or "") == str(regime.verifier_fingerprint or "")
    replay_verified = bool(replay.get("replay_verified", raw_result.replayable))
    replay_score = 1.0 if artifact_match and fingerprint_match and replay_verified else 0.0
    if replay_score <= 0:
        diagnostics.append("replay_record_missing_or_mismatched")

    return HonestyMeasurements(
        exogeneity_score=exogeneity_score,
        variety_score=variety_score,
        falsification_score=falsification_score,
        replay_score=replay_score,
        diagnostics=diagnostics,
    )


def strength_from_measurements(
    measurements: HonestyMeasurements,
    *,
    oracle_kind: str = "",
    raw_strength_upper_bound: VerificationStrength | int | str = VerificationStrength.EXECUTABLE,
) -> VerificationStrength:
    if measurements.exogeneity_score == 0 or measurements.variety_score == 0:
        cap = VerificationStrength.NONE
    elif measurements.replay_score == 0:
        cap = VerificationStrength.ADVERSARIAL
    elif measurements.falsification_score < 0.25:
        cap = VerificationStrength.ADVERSARIAL
    elif measurements.falsification_score < 0.50:
        cap = VerificationStrength.DECOMPOSED
    elif measurements.falsification_score < 0.75:
        cap = VerificationStrength.EMPIRICAL
    else:
        cap = VerificationStrength.EXECUTABLE
    cap = min(cap, _oracle_cap(oracle_kind))
    return min(cap, VerificationStrength.from_value(raw_strength_upper_bound))


def measure_verification_result(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    actual_probe_verdicts: dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
) -> MeasuredVerificationResult:
    measurements = measure_honesty(raw_result, regime, actual_probe_verdicts, replay_record)
    upper_bound = raw_result.strength
    if raw_result.metadata.get("diagnostics_only") or raw_result.metadata.get("legacy") or upper_bound == VerificationStrength.NONE:
        upper_bound = VerificationStrength.EXECUTABLE
    measured_strength = strength_from_measurements(
        measurements,
        oracle_kind=regime.oracle_kind,
        raw_strength_upper_bound=upper_bound,
    )
    return MeasuredVerificationResult(
        raw_result=raw_result,
        measurements=measurements,
        measured_strength=measured_strength,
        grounding_regime_id=regime.regime_id,
        replay_record=coerce_dict(replay_record),
    )


def _oracle_cap(oracle_kind: str) -> VerificationStrength:
    kind = str(oracle_kind or "").strip().lower()
    if kind == "formal":
        return VerificationStrength.FORMAL
    if kind in {"executable", "toolrunner", "tool_runner"}:
        return VerificationStrength.EXECUTABLE
    if kind == "empirical":
        return VerificationStrength.EMPIRICAL
    if kind == "decomposed":
        return VerificationStrength.DECOMPOSED
    if kind in {"adversarial", "text", "diagnostic_matcher"}:
        return VerificationStrength.ADVERSARIAL
    return VerificationStrength.ADVERSARIAL


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "GroundingRegime",
    "HonestyMeasurements",
    "MeasuredVerificationResult",
    "ProbeCase",
    "measure_honesty",
    "measure_verification_result",
    "strength_from_measurements",
]
