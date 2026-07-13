from __future__ import annotations

from cognitive_evolve_runtime.verification.honesty_core import (
    GroundingRegime,
    ProbeCase,
    measure_honesty,
    strength_from_measurements,
)
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult


def _regime(**kwargs: object) -> GroundingRegime:
    base = {
        "regime_id": "r1",
        "probes": [ProbeCase("p1", "engine secret", "engine", True)],
        "adversarial_budget": 4,
        "isolation_enforced": True,
        "replay_artifact_hash": "artifact-1",
        "verifier_fingerprint": "vf",
        "oracle_kind": "formal",
    }
    base.update(kwargs)
    return GroundingRegime(**base)  # type: ignore[arg-type]


def test_exogeneity_requires_engine_injected_probe() -> None:
    raw = VerificationResult(True, replayable=True, metadata={"model_claimed_exogenous": True})
    measurements = measure_honesty(raw, _regime(probes=[]), {"p1": True}, {"frozen_artifact_hash": "artifact-1", "verifier_fingerprint": "vf", "replay_verified": True})
    assert measurements.exogeneity_score == 0.0


def test_variety_isolated_flag_must_come_from_engine() -> None:
    raw = VerificationResult(True, replayable=True, metadata={"isolated": True})
    measurements = measure_honesty(raw, _regime(isolation_enforced=False), {"p1": True, "known_good_bad_distinguishable": True}, {"frozen_artifact_hash": "artifact-1", "verifier_fingerprint": "vf", "replay_verified": True})
    assert measurements.variety_score == 0.0


def test_falsification_budget_must_be_engine_injected() -> None:
    raw = VerificationResult(True, replayable=True, metadata={"falsification_rounds": 99})
    measurements = measure_honesty(raw, _regime(adversarial_budget=0), {"p1": True, "known_good_bad_distinguishable": True, "survived_count": 99}, {"frozen_artifact_hash": "artifact-1", "verifier_fingerprint": "vf", "replay_verified": True})
    assert measurements.falsification_score == 0.0


def test_replay_record_matching_hash_enables_replay_score() -> None:
    raw = VerificationResult(True, replayable=True)
    measurements = measure_honesty(raw, _regime(), {"p1": True, "known_good_bad_distinguishable": True, "survived_count": 4}, {"frozen_artifact_hash": "artifact-1", "verifier_fingerprint": "vf", "replay_verified": True})
    assert measurements.replay_score == 1.0
    assert strength_from_measurements(measurements, oracle_kind="formal") == VerificationStrength.FORMAL


def test_oracle_kind_caps_measured_strength() -> None:
    raw = VerificationResult(True, replayable=True)
    measurements = measure_honesty(raw, _regime(oracle_kind="empirical"), {"p1": True, "known_good_bad_distinguishable": True, "survived_count": 4}, {"frozen_artifact_hash": "artifact-1", "verifier_fingerprint": "vf", "replay_verified": True})
    assert strength_from_measurements(measurements, oracle_kind="empirical") == VerificationStrength.EMPIRICAL
