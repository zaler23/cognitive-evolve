from __future__ import annotations

import math

from cognitive_evolve_runtime.outcomes.anytime_valid import (
    EProcessConfig,
    EProcessObservation,
    build_anytime_valid_certificate,
    run_e_process,
)
from cognitive_evolve_runtime.outcomes.calibration import (
    CalibrationEvent,
    CalibrationPolicy,
    calibration_allows_solve,
    calibration_block_reasons,
    materialize_calibration_snapshot,
)


def _observations(delta: float, n: int = 40) -> list[EProcessObservation]:
    return [
        EProcessObservation(
            trial_id=f"t{i}",
            sequence_no=i + 1,
            candidate_id="C",
            baseline_id="B",
            metric_id="quality",
            candidate_value=0.5 + delta,
            baseline_value=0.5,
            evidence_ref=f"raw:t{i}",
        )
        for i in range(n)
    ]


def test_e_process_seals_under_repeated_trusted_improvement_and_replays() -> None:
    config = EProcessConfig(metric_id="quality", value_min=0.0, value_max=1.0, alpha=0.05, min_trials=3)
    state = run_e_process(config, _observations(0.9, n=40))
    replayed = run_e_process(config, [item.to_dict() for item in _observations(0.9, n=40)])

    assert state.crossed is True
    assert state.log_e_value >= math.log(1 / config.alpha)
    assert state.state_hash() == replayed.state_hash()


def test_e_process_rejects_duplicate_untrusted_and_unreplayable_observations() -> None:
    config = EProcessConfig(metric_id="quality", value_min=0.0, value_max=1.0, alpha=0.05)
    observations = [
        EProcessObservation("t1", 1, "C", "B", "quality", 0.9, 0.1, evidence_ref="raw:t1"),
        EProcessObservation("t1", 2, "C", "B", "quality", 0.9, 0.1, evidence_ref="raw:t1b"),
        EProcessObservation("t3", 3, "C", "B", "quality", 0.9, 0.1, evidence_ref="", trusted=True),
        EProcessObservation("t4", 4, "C", "B", "quality", 0.9, 0.1, evidence_ref="raw:t4", trusted=False),
    ]
    state = run_e_process(config, observations)

    assert state.count == 1
    reasons = {item["reason"] for item in state.rejected_observations}
    assert "duplicate_trial_id" in reasons
    assert "missing_replayable_evidence_ref" in reasons
    assert "untrusted_observation" in reasons


def test_anytime_certificate_needs_all_external_gate_hashes_to_verify() -> None:
    config = EProcessConfig(metric_id="quality", value_min=0.0, value_max=1.0, alpha=0.05, min_trials=3)
    unbound = build_anytime_valid_certificate(
        scope="latent-intent:quality",
        candidate_id="C",
        baseline_id="B",
        problem_model_snapshot_hash="pms:1",
        config=config,
        observations=_observations(0.9, n=40),
    )
    bound = build_anytime_valid_certificate(
        scope="latent-intent:quality",
        candidate_id="C",
        baseline_id="B",
        problem_model_snapshot_hash="pms:1",
        config=config,
        observations=_observations(0.9, n=40),
        calibration_snapshot_hash="cals:ok",
        falsification_bundle_hash="fb:ok",
        structural_replay_bundle_hash="srb:ok",
    )

    assert unbound.e_process_state.crossed is True
    assert unbound.verified is False
    assert bound.verified is True


def _calibrated_events(n: int = 100) -> list[CalibrationEvent]:
    events: list[CalibrationEvent] = []
    for i in range(n):
        prediction = 0.2 if i % 2 == 0 else 0.8
        realization = prediction
        events.append(
            CalibrationEvent(
                event_id=f"cal{i}",
                prediction=prediction,
                realization=realization,
                lower_confidence_bound=max(0.0, prediction - 0.1),
                evidence_ref=f"raw:cal{i}",
            )
        )
    return events


def test_calibration_blocks_missing_sparse_and_miscalibrated_snapshots() -> None:
    policy = CalibrationPolicy(min_total_count=20, min_count_per_required_bin=2, max_ece=0.10, max_mce=0.20, max_brier_score=0.20)
    sparse = materialize_calibration_snapshot(_calibrated_events(4), bin_count=2)
    bad = materialize_calibration_snapshot(
        [CalibrationEvent(f"bad{i}", prediction=0.95, realization=0.0, lower_confidence_bound=0.9, evidence_ref=f"raw:bad{i}") for i in range(50)],
        bin_count=2,
    )

    assert "missing_calibration_snapshot" in calibration_block_reasons(None, policy)
    assert "calibration_total_count_below_minimum" in calibration_block_reasons(sparse, policy)
    bad_reasons = calibration_block_reasons(bad, policy)
    assert "calibration_ece_above_limit" in bad_reasons
    assert "calibration_brier_above_limit" in bad_reasons


def test_calibrated_snapshot_allows_solve_with_supported_confidence() -> None:
    policy = CalibrationPolicy(
        min_total_count=20,
        min_count_per_required_bin=2,
        max_ece=0.05,
        max_mce=0.05,
        max_brier_score=0.01,
        min_lower_confidence_coverage=0.80,
    )
    snapshot = materialize_calibration_snapshot(_calibrated_events(80), bin_count=10)

    assert calibration_block_reasons(snapshot, policy, candidate_confidence=0.8) == ()
    assert calibration_allows_solve(snapshot, policy, candidate_confidence=0.8) is True
    assert "candidate_confidence_outside_calibrated_support" in calibration_block_reasons(snapshot, policy, candidate_confidence=0.5)
