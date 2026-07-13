from __future__ import annotations

from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    CalibrationEvent,
    CalibrationPolicy,
    EProcessConfig,
    EProcessObservation,
    FalsificationCase,
    FalsificationOutcome,
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    bind_closure_gate_digests,
    build_anytime_valid_certificate,
    build_falsification_gauntlet,
    evaluate_m6_closure_gate,
    materialize_calibration_snapshot,
    materialize_contract_problem_model_snapshot,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="quality", statement="improve quality", posterior=1.0, uncertainty=0.3),
        )
    )


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve best quality",
        normalized_goal="evolve best quality",
        metadata={"latent_problem_state": _state().to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def _observations() -> list[EProcessObservation]:
    return [
        EProcessObservation(f"t{i}", i + 1, "C", "B", "quality", 1.0, 0.0, evidence_ref=f"raw:t{i}")
        for i in range(40)
    ]


def _calibration():
    events = [
        CalibrationEvent(f"cal{i}", prediction=0.8, realization=0.8, lower_confidence_bound=0.7, evidence_ref=f"raw:cal{i}")
        for i in range(40)
    ]
    return materialize_calibration_snapshot(events, bin_count=5)


def _falsification(problem_hash: str):
    return build_falsification_gauntlet(
        scope="latent-intent:quality",
        candidate_id="C",
        problem_model_snapshot_hash=problem_hash,
        cases=(FalsificationCase(case_id="held-out", challenge="held-out boundary probe"),),
        outcomes=(FalsificationOutcome(case_id="held-out", survived=True, evidence_ref="raw:held-out"),),
    )


def _gate_inputs(contract: NexusObjectiveContract):
    problem_snapshot = materialize_contract_problem_model_snapshot(contract, force=True)
    assert problem_snapshot is not None
    calibration = _calibration()
    falsification = _falsification(problem_snapshot.snapshot_hash())
    config = EProcessConfig(metric_id="quality", value_min=0.0, value_max=1.0, alpha=0.05, min_trials=3)
    interim = build_anytime_valid_certificate(
        scope="latent-intent:quality",
        candidate_id="C",
        baseline_id="B",
        problem_model_snapshot_hash=problem_snapshot.snapshot_hash(),
        config=config,
        observations=_observations(),
        calibration_snapshot_hash=calibration.snapshot_hash(),
        falsification_bundle_hash=falsification.bundle_hash(),
        structural_replay_bundle_hash="srb:placeholder",
    )
    structural = bind_closure_gate_digests(
        scope="latent-intent:quality",
        candidate_id="C",
        baseline_id="B",
        e_process_digest=interim.e_process_state.state_hash(),
        calibration_digest=calibration.snapshot_hash(),
        falsification_digest=falsification.bundle_hash(),
        problem_model_digest=problem_snapshot.snapshot_hash(),
        latent_digest=contract.metadata.get("latent_problem_state_hash", "latent:none"),
        pareto_digest="pareto:none",
        compaction_digest="pmc:none",
        replay_steps=({"step": "m6_full_gate"},),
        evidence_refs=("raw:closure",),
    )
    certificate = build_anytime_valid_certificate(
        scope="latent-intent:quality",
        candidate_id="C",
        baseline_id="B",
        problem_model_snapshot_hash=problem_snapshot.snapshot_hash(),
        config=config,
        observations=_observations(),
        calibration_snapshot_hash=calibration.snapshot_hash(),
        falsification_bundle_hash=falsification.bundle_hash(),
        structural_replay_bundle_hash=structural.bundle_hash(),
    )
    return certificate, calibration, falsification, structural


def test_m6_closure_gate_passes_only_when_all_full_sprint_gates_pass() -> None:
    contract = _contract()
    certificate, calibration, falsification, structural = _gate_inputs(contract)

    result = evaluate_m6_closure_gate(
        contract=contract,
        anytime_certificate=certificate,
        calibration_snapshot=calibration,
        calibration_policy=CalibrationPolicy(min_total_count=20, min_count_per_required_bin=1, max_ece=0.01, max_mce=0.01, max_brier_score=0.05),
        falsification_bundle=falsification,
        structural_replay_bundle=structural,
        candidate_confidence=0.8,
    )

    assert result["passed"] is True
    assert result["failure_reasons"] == []
    assert contract.metadata["m6_closure_gate"]["gate_hash"] == result["gate_hash"]


def test_m6_closure_gate_fails_closed_when_any_required_gate_is_missing() -> None:
    contract = _contract()
    certificate, _calibration, falsification, structural = _gate_inputs(contract)

    result = evaluate_m6_closure_gate(
        contract=contract,
        anytime_certificate=certificate,
        calibration_snapshot=None,
        falsification_bundle=falsification,
        structural_replay_bundle=structural,
    )

    assert result["passed"] is False
    assert "missing_calibration_snapshot" in result["failure_reasons"]
