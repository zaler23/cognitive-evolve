from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan, MutationOperator
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    MODEL_ADDED,
    MODEL_DEDUPLICATED,
    MODEL_PROMOTED,
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    PreferenceEvidence,
    ProblemModelLedger,
    ProblemModelPrediction,
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    attach_latent_state_if_needed,
    detect_problem_residuals,
    initial_problem_model_from_latent_state,
    materialize_contract_problem_model_snapshot,
    materialize_problem_model_snapshot,
    problem_model_from_contract,
    problem_model_ledger_from_contract,
    propose_problem_models_for_contract,
    propose_structural_models,
    select_model_discrimination_action,
    validate_problem_model_promotion,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(
                id="clarity",
                statement="make the answer clearer",
                posterior=0.5,
                utility_dimensions=("clarity",),
                uncertainty=0.65,
            ),
            IntentHypothesis(
                id="impact",
                statement="make the answer more useful",
                posterior=0.5,
                utility_dimensions=("impact",),
                uncertainty=0.65,
            ),
        )
    )


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata={"latent_problem_state": _state().to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def test_initial_problem_model_lifts_latent_state_without_contract_hash_drift() -> None:
    contract = _contract()
    before = contract.contract_hash()

    model = attach_latent_state_if_needed(contract)
    problem_model = problem_model_from_contract(contract)

    assert model is not None
    assert problem_model is not None
    assert problem_model.model_hash() == contract.metadata["problem_model_hash"]
    assert contract.metadata["problem_model_summary"]["objective_count"] == 2
    assert contract.contract_hash() == before
    assert problem_model.unknown_mass > 0.0


def test_problem_model_ledger_append_only_replay_dedup_and_promotion() -> None:
    base = initial_problem_model_from_latent_state(_state())
    ledger = ProblemModelLedger()

    first = ledger.add_model(base, idempotency_key="initial")
    duplicate = ledger.add_model(base, idempotency_key="initial")
    validation = validate_problem_model_promotion(
        base,
        frozen_model_hash=base.model_hash(),
        validation_evidence_refs=("raw:future-validation",),
        trusted_verifier_refs=("verifier:independent",),
        predictive_gain=0.40,
        falsification_survived=True,
        calibration_status="within_bounds",
    )
    promoted = ledger.promote_model(validation)

    replay = ledger.replay()
    replayed = ProblemModelLedger.from_dict(ledger.to_dict()).replay()

    assert first.event_type == MODEL_ADDED
    assert duplicate.event_type == MODEL_DEDUPLICATED
    assert promoted.event_type == MODEL_PROMOTED
    assert validation.promoted is True
    assert [event.sequence for event in ledger.events] == [1, 2, 3]
    assert replay.active_model_hashes == replayed.active_model_hashes
    assert replay.promoted_model_hashes == replayed.promoted_model_hashes
    assert replay.ledger_replay_hash == replayed.ledger_replay_hash


def test_omitted_truth_residual_births_new_problem_objective_but_cannot_self_promote() -> None:
    base = initial_problem_model_from_latent_state(_state())
    evidence = PreferenceEvidence(
        intent_id="maintainability",
        support=0.9,
        weight=1.0,
        confidence=1.0,
        evidence_ref="verifier:omitted-truth",
        source_type="verifier",
    )

    residuals = detect_problem_residuals(base, evidence=[evidence])
    proposals = propose_structural_models(base, residuals)
    proposed = next(
        proposal.proposed_model
        for proposal in proposals
        if any(objective.id == "maintainability" for objective in proposal.proposed_model.objectives)
    )
    weak_validation = validate_problem_model_promotion(
        proposed,
        parent_model=base,
        frozen_model_hash=proposed.model_hash(),
        predictive_gain=0.9,
        falsification_survived=True,
    )
    trusted_validation = validate_problem_model_promotion(
        proposed,
        parent_model=base,
        frozen_model_hash=proposed.model_hash(),
        validation_evidence_refs=("raw:future-maintainability",),
        trusted_verifier_refs=("verifier:independent",),
        predictive_gain=0.9,
        falsification_survived=True,
        calibration_status="within_bounds",
    )

    assert residuals[0].suggested_operator == "birth"
    assert any(objective.id == "maintainability" for objective in proposed.objectives)
    assert weak_validation.promoted is False
    assert "missing_validation_evidence" in weak_validation.reason_codes
    assert "missing_trusted_verifier" in weak_validation.reason_codes
    assert trusted_validation.promoted is True


def test_overfit_structure_rejected_after_complexity_correction_and_falsification_gate() -> None:
    base = initial_problem_model_from_latent_state(_state())
    residuals = detect_problem_residuals(
        base,
        certificates=[{"certificate_hash": "cert:bad", "critical_failures": ["manifest_drift", "self_verification", "raw_evidence_replayable"]}],
    )
    proposal = propose_structural_models(base, residuals)[0]

    validation = validate_problem_model_promotion(
        proposal.proposed_model,
        parent_model=base,
        frozen_model_hash=proposal.proposed_model.model_hash(),
        validation_evidence_refs=("raw:training-only",),
        trusted_verifier_refs=("verifier:independent",),
        predictive_gain=0.03,
        complexity_penalty=0.20,
        parent_delta_penalty=0.20,
        falsification_survived=False,
        calibration_status="overconfident",
    )

    assert validation.promoted is False
    assert "predictive_gain_below_threshold" in validation.reason_codes
    assert "complexity_corrected_gain_below_threshold" in validation.reason_codes
    assert "falsification_not_survived" in validation.reason_codes
    assert "calibration_failed" in validation.reason_codes


def test_active_model_discrimination_selects_divergent_prediction_action() -> None:
    base = initial_problem_model_from_latent_state(_state())
    residuals = detect_problem_residuals(base, evidence=[PreferenceEvidence(intent_id="new_axis", support=1.0, evidence_ref="raw:new")])
    challenger = propose_structural_models(base, residuals)[0].proposed_model
    snapshot = materialize_problem_model_snapshot(ProblemModelLedger(events=[]))
    snapshot = snapshot.__class__(active_models=(base, challenger), ledger_cursor=2)

    action = select_model_discrimination_action(
        [
            ProblemModelPrediction(base.model_hash(), "probe_boundary_case", "fails", probability=1.0),
            ProblemModelPrediction(challenger.model_hash(), "probe_boundary_case", "passes", probability=1.0),
            ProblemModelPrediction(base.model_hash(), "generic_probe", "passes", probability=1.0),
            ProblemModelPrediction(challenger.model_hash(), "generic_probe", "passes", probability=1.0),
        ],
        cost_by_action={"probe_boundary_case": 0.1, "generic_probe": 0.0},
        snapshot=snapshot,
    )

    assert action is not None
    assert action.action_id == "probe_boundary_case"
    assert action.expected_information_gain > 0.0
    assert action.decision_trace["problem_model_snapshot_hash"] == snapshot.snapshot_hash()


def test_runtime_bridge_records_problem_model_trace_on_ranking_and_mutation_plan() -> None:
    contract = _contract()
    attach_latent_state_if_needed(contract)
    base = problem_model_from_contract(contract)
    assert base is not None
    proposed = propose_problem_models_for_contract(
        contract=contract,
        evidence=[PreferenceEvidence(intent_id="maintainability", support=0.9, evidence_ref="raw:maintainability")],
    )
    assert proposed["problem_model_proposal_count"] >= 1
    snapshot = materialize_contract_problem_model_snapshot(contract, force=True)
    assert snapshot is not None and len(snapshot.active_model_hashes) >= 2
    contract.metadata["problem_model_predictions"] = [
        {"model_hash": snapshot.active_model_hashes[0], "action_id": "probe_boundary_case", "predicted_outcome": "fails"},
        {"model_hash": snapshot.active_model_hashes[1], "action_id": "probe_boundary_case", "predicted_outcome": "passes"},
    ]

    clear = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.9, "impact": 0.2}})
    impact = CandidateGenome(id="impact", metadata={"latent_intent_scores": {"clarity": 0.2, "impact": 0.9}})
    ranking = annotate_candidates_with_latent_signals([clear, impact], contract)
    plans, exploration = apply_latent_exploration_to_mutation_plans(
        [MutationPlan(operator=MutationOperator.CASE_SPLIT, instruction="test plan")],
        contract,
    )

    assert ranking["problem_model_snapshot_hash"]
    assert clear.metadata["problem_model_decision_trace"]["problem_model_snapshot_hash"] == ranking["problem_model_snapshot_hash"]
    assert exploration["problem_model_discrimination"]["problem_model_discrimination_action"]["action_id"] == "probe_boundary_case"
    assert plans[0].metadata["problem_model_discrimination_action"]["action_id"] == "probe_boundary_case"
    assert "structural novelty alone is not improvement" in plans[0].instruction
    assert problem_model_ledger_from_contract(contract).cursor >= snapshot.ledger_cursor
