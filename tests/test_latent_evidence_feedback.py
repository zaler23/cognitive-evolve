from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    FrontierCandidate,
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    OutcomeContract,
    OutcomeMetric,
    TrialObservation,
    adapt_archive_observation,
    adapt_critique_result,
    adapt_improvement_certificate,
    adapt_trial_observation,
    adapt_verifier_result,
    annotate_candidates_with_latent_signals,
    compare_outcomes,
    ingest_latent_feedback,
    latent_ledger_from_contract,
    materialize_contract_latent_posterior,
    materialize_posterior_snapshot,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make the result clearer", posterior=0.5, utility_dimensions=("clarity",), uncertainty=0.5),
            IntentHypothesis(id="impact", statement="make the result more impactful", posterior=0.5, utility_dimensions=("impact",), uncertainty=0.5),
        ),
        frontier_candidates=(
            FrontierCandidate(candidate_id="clear", utility_by_intent={"clarity": 0.8, "impact": 0.2}),
            FrontierCandidate(candidate_id="punchy", utility_by_intent={"clarity": 0.2, "impact": 0.8}),
        ),
    )


def _contract(state: LatentProblemState | None = None) -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata={"latent_problem_state": (state or _state()).to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def _m5_certificate(intent_id: str = "clarity", *, verified: bool = True) -> dict[str, Any]:
    contract = OutcomeContract(
        objective=f"improve {intent_id}",
        scope=f"latent-intent:{intent_id}",
        metrics=(OutcomeMetric(id=intent_id, weight=1.0),),
        min_effect=0.05,
    )
    baseline = TrialObservation(
        artifact_id="baseline",
        contract_hash=contract.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={intent_id: 0.50},
        uncertainty_radius={intent_id: 0.01},
        raw_observation_ref="raw:baseline",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    challenger = TrialObservation(
        artifact_id="challenger",
        contract_hash=contract.contract_hash(),
        manifest_hash="manifest:v1" if verified else "manifest:v2",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={intent_id: 0.72},
        uncertainty_radius={intent_id: 0.01},
        raw_observation_ref="raw:challenger",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    payload = compare_outcomes(contract, baseline, challenger).to_dict() | {"intent_id": intent_id}
    if verified:
        payload.update({
            "source_type": "runtime_verifier",
            "provenance_ref": "verifier-run:evidence-feedback",
            "verifier_run_id": "evidence-feedback",
            "trial_pair_container_source": "candidate.verification_result",
        })
    return payload


def test_critique_verifier_archive_certificate_and_trial_adapters() -> None:
    state = _state()

    critique = adapt_critique_result({"candidate_id": "C1", "strengths": ["clear structure"], "flaws": [], "intent_id": "clarity"}, state)
    verifier = adapt_verifier_result({"candidate_id": "C1", "passed": True, "diagnostics": ["clarity evidence"], "intent_id": "clarity"}, state)
    archive = adapt_archive_observation({"candidate_id": "C1", "fate": "Elite", "intent_id": "clarity"}, state)
    certificate = adapt_improvement_certificate(_m5_certificate("clarity"), state)
    trial = adapt_trial_observation(
        TrialObservation(
            artifact_id="C1",
            contract_hash="contract",
            manifest_hash="manifest",
            environment_hash="env",
            evaluator_hash="eval",
            scores={"clarity": 0.8},
            raw_observation_ref="trial:C1",
        ),
        state,
    )

    assert critique.evidence[0].source_type == "critique"
    assert critique.evidence[0].weight <= 0.25
    assert verifier.evidence[0].source_type == "verifier"
    assert archive.evidence[0].calibration == "archive_frequency_is_not_desirability"
    assert certificate.evidence[0].source_type == "verified_improvement_certificate"
    assert certificate.evidence[0].support >= 0.9
    assert trial.evidence[0].source_type == "trial_observation"
    assert trial.evidence[0].support < 0.2


def test_malformed_adapter_payload_is_quarantined() -> None:
    state = _state()

    output = adapt_verifier_result("not a verifier result", state)

    assert output.evidence == ()
    assert output.quarantined[0].reason == "malformed_verifier_result"


def test_closed_loop_feedback_changes_ranking_with_pinned_decision_trace() -> None:
    contract = _contract()
    clear = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.82, "impact": 0.20}})
    punchy = CandidateGenome(id="punchy", metadata={"latent_intent_scores": {"clarity": 0.20, "impact": 0.80}})

    before = annotate_candidates_with_latent_signals([clear, punchy], contract)
    assert before["ranked_candidate_ids"][0] == "clear"

    feedback = ingest_latent_feedback(
        contract=contract,
        verifier_results=[{"candidate_id": "punchy", "passed": True, "diagnostics": ["impact evidence"], "intent_id": "impact"}],
    )
    clear2 = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.82, "impact": 0.20}})
    punchy2 = CandidateGenome(id="punchy", metadata={"latent_intent_scores": {"clarity": 0.20, "impact": 0.80}})
    after = annotate_candidates_with_latent_signals([clear2, punchy2], contract)

    assert feedback["evidence_added"] == 1
    assert after["ranked_candidate_ids"][0] == "punchy"
    assert after["latent_posterior_snapshot_hash"]
    assert clear2.metadata["latent_decision_trace_ref"] == punchy2.metadata["latent_decision_trace_ref"]
    assert clear2.metadata["latent_ledger_cursor"] == after["latent_ledger_cursor"]


def test_duplicate_feedback_does_not_change_posterior_and_replay_keeps_trace_consistent() -> None:
    contract = _contract()
    raw = {"candidate_id": "C1", "passed": True, "diagnostics": ["impact evidence"], "intent_id": "impact"}

    first = ingest_latent_feedback(contract=contract, verifier_results=[raw])
    snapshot1 = materialize_contract_latent_posterior(contract)
    second = ingest_latent_feedback(contract=contract, verifier_results=[raw])
    snapshot2 = materialize_contract_latent_posterior(contract)

    assert first["evidence_added"] == 1
    assert second["evidence_deduplicated"] == 1
    assert snapshot1 is not None and snapshot2 is not None
    assert snapshot1.state.state_hash() == snapshot2.state.state_hash()

    replayed = LatentLedger.from_dict(latent_ledger_from_contract(contract).to_dict())
    replay_snapshot = materialize_posterior_snapshot(_state(), replayed, cursor=snapshot2.ledger_cursor)
    assert replayed.replay().active_evidence_ids == latent_ledger_from_contract(contract).replay().active_evidence_ids
    assert replay_snapshot.snapshot_hash() == snapshot2.snapshot_hash()


def test_retracted_feedback_rebuilds_expected_posterior() -> None:
    contract = _contract()
    ingest_latent_feedback(contract=contract, verifier_results=[{"candidate_id": "C1", "passed": True, "intent_id": "impact"}])
    shifted = materialize_contract_latent_posterior(contract)
    ledger = latent_ledger_from_contract(contract)
    evidence_id = ledger.replay().active_evidence_ids[0]
    ledger.retract_evidence(evidence_id)
    contract.metadata["latent_ledger"] = ledger.to_dict()
    rebuilt = materialize_contract_latent_posterior(contract, force=True)

    assert shifted is not None and shifted.state.top_intent().id == "impact"
    assert rebuilt is not None
    assert all(abs(intent.posterior - 0.5) < 1e-9 for intent in rebuilt.state.intents)
