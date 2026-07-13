from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    extract_runtime_trial_observations,
    freeze_outcome_contract,
    ingest_runtime_trial_feedback,
    latent_ledger_from_contract,
    materialize_contract_latent_posterior,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make the answer clearer", posterior=0.5, utility_dimensions=("clarity",)),
            IntentHypothesis(id="impact", statement="make the answer more useful", posterior=0.5, utility_dimensions=("impact",)),
        )
    )


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve best answer",
        normalized_goal="evolve best answer",
        metadata={"latent_problem_state": _state().to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def _trial_pair() -> dict[str, object]:
    outcome = freeze_outcome_contract(_state(), intent_id="clarity", min_effect=0.05)
    base = {
        "artifact_id": "baseline",
        "contract_hash": outcome.contract_hash(),
        "manifest_hash": "manifest:v1",
        "environment_hash": "env:v1",
        "evaluator_hash": "eval:v1",
        "scores": {"clarity": 0.50},
        "uncertainty_radius": {"clarity": 0.01},
        "raw_observation_ref": "raw:baseline",
        "proposer_ref": "generator",
        "verifier_ref": "independent-verifier",
    }
    challenger = dict(base) | {
        "artifact_id": "C1",
        "scores": {"clarity": 0.70},
        "raw_observation_ref": "raw:C1",
    }
    return {
        "intent_id": "clarity",
        "source_type": "runtime_verifier",
        "provenance_ref": "verifier-run:trial-feedback",
        "verifier_run_id": "trial-feedback",
        "baseline": base,
        "challenger": challenger,
    }


def test_runtime_trial_pair_is_extracted_frozen_attached_and_ingested() -> None:
    contract = _contract()
    outcome = freeze_outcome_contract(_state(), intent_id="clarity", min_effect=0.05)
    candidate = CandidateGenome(id="C1", verification_result={"m5_trial_pair": _trial_pair()})

    extracted = extract_runtime_trial_observations([candidate], outcome_contract=outcome)
    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate], outcome_contract=outcome)
    snapshot = materialize_contract_latent_posterior(contract)
    ledger = latent_ledger_from_contract(contract)

    assert len(extracted["trial_pairs"]) == 1
    assert len(extracted["trial_observations"]) == 2
    assert candidate.metadata["improvement_verified"] is True
    assert candidate.verification_result["improvement_certificate"]["intent_id"] == "clarity"
    assert feedback["certificates"] == 1
    assert feedback["verified_certificates"] == 1
    assert feedback["evidence_added"] >= 2
    assert ledger.replay().active_evidence_ids
    assert snapshot is not None and snapshot.state.top_intent().id == "clarity"


def test_partial_trial_scores_are_weak_evidence_not_verified_certificate() -> None:
    contract = _contract()
    candidate = CandidateGenome(
        id="C2",
        verification_result={"trial_observation": {"scores": {"impact": 0.8}, "intent_id": "impact", "round": 3}},
    )

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])

    assert "improvement_certificate" not in candidate.metadata
    assert feedback["trial_observations"] == 1
    assert feedback["certificates"] == 0
    assert feedback["evidence_added"] == 1
