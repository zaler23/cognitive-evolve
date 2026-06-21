from __future__ import annotations

import json

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop import _attach_latent_replay_audit_to_closure
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.outcomes import (
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    extract_runtime_trial_observations,
    freeze_outcome_contract,
    ingest_runtime_trial_feedback,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make the result clearer", posterior=0.5, utility_dimensions=("clarity",)),
            IntentHypothesis(id="impact", statement="make the result more useful", posterior=0.5, utility_dimensions=("impact",)),
        )
    )


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata={"latent_problem_state": _state().to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def test_nexus_final_result_persists_latent_replay_audit_and_closure_binding(tmp_path) -> None:
    result = NexusRuntime(output_dir=tmp_path).run_text(
        "Find the best, most useful and most elegant answer across unclear criteria.",
        max_rounds=1,
    )

    audit = result.evolution["latent_replay_audit"]
    closure = result.evolution["synthesis"]["closure_certificate"]
    persisted = json.loads((tmp_path / "run-result.json").read_text())

    assert audit["passed"] is True
    assert audit["total"] > 0
    assert closure["latent_replay_audit"]["passed"] is True
    assert closure["latent_replay_audit"]["total"] == audit["total"]
    assert any(check["check"] == "latent_replay_audit_passed" for check in closure["checks"])
    assert persisted["evolution"]["latent_replay_audit"] == audit
    assert persisted["evolution"]["synthesis"]["closure_certificate"]["latent_replay_audit"] == closure["latent_replay_audit"]


def test_closure_audit_failure_blocks_objective_solved() -> None:
    synthesis = SynthesizedResult(
        status="completed",
        final_answer="answer",
        closure_certificate={"objective_solved": True, "checks": [], "critical_failures": []},
    )

    _attach_latent_replay_audit_to_closure(
        synthesis,
        {
            "passed": False,
            "total": 1,
            "failed_count": 1,
            "trace_refs": ["trace:bad"],
            "failure_reasons": ["snapshot_hash_mismatch"],
        },
    )

    assert synthesis.closure_certificate["objective_solved"] is True
    assert synthesis.closure_certificate["latent_replay_audit"]["passed"] is False
    assert synthesis.closure_certificate["latent_replay_audit_advisory"] == "failed_nonblocking"
    assert "latent_replay_audit_failed_advisory_only" in synthesis.warnings


def test_semistructured_metric_observation_becomes_weak_feedback_not_certificate() -> None:
    contract = _contract()
    candidate = CandidateGenome(
        id="C-metric",
        verification_result={"metric_observation": {"intent_id": "impact", "score": 0.82, "round": 2}},
    )

    extracted = extract_runtime_trial_observations([candidate])
    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])

    assert len(extracted["trial_observations"]) == 1
    assert extracted["trial_observations"][0].scores == {"impact": 0.82}
    assert feedback["trial_observations"] == 1
    assert feedback["certificates"] == 0
    assert feedback["evidence_added"] == 1
    assert "improvement_certificate" not in candidate.metadata
    assert "improvement_certificate" not in candidate.verification_result


def test_semistructured_metric_pair_can_freeze_verified_certificate() -> None:
    state = _state()
    contract = _contract()
    outcome = freeze_outcome_contract(state, intent_id="clarity", min_effect=0.05)
    base = {
        "artifact_id": "baseline",
        "contract_hash": outcome.contract_hash(),
        "manifest_hash": "manifest:v1",
        "environment_hash": "env:v1",
        "evaluator_hash": "eval:v1",
        "metrics": [{"id": "clarity", "score": 0.50}],
        "uncertainty_radius": {"clarity": 0.01},
        "raw_observation_ref": "raw:baseline",
        "proposer_ref": "generator",
        "verifier_ref": "independent-verifier",
    }
    challenger = dict(base) | {
        "artifact_id": "C-pair",
        "metrics": [{"id": "clarity", "score": 0.72}],
        "raw_observation_ref": "raw:C-pair",
    }
    candidate = CandidateGenome(
        id="C-pair",
        verification_result={
            "m5_trial_pair": {
                "intent_id": "clarity",
                "source_type": "runtime_verifier",
                "provenance_ref": "verifier-run:metric-pair",
                "verifier_run_id": "metric-pair",
                "baseline": base,
                "challenger": challenger,
            }
        },
    )

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate], outcome_contract=outcome)

    assert feedback["certificates"] == 1
    assert feedback["verified_certificates"] == 1
    assert candidate.metadata["improvement_verified"] is True
    assert candidate.verification_result["improvement_certificate"]["intent_id"] == "clarity"


def test_incomplete_basis_trial_pair_is_rejected_not_verified() -> None:
    contract = _contract()
    base = {
        "artifact_id": "baseline",
        "scores": {"clarity": 0.50},
        "raw_observation_ref": "model-claimed-baseline",
        "proposer_ref": "generator",
        "verifier_ref": "independent-verifier",
    }
    challenger = dict(base) | {
        "artifact_id": "C-claimed",
        "scores": {"clarity": 0.95},
        "raw_observation_ref": "model-claimed-challenger",
    }
    candidate = CandidateGenome(
        id="C-claimed",
        verification_result={
            "m5_trial_pair": {
                "intent_id": "clarity",
                "source_type": "runtime_verifier",
                "provenance_ref": "verifier-run:tiny-effect",
                "verifier_run_id": "tiny-effect",
                "baseline": base,
                "challenger": challenger,
            }
        },
    )

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])
    certificate = candidate.verification_result["improvement_certificate"]

    assert feedback["certificates"] == 1
    assert feedback["verified_certificates"] == 0
    assert candidate.metadata["improvement_verified"] is False
    assert certificate["status"] == "rejected"
    assert "basis_present:manifest_hash" in certificate["critical_failures"]
    assert "basis_present:environment_hash" in certificate["critical_failures"]
    assert "basis_present:evaluator_hash" in certificate["critical_failures"]


def test_auto_frozen_trial_pair_requires_meaningful_min_effect() -> None:
    contract = _contract()
    outcome = freeze_outcome_contract(_state(), intent_id="clarity", min_effect=0.05)
    base = {
        "artifact_id": "baseline",
        "contract_hash": outcome.contract_hash(),
        "manifest_hash": "manifest:v1",
        "environment_hash": "env:v1",
        "evaluator_hash": "eval:v1",
        "scores": {"clarity": 0.500},
        "uncertainty_radius": {"clarity": 0.0},
        "raw_observation_ref": "raw:baseline",
        "proposer_ref": "generator",
        "verifier_ref": "independent-verifier",
    }
    challenger = dict(base) | {
        "artifact_id": "C-tiny",
        "contract_hash": "",  # force runtime auto-freeze path rather than caller-supplied contract
        "scores": {"clarity": 0.501},
        "raw_observation_ref": "raw:C-tiny",
    }
    base = dict(base) | {"contract_hash": ""}
    candidate = CandidateGenome(
        id="C-tiny",
        verification_result={
            "m5_trial_pair": {
                "intent_id": "clarity",
                "source_type": "runtime_verifier",
                "provenance_ref": "verifier-run:tiny-effect",
                "verifier_run_id": "tiny-effect",
                "baseline": base,
                "challenger": challenger,
            }
        },
    )

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])
    certificate = candidate.verification_result["improvement_certificate"]

    assert feedback["certificates"] == 1
    assert feedback["verified_certificates"] == 0
    assert certificate["status"] == "rejected"
    assert any(check["check"] == "dominance_lcb_threshold" and check["detail"]["min_effect"] == 0.05 for check in certificate["checks"])
