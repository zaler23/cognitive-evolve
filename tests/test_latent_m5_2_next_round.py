from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    IntentHypothesis,
    LatentLedger,
    LatentLedgerStore,
    LatentProblemState,
    PreferenceEvidence,
    annotate_candidates_with_latent_signals,
    freeze_outcome_contract,
    ingest_runtime_trial_feedback,
    OutcomeContract,
    OutcomeMetric,
    TrialObservation,
    compare_outcomes,
    materialize_posterior_snapshot,
)
from cognitive_evolve_runtime.outcomes.evidence_feedback import adapt_improvement_certificate
from cognitive_evolve_runtime.outcomes.latent_audit import audit_latent_replay_bundle


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


def _trusted_pair(*, raw_prefix: str = "raw") -> dict[str, object]:
    outcome = freeze_outcome_contract(_state(), intent_id="clarity", min_effect=0.05)
    base = {
        "artifact_id": "baseline",
        "contract_hash": outcome.contract_hash(),
        "manifest_hash": "manifest:v1",
        "environment_hash": "env:v1",
        "evaluator_hash": "eval:v1",
        "scores": {"clarity": 0.50},
        "uncertainty_radius": {"clarity": 0.01},
        "raw_observation_ref": f"{raw_prefix}:baseline",
        "proposer_ref": "generator",
        "verifier_ref": "independent-verifier",
    }
    challenger = dict(base) | {
        "artifact_id": "C1",
        "scores": {"clarity": 0.72},
        "raw_observation_ref": f"{raw_prefix}:C1",
    }
    return {
        "intent_id": "clarity",
        "source_type": "runtime_verifier",
        "provenance_ref": "verifier-run:trusted-pair",
        "verifier_run_id": "trusted-pair",
        "baseline": base,
        "challenger": challenger,
    }



def test_verified_certificate_adapter_quarantines_missing_provenance() -> None:
    outcome = OutcomeContract(
        objective="improve clarity",
        scope="latent-intent:clarity",
        metrics=(OutcomeMetric(id="clarity", weight=1.0),),
        min_effect=0.05,
    )
    baseline = TrialObservation(
        artifact_id="baseline",
        contract_hash=outcome.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={"clarity": 0.50},
        uncertainty_radius={"clarity": 0.01},
        raw_observation_ref="raw:baseline",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    challenger = TrialObservation(
        artifact_id="C1",
        contract_hash=outcome.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={"clarity": 0.72},
        uncertainty_radius={"clarity": 0.01},
        raw_observation_ref="raw:C1",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    payload = compare_outcomes(outcome, baseline, challenger).to_dict() | {"intent_id": "clarity"}

    adapted = adapt_improvement_certificate(payload, _state())

    assert adapted.evidence == ()
    assert adapted.quarantined[0].reason == "untrusted_verified_certificate_provenance"

def test_trial_pair_in_metadata_cannot_self_certify_even_with_claimed_source() -> None:
    contract = _contract()
    candidate = CandidateGenome(id="C-meta", metadata={"m5_trial_pair": _trusted_pair()})

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])
    certificate = candidate.metadata["improvement_certificate"]

    assert feedback["certificates"] == 1
    assert feedback["verified_certificates"] == 0
    assert candidate.metadata["improvement_verified"] is False
    assert certificate["status"] == "rejected"
    assert "untrusted_trial_pair_container" in certificate["critical_failures"]


def test_verifier_trial_pair_requires_trusted_source_and_provenance() -> None:
    contract = _contract()
    pair = dict(_trusted_pair())
    pair.pop("source_type")
    pair.pop("provenance_ref")
    pair.pop("verifier_run_id")
    candidate = CandidateGenome(id="C-unprovenanced", verification_result={"m5_trial_pair": pair})

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])
    certificate = candidate.verification_result["improvement_certificate"]

    assert feedback["verified_certificates"] == 0
    assert candidate.metadata["improvement_verified"] is False
    assert "untrusted_trial_pair_source" in certificate["critical_failures"]
    assert "missing_trial_pair_provenance" in certificate["critical_failures"]


def test_runtime_certificate_requires_replayable_raw_evidence_refs() -> None:
    contract = _contract()
    candidate = CandidateGenome(id="C-narrative", verification_result={"m5_trial_pair": _trusted_pair(raw_prefix="model-narrative")})

    feedback = ingest_runtime_trial_feedback(contract=contract, candidates=[candidate])
    certificate = candidate.verification_result["improvement_certificate"]

    assert feedback["verified_certificates"] == 0
    assert candidate.metadata["improvement_verified"] is False
    assert "raw_evidence_replayable" in certificate["critical_failures"]
    assert "non_replayable_baseline_raw_evidence" in certificate["critical_failures"]
    assert "non_replayable_challenger_raw_evidence" in certificate["critical_failures"]


def test_latent_ledger_compaction_snapshot_restores_active_posterior_state() -> None:
    state = _state()
    ledger = LatentLedger()
    first = PreferenceEvidence(intent_id="clarity", support=0.7, evidence_ref="verifier:first", source_type="verifier")
    second = PreferenceEvidence(intent_id="impact", support=0.9, evidence_ref="verifier:second", source_type="verifier")
    ledger.add_evidence(first, idempotency_key="first")
    ledger.add_evidence(second, idempotency_key="second")
    ledger.retract_evidence("first", reason="stale_first")
    ledger.reject_evidence({"bad": True}, reason="malformed", source_type="verifier", provenance_ref="bad")

    full_snapshot = materialize_posterior_snapshot(state, ledger)
    compaction = ledger.compaction_snapshot()
    compacted = LatentLedger.from_compaction_snapshot(compaction)
    compacted_snapshot = materialize_posterior_snapshot(state, compacted)

    assert compaction["source_cursor"] == ledger.cursor
    assert compaction["active_evidence_ids"] == list(full_snapshot.active_evidence_ids)
    assert compacted.replay().active_evidence_ids == full_snapshot.active_evidence_ids
    assert compacted_snapshot.state.state_hash() == full_snapshot.state.state_hash()


def test_latent_ledger_store_persists_and_loads_compaction_snapshot(tmp_path) -> None:
    ledger = LatentLedger()
    ledger.add_evidence(PreferenceEvidence(intent_id="clarity", support=0.8, evidence_ref="verifier:clarity", source_type="verifier"))
    store = LatentLedgerStore(tmp_path)

    persisted = store.persist_compaction_snapshot(ledger)
    loaded = store.load_compaction_snapshot()
    restored = LatentLedger.from_compaction_snapshot(loaded)

    assert persisted["source_cursor"] == ledger.cursor
    assert loaded["compaction_hash"] == persisted["compaction_hash"]
    assert restored.replay().active_evidence_ids == ledger.replay().active_evidence_ids


def test_latent_pareto_archive_traces_are_included_in_replay_audit() -> None:
    contract = _contract()
    clear = CandidateGenome(
        id="clear",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={"latent_intent_scores": {"clarity": 0.9, "impact": 0.2}},
        multihead_scores={"latent_reproductive_signal": 0.7},
    )
    impact = CandidateGenome(
        id="impact",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={"latent_intent_scores": {"clarity": 0.2, "impact": 0.9}},
        multihead_scores={"latent_reproductive_signal": 0.7},
    )
    annotate_candidates_with_latent_signals([clear, impact], contract)
    archives = ArchiveManager()
    archives.update([clear, impact])

    audit = audit_latent_replay_bundle(contract, archives=archives)

    assert audit["passed"] is True
    assert audit["total"] >= 2
    assert any(result["source"].startswith("archives.latent_pareto_archive") for result in audit["results"])


def test_bad_latent_pareto_archive_trace_fails_closed() -> None:
    contract = _contract()
    candidate = CandidateGenome(
        id="bad-frontier",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={"latent_intent_scores": {"clarity": 0.9, "impact": 0.2}},
    )
    annotate_candidates_with_latent_signals([candidate], contract)
    candidate.metadata["latent_decision_trace"] = candidate.metadata["latent_decision_trace"] | {
        "latent_posterior_snapshot_hash": "bad"
    }
    archives = ArchiveManager()
    archives.update([candidate])

    audit = audit_latent_replay_bundle(contract, archives=archives)

    assert audit["passed"] is False
    assert audit["failed"] >= 1
    assert "snapshot_hash_mismatch" in audit["failure_reasons"]
