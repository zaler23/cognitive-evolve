from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    ExplorationAction,
    IntentHypothesis,
    LatentLedger,
    LatentLedgerStore,
    LatentProblemState,
    OutcomeContract,
    OutcomeMetric,
    PreferenceEvidence,
    TrialObservation,
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    audit_latent_decision_replay,
    freeze_improvement_certificate_from_trials,
    ingest_latent_feedback,
    materialize_contract_latent_posterior,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make it clear", posterior=0.5, utility_dimensions=("clarity",)),
            IntentHypothesis(id="impact", statement="make it impactful", posterior=0.5, utility_dimensions=("impact",)),
        ),
        actions=(
            ExplorationAction(
                action_id="probe_impact",
                kind="intent_disambiguation",
                target_intent_ids=("impact",),
                information_gain=0.7,
                expected_improvement=0.1,
            ),
        ),
    )


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata={"latent_problem_state": _state().to_dict(), "latent_ledger": LatentLedger().to_dict()},
    )


def test_latent_ledger_store_persists_jsonl_and_snapshot(tmp_path) -> None:
    ledger = LatentLedger()
    ledger.add_evidence(
        PreferenceEvidence(
            intent_id="impact",
            support=0.8,
            evidence_ref="verifier:C1",
            source_type="verifier",
            provenance_ref="verifier:C1",
        )
    )
    store = LatentLedgerStore(tmp_path)
    persisted = store.persist_ledger(ledger)
    loaded = store.load_ledger()

    assert persisted["events_appended"] == 1
    assert (tmp_path / "latent-events.jsonl").exists()
    assert loaded.replay().active_evidence_ids == ledger.replay().active_evidence_ids

    snapshot = materialize_contract_latent_posterior(_contract())
    assert snapshot is not None
    snapshot_result = store.persist_snapshot(snapshot)
    assert snapshot_result["snapshot_hash"] == snapshot.snapshot_hash()
    assert (tmp_path / "latent-posterior-snapshot.json").exists()


def test_exploration_action_reaches_mutation_plan_metadata() -> None:
    contract = _contract()
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["P1"], instruction="Deepen current route.")

    plans, summary = apply_latent_exploration_to_mutation_plans([plan], contract)

    assert summary["mutation_actions"] == ["case_split"]
    assert plans[0].metadata["latent_exploration_action"]["action_id"] == "probe_impact"
    assert "Latent exploration directive probe_impact" in plans[0].instruction
    assert plans[0].metadata["latent_decision_trace"]["latent_posterior_snapshot_hash"]


def test_trial_observations_freeze_and_attach_improvement_certificate() -> None:
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
        scores={"clarity": 0.70},
        uncertainty_radius={"clarity": 0.01},
        raw_observation_ref="raw:C1",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    candidate = CandidateGenome(id="C1")

    certificate = freeze_improvement_certificate_from_trials(
        outcome_contract=outcome,
        baseline=baseline,
        challenger=challenger,
        candidate=candidate,
        intent_id="clarity",
    )

    assert certificate.verified is True
    assert candidate.metadata["improvement_verified"] is True
    assert candidate.metadata["improvement_certificate_hash"] == certificate.certificate_hash()
    assert candidate.verification_result["improvement_certificate"]["intent_id"] == "clarity"


def test_latent_decision_replay_audit_passes_for_pinned_cursor() -> None:
    contract = _contract()
    ingest_latent_feedback(contract=contract, verifier_results=[{"candidate_id": "C1", "passed": True, "intent_id": "impact"}])
    clear = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.8, "impact": 0.2}})
    punchy = CandidateGenome(id="punchy", metadata={"latent_intent_scores": {"clarity": 0.2, "impact": 0.8}})

    annotate_candidates_with_latent_signals([clear, punchy], contract)
    trace = punchy.metadata["latent_decision_trace"]
    audit = audit_latent_decision_replay(contract, trace)
    bad = audit_latent_decision_replay(contract, trace | {"latent_posterior_snapshot_hash": "bad"})

    assert audit["passed"] is True
    assert bad["passed"] is False
