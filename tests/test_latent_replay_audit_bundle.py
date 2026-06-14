from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.outcomes import (
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    annotate_candidates_with_latent_signals,
    ingest_latent_feedback,
)
from cognitive_evolve_runtime.outcomes.latent_audit import (
    audit_latent_replay_bundle,
    collect_latent_decision_traces,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make it clear", posterior=0.5, utility_dimensions=("clarity",)),
            IntentHypothesis(id="impact", statement="make it impactful", posterior=0.5, utility_dimensions=("impact",)),
        )
    )


def _contract(*, with_latent_state: bool = True) -> NexusObjectiveContract:
    metadata = {"latent_ledger": LatentLedger().to_dict()}
    if with_latent_state:
        metadata["latent_problem_state"] = _state().to_dict()
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata=metadata,
    )


def _ranked_candidate_with_trace(contract: NexusObjectiveContract) -> CandidateGenome:
    ingest_latent_feedback(contract=contract, verifier_results=[{"candidate_id": "punchy", "passed": True, "intent_id": "impact"}])
    clear = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.8, "impact": 0.2}})
    punchy = CandidateGenome(id="punchy", metadata={"latent_intent_scores": {"clarity": 0.2, "impact": 0.8}})
    annotate_candidates_with_latent_signals([clear, punchy], contract)
    return punchy


def test_candidate_and_contract_latent_decision_trace_passes_bundle_audit() -> None:
    contract = _contract()
    candidate = _ranked_candidate_with_trace(contract)

    traces = collect_latent_decision_traces(contract=contract, candidates=[candidate])
    audit = audit_latent_replay_bundle(contract, candidates=[candidate])

    assert [item["source"] for item in traces] == [
        "contract.metadata.latent_decision_trace",
        "candidates[punchy].metadata.latent_decision_trace",
    ]
    assert audit["passed"] is True
    assert audit["total"] == 2
    assert audit["passed_count"] == 2
    assert audit["failed"] == 0
    assert audit["trace_refs"] == [candidate.metadata["latent_decision_trace_ref"], candidate.metadata["latent_decision_trace_ref"]]


def test_wrong_latent_snapshot_hash_fails_bundle_audit() -> None:
    contract = _contract()
    candidate = _ranked_candidate_with_trace(contract)
    contract.metadata.pop("latent_decision_trace", None)
    bad_candidate = CandidateGenome(
        id="punchy",
        metadata={
            "latent_decision_trace": candidate.metadata["latent_decision_trace"]
            | {"latent_posterior_snapshot_hash": "bad"},
        },
    )

    audit = audit_latent_replay_bundle(contract, candidates=[bad_candidate])

    assert audit["passed"] is False
    assert audit["total"] == 1
    assert audit["failed"] == 1
    assert audit["failures"][0]["reason"] == "snapshot_hash_mismatch"


def test_expected_hash_without_latent_state_fails_closed() -> None:
    source_contract = _contract()
    candidate = _ranked_candidate_with_trace(source_contract)

    audit = audit_latent_replay_bundle(_contract(with_latent_state=False), candidates=[candidate])

    assert audit["passed"] is False
    assert audit["failed"] == 1
    assert audit["failures"][0]["reason"] == "no_latent_state"


def test_collects_generation_plan_and_budget_history_traces() -> None:
    contract = _contract()
    candidate = _ranked_candidate_with_trace(contract)
    contract.metadata.pop("latent_decision_trace", None)
    trace = dict(candidate.metadata["latent_decision_trace"])
    generation_plan = {"ranking_summary": {"latent_ranking": trace}}
    budget_history = [
        {
            "round": 1,
            "generation_plan": {
                "latent_exploration_planning": {
                    "latent_decision_trace": trace,
                    "latent_posterior_snapshot_hash": trace["latent_posterior_snapshot_hash"],
                    "latent_ledger_cursor": trace["latent_ledger_cursor"],
                }
            },
        }
    ]

    traces = collect_latent_decision_traces(contract=contract, generation_plan=generation_plan, budget_history=budget_history)
    audit = audit_latent_replay_bundle(contract, generation_plan=generation_plan, budget_history=budget_history)

    assert [item["source"] for item in traces] == [
        "generation_plan.ranking_summary.latent_ranking",
        "budget_history[0].generation_plan.latent_exploration_planning.latent_decision_trace",
    ]
    assert audit["passed"] is True
    assert audit["total"] == 2
    assert audit["failed"] == 0


def test_no_latent_decision_traces_is_empty_pass() -> None:
    audit = audit_latent_replay_bundle(_contract(), candidates=[], generation_plan={}, budget_history=[])

    assert audit["passed"] is True
    assert audit["total"] == 0
    assert audit["passed_count"] == 0
    assert audit["failed"] == 0
    assert audit["trace_refs"] == []
    assert audit["failures"] == []
