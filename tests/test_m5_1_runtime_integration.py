from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract, NexusObjectiveContractBuilder
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, _closure_certificate, _completion_status_for_budget
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.outcomes import (
    ExplorationAction,
    FrontierCandidate,
    IntentHypothesis,
    LatentProblemState,
    OutcomeContract,
    OutcomeMetric,
    TrialObservation,
    annotate_candidates_with_latent_signals,
    compare_outcomes,
)


class SolvedStopModel:
    def should_stop(self, **_: Any) -> dict[str, Any]:
        return {"stop": True, "solved": True, "reason": "objective_solved"}


def _text_world(text: str) -> tuple[TextInputPacket, TextWorldModel]:
    packet = TextInputPacket.from_text(text)
    return packet, TextWorldModel.from_packet(packet)


def _improvement_certificate(challenger_id: str = "C1"):
    contract = OutcomeContract(
        objective="verified local improvement",
        scope="unit-test",
        metrics=(OutcomeMetric(id="quality", weight=1.0),),
        min_effect=0.05,
    )
    baseline = TrialObservation(
        artifact_id="baseline",
        contract_hash=contract.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={"quality": 0.50},
        uncertainty_radius={"quality": 0.01},
        raw_observation_ref="raw:baseline",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    challenger = TrialObservation(
        artifact_id=challenger_id,
        contract_hash=contract.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={"quality": 0.70},
        uncertainty_radius={"quality": 0.01},
        raw_observation_ref=f"raw:{challenger_id}",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
    )
    return compare_outcomes(contract, baseline, challenger)


def test_contract_builder_attaches_latent_state_for_ambiguous_goal() -> None:
    packet, world = _text_world("Find the best, simplest and most useful evolution strategy.")

    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal=packet.raw_text, packet=packet, world=world)

    assert contract.metadata["latent_problem_state_hash"]
    summary = contract.metadata["latent_problem_state_summary"]
    assert summary["intent_count"] >= 2
    assert "posterior_entropy" in summary
    roundtrip = NexusObjectiveContract.from_dict(contract.to_dict())
    assert roundtrip.metadata["latent_problem_state_hash"] == contract.metadata["latent_problem_state_hash"]
    assert roundtrip.contract_hash() == contract.contract_hash()


def test_clear_goal_does_not_force_latent_state() -> None:
    packet, world = _text_world("Return exactly the word OK.")

    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal=packet.raw_text, packet=packet, world=world)

    assert "latent_problem_state_hash" not in contract.metadata


def test_latent_ranking_penalizes_high_risk_candidate_in_runtime_metadata() -> None:
    state = LatentProblemState(
        intents=(IntentHypothesis(id="clarity", statement="make it clear", posterior=1.0),),
    )
    contract = NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve", metadata={"latent_problem_state": state.to_dict()})
    flashy = CandidateGenome(
        id="flashy",
        metadata={"latent_intent_scores": {"clarity": 0.92}, "latent_uncertainty": 0.9, "latent_risk": 0.9},
        multihead_scores={"objective_alignment": 0.95, "novelty": 0.2},
    )
    stable = CandidateGenome(
        id="stable",
        metadata={"latent_intent_scores": {"clarity": 0.76}, "latent_uncertainty": 0.03, "latent_risk": 0.01},
        multihead_scores={"objective_alignment": 0.76, "novelty": 0.1},
    )

    summary = annotate_candidates_with_latent_signals([flashy, stable], contract)

    assert summary["ranked_candidate_ids"][0] == "stable"
    assert stable.multihead_scores["latent_reproductive_signal"] > flashy.multihead_scores["latent_reproductive_signal"]
    assert stable.metadata["latent_ranking"]["rank"] == 1


def test_latent_pareto_frontier_keeps_distinct_intent_candidates() -> None:
    state = LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make it clear", posterior=0.5),
            IntentHypothesis(id="impact", statement="make it impactful", posterior=0.5),
        ),
    )
    contract = NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve", metadata={"latent_problem_state": state.to_dict()})
    clear = CandidateGenome(id="clear", metadata={"latent_intent_scores": {"clarity": 0.85, "impact": 0.40}})
    punchy = CandidateGenome(id="punchy", metadata={"latent_intent_scores": {"clarity": 0.40, "impact": 0.85}})
    weak = CandidateGenome(id="weak", metadata={"latent_intent_scores": {"clarity": 0.40, "impact": 0.40}})

    summary = annotate_candidates_with_latent_signals([clear, punchy, weak], contract)

    assert set(summary["pareto_frontier_ids"]) == {"clear", "punchy"}
    assert clear.metadata["latent_pareto_frontier"] is True
    assert punchy.metadata["latent_pareto_frontier"] is True
    assert weak.metadata["latent_pareto_frontier"] is False


def test_requires_verified_solution_blocks_solved_without_m5_certificate() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="verified result",
        normalized_goal="verified result",
        outcome_policy={"requires_verified_solution": True, "accepts_best_current_route": False},
    )
    budget = EvolutionBudget(max_rounds=1)
    budget.stop_reason = "objective_solved"
    synthesis = SynthesizedResult(status="completed", final_answer="answer", best_candidate_id="C1")
    completion = _completion_status_for_budget(budget=budget, interrupted=False, synthesis=synthesis)

    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=synthesis,
        completion_status=completion,
        contract=contract,
    )

    assert completion == "solved"
    assert certificate["objective_solved"] is False
    assert certificate["improvement_verified"] is False
    assert "missing_verified_improvement_certificate" in certificate["critical_failures"]


def test_verified_m5_certificate_is_exposed_in_closure_certificate() -> None:
    m5 = _improvement_certificate("C1")
    contract = NexusObjectiveContract(
        original_user_goal="verified result",
        normalized_goal="verified result",
        outcome_policy={"requires_verified_solution": True, "accepts_best_current_route": False},
    )
    budget = EvolutionBudget(max_rounds=1)
    budget.stop_reason = "objective_solved"
    synthesis = SynthesizedResult(status="completed", final_answer="answer", best_candidate_id="C1")
    completion = _completion_status_for_budget(budget=budget, interrupted=False, synthesis=synthesis)

    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=synthesis,
        completion_status=completion,
        contract=contract,
        improvement_certificate=m5,
    )

    assert certificate["objective_solved"] is True
    assert certificate["improvement_verified"] is True
    assert certificate["improvement_certificate_hash"] == m5.certificate_hash()
    assert certificate["baseline_id"] == "baseline"
    assert certificate["challenger_id"] == "C1"
    assert certificate["aggregate_lcb"] > 0.0


def test_stop_decision_blocks_solved_when_latent_space_unresolved() -> None:
    state = LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="make it clear", posterior=0.5),
            IntentHypothesis(id="impact", statement="make it impactful", posterior=0.5),
        ),
        actions=(ExplorationAction(action_id="probe", kind="intent_disambiguation", information_gain=0.4),),
    )
    contract = NexusObjectiveContract(original_user_goal="find best", normalized_goal="find best", metadata={"latent_problem_state": state.to_dict()})

    reason = StopDecisionEngine().stop_reason_after_round(
        budget=EvolutionBudget(max_rounds=2, stop_policy="llm_after_minimum", min_rounds_before_stop=1),
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        model=SolvedStopModel(),
        contract=contract,
    )

    assert reason == "latent_problem_space_needs_continuation"


def test_stop_decision_allows_solved_when_latent_converged_and_verified() -> None:
    m5 = _improvement_certificate("C1")
    state = LatentProblemState(
        intents=(IntentHypothesis(id="quality", statement="improve quality", posterior=1.0, uncertainty=0.0),),
        frontier_candidates=(FrontierCandidate(candidate_id="C1", utility_by_intent={"quality": 0.8}),),
    )
    contract = NexusObjectiveContract(original_user_goal="verified best", normalized_goal="verified best", metadata={"latent_problem_state": state.to_dict()})
    candidate = CandidateGenome(id="C1", metadata={"improvement_certificate": m5.to_dict()})
    candidate.verification_trace = [_measured_formal_result().to_dict()]

    reason = StopDecisionEngine().stop_reason_after_round(
        budget=EvolutionBudget(max_rounds=2, stop_policy="llm_after_minimum", min_rounds_before_stop=1),
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([candidate]),
        model=SolvedStopModel(),
        contract=contract,
    )

    assert reason == "objective_solved"


def _measured_formal_result() -> VerificationResult:
    return VerificationResult(
        True,
        score=1.0,
        strength=VerificationStrength.FORMAL,
        evidence_ref="evidence",
        replayable=True,
        metadata={
            "verifier_fingerprint": "vf",
            "measured_strength": "FORMAL",
            "measured_strength_value": 4,
            "honesty_measurements": {
                "exogeneity_score": 1.0,
                "variety_score": 1.0,
                "falsification_score": 1.0,
                "replay_score": 1.0,
            },
            "diagnostics_only": False,
            "legacy": False,
        },
    )
