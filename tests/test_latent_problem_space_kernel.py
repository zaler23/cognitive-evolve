from __future__ import annotations

from cognitive_evolve_runtime.outcomes import (
    ExplorationAction,
    FrontierCandidate,
    IntentHypothesis,
    LatentProblemState,
    PreferenceEvidence,
    TrialObservation,
    assess_convergence,
    compare_outcomes,
    freeze_outcome_contract,
    pareto_frontier,
    rank_candidates,
    select_exploration_action,
    update_intent_posteriors,
)


def _ambiguous_state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(
                id="clarity",
                statement="make the result easier to understand without changing its core claim",
                posterior=0.5,
                utility_dimensions=("clarity", "faithfulness"),
                hard_constraints=("preserve core claim",),
            ),
            IntentHypothesis(
                id="force",
                statement="make the result more forceful and memorable",
                posterior=0.5,
                utility_dimensions=("impact", "style"),
                hard_constraints=("avoid distortion",),
            ),
        ),
        frontier_candidates=(
            FrontierCandidate(
                candidate_id="clear_version",
                utility_by_intent={"clarity": 0.82, "force": 0.45},
                uncertainty_by_intent={"clarity": 0.04, "force": 0.08},
                risk=0.05,
                cost=0.10,
            ),
            FrontierCandidate(
                candidate_id="forceful_version",
                utility_by_intent={"clarity": 0.48, "force": 0.86},
                uncertainty_by_intent={"clarity": 0.08, "force": 0.04},
                risk=0.05,
                cost=0.10,
            ),
            FrontierCandidate(
                candidate_id="weak_middle",
                utility_by_intent={"clarity": 0.40, "force": 0.40},
                uncertainty_by_intent={"clarity": 0.05, "force": 0.05},
                risk=0.10,
                cost=0.20,
            ),
        ),
        actions=(
            ExplorationAction(
                action_id="polish_current_best",
                kind="candidate_improvement",
                expected_improvement=0.18,
                information_gain=0.01,
                cost=0.05,
            ),
            ExplorationAction(
                action_id="ask_pairwise_preference_probe",
                kind="intent_disambiguation",
                target_intent_ids=("clarity", "force"),
                expected_improvement=0.04,
                information_gain=0.35,
                diversity_gain=0.05,
                cost=0.05,
            ),
        ),
    )


def _observation(contract, artifact_id: str, *, quality: float) -> TrialObservation:
    return TrialObservation(
        artifact_id=artifact_id,
        contract_hash=contract.contract_hash(),
        manifest_hash="manifest:v1",
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={metric.id: quality for metric in contract.metrics},
        uncertainty_radius={metric.id: 0.01 for metric in contract.metrics},
        constraints_passed=True,
        raw_observation_ref=f"raw:{artifact_id}",
        proposer_ref="generator",
        verifier_ref="independent-verifier",
        seed="seed:v1",
    )


def test_intent_posteriors_update_without_collapsing_other_hypotheses() -> None:
    state = _ambiguous_state()

    updated = update_intent_posteriors(
        state,
        [
            PreferenceEvidence(intent_id="clarity", support=0.9, weight=2.0, evidence_ref="pairwise:1"),
            PreferenceEvidence(intent_id="force", contradiction=0.5, weight=1.0, evidence_ref="pairwise:2"),
        ],
    )

    clarity = next(intent for intent in updated.intents if intent.id == "clarity")
    force = next(intent for intent in updated.intents if intent.id == "force")
    assert clarity.posterior > force.posterior
    assert 0 < force.posterior < 0.5
    assert abs(sum(intent.posterior for intent in updated.intents) - 1.0) < 1e-9
    assert updated.evidence_refs == ("pairwise:1", "pairwise:2")


def test_ambiguous_state_prefers_information_gain_over_premature_polish() -> None:
    state = _ambiguous_state()

    action = select_exploration_action(state)

    assert action is not None
    assert action.action_id == "ask_pairwise_preference_probe"


def test_candidate_ranking_penalizes_uncertainty_risk_and_cost() -> None:
    state = LatentProblemState(
        intents=(IntentHypothesis(id="clarity", statement="make it clear", posterior=1.0),),
        frontier_candidates=(
            FrontierCandidate(
                candidate_id="flashy_uncertain",
                utility_by_intent={"clarity": 0.90},
                uncertainty_by_intent={"clarity": 0.50},
                risk=0.05,
                cost=0.05,
            ),
            FrontierCandidate(
                candidate_id="stable_improvement",
                utility_by_intent={"clarity": 0.76},
                uncertainty_by_intent={"clarity": 0.04},
                risk=0.01,
                cost=0.02,
            ),
        ),
    )

    ranked = rank_candidates(state)

    assert ranked[0].candidate_id == "stable_improvement"
    assert ranked[0].score > ranked[1].score


def test_pareto_frontier_preserves_distinct_high_value_interpretations() -> None:
    state = _ambiguous_state()

    frontier = pareto_frontier(state)

    assert {candidate.candidate_id for candidate in frontier} == {"clear_version", "forceful_version"}


def test_freezing_top_intent_creates_m5_outcome_contract() -> None:
    state = update_intent_posteriors(
        _ambiguous_state(),
        [PreferenceEvidence(intent_id="clarity", support=1.0, weight=3.0)],
    )

    contract = freeze_outcome_contract(state, min_effect=0.05)

    assert contract.scope == "latent-intent:clarity"
    assert contract.objective.startswith("make the result easier")
    assert {metric.id for metric in contract.metrics} == {"clarity", "faithfulness"}
    assert contract.contract_hash() == freeze_outcome_contract(state, min_effect=0.05).contract_hash()


def test_convergence_requires_low_entropy_low_value_next_action_and_m5_certificate() -> None:
    state = LatentProblemState(
        intents=(IntentHypothesis(id="clarity", statement="make it clear", posterior=1.0),),
        frontier_candidates=(
            FrontierCandidate(candidate_id="candidate_b", utility_by_intent={"clarity": 0.85}, uncertainty_by_intent={"clarity": 0.02}),
        ),
        actions=(ExplorationAction(action_id="tiny_cleanup", kind="candidate_improvement", expected_improvement=0.01, cost=0.02),),
    )
    contract = freeze_outcome_contract(state, min_effect=0.05)
    baseline = _observation(contract, "baseline", quality=0.60)
    challenger = _observation(contract, "candidate_b", quality=0.75)
    certificate = compare_outcomes(contract, baseline, challenger)

    assessment = assess_convergence(state, improvement_certificate=certificate)

    assert certificate.verified is True
    assert assessment.converged is True
    assert assessment.selected_candidate_id == "candidate_b"
    assert assessment.reason_codes == ()


def test_convergence_rejects_unresolved_latent_space_even_with_good_candidate() -> None:
    state = _ambiguous_state()

    assessment = assess_convergence(state)

    assert assessment.converged is False
    assert "latent_intent_entropy_high" in assessment.reason_codes
    assert "valuable_exploration_remains" in assessment.reason_codes
    assert "missing_verified_improvement_certificate" in assessment.reason_codes
