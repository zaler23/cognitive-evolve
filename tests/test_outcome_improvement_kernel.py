from __future__ import annotations

from cognitive_evolve_runtime.outcomes import (
    OutcomeContract,
    OutcomeMetric,
    TrialObservation,
    compare_outcomes,
    improvement_edge,
    verify_certificate,
)


def _contract(*, rule: str = "weighted_lcb") -> OutcomeContract:
    return OutcomeContract(
        objective="produce a verifiably better result for the declared input",
        scope="unit-test-scope",
        comparison_rule=rule,  # type: ignore[arg-type]
        min_effect=0.05,
        metrics=(
            OutcomeMetric(id="quality", weight=1.0, direction="maximize"),
            OutcomeMetric(id="cost", weight=0.2, direction="minimize", protected_regression_tolerance=0.1),
        ),
        hard_constraints=("same declared objective", "raw evidence retained"),
    )


def _observation(
    contract: OutcomeContract,
    artifact_id: str,
    *,
    quality: float,
    cost: float,
    manifest_hash: str = "manifest:v1",
    verifier_ref: str = "independent-verifier",
    proposer_ref: str = "generator",
    constraints_passed: bool = True,
) -> TrialObservation:
    return TrialObservation(
        artifact_id=artifact_id,
        contract_hash=contract.contract_hash(),
        manifest_hash=manifest_hash,
        environment_hash="env:v1",
        evaluator_hash="eval:v1",
        scores={"quality": quality, "cost": cost},
        uncertainty_radius={"quality": 0.01, "cost": 0.01},
        constraints_passed=constraints_passed,
        raw_observation_ref=f"raw:{artifact_id}",
        proposer_ref=proposer_ref,
        verifier_ref=verifier_ref,
        evidence_refs=(f"evidence:{artifact_id}",),
        seed="seed:v1",
    )


def test_certifies_scoped_improvement_over_same_baseline() -> None:
    contract = _contract()
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.50)
    challenger = _observation(contract, "challenger", quality=0.75, cost=0.45)

    certificate = compare_outcomes(contract, baseline, challenger)
    edge = improvement_edge("baseline", "challenger", certificate)

    assert certificate.verified is True
    assert certificate.status == "verified"
    assert certificate.aggregate_lcb > contract.min_effect
    assert edge.status == "verified"
    assert edge.reason_codes == ()


def test_rejects_manifest_drift_even_when_score_improves() -> None:
    contract = _contract()
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.50, manifest_hash="manifest:v1")
    challenger = _observation(contract, "challenger", quality=0.95, cost=0.40, manifest_hash="manifest:v2")

    certificate = compare_outcomes(contract, baseline, challenger)

    assert certificate.verified is False
    assert certificate.status == "rejected"
    assert "same_manifest" in certificate.critical_failures


def test_rejects_protected_metric_regression_hidden_by_quality_gain() -> None:
    contract = _contract()
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.30)
    challenger = _observation(contract, "challenger", quality=0.95, cost=0.80)

    certificate = compare_outcomes(contract, baseline, challenger)

    assert certificate.status == "rejected"
    assert "protected_metric_non_regression:cost" in certificate.critical_failures
    assert "protected_metrics_ok" in certificate.critical_failures


def test_rejects_self_certification() -> None:
    contract = _contract()
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.50)
    challenger = _observation(
        contract,
        "challenger",
        quality=0.75,
        cost=0.45,
        proposer_ref="same-agent",
        verifier_ref="same-agent",
    )

    certificate = compare_outcomes(contract, baseline, challenger)

    assert certificate.status == "rejected"
    assert "independent_verifier" in certificate.critical_failures


def test_certificate_verification_detects_evidence_drift() -> None:
    contract = _contract()
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.50)
    challenger = _observation(contract, "challenger", quality=0.75, cost=0.45)
    certificate = compare_outcomes(contract, baseline, challenger)
    tampered_challenger = _observation(contract, "challenger", quality=0.62, cost=0.45)

    verified = verify_certificate(certificate, contract=contract, baseline=baseline, challenger=tampered_challenger)

    assert verified.status == "rejected"
    assert "challenger_observation_hash_drift" in verified.critical_failures
    assert "evidence_hash_drift" in verified.critical_failures


def test_pareto_rule_requires_no_negative_lcb_and_one_practical_gain() -> None:
    contract = _contract(rule="pareto_lcb")
    baseline = _observation(contract, "baseline", quality=0.60, cost=0.50)
    challenger = _observation(contract, "challenger", quality=0.70, cost=0.46)

    certificate = compare_outcomes(contract, baseline, challenger)

    assert certificate.status == "verified"
    assert certificate.metric_lcbs["quality"] > contract.min_effect
    assert certificate.metric_lcbs["cost"] >= 0
