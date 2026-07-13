from __future__ import annotations

import pytest

from cognitive_evolve_runtime.outcomes.closure_bundle import (
    REQUIRED_DIGEST_FIELDS,
    audit_structural_replay_bundle,
    bind_closure_gate_digests,
    verify_structural_replay_bundle,
)
from cognitive_evolve_runtime.outcomes.falsification import (
    FalsificationCase,
    FalsificationOutcome,
    audit_falsification_gauntlet,
    build_falsification_gauntlet,
    verify_falsification_gauntlet,
)


def _falsification_bundle(**overrides: object):
    base = {
        "scope": "unit-closure",
        "candidate_id": "candidate",
        "problem_model_snapshot_hash": "pms:abc",
        "cases": (
            FalsificationCase(
                case_id="counterexample-search",
                assertion_ref="claim:solve",
                challenge="Search boundary cases that would invalidate the solve claim.",
            ),
            FalsificationCase(
                case_id="adversarial-replay",
                assertion_ref="claim:solve",
                challenge="Replay with adversarial examples.",
            ),
        ),
        "outcomes": (
            FalsificationOutcome(case_id="counterexample-search", survived=True, evidence_ref="raw:counterexample-search"),
            FalsificationOutcome(case_id="adversarial-replay", survived=True, evidence_ref="raw:adversarial-replay"),
        ),
    }
    base.update(overrides)
    return build_falsification_gauntlet(**base)  # type: ignore[arg-type]


def _structural_bundle():
    return bind_closure_gate_digests(
        scope="unit-closure",
        candidate_id="candidate",
        e_process_digest="eps:ok",
        calibration_digest="cals:ok",
        falsification_digest="fgb:ok",
        problem_model_digest="pms:ok",
        latent_digest="lrs:ok",
        pareto_digest="pareto:ok",
        compaction_digest="compact:ok",
        replay_steps=({"step": "bind-all-closure-digests"},),
        evidence_refs=("raw:closure",),
    )


def test_falsification_gauntlet_passes_only_with_trusted_replayable_survivals() -> None:
    bundle = _falsification_bundle()
    audit = audit_falsification_gauntlet(bundle)

    assert audit.passed is True
    assert verify_falsification_gauntlet(bundle.to_dict()) is True
    assert audit.survived_case_ids == ("counterexample-search", "adversarial-replay")


def test_falsification_gauntlet_fails_closed_on_missing_required_outcome() -> None:
    bundle = _falsification_bundle(
        outcomes=(FalsificationOutcome(case_id="counterexample-search", survived=True, evidence_ref="raw:counterexample-search"),)
    )
    audit = audit_falsification_gauntlet(bundle)

    assert audit.passed is False
    assert "missing_required_falsification_outcome" in audit.failure_reasons
    assert audit.missing_case_ids == ("adversarial-replay",)


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (FalsificationOutcome(case_id="counterexample-search", survived=True, trusted=False, evidence_ref="raw:x"), "untrusted_falsification_outcome"),
        (FalsificationOutcome(case_id="counterexample-search", survived=True, evidence_ref=""), "missing_replayable_falsification_evidence"),
        (FalsificationOutcome(case_id="counterexample-search", status="inconclusive", evidence_ref="raw:x"), "inconclusive_falsification_outcome"),
        (FalsificationOutcome(case_id="counterexample-search", status="falsified", counterexample_ref="raw:cex", evidence_ref="raw:x"), "falsification_counterexample_found"),
    ],
)
def test_falsification_gauntlet_fails_closed_on_bad_outcome(outcome: FalsificationOutcome, reason: str) -> None:
    bundle = _falsification_bundle(
        cases=(FalsificationCase(case_id="counterexample-search"),),
        outcomes=(outcome,),
    )
    audit = audit_falsification_gauntlet(bundle)

    assert audit.passed is False
    assert reason in audit.failure_reasons
    assert verify_falsification_gauntlet(bundle.to_dict()) is False


def test_falsification_bundle_hash_detects_tampered_survival_claim() -> None:
    bundle = _falsification_bundle()
    tampered = bundle.to_dict()
    tampered["outcomes"][0]["status"] = "falsified"
    tampered["outcomes"][0]["survived"] = False
    tampered["outcomes"][0]["counterexample_ref"] = "raw:new-counterexample"

    audit = audit_falsification_gauntlet(tampered)

    assert audit.passed is False
    assert "falsification_bundle_hash_mismatch" in audit.failure_reasons
    assert "falsification_counterexample_found" in audit.failure_reasons


def test_structural_replay_bundle_binds_all_required_closure_digests() -> None:
    bundle = _structural_bundle()
    audit = audit_structural_replay_bundle(bundle)

    assert audit.passed is True
    assert verify_structural_replay_bundle(bundle.to_dict()) is True
    assert set(bundle.digests.to_dict()) == set(REQUIRED_DIGEST_FIELDS)


@pytest.mark.parametrize("digest_field", REQUIRED_DIGEST_FIELDS)
def test_structural_replay_bundle_verification_fails_when_any_digest_is_tampered(digest_field: str) -> None:
    bundle = _structural_bundle()
    tampered = bundle.to_dict()
    tampered["digests"][digest_field] = f"tampered:{digest_field}"

    audit = audit_structural_replay_bundle(tampered)

    assert audit.passed is False
    assert verify_structural_replay_bundle(tampered) is False
    assert "structural_replay_bundle_hash_mismatch" in audit.failure_reasons


def test_structural_replay_bundle_fails_closed_when_required_digest_is_missing() -> None:
    bundle = _structural_bundle()
    tampered = bundle.to_dict()
    tampered["digests"]["calibration_digest"] = ""

    audit = audit_structural_replay_bundle(tampered)

    assert audit.passed is False
    assert "missing_required_closure_digest" in audit.failure_reasons
    assert audit.missing_digest_fields == ("calibration_digest",)


def test_structural_replay_bundle_can_compare_against_external_expected_digests() -> None:
    bundle = _structural_bundle()
    expected = bundle.digests.to_dict() | {"pareto_digest": "pareto:other"}

    audit = audit_structural_replay_bundle(bundle, expected_digests=expected)

    assert audit.passed is False
    assert "closure_digest_mismatch" in audit.failure_reasons
    assert audit.mismatched_digest_fields == ("pareto_digest",)
