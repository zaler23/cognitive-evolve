from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.obligations import candidate_formal_artifacts, looks_like_formal_artifact
from cognitive_evolve_runtime.nexus.stage_policy import parse_metric_value, stage_eligibility
from cognitive_evolve_runtime.ranking.relative_rater import _coerce_score, _verification_score
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Improve runtime correctness with proof-style verification evidence.",
        normalized_goal="prove runtime metric parsing is robust with executable assertions",
        expected_output_forms=["proof", "assertion_set"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def test_parse_metric_value_handles_scientific_notation_without_overflow() -> None:
    assert parse_metric_value(" 1.5E-05 ") == 1.5e-05
    assert parse_metric_value("1e-300") == 1e-300
    assert parse_metric_value("1e309") == 1.0
    assert parse_metric_value("-1e309") == 0.0
    assert parse_metric_value("1.5e-05-2") is None
    assert parse_metric_value("nan") is None
    assert parse_metric_value({"score": "2.0"}) == 1.0
    assert parse_metric_value(["0.2", "4e-1"]) == 0.30000000000000004


def test_relative_rater_uses_robust_metric_parser_for_nested_scores() -> None:
    candidate = CandidateGenome(
        id="C-score",
        multihead_scores={"answer_likelihood": "9.1e-1"},
        verification_result={
            "passed": True,
            "proof_progress": {"score": "8e-1"},
            "evidence_obligation": {"score": "1e309"},
        },
    )

    assert _coerce_score("9.1e-1") == 0.91
    assert _coerce_score("1.5e-05-2") is None
    assert _verification_score(candidate) == 0.9
    assert stage_eligibility(candidate).stage in {"early", "middle", "late", "final"}


def test_assertion_set_formal_artifact_is_structurally_checkable() -> None:
    artifact = {
        "kind": "assertion_set",
        "target_obligation_id": "obl_metric_parser",
        "assertions": [
            "assert parse_metric_value('1e-300') == 1e-300",
            "assert parse_metric_value('1.5e-05-2') is None",
        ],
    }

    assert looks_like_formal_artifact(artifact) is True

    candidate = CandidateGenome(
        id="C-assertions",
        artifact="Runtime metric parser proof witness.",
        concise_claim="Scientific notation parsing is protected by executable assertions.",
        core_mechanism="Use Decimal-backed parse_metric_value and bind it to assertion_set formal_artifacts.",
        formal_artifacts=[artifact],
        proof_obligations=[{"id": "obl_metric_parser", "status": "discharged", "description": "verify robust metric parser"}],
        obligation_delta={"targeted": ["obl_metric_parser"], "discharged": ["obl_metric_parser"]},
        multihead_scores={"objective_alignment": "9e-1", "answer_likelihood": "8e-1", "verifiability": "8e-1"},
    )

    verification = NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract())

    assert candidate_formal_artifacts(candidate)[0]["kind"] == "assertion_set"
    assert verification.passed is True
    assert "proof_object_structurally_weak" not in verification.diagnostics
