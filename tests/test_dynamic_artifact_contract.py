from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.artifact_contract import (
    DynamicArtifactContract,
    evaluate_candidate_against_dynamic_contract,
    validate_dynamic_artifact_contract,
)
from cognitive_evolve_runtime.nexus.prompt_view import contract_prompt_view
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _contract(**overrides: object) -> NexusObjectiveContract:
    dac = {
        "objective": "Explore a better theory search mechanism",
        "allowed_artifact_shapes": [{"name": "mathematical_model"}],
        "final_gate": {"check": "advisory only"},
    }
    dac.update(overrides)
    return NexusObjectiveContract(
        original_user_goal="Find a high-ceiling mathematical model.",
        normalized_goal="find high-ceiling mathematical model",
        dynamic_artifact_contract=dac,
    )


def test_missing_contract_fields_are_advisory_not_diagnostics() -> None:
    dac = DynamicArtifactContract(objective="Find a theory")

    summary = validate_dynamic_artifact_contract(dac)

    assert summary.required is True
    assert summary.valid is True
    assert summary.diagnostics == []


def test_answer_only_candidate_is_rank_and_final_eligible_without_delta_or_gate() -> None:
    contract = _contract(required_work_product={}, minimum_concrete_delta={}, final_gate={})
    candidate = CandidateGenome(
        id="C-answer",
        artifact="A novelty-weighted control process over theorem-building moves.",
        concise_claim="novelty-weighted control process",
        core_mechanism="state is a hypothesis frontier; actions mutate theory moves; value mixes novelty and compression",
    )

    summary = evaluate_candidate_against_dynamic_contract(candidate, contract=contract)
    result = NexusVerifierStack().verify_candidate(candidate, contract=contract)

    assert summary.required is True
    assert summary.rank_eligible is True
    assert summary.final_eligible is True
    assert summary.diagnostics == []
    assert result.passed is True
    assert result.rank_eligible is True
    assert result.final_eligible is True


def test_design_candidate_can_be_final_answer_material() -> None:
    contract = _contract(allowed_artifact_shapes=[{"name": "design_candidate", "stage": "exploration"}])
    candidate = CandidateGenome(
        id="C-design",
        artifact_type="design_candidate",
        concise_claim="Use a stepping-stone archive for mathematical mechanisms.",
        core_mechanism="rank mechanisms by future option value rather than immediate proofability",
        artifact={"kind": "design_candidate", "mechanism": "stepping-stone archive"},
    )

    summary = evaluate_candidate_against_dynamic_contract(candidate, contract=contract)

    assert summary.rank_eligible is True
    assert summary.final_eligible is True
    assert "design_candidate_non_final" not in summary.diagnostics


def test_prompt_view_exposes_answer_first_default_contract() -> None:
    contract = _contract()
    view = contract_prompt_view(contract)

    assert view["dynamic_artifact_contract"]["objective"] == "Explore a better theory search mechanism"
    assert view["dynamic_artifact_contract"]["final_gate"]["check"] == "advisory only"
    assert "mathematical_model" in str(view["dynamic_artifact_contract"]["allowed_artifact_shapes"])
