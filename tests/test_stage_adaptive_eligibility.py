from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.loop import _attach_policy_directives_to_plans
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Find a high-ceiling mathematical model.",
        normalized_goal="find high-ceiling mathematical model",
        expected_output_forms=["mathematical_model", "algorithmic_hypothesis"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def _math_candidate(candidate_id: str = "C-math") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=2,
        artifact="A bounded optimal-control bandit over theory moves with novelty-weighted value iteration.",
        concise_claim="bounded optimal-control bandit for theory search",
        core_mechanism="treat hypotheses as actions and update a novelty-weighted value function",
        missing_parts=["formal proof", "local verification"],
        multihead_scores={"objective_alignment": 0.75, "answer_likelihood": 0.72, "verifiability": 0.0, "novelty": 0.8},
    )


def test_math_candidate_without_proof_is_final_eligible_answer_material() -> None:
    candidate = _math_candidate()
    result = NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=24, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=24, round_limit=48), candidates=[candidate])

    assert result.passed is True
    assert result.rank_eligible is True
    assert result.final_eligible is True
    assert candidate.metadata["stage_eligibility"]["repair_required"] is False
    assert candidate.current_fate == CandidateFate.ELITE.value
    assert archives.is_final_answer_eligible(candidate) is True


def test_parent_selector_keeps_answer_candidates_even_with_legacy_verifier_notes() -> None:
    candidates = [_math_candidate(f"C{i}") for i in range(3)]
    for candidate in candidates:
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
        candidate.verification_result["diagnostics"].extend(["proof_object_absent", "source_binding_absent"])
        candidate.mark_fate(CandidateFate.ACTIVE.value)

    selected = ParentSelector().select(candidates, ArchiveManager(), limit=2)

    assert len(selected) == 2
    assert all(candidate.current_fate == CandidateFate.ACTIVE.value for candidate in selected)


def test_legacy_repair_directives_are_advisory_not_forced_patch_or_proof() -> None:
    parent = _math_candidate()
    parent.metadata["repair_required"] = {"blockers": ["proof_object_absent"], "evidence_needed": ["formal_artifact"]}
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=[parent.id], instruction="Deepen the route.")

    [updated] = _attach_policy_directives_to_plans([plan], EvolutionPolicy(), parents=[parent])

    assert updated.operator == MutationOperator.DEEPEN
    assert updated.metadata["targeted_repair_lane"] is False
    assert "answer mechanism" in updated.instruction


def test_diagnosis_retires_proof_object_absence_as_policy_driver() -> None:
    raw = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="ProofObjectAbsence",
        recommended_actions=["instantiate_formal_artifact", "discharge_obligation"],
    )

    assert raw.stagnation_type == "DiversityCollapse"
    assert raw.recommended_actions == ["continue"]
    assert raw.metadata["raw_retired_stagnation_type"] == "ProofObjectAbsence"


def test_diagnoser_does_not_convert_missing_proofs_into_proof_object_absence() -> None:
    candidates = [_math_candidate(f"D{i}") for i in range(3)]
    for candidate in candidates:
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
    diagnosis = SearchStateDiagnoser().diagnose(population=candidates, archives=ArchiveManager(), contract=_proof_contract())

    assert diagnosis.stagnation_type != "ProofObjectAbsence"
    assert "instantiate_formal_artifact" not in diagnosis.recommended_actions


def test_boundary_hard_reject_still_blocks_second_runtime_scaffold() -> None:
    candidate = _math_candidate("C-boundary")
    candidate.artifact = "Add a second runtime and new ranking authority."
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=8, round_limit=48), candidates=[candidate])

    assert "second_runtime_or_ranking_authority" in candidate.metadata["stage_eligibility"]["hard_reject_reason"]
    assert archives.is_final_answer_eligible(candidate) is False
