from __future__ import annotations

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.loop import _attach_policy_directives_to_plans, _elite_gap_merge_offspring
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Prove the theorem and give explicit coordinate equations.",
        normalized_goal="prove theorem with explicit coordinate equations",
        expected_output_forms=["proof", "equation_set"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def _missing_proof(candidate_id: str = "C-missing", *, created_in_round: int = 0) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=2,
        artifact="A promising route, but the formal equations are not instantiated yet.",
        concise_claim="promising but incomplete proof route",
        core_mechanism="translation-only constraints",
        missing_parts=["concrete formal_artifact", "obligation_delta"],
        multihead_scores={"objective_alignment": 0.75, "answer_likelihood": 0.72, "verifiability": 0.7},
        metadata={"created_in_round": created_in_round},
    )


def _verified(candidate_id: str, *, fate: str = CandidateFate.ACTIVE.value, score: float = 0.8) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=3,
        artifact="verified formal progress",
        concise_claim="verified",
        core_mechanism="verified route",
        current_fate=fate,
        formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}],
        proof_obligations=[{"id": "obl", "status": "discharged"}],
        obligation_delta={"discharged": ["obl"]},
        multihead_scores={"objective_alignment": score, "answer_likelihood": score, "verifiability": score},
        verification_result={"passed": True, "rank_eligible": True, "final_eligible": True, "diagnostics": []},
    )


def _fully_verified(candidate_id: str = "C-verified") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=4,
        artifact="verified formal progress with evidence",
        concise_claim="verified formal route",
        core_mechanism="verified route",
        formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}],
        proof_obligations=[{"id": "obl", "status": "discharged"}],
        obligation_delta={"targeted": ["obl"], "discharged": ["obl"]},
        evidence_refs=[{"id": "formal-check", "kind": "formal_artifact", "status": "verified"}],
        source_bindings=[{"path": "proof.md", "kind": "artifact", "required": True}],
        evidence_delta={"verified": ["formal-check"]},
        multihead_scores={"objective_alignment": 0.78, "answer_likelihood": 0.7, "verifiability": 0.82},
    )


def test_mid_stage_missing_proof_enters_incubating_repair_lane_not_dormant() -> None:
    candidate = _missing_proof(created_in_round=0)
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=24, round_limit=48)

    archives = ArchiveManager()
    assignments = archives.assign_by_policy([candidate], current_round=24, round_limit=48)
    archives.update(assignments, candidates=[candidate])

    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert candidate.id not in archives.dormant_archive.candidates
    assert archives.is_final_answer_eligible(candidate) is False
    assert candidate.metadata["stage_eligibility"]["repair_required"] is True
    assert candidate.metadata["repair_required"]["blockers"]


def test_old_unrepaired_candidate_stays_repair_material_until_final_answer_gate() -> None:
    candidate = _missing_proof(created_in_round=0)
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=48, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=48, round_limit=48), candidates=[candidate])

    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert candidate.id not in archives.dormant_archive.candidates
    assert archives.is_final_answer_eligible(candidate) is False
    assert candidate.metadata["stage_eligibility"]["stage"] == "final"
    assert candidate.metadata["stage_eligibility"]["parent_eligible"] is True
    assert candidate.metadata["stage_eligibility"]["final_eligible"] is False


def test_late_newborn_candidate_uses_model_policy_not_hardcoded_final_boundary() -> None:
    candidate = _missing_proof(created_in_round=45)
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=45, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=45, round_limit=48), candidates=[candidate])

    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert candidate.metadata["stage_eligibility"]["global_stage"] == "middle"
    assert candidate.metadata["stage_eligibility"]["candidate_age_stage"] == "early"
    assert candidate.metadata["stage_eligibility"]["incubating"] is True


def test_model_supplied_stage_boundary_can_tighten_late_pressure() -> None:
    candidate = _missing_proof(created_in_round=45)
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=45, round_limit=48)
    policy = {"stage_fractions": {"early_until": 0.1, "middle_until": 0.5, "late_until": 0.9}}

    archives = ArchiveManager()
    archives.update(
        archives.assign_by_policy([candidate], current_round=45, round_limit=48, eligibility_policy=policy),
        candidates=[candidate],
    )

    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert candidate.metadata["stage_eligibility"]["global_stage"] == "final"
    assert candidate.metadata["stage_eligibility"]["incubating"] is True
    assert candidate.metadata["stage_eligibility"]["final_eligible"] is False


def test_prefinal_contract_gate_failures_create_repair_lane_not_hard_reject() -> None:
    candidate = CandidateGenome(
        id="C-contract-repair",
        generation=1,
        artifact={"type": "design_candidate", "mechanism": "observe candidate vitality before tightening gates"},
        artifact_type="design_candidate",
        concise_claim="final gate missing but exploration material is useful",
        core_mechanism="turn missing final-answer contract fields into repair obligations",
        missing_parts=["final gate", "work product materialization"],
        metadata={
            "dynamic_artifact_contract": {
                "objective": "Explore a self-evolution mechanism",
                "allowed_artifact_shapes": [{"name": "design_candidate"}],
                "invalid_outputs": ["empty output", "meta commentary only", "restating objective without artifact"],
            }
        },
        multihead_scores={"objective_alignment": 0.62, "novelty": 0.5},
    )

    NexusVerifierStack().verify_candidate(candidate, current_round=5, round_limit=48)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=5, round_limit=48), candidates=[candidate])

    decision = candidate.metadata["stage_eligibility"]
    assert "required_work_product_absent" in candidate.verification_result["diagnostics"]
    assert "final_gate_absent" in candidate.verification_result["diagnostics"]
    assert decision["hard_reject_reason"] == ""
    assert decision["repair_required"] is True
    assert decision["parent_eligible"] is True
    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert archives.is_final_answer_eligible(candidate) is False


def test_incubating_parent_selection_is_bounded_repair_lane() -> None:
    active = [_verified(f"A{i}", score=0.7 + i * 0.01) for i in range(4)]
    incubating = [_missing_proof(f"I{i}") for i in range(4)]
    for candidate in incubating:
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=10, round_limit=48)
        candidate.mark_fate(CandidateFate.INCUBATING.value)

    selected = ParentSelector().select([*active, *incubating], ArchiveManager(), limit=4)

    selected_incubating = [candidate for candidate in selected if candidate.current_fate == CandidateFate.INCUBATING.value]
    assert len(selected) == 4
    assert len(selected_incubating) <= 1
    assert any(candidate.current_fate == CandidateFate.ACTIVE.value for candidate in selected)


def test_all_incubating_population_still_has_repair_parents() -> None:
    candidates = [_missing_proof(f"I{i}") for i in range(3)]
    for candidate in candidates:
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
        candidate.mark_fate(CandidateFate.INCUBATING.value)

    selected = ParentSelector().select(candidates, ArchiveManager(), limit=2)

    assert [candidate.current_fate for candidate in selected] == [CandidateFate.INCUBATING.value, CandidateFate.INCUBATING.value]


def test_negative_scored_repairable_candidates_do_not_end_as_no_parents() -> None:
    candidates = [_missing_proof(f"R{i}") for i in range(4)]
    archives = ArchiveManager()
    for candidate in candidates:
        candidate.multihead_scores = {"answer_likelihood": 0.0, "verifiability": 0.0, "objective_alignment": 0.0}
        candidate.failure_lessons.extend(["proof-like objective requires a concrete formal object", "candidate must bind named obligations"])
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
        candidate.mark_fate(CandidateFate.ACTIVE.value)
    archives.update(archives.assign_by_policy(candidates, current_round=8, round_limit=48), candidates=candidates)

    selected = ParentSelector().select(candidates, archives, limit=2)

    assert len(selected) == 2
    assert all(candidate.metadata.get("repair_required") for candidate in selected)


def test_archived_incubating_constraint_does_not_block_repair_parent_selection() -> None:
    candidate = _missing_proof("I-archived")
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=8, round_limit=48), candidates=[candidate])

    selected = ParentSelector().select([candidate], archives, limit=1)

    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert archives.constraint_records
    assert [item.id for item in selected] == [candidate.id]


def test_post_critique_active_floor_prevents_middle_stage_total_collapse() -> None:
    candidates = [_missing_proof(f"C-floor-{i}") for i in range(6)]
    for candidate in candidates:
        NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=24, round_limit=48)

    archives = ArchiveManager()
    assignments = archives.assign_by_policy(candidates, current_round=24, round_limit=48)
    archives.update(assignments, candidates=candidates)

    active = [candidate for candidate in candidates if candidate.current_fate == CandidateFate.ACTIVE.value]
    incubating = [candidate for candidate in candidates if candidate.current_fate == CandidateFate.INCUBATING.value]
    assert len(active) >= 2
    assert incubating
    assert all(candidate.metadata.get("final_answer_blocked_until_repaired") for candidate in active)
    assert all(archives.is_final_answer_eligible(candidate) is False for candidate in active)


def test_incubating_repair_requirement_forces_targeted_mutation_operator() -> None:
    parent = _missing_proof()
    NexusVerifierStack().verify_candidate(parent, contract=_proof_contract(), current_round=8, round_limit=48)
    parent.mark_fate(CandidateFate.INCUBATING.value)
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=[parent.id], instruction="Deepen the route.")

    [updated] = _attach_policy_directives_to_plans([plan], EvolutionPolicy(), parents=[parent])

    assert updated.operator in {MutationOperator.INSTANTIATE_FORMAL_ARTIFACT, MutationOperator.DISCHARGE_OBLIGATION, MutationOperator.TOOL_GROUND, MutationOperator.REPAIR}
    assert updated.metadata["targeted_repair_lane"] is True
    assert updated.metadata["repair_required"]["blockers"]
    assert "Targeted repair lane" in updated.instruction


def test_incubating_candidate_never_final_eligible_even_with_high_scores() -> None:
    candidate = _missing_proof()
    candidate.multihead_scores.update({"objective_alignment": 1.0, "answer_likelihood": 1.0, "verifiability": 1.0})
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)
    candidate.mark_fate(CandidateFate.INCUBATING.value)
    archives = ArchiveManager()
    archives.update([candidate])

    assert archives.is_final_answer_eligible(candidate) is False
    assert candidate.id not in archives.answer_archive


@pytest.mark.parametrize(
    "metadata, artifact, expected_reason",
    [
        ({"unrelated_drift": True}, "unrelated idea", "unrelated_semantic_drift"),
        ({}, "Add a second runtime and new ranking authority.", "second_runtime_or_ranking_authority"),
        ({}, "Use a hidden fallback router when verification fails.", "hidden_fallback"),
        ({}, "Prompt-only gate that says the proof is done.", "prompt_only_gate"),
        ({}, "Docs-only essay with no source or proof object.", "docs_only_essay"),
        ({}, "Seed-note-only artifact pretending to be a fix.", "seed_note_only_artifact"),
        ({"final_claim": True}, "Final answer: solved without evidence.", "source_free_final_claim"),
    ],
)
def test_hard_reject_candidates_do_not_enter_incubating(metadata: dict[str, object], artifact: str, expected_reason: str) -> None:
    candidate = _missing_proof("C-hard")
    candidate.artifact = artifact
    candidate.concise_claim = artifact
    candidate.metadata.update(metadata)
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=8, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=8, round_limit=48), candidates=[candidate])
    selected = ParentSelector().select([candidate], archives, limit=1)

    assert candidate.current_fate != CandidateFate.INCUBATING.value
    assert expected_reason in candidate.metadata["stage_eligibility"]["hard_reject_reason"]
    assert selected == []
    assert archives.is_final_answer_eligible(candidate) is False


def test_middle_stage_candidate_promotes_after_evidence_and_obligation_repair() -> None:
    incomplete = _missing_proof("C-mid")
    NexusVerifierStack().verify_candidate(incomplete, contract=_proof_contract(), current_round=24, round_limit=48)

    assert incomplete.metadata["stage_eligibility"]["stage"] == "middle"
    assert incomplete.metadata["stage_eligibility"]["exploration_eligible"] is True
    assert incomplete.metadata["stage_eligibility"]["final_eligible"] is False
    assert "obligation_delta_absent" in incomplete.metadata["repair_required"]["blockers"]

    repaired = _fully_verified("C-mid-fixed")
    verification = NexusVerifierStack().verify_candidate(repaired, contract=_proof_contract(), current_round=24, round_limit=48)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([repaired], current_round=24, round_limit=48), candidates=[repaired])

    assert verification.passed is True
    assert repaired.verification_result["rank_eligible"] is True
    assert repaired.current_fate in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}
    assert archives.is_final_answer_eligible(repaired) is True


def test_incubating_expiration_demotes_to_dormant_without_tombstone() -> None:
    candidate = _missing_proof(
        "C-expired",
        created_in_round=1,
    )
    candidate.metadata.update({"incubation_started_round": 1, "repair_attempts": 3, "max_incubation_attempts": 3})
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=10, round_limit=48)

    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=10, round_limit=48), candidates=[candidate])

    assert candidate.current_fate == CandidateFate.DORMANT.value
    assert candidate.id in archives.dormant_archive.candidates
    assert candidate.id not in archives.terminal_tombstones
    assert candidate.metadata["stage_eligibility"]["repair_exhausted"] is True
    assert "incubation_budget_exhausted" in candidate.metadata["stage_eligibility"]["state_transition_reason"]


def test_incubating_checkpoint_archive_roundtrip_preserves_contract_fields() -> None:
    candidate = _missing_proof("C-checkpoint")
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=12, round_limit=48)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=12, round_limit=48), candidates=[candidate])

    reloaded_candidate = CandidateGenome.from_dict(candidate.to_dict())
    reloaded_archives = ArchiveManager.from_dict(archives.to_dict())

    assert reloaded_candidate.current_fate == CandidateFate.INCUBATING.value
    assert reloaded_candidate.metadata["repair_required"]["blockers"]
    assert "repair_attempts" in reloaded_candidate.metadata
    assert reloaded_archives.fates[candidate.id] == CandidateFate.INCUBATING.value
    assert candidate.id not in reloaded_archives.terminal_tombstones
    assert candidate.id not in reloaded_archives.dormant_archive.candidates


def test_diagnosis_continues_repair_when_active_zero_but_incubating_exists() -> None:
    candidate = _missing_proof("C-diagnose")
    NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract(), current_round=12, round_limit=48)
    candidate.mark_fate(CandidateFate.INCUBATING.value)

    diagnosis = SearchStateDiagnoser().diagnose(population=[candidate], archives=ArchiveManager(), history=[], contract=_proof_contract(), policy=EvolutionPolicy())

    assert diagnosis.stagnation_type in {"ProofObjectAbsence", "VerificationBottleneck", "ObligationBottleneck"}
    assert "incubating_count=1" in diagnosis.notes or "1 Incubating" in diagnosis.notes
    assert "repair" in diagnosis.recommended_actions or "instantiate_formal_artifact" in diagnosis.recommended_actions


def test_elite_gap_merge_combines_auxiliary_repair_material_without_final_relaxation() -> None:
    elite = _verified("C-elite", fate=CandidateFate.ELITE.value, score=0.9)
    elite.missing_parts = ["scientific notation parser test", "list observation repair"]
    donor = _verified("C-aux", fate=CandidateFate.AUXILIARY.value, score=0.7)
    donor.evidence_delta = {"verified": ["scientific notation parser test"]}
    donor.evidence_refs = [{"id": "sci-notation-test", "kind": "test", "status": "verified"}]
    archives = ArchiveManager()
    archives.update({elite.id: CandidateFate.ELITE.value, donor.id: CandidateFate.AUXILIARY.value}, candidates=[elite, donor])

    children = _elite_gap_merge_offspring([elite, donor], archives=archives, policy=EvolutionPolicy(), branch_factor=2)

    assert children
    child = children[0]
    assert child.current_fate == CandidateFate.ACTIVE.value
    assert child.metadata["elite_gap_merge"]["elite_parent_id"] == elite.id
    assert child.metadata["elite_gap_merge"]["donor_parent_id"] == donor.id
    assert child.metadata["merged_from"] == [elite.id, donor.id]
    assert archives.is_final_answer_eligible(child) is False
