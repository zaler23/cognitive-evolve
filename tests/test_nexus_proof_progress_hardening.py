from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Prove a theorem",
        normalized_goal="prove theorem",
        expected_output_forms=["proof", "equation_set"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def test_proof_candidate_without_formal_object_can_rank_and_synthesize_answer() -> None:
    contract = _proof_contract()
    candidate = CandidateGenome(
        id="C-narrative",
        generation=2,
        artifact="Laman rigidity may control dimensional leakage via translation-only constraints.",
        concise_claim="Laman rigidity route",
        core_mechanism="translation-only Laman constraints",
        missing_parts=["explicit coordinate-level algebraic equations"],
        multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.9, "verifiability": 0.0},
    )

    verification = NexusVerifierStack().verify_candidate(candidate, contract=contract)
    ranking = RelativeRater().rank(candidates=[candidate], contract=contract)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], ranking), candidates=[candidate])
    synthesis = synthesize_result(population=CandidatePopulation([candidate]), archives=archives, contract=contract)

    assert verification.passed is True
    assert verification.final_eligible is True
    assert ranking.best_final_answer_id == "C-narrative"
    assert archives.is_final_answer_eligible(candidate) is True
    assert synthesis.answer_produced is True
    assert synthesis.objective_solved is False
    assert synthesis.best_candidate_id == "C-narrative"


def test_duplicate_formal_signature_is_advisory_only() -> None:
    contract = _proof_contract()
    first = CandidateGenome(id="C-a", artifact="eq", formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}], obligation_delta={"discharged": ["obl_a"]})
    second = CandidateGenome(id="C-b", artifact="eq", formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}], obligation_delta={"discharged": ["obl_b"]})

    results = NexusVerifierStack().verify_population([first, second], contract=contract)

    assert results[0].passed is True
    assert results[1].passed is True
    assert "duplicate_formal_signature" in results[1].diagnostics
    assert results[1].final_eligible is True
