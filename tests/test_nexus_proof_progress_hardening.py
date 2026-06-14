from __future__ import annotations

from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, evolve_once
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Prove the theorem and give explicit coordinate equations.",
        normalized_goal="prove theorem with explicit coordinate equations",
        expected_output_forms=["proof", "equation_set"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def test_candidate_genome_preserves_proof_progress_fields() -> None:
    candidate = CandidateGenome(
        id="C-proof",
        formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}],
        proof_obligations=[{"id": "obl_equations", "status": "introduced", "description": "derive equations"}],
        obligation_delta={"introduced": ["obl_equations"]},
        verification_result={"passed": True, "proof_progress": {"score": 0.7}},
    )

    decoded = CandidateGenome.from_json(candidate.to_json())

    assert decoded.formal_artifacts == candidate.formal_artifacts
    assert decoded.proof_obligations == candidate.proof_obligations
    assert decoded.obligation_delta == candidate.obligation_delta
    assert decoded.verification_result["proof_progress"]["score"] == 0.7


def test_proof_candidate_without_formal_object_cannot_rank_or_synthesize() -> None:
    contract = _proof_contract()
    candidate = CandidateGenome(
        id="C-narrative",
        generation=2,
        artifact="Laman rigidity might control dimensional leakage, but equations are still missing.",
        concise_claim="Laman rigidity route",
        core_mechanism="translation-only Laman constraints",
        missing_parts=["explicit coordinate-level algebraic equations"],
        multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.9, "verifiability": 0.9},
    )

    verification = NexusVerifierStack().verify_candidate(candidate, contract=contract)
    ranking = RelativeRater().rank(candidates=[candidate], contract=contract)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], ranking), candidates=[candidate])
    synthesis = synthesize_result(population=CandidatePopulation([candidate]), archives=archives, contract=contract)

    assert verification.passed is False
    assert "proof_object_absent" in verification.diagnostics
    assert candidate.verification_result["rank_eligible"] is False
    assert ranking.best_final_answer_id == ""
    assert candidate.current_fate == CandidateFate.DORMANT
    assert archives.is_final_answer_eligible(candidate) is False
    assert synthesis.status == "route_incomplete"


def test_concrete_formal_object_with_obligation_delta_passes_proof_gate() -> None:
    contract = _proof_contract()
    candidate = CandidateGenome(
        id="C-equations",
        generation=2,
        artifact="Coordinate equations instantiated.",
        concise_claim="explicit equation set",
        core_mechanism="coordinate equation discharge",
        formal_artifacts=[
            {
                "kind": "equation_set",
                "target_obligation_id": "obl_equations",
                "equations": ["x_i - x_j = a_ij", "det(J_F(x)) != 0"],
            }
        ],
        proof_obligations=[{"id": "obl_equations", "status": "discharged", "description": "derive coordinate equations"}],
        obligation_delta={"targeted": ["obl_equations"], "discharged": ["obl_equations"]},
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7, "verifiability": 0.7},
    )

    verification = NexusVerifierStack().verify_candidate(candidate, contract=contract)
    ranking = RelativeRater().rank(candidates=[candidate], contract=contract)

    assert verification.passed is True
    assert verification.proof_progress["score"] > 0.7
    assert ranking.best_final_answer_id == "C-equations"


def test_duplicate_formal_signature_is_rejected() -> None:
    contract = _proof_contract()
    first = CandidateGenome(
        id="C-a",
        formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}],
        obligation_delta={"discharged": ["obl_a"]},
    )
    second = CandidateGenome(
        id="C-b",
        formal_artifacts=[{"kind": "equation_set", "equations": ["x_i - x_j = a_ij"]}],
        obligation_delta={"discharged": ["obl_b"]},
    )

    results = NexusVerifierStack().verify_population([first, second], contract=contract)

    assert results[0].passed is True
    assert results[1].passed is False
    assert "duplicate_formal_signature" in results[1].diagnostics


class QuotaOnRankModel:
    def __init__(self) -> None:
        self.rank_calls = 0

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        self.rank_calls += 1
        raise LLMResponseError("RESOURCE_EXHAUSTED 429 provider quota exhausted")


def test_quota_error_pauses_without_deterministic_continuation(tmp_path: Path) -> None:
    contract = NexusObjectiveContract(original_user_goal="Answer plainly.", normalized_goal="answer plainly")
    population = CandidatePopulation([CandidateGenome(id="C0", artifact="candidate", concise_claim="candidate", core_mechanism="direct")])
    model = QuotaOnRankModel()

    result = evolve_once(
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=contract,
        world={},
        budget=EvolutionBudget(max_rounds=3, recover_model_errors=True),
        model=model,
    )

    assert model.rank_calls == 1
    assert result.interrupted is True
    assert result.stop_reason == "model_quota_pause_checkpointed"
    assert result.completion_status == "paused_quota"
    assert result.synthesis.status == "paused_quota"
