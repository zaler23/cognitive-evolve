from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop import _attach_policy_directives_to_plans
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Prove the theorem with explicit equations.",
        normalized_goal="prove theorem with explicit equations",
        expected_output_forms=["proof", "equation_set"],
        verification_preferences=["formal_artifact", "obligation_delta"],
    )


def test_verifier_writes_failure_micro_guidance() -> None:
    candidate = CandidateGenome(
        id="needs-proof",
        generation=2,
        artifact="Narrative route without equations.",
        core_mechanism="narrative proof route",
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.75},
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract())

    assert result.passed is False
    guidance = candidate.metadata["failure_micro_guidance"]
    blockers = {item["blocker"] for item in guidance}
    assert "proof_object_absent" in blockers
    assert any("formal_artifact" in item["evidence_needed"] for item in guidance)


def test_failure_micro_guidance_becomes_compact_mutation_repair_directive() -> None:
    parent = CandidateGenome(
        id="needs-proof",
        generation=2,
        artifact="Narrative route without equations.",
        core_mechanism="narrative proof route",
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.75},
    )
    NexusVerifierStack().verify_candidate(parent, contract=_proof_contract())
    plan = MutationPlan(operator=MutationOperator.REPAIR, parent_ids=[parent.id], instruction="Repair the parent.")

    [updated] = _attach_policy_directives_to_plans([plan], EvolutionPolicy(), parents=[parent])

    assert "repair_directives" in updated.metadata
    assert updated.metadata["repair_directives"][0]["blocker"] == "proof_object_absent"
    assert "Repair directive: fix proof_object_absent" in updated.instruction
    assert len(updated.instruction) < 600

