from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan


def test_structured_artifact_fallback_deep_copies_without_stringifying() -> None:
    artifact = {"answer": {"steps": ["observe", "intervene"]}}
    parent = CandidateGenome(id="P", artifact=artifact, artifact_type="causal_plan")

    child = MutationEngine().mutate(parent, MutationPlan(operator=MutationOperator.CASE_SPLIT))

    assert child.artifact == artifact
    assert child.artifact is not artifact
    assert child.artifact["answer"] is not artifact["answer"]
    assert child.artifact_type == "causal_plan"


def test_explicit_candidate_transform_still_precedes_structured_fallback() -> None:
    parent = CandidateGenome(id="P", artifact={"params": {"width": 1}})
    plan = MutationPlan(
        operator=MutationOperator.CASE_SPLIT,
        metadata={
            "candidate_transforms": [
                {
                    "kind": "collapse_params",
                    "payload": {
                        "assignment": {"width": 2},
                        "parameter_slots": {"width": {"path": "params.width"}},
                    },
                }
            ]
        },
    )

    child = MutationEngine().mutate(parent, plan)

    assert child.artifact == {"params": {"width": 2}}


def test_non_applicable_transform_cannot_force_structured_artifact_into_prose() -> None:
    artifact = {"params": {"width": 1}}
    parent = CandidateGenome(id="P", artifact=artifact, artifact_type="machine")
    plan = MutationPlan(
        operator=MutationOperator.CASE_SPLIT,
        metadata={"candidate_transforms": [{"kind": "unknown_transform"}]},
    )

    child = MutationEngine().mutate(parent, plan)

    assert child.artifact == artifact
    assert child.artifact is not artifact


def test_string_case_split_keeps_existing_mutation_behavior() -> None:
    parent = CandidateGenome(id="P", artifact="solve from the causal graph")

    child = MutationEngine().mutate(parent, MutationPlan(operator=MutationOperator.CASE_SPLIT, instruction="split on sensor state"))

    assert child.artifact != parent.artifact
    assert child.artifact.startswith("CaseSplit mutation of P:")
    assert "solve from the causal graph" in child.artifact
    assert "split on sensor state" in child.artifact
