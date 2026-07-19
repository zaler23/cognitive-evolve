from __future__ import annotations

from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.loop import offspring as offspring_module
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


class _ScriptedModel:
    def __init__(self, outcomes: list[list[CandidateGenome] | Exception]) -> None:
        self.outcomes = outcomes

    def generate_offspring(
        self,
        *,
        plans: list[MutationPlan],
        parents: list[CandidateGenome],
        world: Any,
        contract: Any,
        policy: EvolutionPolicy,
        provided_context: dict[str, Any] | None = None,
    ) -> list[CandidateGenome]:
        outcome = self.outcomes[int(policy.metadata.get("offspring_batch_index", 0))]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FailingPlanModel:
    def plan_mutations(self, **_: Any) -> list[MutationPlan]:
        raise LLMResponseError("planner provider failed")


class _EmptyPlanModel:
    def plan_mutations(self, **_: Any) -> list[MutationPlan]:
        return []


class _StaticPlanModel:
    def __init__(self, plans: list[MutationPlan]) -> None:
        self.plans = plans

    def plan_mutations(self, **_: Any) -> list[MutationPlan]:
        return self.plans


def _model_child() -> CandidateGenome:
    return CandidateGenome(id="M", parent_ids=["P"], artifact="model child", concise_claim="model child", core_mechanism="model child", metadata={"plan_id": "plan-0"})


def _run(monkeypatch, outcomes: list[list[CandidateGenome] | Exception]) -> tuple[list[CandidateGenome], list[bool]]:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "2")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "2")
    fallback_calls: list[bool] = []

    def _fallback(**_: Any) -> list[CandidateGenome]:
        fallback_calls.append(True)
        return [CandidateGenome(id="F", artifact="fallback")]

    monkeypatch.setattr(offspring_module, "_deterministic_fallback_offspring", _fallback)
    parent = CandidateGenome(id="P", artifact="parent", concise_claim="parent", core_mechanism="parent")
    plans = [
        MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=[parent.id], instruction=f"deepen {index}", metadata={"plan_id": f"plan-{index}"})
        for index in range(2)
    ]
    generated = offspring_module._generate_offspring(
        model=_ScriptedModel(outcomes),
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=plans,
        world={},
        contract=NexusObjectiveContract(original_user_goal="model child", normalized_goal="model child"),
        policy=EvolutionPolicy(),
    )
    return generated, fallback_calls


def test_model_accepted_is_pure_and_does_not_build_fallback(monkeypatch) -> None:
    offspring, fallback_calls = _run(monkeypatch, [[_model_child()], []])

    assert len(offspring) == 1
    assert offspring[0].id != "M"
    assert offspring[0].metadata["model_claimed_candidate_id"] == "M"
    assert offspring[0].generation == 1
    assert offspring[0].lineage == ["P", offspring[0].id]
    assert fallback_calls == []


def test_empty_model_result_is_abstention_without_deterministic_fallback(monkeypatch) -> None:
    offspring, fallback_calls = _run(monkeypatch, [[], []])

    assert offspring == []
    assert fallback_calls == []


def test_model_boundary_error_raises_without_deterministic_fallback(monkeypatch) -> None:
    with pytest.raises(LLMResponseError, match="provider failed"):
        _run(monkeypatch, [LLMResponseError("provider failed"), []])


def test_partial_model_result_survives_later_fatal_error_without_fallback(monkeypatch) -> None:
    offspring, fallback_calls = _run(monkeypatch, [[_model_child()], LLMResponseError("later batch failed")])

    assert len(offspring) == 1
    assert offspring[0].id != "M"
    assert offspring[0].metadata["model_claimed_candidate_id"] == "M"
    assert fallback_calls == []
    assert offspring[0].metadata["partial_model_offspring_error"] == "LLMResponseError: later batch failed"


def test_model_child_without_artifact_is_rejected(monkeypatch) -> None:
    child = _model_child()
    child.artifact = ""

    with pytest.raises(ModelResponseSchemaError, match="concrete artifact"):
        _run(monkeypatch, [[child], []])


def test_model_child_without_parent_binding_is_rejected(monkeypatch) -> None:
    child = _model_child()
    child.parent_ids = []

    with pytest.raises(ModelResponseSchemaError, match="parent_ids"):
        _run(monkeypatch, [[child], []])


def test_model_plan_boundary_error_does_not_use_deterministic_plans(monkeypatch) -> None:
    planner = MutationPlanner()
    monkeypatch.setattr(planner, "plan_from_actions", lambda *_args, **_kwargs: pytest.fail("configured model must not use deterministic plans"))

    with pytest.raises(LLMResponseError, match="planner provider failed"):
        offspring_module._plan_mutations(
            model=_FailingPlanModel(),
            mutation_planner=planner,
            parents=[CandidateGenome(id="P", artifact="parent")],
            actions=["deepen"],
            archives=ArchiveManager(),
            diagnosis=SearchDiagnosis(),
            policy=EvolutionPolicy(),
        )


def test_empty_model_plan_result_does_not_use_deterministic_plans(monkeypatch) -> None:
    planner = MutationPlanner()
    monkeypatch.setattr(planner, "plan_from_actions", lambda *_args, **_kwargs: pytest.fail("configured model must not use deterministic plans"))

    with pytest.raises(ModelResponseSchemaError, match="no valid mutation plans"):
        offspring_module._plan_mutations(
            model=_EmptyPlanModel(),
            mutation_planner=planner,
            parents=[CandidateGenome(id="P", artifact="parent")],
            actions=["deepen"],
            archives=ArchiveManager(),
            diagnosis=SearchDiagnosis(),
            policy=EvolutionPolicy(),
        )


def test_mixed_available_and_dormant_plan_keeps_available_parent() -> None:
    parent = CandidateGenome(id="P", artifact="parent")
    plan = MutationPlan(operator="ModelDirected", parent_ids=["P", "DORMANT"], instruction="try both")

    accepted = offspring_module._plan_mutations(
        model=_StaticPlanModel([plan]),
        mutation_planner=MutationPlanner(),
        parents=[parent],
        actions=["explore"],
        archives=ArchiveManager(),
        diagnosis=SearchDiagnosis(),
        policy=EvolutionPolicy(),
    )

    assert accepted[0].parent_ids == ["P"]
    assert accepted[0].metadata["dropped_unavailable_parent_ids"] == ["DORMANT"]
    assert accepted[0].metadata["model_claimed_parent_ids"] == ["P", "DORMANT"]


def test_model_plan_cannot_claim_runtime_owned_plan_identity() -> None:
    parent = CandidateGenome(id="P", artifact="parent")
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        instruction="spoof direct path",
        metadata={"plan_id": "model-plan", "plan_source": "runtime_lineage_envelope"},
    )

    accepted = offspring_module._plan_mutations(
        model=_StaticPlanModel([plan]),
        mutation_planner=MutationPlanner(),
        parents=[parent],
        actions=["explore"],
        archives=ArchiveManager(),
        diagnosis=SearchDiagnosis(),
        policy=EvolutionPolicy(),
    )

    assert accepted[0].metadata["plan_id"] != "model-plan"
    assert accepted[0].metadata["model_claimed_plan_id"] == "model-plan"
    assert "plan_source" not in accepted[0].metadata
    assert accepted[0].metadata["model_claimed_plan_source"] == "runtime_lineage_envelope"


def test_plan_with_only_unavailable_parents_is_rejected() -> None:
    plan = MutationPlan(operator="ModelDirected", parent_ids=["DORMANT"], instruction="invalid")

    with pytest.raises(ModelResponseSchemaError, match="no valid mutation plans"):
        offspring_module._plan_mutations(
            model=_StaticPlanModel([plan]),
            mutation_planner=MutationPlanner(),
            parents=[CandidateGenome(id="P", artifact="parent")],
            actions=["explore"],
            archives=ArchiveManager(),
            diagnosis=SearchDiagnosis(),
            policy=EvolutionPolicy(),
        )


def test_same_parent_multi_plan_child_is_accepted_without_fabricated_plan_binding() -> None:
    parent = CandidateGenome(id="P", artifact={"value": 1}, generation=3, lineage=["G0", "P"])
    plans = [
        MutationPlan(operator="Deepen", parent_ids=["P"], metadata={"plan_id": "plan-a"}),
        MutationPlan(operator="Repair", parent_ids=["P"], metadata={"plan_id": "plan-b"}),
    ]
    child = CandidateGenome(id="C", parent_ids=["P"], artifact={"value": 2})

    offspring_module._merge_plan_metadata_into_model_offspring([child], plans, [parent])

    assert child.parent_ids == ["P"]
    assert child.generation == 4
    assert child.lineage == ["G0", "P", "C"]
    assert child.metadata["plan_binding_status"] == "ambiguous_parent_lineage_only"
    assert child.metadata["candidate_plan_ids"] == ["plan-a", "plan-b"]
    assert "plan_id" not in child.metadata


def test_child_lineage_uses_claimed_parent_subset_not_entire_plan() -> None:
    parent_a = CandidateGenome(id="PA", artifact="a", generation=1, lineage=["PA"])
    parent_b = CandidateGenome(id="PB", artifact="b", generation=8, lineage=["PB"])
    plan = MutationPlan(operator="Crossover", parent_ids=["PA", "PB"], metadata={"plan_id": "joint"})
    child = CandidateGenome(id="C", parent_ids=["PA"], artifact="changed", metadata={"plan_id": "joint"})

    offspring_module._merge_plan_metadata_into_model_offspring([child], [plan], [parent_a, parent_b])

    assert child.parent_ids == ["PA"]
    assert child.generation == 2
    assert child.lineage == ["PA", "C"]


def test_unplanned_model_variation_with_real_parent_is_evaluator_eligible() -> None:
    parent = CandidateGenome(id="P", artifact={"value": 1}, lineage=["P"])
    other = CandidateGenome(id="Q", artifact={"value": 9}, lineage=["Q"])
    plans = [
        MutationPlan(operator="Deepen", parent_ids=["Q"], metadata={"plan_id": "q-a"}),
        MutationPlan(operator="Repair", parent_ids=["Q"], metadata={"plan_id": "q-b"}),
    ]
    child = CandidateGenome(id="C", parent_ids=["P"], artifact={"value": 2})

    offspring_module._merge_plan_metadata_into_model_offspring([child], plans, [parent, other])

    assert child.metadata["plan_binding_status"] == "unplanned_model_variation"
    assert child.parent_ids == ["P"]
    assert child.lineage == ["P", "C"]


def test_unplanned_parent_identity_copy_is_rejected() -> None:
    parent = CandidateGenome(id="P", artifact={"value": 1})
    plans = [
        MutationPlan(operator="Deepen", parent_ids=["Q"], metadata={"plan_id": "q-a"}),
        MutationPlan(operator="Repair", parent_ids=["Q"], metadata={"plan_id": "q-b"}),
    ]
    child = CandidateGenome(id="C", parent_ids=["P"], artifact={"value": 1})

    with pytest.raises(ModelResponseSchemaError, match="cannot be resolved"):
        offspring_module._merge_plan_metadata_into_model_offspring([child], plans, [parent])


def test_runtime_plan_identity_overrides_unknown_model_claim() -> None:
    parent = CandidateGenome(id="P", artifact="parent")
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["P"],
        metadata={"plan_id": "runtime-plan", "plan_source": "runtime_lineage_envelope"},
    )
    child = CandidateGenome(
        id="C",
        parent_ids=["P"],
        artifact="changed",
        metadata={"plan_id": "model-invented", "plan_source": "model-invented-source"},
    )

    offspring_module._merge_plan_metadata_into_model_offspring([child], [plan], [parent])

    assert child.metadata["plan_id"] == "runtime-plan"
    assert child.metadata["plan_source"] == "runtime_lineage_envelope"
    assert child.metadata["model_claimed_plan_id"] == "model-invented"
    assert child.metadata["model_claimed_plan_source"] == "model-invented-source"


def test_evaluator_led_offspring_rejects_incremental_patch_without_complete_artifact() -> None:
    parent = CandidateGenome(
        id="P",
        artifact={
            "patch_set": [
                {
                    "path": "heuristic.py",
                    "operation": "write",
                    "content": "def heuristic():\n    return []\n",
                }
            ]
        },
    )
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["P"],
        metadata={
            "plan_id": "runtime-plan",
            "plan_source": "runtime_lineage_envelope",
            "completion_mode": "complete_task_artifact_only",
        },
    )
    child = CandidateGenome(
        id="C",
        parent_ids=["P"],
        artifact={
            "unified_diff": "--- a/heuristic.py\n+++ b/heuristic.py\n@@ -1,2 +1,2 @@\n-def heuristic():\n+def heuristic(x):\n",
        },
        artifact_type="project_patch",
    )

    with pytest.raises(ModelResponseSchemaError, match="complete evaluator-visible artifact"):
        offspring_module._merge_plan_metadata_into_model_offspring([child], [plan], [parent])

    plan.metadata["completion_mode"] = "concrete_progress_allowed"
    offspring_module._merge_plan_metadata_into_model_offspring([child], [plan], [parent])
    assert child.parent_ids == ["P"]
