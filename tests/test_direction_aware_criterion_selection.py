from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContractBuilder
from cognitive_evolve_runtime.evaluators.evidence import evaluator_selection_key, select_preliminary_incumbent
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.adaptive.config import AdaptiveConfig
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import productive_outcomes
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.stop_reasons import (
    CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
    DIMINISHING_RETURNS_CHECKPOINT,
    GOAL_REACHED,
    STAGNATION_EXHAUSTED,
    stop_reason_class,
)
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def _measured(candidate_id: str, value: float, *, direction: str, parent_id: str = "") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        parent_ids=[parent_id] if parent_id else [],
        generation=1 if parent_id else 0,
        lineage=[parent_id, candidate_id] if parent_id else [candidate_id],
        artifact={"candidate": candidate_id},
        concise_claim=candidate_id,
        core_mechanism=candidate_id,
        metadata={
            "evaluator": {
                "status": "measured",
                "metrics": {"loss": value},
                "selection_binding": {
                    "id": "criterion:loss",
                    "kind": "criterion_metric",
                    "metric": "loss",
                    "direction": direction,
                    "value_type": "number",
                    "source_span": {"start": 0, "end": 4, "spec_sha256": "fixture"},
                },
            }
        },
    )


class _CaptureStopModel:
    def __init__(self) -> None:
        self.best_answer_id = ""

    def should_stop(self, **kwargs: Any) -> dict[str, Any]:
        self.best_answer_id = str(kwargs["best_answer_id"])
        return {"stop": False, "solved": False, "reason": "continue"}


@pytest.mark.parametrize(
    ("direction", "expected_id", "baseline_value", "child_value"),
    [
        ("minimize", "low", 0.9, 0.1),
        ("maximize", "high", 0.1, 0.9),
    ],
)
def test_one_direction_binding_drives_incumbent_parent_pba_and_stop(
    direction: str,
    expected_id: str,
    baseline_value: float,
    child_value: float,
) -> None:
    low = _measured("low", 0.1, direction=direction)
    high = _measured("high", 0.9, direction=direction)

    assert select_preliminary_incumbent([high, low]).id == expected_id
    assert ParentSelector().select([high, low], limit=1)[0].id == expected_id

    baseline = _measured("baseline", baseline_value, direction=direction)
    child = _measured("child", child_value, direction=direction, parent_id="root")
    [outcome] = productive_outcomes([baseline, child])
    assert outcome.reward == 1.0
    assert outcome.reason_codes == ("same_cell_evaluator_elite_improvement",)

    model = _CaptureStopModel()
    assert StopDecisionEngine().stop_reason_after_round(
        budget=EvolutionBudget(max_rounds=3, stop_policy="llm_after_minimum"),
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="wrong-ranking-id",
        population=CandidatePopulation([high, low]),
        model=model,
    ) == ""
    assert model.best_answer_id == expected_id


def test_minimize_loss_key_selects_point_one_everywhere() -> None:
    low = _measured("loss-0.1", 0.1, direction="minimize")
    high = _measured("loss-0.9", 0.9, direction="minimize")

    assert evaluator_selection_key(low) > evaluator_selection_key(high)
    assert select_preliminary_incumbent([high, low]) is low
    assert ParentSelector().select([high, low], limit=2)[0] is low


def test_legacy_maximize_score_regression_is_unchanged() -> None:
    low = CandidateGenome(id="low", metadata={"evaluator": {"status": "passed", "passed": True, "metrics": {"score": 0.1}}})
    high = CandidateGenome(id="high", metadata={"evaluator": {"status": "passed", "passed": True, "metrics": {"score": 0.9}}})

    assert select_preliminary_incumbent([low, high]) is high


def test_complete_criterion_compiles_to_evaluator_binding_bound_to_frozen_input() -> None:
    problem = "Minimize loss."

    class Model:
        def build_objective_contract(self, **_: Any) -> dict[str, Any]:
            return {
                "original_user_goal": problem,
                "normalized_goal": problem,
                "criteria": [
                    {
                        "metric": "loss",
                        "direction": "minimize",
                        "value_type": "number",
                        "source_span": {"start": 0, "end": len(problem)},
                    }
                ],
            }

    contract = NexusObjectiveContractBuilder().build_text_contract(
        user_goal=problem,
        packet=SimpleNamespace(raw_text=problem, constraints=[]),
        model=Model(),
    )

    [binding] = contract.evaluators
    assert binding.metric == "loss"
    assert binding.direction == "minimize"
    assert binding.value_type == "number"
    assert binding.source_span == {
        "start": 0,
        "end": len(problem),
        "spec_sha256": contract.frozen_spec["spec_sha256"],
    }
    assert AdaptiveConfig.from_sources(contract=contract).evaluator["metrics"] == [
        {
            "name": "loss",
            "direction": "minimize",
            "value_type": "number",
            "source_span": binding.source_span,
        }
    ]


def test_missing_criterion_direction_is_rejected_and_audited() -> None:
    problem = "Minimize loss."

    class Model:
        def build_objective_contract(self, **_: Any) -> dict[str, Any]:
            return {
                "original_user_goal": problem,
                "normalized_goal": problem,
                "criteria": [
                    {
                        "metric": "loss",
                        "value_type": "number",
                        "source_span": {"start": 0, "end": len(problem)},
                    }
                ],
            }

    contract = NexusObjectiveContractBuilder().build_text_contract(
        user_goal=problem,
        packet=SimpleNamespace(raw_text=problem, constraints=[]),
        model=Model(),
    )

    assert contract.evaluators == []
    [audit] = contract.metadata["criterion_binding_audit"]
    assert audit["status"] == "rejected"
    assert audit["reason"] == "criterion_direction_required"
    assert audit["frozen_input_sha256"] == contract.frozen_spec["spec_sha256"]
    state = AdaptiveRuntimeController.from_sources(contract=contract).to_dict()
    assert any(
        event.get("type") == "criterion_binding_rejected"
        and event.get("reason") == "criterion_direction_required"
        for event in state["events"]
    )


def test_goal_and_stagnation_stop_reasons_remain_distinct_without_self_finality() -> None:
    assert CANDIDATE_READY_FOR_EXTERNAL_REVIEW != DIMINISHING_RETURNS_CHECKPOINT
    assert stop_reason_class(CANDIDATE_READY_FOR_EXTERNAL_REVIEW) == GOAL_REACHED
    assert stop_reason_class(DIMINISHING_RETURNS_CHECKPOINT) == STAGNATION_EXHAUSTED
    assert stop_reason_class("max_rounds") == STAGNATION_EXHAUSTED


def test_budget_exhaustion_reason_is_written_to_event_and_decision(tmp_path: Path) -> None:
    result = NexusRuntime(output_dir=tmp_path).run_text(
        "Return a concise answer.",
        max_rounds=1,
        stop_policy="max_rounds",
    )

    [history] = result.evolution["budget_history"]
    decision = history["progress_event"]["metadata"]["stop_decision"]
    assert history["stop_reason"] == "max_rounds"
    assert decision == {
        "stop": True,
        "reason": "max_rounds",
        "reason_class": STAGNATION_EXHAUSTED,
        "best_candidate_id": history["ranking"]["best_final_answer_id"],
    }
