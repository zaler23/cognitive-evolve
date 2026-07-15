from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlan
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.telemetry import build_round_cost_ledger
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.controller import _apply_gain_token_control, _gain_token_control
from cognitive_evolve_runtime.nexus.loop.evaluate_stage import _pre_rank_admission_view
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _offspring_transport_decision
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.tools.feedback import ToolFeedback


class _World:
    kind = "text"


class _Contract:
    objective = "transport admission"

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective}


def _candidate(
    candidate_id: str,
    *,
    axis: str = "direct_mainstream",
    family: str = "family-a",
    uncertainty: int = 0,
) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact={"score": int(candidate_id.rsplit("-", 1)[-1]) if candidate_id.rsplit("-", 1)[-1].isdigit() else 1},
        artifact_type="machine",
        concise_claim=f"claim-{candidate_id}",
        core_mechanism=family,
        uncertainty_notes=[f"uncertainty-{index}" for index in range(uncertainty)],
        multihead_scores={"answer_likelihood": 0.5, "objective_alignment": 0.5},
        metadata={"search_space": {"seed_axis": axis, "family_id": family}},
    )


def _slots(count: int = 2, *, intent: str = "explore_fresh") -> list[dict[str, Any]]:
    return [
        {
            "slot_id": f"slot-{index}",
            "arm_id": f"P{index}",
            "parent_id": f"P{index}",
            "intent": intent,
            "variation_index": index,
        }
        for index in range(count)
    ]


def _gate_history(
    *,
    completion_tokens: int = 200,
    valid_children: int = 2,
    physical_calls: int = 4,
    truncations: int = 0,
) -> list[dict[str, Any]]:
    return [
        {
            "round": 1,
            "cost_ledger": {
                "round": 1,
                "totals": {
                    "physical_calls": physical_calls,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens * 2,
                    "estimated_cost_usd": 0.01,
                },
                "observations": {
                    "unique_valid_children": valid_children,
                    "truncation_count": truncations,
                },
            },
        }
    ]


def _gate_policy(
    *,
    profiles: list[dict[str, Any]] | None = None,
    output_limit: int = 1_000,
    truncation_threshold: float = 0.05,
) -> EvolutionPolicy:
    return EvolutionPolicy(
        metadata={
            "offspring_parallel_mode": "single_batch",
            "search_phase": "explore",
            "slot_sampling_profiles": {
                "explore": {"explore_fresh": profiles or [{"temperature": 0.2, "top_p": 0.8, "seed": 7}]},
                "exit_sweep": {"explore_fresh": [{"temperature": 0.1, "top_p": 0.5, "seed": 8}]},
            },
            "single_batch_output_token_limit": output_limit,
            "single_batch_truncation_rate_threshold": truncation_threshold,
        }
    )


@pytest.mark.parametrize(
    ("policy", "history", "failed_gate"),
    [
        (
            _gate_policy(
                profiles=[
                    {"temperature": 0.2, "top_p": 0.8, "seed": 7},
                    {"temperature": 0.7, "top_p": 0.9, "seed": 9},
                ]
            ),
            _gate_history(),
            "sampling_homogeneous",
        ),
        (_gate_policy(output_limit=150), _gate_history(completion_tokens=200, valid_children=2), "batch_output_budget"),
        (_gate_policy(truncation_threshold=0.10), _gate_history(physical_calls=4, truncations=1), "historical_truncation_rate"),
    ],
)
def test_single_batch_gate_rejects_each_failed_condition(
    policy: EvolutionPolicy,
    history: list[dict[str, Any]],
    failed_gate: str,
) -> None:
    decision = _offspring_transport_decision(
        policy=policy,
        branch_slots=_slots(),
        budget_history=history,
        target_size=2,
    )

    assert decision["selected_mode"] == "slot"
    assert decision["gates"][failed_gate]["passed"] is False
    assert decision["reason"] == f"gate_failed:{failed_gate}"


def test_single_batch_gate_selects_batch_only_when_all_conditions_pass_and_audits_inputs() -> None:
    decision = _offspring_transport_decision(
        policy=_gate_policy(output_limit=500, truncation_threshold=0.1),
        branch_slots=_slots(),
        budget_history=_gate_history(completion_tokens=200, valid_children=2, physical_calls=5),
        target_size=2,
    )

    assert decision["schema"] == "cogev.offspring_transport_gate.v1"
    assert decision["requested_mode"] == "single_batch"
    assert decision["selected_mode"] == "single_batch"
    assert all(item["passed"] for item in decision["gates"].values())
    assert decision["gates"]["batch_output_budget"] == {
        "passed": True,
        "history_available": True,
        "history_completion_tokens": 200,
        "history_valid_children": 2,
        "observed_tokens_per_valid_child": 100.0,
        "required_tokens": 200,
        "output_token_limit": 500,
        "estimate_source": "round_cost_ledger",
    }
    assert decision["gates"]["historical_truncation_rate"]["observed_rate"] == 0.0


def test_single_batch_gate_fails_closed_without_cost_or_truncation_history() -> None:
    decision = _offspring_transport_decision(
        policy=_gate_policy(),
        branch_slots=_slots(),
        budget_history=[],
        target_size=2,
    )

    assert decision["selected_mode"] == "slot"
    assert decision["gates"]["batch_output_budget"]["history_available"] is False
    assert decision["gates"]["batch_output_budget"]["passed"] is False
    assert decision["gates"]["historical_truncation_rate"]["history_available"] is False
    assert decision["gates"]["historical_truncation_rate"]["passed"] is False


class _BatchThenSlotModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_offspring(
        self,
        *,
        plans: list[MutationPlan],
        parents: list[CandidateGenome],
        world: Any,
        contract: Any,
        policy: EvolutionPolicy,
    ) -> list[dict[str, Any]]:
        branch_slots = list(plans[0].metadata.get("branch_slots") or [])
        if len(branch_slots) > 1:
            self.calls.append("batch")
            raise LLMResponseError("batch transport failed")
        slot_id = str(branch_slots[0]["slot_id"])
        self.calls.append(slot_id)
        parent = parents[0]
        return [
            {
                "id": f"child-{slot_id}",
                "parent_ids": [parent.id],
                "artifact": {"slot": slot_id},
                "artifact_type": "machine",
                "concise_claim": slot_id,
                "core_mechanism": f"mechanism-{slot_id}",
            }
        ]


def test_batch_failure_falls_back_to_one_slot_pass_and_records_blast_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    parents = [_candidate("P0"), _candidate("P1")]
    slots = _slots()
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id for parent in parents],
        metadata={"plan_id": "runtime-plan", "plan_source": "runtime_lineage_envelope", "branch_slots": slots},
    )
    model = _BatchThenSlotModel()
    audit: dict[str, Any] = {}

    offspring = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=parents,
        plans=[plan],
        world=_World(),
        contract=_Contract(),
        policy=_gate_policy(),
        target_size=2,
        budget_history=_gate_history(),
        harvest_outcome=audit,
    )

    assert model.calls == ["batch", "slot-0", "slot-1"]
    assert [candidate.metadata["branch_slot_id"] for candidate in offspring] == ["slot-0", "slot-1"]
    assert audit["transport"]["fallback"] == {
        "attempted": True,
        "passes": 1,
        "from": "single_batch",
        "to": "slot",
        "status": "succeeded",
        "error_type": "LLMResponseError",
        "error": "batch transport failed",
    }
    assert audit["partial_failure_blast_radius"] == 2


def test_pre_rank_admission_defers_without_dropping_and_rotates_deferred_candidates_back_in() -> None:
    candidates = [_candidate(f"C-{index}", uncertainty=int(index == 0)) for index in range(4)]
    admitted, first_audit = _pre_rank_admission_view(
        candidates,
        limit=2,
        history=[],
        uncertainty_floor=1,
    )

    assert len(admitted) == 2
    assert sorted(first_audit["admitted_candidate_ids"] + first_audit["deferred_candidate_ids"]) == sorted(
        candidate.id for candidate in candidates
    )
    assert len(first_audit["deferred_candidate_ids"]) == 2

    history = [{"round": 1, "generation_plan": {"pre_rank_admission": first_audit}}]
    readmitted, second_audit = _pre_rank_admission_view(
        candidates,
        limit=2,
        history=history,
        uncertainty_floor=1,
    )

    assert set(first_audit["deferred_candidate_ids"]) & {candidate.id for candidate in readmitted}
    assert sorted(second_audit["admitted_candidate_ids"] + second_audit["deferred_candidate_ids"]) == sorted(
        candidate.id for candidate in candidates
    )


def test_pre_rank_admission_preserves_axis_family_and_uncertainty_floors() -> None:
    candidates = [
        _candidate("axis-a", axis="axis-a", family="family-a"),
        _candidate("axis-b", axis="axis-b", family="family-a"),
        _candidate("family-b", axis="axis-a", family="family-b"),
        _candidate("uncertain", axis="axis-a", family="family-a", uncertainty=3),
    ]

    admitted, audit = _pre_rank_admission_view(candidates, limit=1, history=[], uncertainty_floor=1)

    admitted_ids = {candidate.id for candidate in admitted}
    assert set(audit["floor_candidate_ids"]).issubset(admitted_ids)
    assert set(audit["axis_floor_candidate_ids"]).issubset(admitted_ids)
    assert set(audit["family_floor_candidate_ids"]).issubset(admitted_ids)
    assert set(audit["uncertainty_floor_candidate_ids"]).issubset(admitted_ids)
    assert "uncertain" in admitted_ids


def test_tighter_admission_budget_increases_deferred_count() -> None:
    candidates = [_candidate(f"C-{index}") for index in range(6)]
    _, loose = _pre_rank_admission_view(candidates, limit=5, history=[], uncertainty_floor=1)
    _, tight = _pre_rank_admission_view(candidates, limit=2, history=[], uncertainty_floor=1)

    assert len(tight["deferred_candidate_ids"]) > len(loose["deferred_candidate_ids"])


class _DeterministicRunner:
    def run(self, command: list[str], **_: Any) -> ToolFeedback:
        payload = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        score = float(payload["artifact"]["score"])
        return ToolFeedback(
            tool_id="deterministic",
            status="passed",
            raw_output_ref=json.dumps({"passed": score >= 2, "metrics": {"score": score}}),
            cost={"seconds": 0.0, "returncode": 0},
            confidence=1.0,
        )


def test_admission_changes_only_evaluation_round_not_eventual_evaluator_verdict() -> None:
    original = [_candidate(f"C-{index}") for index in range(4)]
    admitted_population = [copy.deepcopy(candidate) for candidate in original]
    baseline_population = [copy.deepcopy(candidate) for candidate in original]
    spec = EvaluatorSpec.from_mapping({"enabled": True, "command": "python evaluator.py", "metrics": [{"name": "score", "direction": "maximize"}]})
    baseline_runner = ExternalEvaluatorRunner(runner=_DeterministicRunner())
    admission_runner = ExternalEvaluatorRunner(runner=_DeterministicRunner())

    baseline_runner.evaluate_population_if_configured(baseline_population, spec=spec, round_index=1)
    history: list[dict[str, Any]] = []
    for round_index in range(1, 4):
        admitted, audit = _pre_rank_admission_view(
            admitted_population,
            limit=2,
            history=history,
            uncertainty_floor=1,
        )
        admission_runner.evaluate_population_if_configured(admitted, spec=spec, round_index=round_index)
        history.append({"round": round_index, "generation_plan": {"pre_rank_admission": audit}})
        if all(isinstance(candidate.metadata.get("evaluator"), dict) for candidate in admitted_population):
            break

    assert all(isinstance(candidate.metadata.get("evaluator"), dict) for candidate in admitted_population)
    assert {
        candidate.id: (candidate.metadata["evaluator"]["passed"], candidate.metadata["evaluator"]["metrics"])
        for candidate in admitted_population
    } == {
        candidate.id: (candidate.metadata["evaluator"]["passed"], candidate.metadata["evaluator"]["metrics"])
        for candidate in baseline_population
    }


@pytest.mark.parametrize(
    ("gain", "tokens", "expected_action", "expected_width_delta", "expected_transport"),
    [
        (0.01, 10_000, "contract", -1, "single_batch"),
        (1.0, 1_000, "expand", 1, "slot"),
    ],
)
def test_gain_token_controller_adjusts_only_next_round_resources(
    gain: float,
    tokens: int,
    expected_action: str,
    expected_width_delta: int,
    expected_transport: str,
) -> None:
    history = [{"round": 2, "direction_aware_gain": {"total_gain": gain, "sample_count": 2}}]
    ledger = {
        "schema_version": "round-cost-ledger/v1",
        "rounds": [
            {
                "round": 2,
                "totals": {"total_tokens": tokens, "estimated_cost_usd": 0.25},
            }
        ],
    }
    verdict = {"status": "passed", "passed": True, "metrics": {"score": 9.0}}
    policy = EvolutionPolicy(metadata={"evaluator_verdict_fixture": copy.deepcopy(verdict)})
    budget = EvolutionBudget(max_rounds=4, branch_factor=4)

    decision = _gain_token_control(
        history=history,
        cost_ledger=ledger,
        current_width=budget.branch_factor,
        current_transport="slot",
        current_retry_limit=3,
        config={"low_gain_per_token": 0.00001, "high_gain_per_token": 0.0001, "min_width": 2, "max_width": 8},
    )
    _apply_gain_token_control(budget=budget, policy=policy, decision=decision)

    assert decision["action"] == expected_action
    assert budget.branch_factor == 4 + expected_width_delta
    assert policy.metadata["offspring_parallel_mode"] == expected_transport
    assert policy.metadata["evaluator_verdict_fixture"] == verdict
    assert decision["authority_boundary"] == "budget_width_transport_retry_only"


def test_gain_token_controller_does_not_reuse_stale_gain_when_latest_round_has_no_outcome() -> None:
    history = [
        {"round": 1, "direction_aware_gain": {"total_gain": 0.01, "sample_count": 1}},
        {"round": 2, "direction_aware_gain": {"total_gain": 0.0, "sample_count": 0}},
    ]
    ledger = {
        "schema_version": "round-cost-ledger/v1",
        "rounds": [
            {"round": 1, "totals": {"total_tokens": 10_000}},
            {"round": 2, "totals": {"total_tokens": 10_000}},
        ],
    }

    decision = _gain_token_control(
        history=history,
        cost_ledger=ledger,
        current_width=4,
        current_transport="slot",
        current_retry_limit=3,
        config={},
    )

    assert decision["source_round"] == 2
    assert decision["action"] == "hold"
    assert decision["gain_per_token"] is None


def test_round_cost_ledger_exposes_all_acceptance_metrics() -> None:
    events = [
        {
            "type": "llm_call",
            "round_id": "2",
            "logical_call_id": "logical-2",
            "physical_call_id": "physical-2",
            "attempts": 1,
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            "estimated_cost_usd": 0.01,
            "slot_ids": ["slot-0", "slot-1"],
        }
    ]
    history = [
        {
            "round": 2,
            "evaluator_qualified_survivors": 1,
            "generation_plan": {
                "offspring_ids": ["child-a", "child-b"],
                "offspring_harvest": {
                    "reservoir_truncated_count": 1,
                    "partial_failure_blast_radius": 2,
                },
            },
        }
    ]

    ledger = build_round_cost_ledger(events, budget_history=history)
    observations = ledger["rounds"][0]["observations"]

    assert observations == {
        "physical_calls": 1,
        "unique_valid_children": 2,
        "evaluator_qualified_survivors": 1,
        "truncation_count": 1,
        "partial_failure_blast_radius": 2,
    }
