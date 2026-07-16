from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveConfig, AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.budget_factory import resume_evolution_budget
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.controller import EvolutionLoopController
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.loop.round_context import RoundEvaluation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import allocate_productive_branches
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def _candidate(candidate_id: str, *, fate: CandidateFate = CandidateFate.ACTIVE) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact={"answer": candidate_id},
        concise_claim=candidate_id,
        core_mechanism=candidate_id,
        current_fate=fate,
        multihead_scores={
            "objective_alignment": 0.8,
            "answer_likelihood": 0.8,
            "core_mechanism_strength": 0.8,
            "verifiability": 0.8,
        },
    )


@pytest.mark.parametrize(
    "blocked_metadata",
    [
        {"structural_failure": True},
        {"failure_classification": {"category": "terminal_credential_path_escape"}},
    ],
)
def test_explore_reserves_repair_parent_but_never_structurally_blocked(
    blocked_metadata: dict[str, Any],
) -> None:
    passed = _candidate("passed")
    passed.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"score": 1.0}}
    repairable = _candidate("repairable", fate=CandidateFate.INCUBATING)
    repairable.metadata.update(
        {
            "evaluator": {"status": "failed", "passed": False, "metrics": {"score": 0.0}},
            "repair_required": {"blockers": ["compileall_failed: SyntaxError"]},
            "failure_micro_guidance": [{"blocker": "compileall_failed", "next_action": "repair syntax"}],
        }
    )
    blocked = _candidate("blocked", fate=CandidateFate.INCUBATING)
    blocked.metadata.update(
        {
            "evaluator": {"status": "failed", "passed": False, "metrics": {"score": 0.0}},
            "repair_required": {"blockers": ["unsafe patch"]},
            **blocked_metadata,
        }
    )

    explore = ParentSelector().select([passed, repairable, blocked], limit=1, search_phase="explore")
    exit_sweep = ParentSelector().select([passed, repairable, blocked], limit=1, search_phase="exit_sweep")

    assert [candidate.id for candidate in explore] == ["repairable"]
    assert [candidate.id for candidate in exit_sweep] == ["passed"]
    assert "blocked" not in {candidate.id for candidate in [*explore, *exit_sweep]}

    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=1, search_phase="explore"))
    parents = round_stage._select_reproduction_parents(
        current_round=1,
        population=CandidatePopulation([passed, repairable, blocked]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        world={},
        rankings=RelativeRankingResult(best_final_answer_id="passed", mutation_worthy_ids=["repairable"]),
        diagnosis=SearchDiagnosis(),
        repair_parent_candidates=[repairable, blocked],
        limit_override=1,
    )
    assert [candidate.id for candidate in parents] == ["repairable"]


def test_exit_sweep_parent_and_stop_semantics_match_non_phase_behavior() -> None:
    low = _candidate("low")
    low.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"loss": 0.1}, "selection_binding": {"metric": "loss", "direction": "minimize", "value_type": "number"}}
    high = _candidate("high")
    high.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"loss": 0.9}, "selection_binding": {"metric": "loss", "direction": "minimize", "value_type": "number"}}

    baseline = ParentSelector().select([high, low], limit=2)
    exit_sweep = ParentSelector().select([high, low], limit=2, search_phase="exit_sweep")
    assert [candidate.id for candidate in exit_sweep] == [candidate.id for candidate in baseline] == ["low", "high"]

    stop = StopDecisionEngine()
    kwargs = {
        "completed_round": 1,
        "diagnosis": SearchDiagnosis(),
        "best_answer_id": "high",
        "population": CandidatePopulation([high, low]),
        "model": None,
    }
    assert stop.stop_reason_after_round(budget=EvolutionBudget(max_rounds=3, stop_policy="max_rounds"), **kwargs) == stop.stop_reason_after_round(
        budget=EvolutionBudget(max_rounds=3, stop_policy="max_rounds", search_phase="exit_sweep"),
        **kwargs,
    )


@pytest.mark.parametrize("search_phase", ["explore", "exit_sweep"])
def test_evaluator_executes_and_records_in_every_phase(search_phase: str, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _candidate("candidate")
    budget = EvolutionBudget(max_rounds=2, branch_factor=1, stop_policy="max_rounds", search_phase=search_phase)
    adaptive = AdaptiveRuntimeController(
        config=AdaptiveConfig.from_sources(
            explicit={"enabled": True, "evaluator": {"enabled": True, "command": "unit-test-evaluator"}}
        )
    )
    pipeline = EvolutionRound(model=None, budget=budget, adaptive=adaptive)
    calls: list[str] = []

    def evaluate(candidates: list[CandidateGenome], **_: Any) -> list[Any]:
        calls.extend(item.id for item in candidates)
        for item in candidates:
            item.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"score": 1.0}}
        return [SimpleNamespace(passed=True) for _item in candidates]

    monkeypatch.setattr(pipeline.evaluator_runner, "evaluate_population_if_configured", evaluate)
    result = pipeline.evaluate(
        current_round=1,
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(metadata={"search_phase": search_phase}),
        contract=NexusObjectiveContract(original_user_goal="phase", normalized_goal="phase"),
    )

    assert calls == ["candidate"]
    assert result.progress_event["metadata"]["search_phase"] == search_phase
    assert candidate.metadata["evaluator"]["passed"] is True


class _PhasePipeline:
    def __init__(self, *, emit_receipt: bool = True) -> None:
        self.emit_receipt = emit_receipt
        self.phases: list[str] = []
        self.last_generation_plan: dict[str, Any] = {}
        self.last_completed_stage_ops: list[str] = []
        self.elo = SimpleNamespace(to_dict=lambda: {})

    def evaluate(self, *, current_round: int, population: CandidatePopulation, policy: EvolutionPolicy, **_: Any) -> RoundEvaluation:
        phase = str(policy.metadata["search_phase"])
        self.phases.append(phase)
        ranking = RelativeRankingResult(
            best_final_answer_id=population.candidates[0].id,
            strongest_mechanism_id=population.candidates[0].id,
            mutation_worthy_ids=[population.candidates[0].id],
        )
        plan = {"plan_id": f"plan-{current_round}", "round_index": current_round}
        self.last_generation_plan = dict(plan)
        return RoundEvaluation(
            rankings=ranking,
            policy=policy,
            diagnosis=SearchDiagnosis(),
            critiques=[],
            verification_results=[],
            progress_event={"type": "evolution_progress", "round": current_round, "metadata": {}},
            pipeline_event={"type": "pipeline_progress", "round": current_round},
            stop_reason="",
            generation_plan=plan,
        )

    def reproduce(self, *, current_round: int, **_: Any) -> tuple[str, list[Any], dict[str, Any]]:
        if current_round == 1 and self.emit_receipt:
            self.last_generation_plan["intervention_receipts"] = [
                {
                    "receipt_id": "intervention-round-1",
                    "diagnosed_pressure": {"stagnation_type": "SemanticLooping", "diagnosis_ref": "diagnosis-round-1"},
                    "intervention_type": "diagnosis_guided_reproduction",
                    "target": {"axis": "", "family": "", "action": "repair", "slot": "slot-1"},
                    "recipient_branch_slot_ids": ["slot-1"],
                    "produced_candidate_ids": ["candidate"],
                    "outcome_refs": [],
                }
            ]
        return "", [], {}


def test_phase_transitions_are_audited_and_resume_preserves_phase_and_slots() -> None:
    candidate = _candidate("candidate")
    budget = EvolutionBudget(max_rounds=4, branch_factor=1, stop_policy="max_rounds")
    controller = EvolutionLoopController(
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="phase", normalized_goal="phase"),
        world={},
        budget=budget,
    )
    phase_pipeline = _PhasePipeline()
    controller.round_pipeline = phase_pipeline  # type: ignore[assignment]

    assert controller._run_direct_epoch(1) is False
    assert controller._run_direct_epoch(2) is False
    assert budget.search_phase == "exit_sweep"
    checkpoint_budget = budget.to_dict()

    parents = [candidate]
    original_slots = allocate_productive_branches(parents=parents, candidates=parents, budget_history=budget.history, total_slots=1).to_dict()["slots"]
    resumed = resume_evolution_budget(
        checkpoint_round=budget.current_round,
        checkpoint_max_rounds=budget.max_rounds,
        budget_data=checkpoint_budget,
        max_rounds=budget.max_rounds,
    )
    resumed.history = [dict(item) for item in budget.history]
    resumed_slots = allocate_productive_branches(parents=parents, candidates=parents, budget_history=resumed.history, total_slots=1).to_dict()["slots"]

    assert resumed.search_phase == "exit_sweep"
    assert resumed_slots == original_slots
    assert controller._run_direct_epoch(3) is False
    assert phase_pipeline.phases[:3] == ["explore", "explore", "exit_sweep"]
    assert budget.search_phase == "explore"

    enter = budget.history[1]["search_phase_transition"]
    leave = budget.history[2]["search_phase_transition"]
    assert (enter["from"], enter["to"], enter["reason"]) == ("explore", "exit_sweep", "stagnation_intervention_completed")
    assert enter["stagnation_receipt_refs"] == ["intervention-round-1"]
    assert (leave["from"], leave["to"], leave["reason"]) == ("exit_sweep", "explore", "exit_conditions_not_met")
    assert enter == budget.history[1]["generation_plan"]["search_phase_transition"]
    assert leave == budget.history[2]["generation_plan"]["search_phase_transition"]
    assert "remaining_budget" in enter


def test_round_budget_tail_marks_the_next_round_exit_sweep() -> None:
    candidate = _candidate("candidate")
    budget = EvolutionBudget(max_rounds=3, branch_factor=1, stop_policy="max_rounds")
    controller = EvolutionLoopController(
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="phase", normalized_goal="phase"),
        world={},
        budget=budget,
    )
    controller.round_pipeline = _PhasePipeline(emit_receipt=False)  # type: ignore[assignment]

    assert controller._run_direct_epoch(1) is False
    assert budget.search_phase == "explore"
    assert controller._run_direct_epoch(2) is False
    assert budget.search_phase == "exit_sweep"
    assert budget.history[1]["search_phase_transition"]["reason"] == "budget_tail"
