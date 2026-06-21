from __future__ import annotations

from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.fabric import TaskGraphScheduler
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopController, evolve_once
from cognitive_evolve_runtime.nexus.loop.round import RoundEvaluation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


class _Elo:
    def to_dict(self) -> dict[str, Any]:
        return {"fake": True}


class _Phase1BRoundPipeline:
    def __init__(self, *, stop_reason: str = "") -> None:
        self.stop_reason = stop_reason
        self.evaluate_calls = 0
        self.reproduce_calls = 0
        self.last_generation_plan: dict[str, Any] = {}
        self.elo = _Elo()

    def evaluate(self, *, current_round: int, population: CandidatePopulation, archives: ArchiveManager, policy: EvolutionPolicy, contract: NexusObjectiveContract) -> RoundEvaluation:
        self.evaluate_calls += 1
        population.candidates[0].metadata["phase1b_evaluated_round"] = current_round
        return RoundEvaluation(
            rankings=RelativeRankingResult(best_final_answer_id=population.candidates[0].id, mutation_worthy_ids=[population.candidates[0].id]),
            policy=policy,
            diagnosis=SearchDiagnosis(notes="phase1b"),
            critiques=[],
            verification_results=[],
            progress_event={"type": "evolution_progress", "round": current_round, "max_rounds": 2},
            pipeline_event={"type": "pipeline_progress", "stage": "phase1b_fake"},
            stop_reason=self.stop_reason,
            population_compaction={"changed": False},
            repair_parent_candidates=list(population.candidates),
            generation_plan={"plan_id": "phase1b"},
        )

    def reproduce(self, *, current_round: int, population: CandidatePopulation, archives: ArchiveManager, policy: EvolutionPolicy, contract: NexusObjectiveContract, world: Any, rankings: RelativeRankingResult, diagnosis: Any, critiques: list[Any], offspring_verifier: Any, repair_parent_candidates: list[CandidateGenome] | None = None) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.reproduce_calls += 1
        self.last_generation_plan = {"plan_id": "phase1b", "completed_ops": ["fake_reproduce"]}
        child = CandidateGenome(id=f"phase1b-child-{current_round}", parent_ids=[population.candidates[0].id], concise_claim="child")
        population.integrate(child)
        return "", [], {"live_population_size": len(population.candidates)}


def _controller(*, budget: EvolutionBudget | None = None, stop_reason: str = "", fabric_state: dict[str, Any] | None = None) -> tuple[EvolutionLoopController, _Phase1BRoundPipeline]:
    controller = EvolutionLoopController(
        population=CandidatePopulation([CandidateGenome(id="C1", artifact="candidate", concise_claim="candidate")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        world={},
        budget=budget or EvolutionBudget(max_rounds=2, branch_factor=1),
        fabric_state=fabric_state,
    )
    pipeline = _Phase1BRoundPipeline(stop_reason=stop_reason)
    controller.round_pipeline = pipeline
    return controller, pipeline


def test_evolve_once_uses_scheduler(monkeypatch) -> None:
    calls: list[str] = []
    original_run = TaskGraphScheduler.run

    def wrapped_run(self):
        calls.append(self.graph.schema_version)
        return original_run(self)

    monkeypatch.setattr(TaskGraphScheduler, "run", wrapped_run)
    controller, pipeline = _controller()
    result = controller.run()
    assert calls == ["fabric-task-graph/v1", "fabric-task-graph/v1"]
    assert pipeline.evaluate_calls == 2
    assert result.fabric_state["graph"]["tasks"]


def test_evolution_loop_controller_is_scheduler_shim() -> None:
    controller, pipeline = _controller(budget=EvolutionBudget(max_rounds=1, branch_factor=1))
    result = controller.run()
    assert pipeline.evaluate_calls == 1
    assert pipeline.reproduce_calls == 0
    assert result.stop_reason == "max_rounds"
    assert result.fabric_state["last_scheduler_result"]["graph"]["schema_version"] == "fabric-task-graph/v1"


def test_old_loop_bodies_removed() -> None:
    assert not hasattr(EvolutionLoopController, "_run_round")
    assert not hasattr(EvolutionLoopController, "_reproduce")


def test_graph_checkpoint_resume() -> None:
    first, _ = _controller(budget=EvolutionBudget(max_rounds=1, branch_factor=1))
    first_result = first.run()
    restored_state = dict(first_result.fabric_state)
    second, _ = _controller(budget=EvolutionBudget(max_rounds=2, branch_factor=1), fabric_state=restored_state)
    second_result = second.run()
    assert second_result.fabric_state["graph"]["schema_version"] == "fabric-task-graph/v1"
    assert second_result.current_round >= 1


def test_scheduler_stop_produces_answer_without_verified_solved_claim() -> None:
    controller, _ = _controller(stop_reason="objective_solved", budget=EvolutionBudget(max_rounds=2, branch_factor=1))
    result = controller.run()
    assert result.stop_reason == "objective_solved"
    assert result.graded_output["mode"] != "verified_result"
    assert result.synthesis.closure_certificate["answer_produced"] is True
    assert result.synthesis.closure_certificate["objective_solved"] is False
    assert result.synthesis.closure_certificate["graded_output_advisory"] == "verification_result_not_required_for_answer_first_completion"


def test_evolve_once_accepts_restored_fabric_state() -> None:
    result = evolve_once(
        population=CandidatePopulation([CandidateGenome(id="C1", artifact="candidate", concise_claim="candidate")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        world={},
        budget=EvolutionBudget(max_rounds=1, branch_factor=1),
        fabric_state={"diagnostics": [{"type": "restored"}]},
    )
    assert result.fabric_state["diagnostics"][0]["type"] == "restored"
    assert result.fabric_state["graph"]["schema_version"] == "fabric-task-graph/v1"
