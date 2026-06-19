from __future__ import annotations

from copy import deepcopy
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.fabric import FabricRuntimeConfig, TaskGraphScheduler, build_round_parity_epoch_graph
from cognitive_evolve_runtime.fabric.executors import FabricExecutionContext, resolve_model_pool
from cognitive_evolve_runtime.fabric.scheduler import EpochConfig
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.round import RoundEvaluation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


class FakeRoundPipeline:
    def __init__(self, *, stop_reason: str = "") -> None:
        self.stop_reason = stop_reason
        self.evaluate_calls = 0
        self.reproduce_calls = 0
        self.offspring_verifier_seen = False

    def evaluate(self, *, current_round: int, population: CandidatePopulation, archives: ArchiveManager, policy: EvolutionPolicy, contract: NexusObjectiveContract) -> RoundEvaluation:
        self.evaluate_calls += 1
        population.candidates[0].metadata["evaluated_round"] = current_round
        archives.history.append({"stage": "evaluate", "round": current_round})
        return RoundEvaluation(
            rankings=RelativeRankingResult(best_final_answer_id=population.candidates[0].id, mutation_worthy_ids=[population.candidates[0].id]),
            policy=policy,
            diagnosis=SearchDiagnosis(notes="fake"),
            critiques=[],
            verification_results=[],
            progress_event={"round": current_round, "max_rounds": 2},
            pipeline_event={"stage": "fake"},
            stop_reason=self.stop_reason,
            population_compaction={"changed": False},
            repair_parent_candidates=list(population.candidates),
            generation_plan={"plan_id": "fake"},
        )

    def reproduce(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        critiques: list[Any],
        offspring_verifier: Any,
        repair_parent_candidates: list[CandidateGenome] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.reproduce_calls += 1
        child = CandidateGenome(id=f"child-{current_round}", parent_ids=[population.candidates[0].id], concise_claim="child")
        population.integrate(child)
        archives.history.append({"stage": "reproduce", "round": current_round, "child": child.id})
        records = offspring_verifier([child]) if offspring_verifier is not None else []
        self.offspring_verifier_seen = offspring_verifier is not None
        return "", [record if isinstance(record, dict) else {"candidate_id": getattr(record, "candidate_id", child.id)} for record in records], {"live_population_size": len(population.candidates)}


def _context(*, pipeline: FakeRoundPipeline, verifier=None) -> FabricExecutionContext:
    return FabricExecutionContext(
        population=CandidatePopulation([CandidateGenome(id="C1", concise_claim="seed")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        world={"kind": "text"},
        budget=EvolutionBudget(max_rounds=2, branch_factor=2),
        offspring_verifier=verifier,
        round_pipeline=pipeline,
    )


def _run_direct(context: FabricExecutionContext) -> None:
    current_round = context.budget.step()
    evaluation = context.pipeline().evaluate(current_round=current_round, population=context.population, archives=context.archives, policy=context.policy, contract=context.contract)
    context.policy = evaluation.policy
    context.diagnosis = evaluation.diagnosis
    context.last_evaluation = evaluation
    if evaluation.stop_reason:
        context.budget.stop_reason = evaluation.stop_reason
        return
    if current_round >= context.budget.round_limit:
        context.budget.stop_reason = "adaptive_safety_checkpoint" if context.budget.adaptive else "max_rounds"
        return
    context.pipeline().reproduce(
        current_round=current_round,
        population=context.population,
        archives=context.archives,
        policy=context.policy,
        contract=context.contract,
        world=context.world,
        rankings=evaluation.rankings,
        diagnosis=context.diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=context.offspring_verifier,
        repair_parent_candidates=evaluation.repair_parent_candidates,
    )


def test_scheduler_shadow_round_parity() -> None:
    direct = _context(pipeline=FakeRoundPipeline())
    scheduled = _context(pipeline=FakeRoundPipeline())
    _run_direct(direct)
    graph = build_round_parity_epoch_graph(round_index=1)
    result = TaskGraphScheduler(graph=graph, context=scheduled, epoch_config=EpochConfig(barrier="full")).run()
    assert result.graph.is_drained()
    assert [candidate.id for candidate in scheduled.population.candidates] == [candidate.id for candidate in direct.population.candidates]
    assert scheduled.budget.stop_reason == direct.budget.stop_reason
    assert scheduled.population.by_id()["C1"].metadata["evaluated_round"] == direct.population.by_id()["C1"].metadata["evaluated_round"]


def test_scheduler_shadow_population_archive_equivalence() -> None:
    direct = _context(pipeline=FakeRoundPipeline())
    scheduled = _context(pipeline=FakeRoundPipeline())
    _run_direct(direct)
    TaskGraphScheduler(graph=build_round_parity_epoch_graph(round_index=1), context=scheduled, epoch_config=EpochConfig(barrier="full")).run()
    assert [candidate.id for candidate in scheduled.population.candidates] == [candidate.id for candidate in direct.population.candidates]
    assert [candidate.parent_ids for candidate in scheduled.population.candidates] == [candidate.parent_ids for candidate in direct.population.candidates]
    assert [candidate.metadata for candidate in scheduled.population.candidates] == [candidate.metadata for candidate in direct.population.candidates]
    assert scheduled.archives.history == direct.archives.history


def test_scheduler_shadow_stop_reason_equivalence_and_reproduce_skipped() -> None:
    pipeline = FakeRoundPipeline(stop_reason="model_stop")
    context = _context(pipeline=pipeline)
    graph = build_round_parity_epoch_graph(round_index=1)
    TaskGraphScheduler(graph=graph, context=context).run()
    assert context.budget.stop_reason == "model_stop"
    assert pipeline.evaluate_calls == 1
    assert pipeline.reproduce_calls == 0
    assert graph.tasks["reproduce-r0001"].status.value == "skipped"


def test_scheduler_uses_bound_offspring_verifier() -> None:
    calls: list[str] = []

    def verifier(candidates: list[CandidateGenome]) -> list[dict[str, Any]]:
        calls.extend(candidate.id for candidate in candidates)
        return [{"candidate_id": candidates[0].id, "passed": True}]

    pipeline = FakeRoundPipeline()
    context = _context(pipeline=pipeline, verifier=verifier)
    TaskGraphScheduler(graph=build_round_parity_epoch_graph(round_index=1), context=context).run()
    assert pipeline.offspring_verifier_seen is True
    assert calls == ["child-1"]


def test_unknown_model_pool_warns_and_falls_back() -> None:
    state: dict[str, Any] = {}
    cfg = FabricRuntimeConfig.from_runtime_context()
    assert resolve_model_pool("typo-pool", config=cfg, fabric_state=state) == "default"
    assert state["diagnostics"][0]["type"] == "unknown_model_pool_fallback"


def test_scheduler_does_not_split_evaluate_or_reproduce() -> None:
    pipeline = FakeRoundPipeline()
    context = _context(pipeline=pipeline)
    TaskGraphScheduler(graph=build_round_parity_epoch_graph(round_index=1), context=context).run()
    assert pipeline.evaluate_calls == 1
    assert pipeline.reproduce_calls == 1
