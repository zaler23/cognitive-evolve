"""Coarse-grained Phase 1A executors for the Exploration Fabric scheduler."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.adaptive.controller import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound, RoundEvaluation
from cognitive_evolve_runtime.nexus._shared import call_with_optional_context
from cognitive_evolve_runtime.nexus.model_routes import NexusModelRoutes
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from .config import FabricRuntimeConfig
from .task import ExplorationTask, TaskKind, TaskResult, TaskStatus


@dataclass
class FabricExecutionContext:
    population: CandidatePopulation
    archives: ArchiveManager
    policy: EvolutionPolicy
    contract: NexusObjectiveContract
    world: Any
    budget: EvolutionBudget
    model: NexusModelLike | None = None
    model_routes: NexusModelRoutes | None = None
    observer: Callable[[dict[str, Any]], None] | None = None
    adaptive: AdaptiveRuntimeController | None = None
    offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None = None
    cancellation_callback: Callable[[], bool] | None = None
    record_evaluation: Callable[[int, RoundEvaluation, "FabricExecutionContext"], None] | None = None
    record_reproduction: Callable[[int, RoundEvaluation, str, list[Any], dict[str, Any], "FabricExecutionContext"], None] | None = None
    fabric_config: FabricRuntimeConfig | None = None
    fabric_state: dict[str, Any] = field(default_factory=dict)
    provided_context: dict[str, Any] = field(default_factory=dict)
    round_pipeline: Any | None = None
    diagnosis: SearchDiagnosis = field(default_factory=SearchDiagnosis)
    last_evaluation: RoundEvaluation | None = None
    should_reproduce: bool = True

    def provide(self, value: Any, *, key: str | None = None) -> Any:
        name = key or type(value).__name__
        self.provided_context[str(name)] = value
        return value

    def resolve(self, protocol_or_key: Any, default: Any = None) -> Any:
        if isinstance(protocol_or_key, str):
            return self.provided_context.get(protocol_or_key, default)
        name = getattr(protocol_or_key, "__name__", str(protocol_or_key))
        return self.provided_context.get(name, default)

    def pipeline(self) -> Any:
        if self.round_pipeline is None:
            self.round_pipeline = EvolutionRound(model=self.model, budget=self.budget, adaptive=self.adaptive)
        return self.round_pipeline

    def diagnostics(self) -> list[Any]:
        diagnostics = self.fabric_state.setdefault("diagnostics", [])
        if not isinstance(diagnostics, list):
            diagnostics = []
            self.fabric_state["diagnostics"] = diagnostics
        return diagnostics

    def raise_if_cancelled(self) -> None:
        if self.cancellation_callback is not None and self.cancellation_callback():
            raise InterruptedError("nexus evolution cancellation requested")


class TaskExecutor(Protocol):
    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult: ...


class PreprocessExecutor:
    """Advisory candidate-pool preprocessing before evaluation."""

    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult:
        from cognitive_evolve_runtime.nexus.pool_preprocessing import (
            annotate_candidate_clusters,
            cluster_candidates,
            pool_coverage_report,
            preprocess_candidate_pool,
        )

        context.raise_if_cancelled()
        cfg = context.fabric_config or FabricRuntimeConfig.from_runtime_context(policy=context.policy, contract=context.contract)
        clusters = cluster_candidates(context.population.candidates, config=cfg.pool)
        annotate_candidate_clusters(context.population.candidates, clusters)
        expected_cells = _expected_descriptor_cells(context.policy, context.contract)
        coverage = pool_coverage_report(context.population.candidates, expected_cells=expected_cells, config=cfg.preprocess)
        model_report = preprocess_candidate_pool(
            context.model,
            candidates=context.population.candidates,
            clusters=clusters,
            coverage_report=coverage,
            contract=context.contract,
            policy=context.policy,
            config=cfg.preprocess,
        )
        model_summary = {key: value for key, value in model_report.items() if key != "prompt_payload"}
        report = {
            "advisory": True,
            "task_id": task.task_id,
            "cluster_count": len(clusters),
            "clusters": [cluster.to_dict() for cluster in clusters[: cfg.pool.representative_limit]],
            "coverage": coverage,
            "model_preprocess": model_summary,
            "config_hash": cfg.config_hash,
        }
        reports = context.fabric_state.setdefault("pool_reports", [])
        if isinstance(reports, list):
            reports.append(report)
        context.policy.metadata["fabric_pool_preprocess"] = {
            "advisory": True,
            "latest_task_id": task.task_id,
            "cluster_count": len(clusters),
            "coverage": {
                "occupied_cell_count": coverage.get("occupied_cell_count"),
                "sparse_cells": list(coverage.get("sparse_cells") or []),
                "overrepresented_cells": list(coverage.get("overrepresented_cells") or []),
                "missing_cells": list(coverage.get("missing_cells") or []),
            },
            "schedule_hints": list(model_summary.get("schedule_hints") or []),
        }
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.DONE,
            advisory_updates={
                "pool_report_index": len(reports) - 1 if isinstance(reports, list) else 0,
                "cluster_count": len(clusters),
                "occupied_cell_count": coverage.get("occupied_cell_count"),
            },
            events=[{"type": "fabric_preprocess_completed", "cluster_count": len(clusters), "occupied_cell_count": coverage.get("occupied_cell_count")}],
        )


class EvaluateExecutor:
    """Direct wrapper around ``EvolutionRound.evaluate``; no internal splitting."""

    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult:
        context.raise_if_cancelled()
        current_round = int(task.payload.get("current_round") or task.payload.get("planned_round") or 0)
        if current_round <= 0:
            current_round = context.budget.step()
        elif context.budget.current_round < current_round:
            context.budget.current_round = current_round
        evaluation = context.pipeline().evaluate(
            current_round=current_round,
            population=context.population,
            archives=context.archives,
            policy=context.policy,
            contract=context.contract,
        )
        context.policy = evaluation.policy
        context.diagnosis = evaluation.diagnosis
        context.last_evaluation = evaluation
        if context.record_evaluation is not None:
            context.record_evaluation(current_round, evaluation, context)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.DONE,
            advisory_updates={"round": current_round, "stop_reason": evaluation.stop_reason, "evaluation_task": task.task_id},
            events=[{"type": "fabric_evaluate_completed", "round": current_round}],
        )


class RoundGateExecutor:
    """Apply the same post-evaluate gate used by the legacy controller."""

    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult:
        context.raise_if_cancelled()
        evaluation = context.last_evaluation
        current_round = int(context.budget.current_round or task.payload.get("current_round") or 0)
        stop_reason = ""
        should_reproduce = True
        if evaluation is not None and evaluation.stop_reason:
            stop_reason = evaluation.stop_reason
            context.budget.stop_reason = stop_reason
            should_reproduce = False
        elif current_round >= context.budget.round_limit:
            stop_reason = "adaptive_safety_checkpoint" if context.budget.adaptive else "max_rounds"
            context.budget.stop_reason = stop_reason
            should_reproduce = False
        context.should_reproduce = should_reproduce
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.DONE,
            advisory_updates={"round": current_round, "stop_reason": stop_reason, "should_reproduce": should_reproduce},
            events=[{"type": "fabric_round_gate", "round": current_round, "stop_reason": stop_reason, "should_reproduce": should_reproduce}],
        )


class ReproduceExecutor:
    """Direct wrapper around ``EvolutionRound.reproduce`` using the bound verifier closure."""

    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult:
        context.raise_if_cancelled()
        evaluation = context.last_evaluation
        current_round = int(context.budget.current_round or task.payload.get("current_round") or 0)
        if not context.should_reproduce or evaluation is None:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SKIPPED,
                advisory_updates={"round": current_round, "skipped": True, "reason": "round_gate_stopped" if not context.should_reproduce else "missing_evaluation"},
            )
        reproduction_stop, offspring_verification, reproduction_compaction = call_with_optional_context(
            context.pipeline().reproduce,
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
            provided_context=context.provided_context,
        )
        if reproduction_stop:
            context.budget.stop_reason = reproduction_stop
        if context.record_reproduction is not None:
            context.record_reproduction(current_round, evaluation, reproduction_stop, offspring_verification, reproduction_compaction, context)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.DONE,
            produced_candidate_ids=[candidate.id for candidate in context.population.candidates],
            advisory_updates={"round": current_round, "reproduction_stop": reproduction_stop, "offspring_verification_count": len(offspring_verification), "reproduction_compaction": reproduction_compaction},
            events=[{"type": "fabric_reproduce_completed", "round": current_round, "stop_reason": reproduction_stop}],
        )


class SynthesizeExecutor:
    """Placeholder executor for Phase 1A graph completeness.

    Production finalization remains owned by the existing controller until Phase
    1B.  This executor only records that a synthesis boundary was reached.
    """

    def execute(self, task: ExplorationTask, context: FabricExecutionContext) -> TaskResult:
        return TaskResult(task_id=task.task_id, status=TaskStatus.DONE, advisory_updates={"synthesis_boundary": True})


def default_fabric_executors() -> dict[TaskKind, TaskExecutor]:
    return {
        TaskKind.PREPROCESS: PreprocessExecutor(),
        TaskKind.EVALUATE: EvaluateExecutor(),
        TaskKind.ROUND_GATE: RoundGateExecutor(),
        TaskKind.REPRODUCE: ReproduceExecutor(),
        TaskKind.SYNTHESIZE: SynthesizeExecutor(),
    }


def resolve_model_pool(pool: str, *, config: FabricRuntimeConfig, fabric_state: dict[str, Any]) -> str:
    known = {"seed", "default", "verify", "local", *config.pool_concurrency.keys()}
    normalized = str(pool or "default")
    if normalized in known:
        return normalized
    diagnostics = fabric_state.setdefault("diagnostics", [])
    if isinstance(diagnostics, list):
        diagnostics.append({"type": "unknown_model_pool_fallback", "requested_pool": normalized, "fallback_pool": "default"})
    return "default"


__all__ = [
    "EvaluateExecutor",
    "FabricExecutionContext",
    "PreprocessExecutor",
    "ReproduceExecutor",
    "RoundGateExecutor",
    "SynthesizeExecutor",
    "TaskExecutor",
    "default_fabric_executors",
    "resolve_model_pool",
]


def _expected_descriptor_cells(policy: EvolutionPolicy | None, contract: NexusObjectiveContract | None) -> list[str]:
    expected: list[str] = []
    for source in (policy, contract):
        metadata = getattr(source, "metadata", {}) if source is not None else {}
        if not isinstance(metadata, dict):
            continue
        raw = metadata.get("expected_descriptor_cells") or metadata.get("fabric_expected_descriptor_cells")
        if isinstance(raw, list):
            expected.extend(str(item) for item in raw if str(item or "").strip())
    return list(dict.fromkeys(expected))
