"""TaskGraphScheduler for Exploration Fabric Phase 1A."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.llm.governor import llm_governor
from .config import FabricRuntimeConfig
from .executors import FabricExecutionContext, TaskExecutor, default_phase1a_executors, resolve_model_pool
from .task import TaskKind, TaskResult, TaskStatus
from .task_graph import TaskGraph


@dataclass(frozen=True)
class EpochConfig:
    barrier: str = "full"
    checkpoint_each_epoch: bool = True
    raise_task_exceptions: bool = False


@dataclass
class FabricSchedulerResult:
    graph: TaskGraph
    task_results: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"graph": self.graph.to_dict(), "task_results": list(self.task_results), "diagnostics": list(self.diagnostics)}


class TaskGraphScheduler:
    """Run a task graph with bounded built-in concurrency.

    Phase 1A uses coarse EVALUATE/ROUND_GATE/REPRODUCE tasks and full barriers
    so it can shadow the existing round loop without splitting method bodies.
    """

    def __init__(
        self,
        *,
        graph: TaskGraph,
        context: FabricExecutionContext,
        executors: dict[TaskKind, TaskExecutor] | None = None,
        config: FabricRuntimeConfig | None = None,
        epoch_config: EpochConfig | None = None,
        governor: Any | None = None,
    ) -> None:
        self.graph = graph
        self.context = context
        self.executors = executors or default_phase1a_executors()
        self.config = config or FabricRuntimeConfig.from_runtime_context(policy=context.policy, contract=context.contract)
        self.epoch_config = epoch_config or EpochConfig(barrier=self.config.scheduler.epoch_barrier)
        self.governor = governor or llm_governor()

    def run(self) -> FabricSchedulerResult:
        self.graph.recover_inflight()
        self._sync_graph_state()
        results: list[dict[str, Any]] = []
        while not self.graph.is_drained():
            ready = self.graph.ready_tasks()
            if not ready:
                break
            max_workers = self._max_workers(len(ready))
            if max_workers <= 1 or len(ready) <= 1:
                for task in ready:
                    results.append(self._run_task(task).to_dict())
            else:
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cogev-fabric") as pool:
                    futures = {pool.submit(self._run_task, task): task.task_id for task in ready}
                    for fut in as_completed(futures):
                        results.append(fut.result().to_dict())
            if self.epoch_config.barrier == "full":
                continue
        self._sync_graph_state()
        return FabricSchedulerResult(graph=self.graph, task_results=results, diagnostics=list(self.context.fabric_state.get("diagnostics", []) or []))

    def _run_task(self, task: Any) -> TaskResult:
        pool = resolve_model_pool(task.model_pool, config=self.config, fabric_state=self.context.fabric_state)
        task.model_pool = pool
        self.graph.mark(task.task_id, TaskStatus.RUNNING)
        self._sync_graph_state()
        executor = self.executors.get(task.kind)
        if executor is None:
            result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED, error={"type": "MissingExecutor", "kind": str(task.kind)})
            self.graph.mark(task.task_id, TaskStatus.FAILED, error=result.error)
            self._sync_graph_state()
            return result
        try:
            result = executor.execute(task, self.context)
        except Exception as exc:
            result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED, error={"type": exc.__class__.__name__, "message": str(exc)})
            self.graph.mark(task.task_id, TaskStatus.FAILED, error=result.error)
            self._sync_graph_state()
            if self.epoch_config.raise_task_exceptions:
                raise
            return result
        self.graph.mark(task.task_id, result.status, result_ref=result.to_dict(), error=result.error or None)
        self._sync_graph_state()
        return result

    def _max_workers(self, ready_count: int) -> int:
        configured = max(1, int(self.config.scheduler.max_active_tasks or 1))
        governor_limit = max(1, int(self.governor._max_concurrent())) if hasattr(self.governor, "_max_concurrent") else configured
        return max(1, min(ready_count, configured, governor_limit))

    def _sync_graph_state(self) -> None:
        self.context.fabric_state["graph"] = self.graph.to_dict()
        self.context.fabric_state["scheduler"] = {
            "barrier": self.epoch_config.barrier,
            "checkpoint_each_epoch": self.epoch_config.checkpoint_each_epoch,
        }


__all__ = ["EpochConfig", "FabricSchedulerResult", "TaskGraphScheduler"]
