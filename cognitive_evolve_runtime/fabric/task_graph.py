"""Serializable task graph for the Exploration Fabric."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, utc_now
from .task import ExplorationTask, TaskStatus

_TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED}
_COMPLETE_FOR_DEPS = {TaskStatus.DONE, TaskStatus.SKIPPED}


@dataclass
class TaskGraph:
    tasks: dict[str, ExplorationTask] = field(default_factory=dict)
    epoch: int = 0
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "fabric-task-graph/v1"
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def add(self, task: ExplorationTask) -> None:
        with self._lock:
            if not task.task_id:
                raise ValueError("task_id is required")
            if task.task_id in self.tasks:
                raise ValueError(f"duplicate task_id: {task.task_id}")
            self.tasks[task.task_id] = task
            self.topological_order()

    def ready_tasks(self) -> list[ExplorationTask]:
        with self._lock:
            ready: list[ExplorationTask] = []
            for task in self.tasks.values():
                if task.status not in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RETRYABLE_FAILED}:
                    continue
                if all(self.tasks.get(dep) is not None and self.tasks[dep].status in _COMPLETE_FOR_DEPS for dep in task.depends_on):
                    if task.status != TaskStatus.READY:
                        task.status = TaskStatus.READY
                        task.updated_at = utc_now()
                    ready.append(task)
            return sorted(ready, key=lambda item: (float(item.priority or 0.0), item.created_at, item.task_id), reverse=True)

    def mark(self, task_id: str, status: TaskStatus, result_ref: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
        with self._lock:
            if task_id not in self.tasks:
                raise KeyError(task_id)
            task = self.tasks[task_id]
            previous_status = task.status
            if status == TaskStatus.RUNNING and previous_status != TaskStatus.RUNNING:
                task.attempts += 1
            task.status = status
            if result_ref is not None:
                task.result_ref = dict(result_ref)
            if error is not None:
                task.error = dict(error)
            if previous_status != status:
                task.updated_at = utc_now()

    def recover_inflight(self) -> None:
        with self._lock:
            for task in self.tasks.values():
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.READY
                    task.error = {**task.error, "recovered_from_inflight": True}
                    task.updated_at = utc_now()

    def is_drained(self) -> bool:
        with self._lock:
            return all(task.status in _TERMINAL for task in self.tasks.values())

    def topological_order(self) -> list[str]:
        with self._lock:
            visiting: set[str] = set()
            visited: set[str] = set()
            order: list[str] = []

            def visit(task_id: str) -> None:
                if task_id in visited:
                    return
                if task_id in visiting:
                    raise ValueError("task graph contains a dependency cycle")
                if task_id not in self.tasks:
                    raise ValueError(f"task depends on missing task: {task_id}")
                visiting.add(task_id)
                for dep in self.tasks[task_id].depends_on:
                    visit(dep)
                visiting.remove(task_id)
                visited.add(task_id)
                order.append(task_id)

            for task_id in sorted(self.tasks):
                visit(task_id)
            return order

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self.schema_version,
                "epoch": int(self.epoch or 0),
                "scheduler_state": dict(self.scheduler_state or {}),
                "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        tasks_raw = coerce_dict(data.get("tasks"))
        graph = cls(
            tasks={task_id: ExplorationTask.from_dict(task_data) for task_id, task_data in tasks_raw.items() if isinstance(task_data, dict)},
            epoch=int(data.get("epoch") or 0),
            scheduler_state=coerce_dict(data.get("scheduler_state")),
            schema_version=str(data.get("schema_version") or "fabric-task-graph/v1"),
        )
        graph.topological_order()
        return graph


__all__ = ["TaskGraph"]
