"""Build coarse round-parity task graphs for Fabric Phase 1A."""
from __future__ import annotations

from .task import ExplorationTask, TaskKind
from .task_graph import TaskGraph


def build_round_parity_epoch_graph(*, round_index: int) -> TaskGraph:
    suffix = f"r{int(round_index):04d}"
    graph = TaskGraph(epoch=int(round_index or 0))
    graph.add(ExplorationTask(task_id=f"evaluate-{suffix}", kind=TaskKind.EVALUATE, payload={"planned_round": int(round_index or 0)}, epoch=int(round_index or 0)))
    graph.add(ExplorationTask(task_id=f"round-gate-{suffix}", kind=TaskKind.ROUND_GATE, depends_on=[f"evaluate-{suffix}"], payload={"current_round": int(round_index or 0)}, epoch=int(round_index or 0)))
    graph.add(ExplorationTask(task_id=f"reproduce-{suffix}", kind=TaskKind.REPRODUCE, depends_on=[f"round-gate-{suffix}"], payload={"current_round": int(round_index or 0)}, epoch=int(round_index or 0)))
    return graph


__all__ = ["build_round_parity_epoch_graph"]
