"""Checkpoint state container for the Exploration Fabric."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict
from .dossier import DossierIndexEntry
from .task_graph import TaskGraph


@dataclass
class FabricCheckpointState:
    graph: TaskGraph | None = None
    dossier_index: dict[str, DossierIndexEntry] = field(default_factory=dict)
    pool_reports: list[dict[str, Any]] = field(default_factory=list)
    best_stream: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any] | str] = field(default_factory=list)
    schema_version: str = "fabric-checkpoint-state/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph": self.graph.to_dict() if self.graph is not None else {},
            "dossier_index": {cid: entry.to_dict() for cid, entry in self.dossier_index.items()},
            "pool_reports": [dict(item) for item in self.pool_reports if isinstance(item, dict)],
            "best_stream": dict(self.best_stream or {}),
            "config": dict(self.config or {}),
            "diagnostics": list(self.diagnostics or []),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FabricCheckpointState":
        graph_raw = coerce_dict(data.get("graph"))
        index_raw = coerce_dict(data.get("dossier_index"))
        reports = data.get("pool_reports") if isinstance(data.get("pool_reports"), list) else []
        diagnostics = data.get("diagnostics") if isinstance(data.get("diagnostics"), list) else []
        return cls(
            graph=TaskGraph.from_dict(graph_raw) if graph_raw.get("tasks") else None,
            dossier_index={str(cid): DossierIndexEntry.from_dict(entry) for cid, entry in index_raw.items() if isinstance(entry, dict)},
            pool_reports=[dict(item) for item in reports if isinstance(item, dict)],
            best_stream=coerce_dict(data.get("best_stream")),
            config=coerce_dict(data.get("config")),
            diagnostics=list(diagnostics),
            schema_version=str(data.get("schema_version") or "fabric-checkpoint-state/v1"),
        )


__all__ = ["FabricCheckpointState"]
