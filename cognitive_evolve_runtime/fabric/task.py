"""Task primitives for the Exploration Fabric scheduler."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, utc_now
from .advisory import assert_advisory_payload


class TaskKind(StrEnum):
    SOURCE = "source"
    PREPROCESS = "preprocess"
    EXPAND = "expand"
    EVALUATE = "evaluate"
    ROUND_GATE = "round_gate"
    REPRODUCE = "reproduce"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    RANK = "rank"
    COMPACT = "compact"
    DIAGNOSE = "diagnose"
    STOP_CHECK = "stop_check"
    TRANSFORM = "transform"
    VERIFY = "verify"
    ATTACK = "attack"
    COMBINE = "combine"


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYABLE_FAILED = "retryable_failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class ExplorationTask:
    task_id: str
    kind: TaskKind
    target_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    priority: float = 0.0
    model_pool: str = "default"
    payload: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    epoch: int = 0
    attempts: int = 0
    result_ref: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)
    advisory: bool = True
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = str(self.kind)
        payload["status"] = str(self.status)
        payload["advisory"] = True
        if self.kind != TaskKind.VERIFY:
            assert_advisory_payload(self.payload)
            assert_advisory_payload(self.result_ref)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplorationTask":
        status = TaskStatus(str(data.get("status") or TaskStatus.PENDING))
        kind = TaskKind(str(data.get("kind") or TaskKind.SOURCE))
        return cls(
            task_id=str(data.get("task_id") or ""),
            kind=kind,
            target_ids=coerce_str_list(data.get("target_ids")),
            depends_on=coerce_str_list(data.get("depends_on")),
            priority=float(data.get("priority") or 0.0),
            model_pool=str(data.get("model_pool") or "default"),
            payload=coerce_dict(data.get("payload")),
            status=status,
            epoch=int(data.get("epoch") or 0),
            attempts=int(data.get("attempts") or 0),
            result_ref=coerce_dict(data.get("result_ref")),
            error=coerce_dict(data.get("error")),
            advisory=True,
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    produced_candidate_ids: list[str] = field(default_factory=list)
    advisory_updates: dict[str, Any] = field(default_factory=dict)
    verification_records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_delta: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        assert_advisory_payload(self.advisory_updates)
        return payload


__all__ = ["ExplorationTask", "TaskKind", "TaskResult", "TaskStatus"]
