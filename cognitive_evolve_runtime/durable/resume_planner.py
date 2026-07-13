"""Resume planning over committed durable checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint_store import CheckpointStore
from .step_state import StepStatus


@dataclass
class ResumeAction:
    action: str
    round_id: str
    step_id: str
    step_name: str
    reason: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResumePlan:
    status: str
    actions: list[ResumeAction] = field(default_factory=list)
    committed_steps: int = 0
    retryable_steps: int = 0
    terminal_failed_steps: int = 0
    inflight_stale_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "actions": [action.to_dict() for action in self.actions],
            "committed_steps": self.committed_steps,
            "retryable_steps": self.retryable_steps,
            "terminal_failed_steps": self.terminal_failed_steps,
            "inflight_stale_steps": self.inflight_stale_steps,
            "no_committed_rerun_policy": True,
        }


class ResumePlanner:
    def __init__(self, root: Path | str, *, inflight_timeout_seconds: float = 900.0) -> None:
        self.store = CheckpointStore(root)
        self.inflight_timeout_seconds = max(1.0, float(inflight_timeout_seconds))

    def plan(self) -> ResumePlan:
        statuses = self.store.iter_statuses()
        actions: list[ResumeAction] = []
        committed = retryable = terminal_failed = inflight_stale = 0
        for status in statuses:
            if status.state == "committed":
                committed += 1
                actions.append(_action("skip", status, "committed_step_will_not_rerun"))
            elif status.state == "retryable_failed":
                retryable += 1
                actions.append(_action("retry", status, "retryable_failed_step_can_resume"))
            elif status.state == "inflight":
                if self._is_stale(status):
                    inflight_stale += 1
                    actions.append(_action("mark_retryable_failed", status, "inflight_step_older_than_timeout"))
                else:
                    actions.append(_action("wait", status, "inflight_step_not_yet_stale"))
            elif status.state == "terminal_failed":
                terminal_failed += 1
                actions.append(_action("stop", status, "terminal_failure_requires_operator_or_new_run"))
            elif status.state in {"created", "ready", "resumed"}:
                retryable += 1
                actions.append(_action("run", status, "uncommitted_step_ready_to_run"))
            elif status.state == "cancelled":
                terminal_failed += 1
                actions.append(_action("stop", status, "cancelled_step_is_terminal"))
        if terminal_failed:
            plan_status = "terminal_failed"
        elif inflight_stale or retryable:
            plan_status = "resume_available"
        elif committed:
            plan_status = "all_committed"
        else:
            plan_status = "empty"
        return ResumePlan(plan_status, actions, committed, retryable, terminal_failed, inflight_stale)

    def apply_stale_inflight(self) -> ResumePlan:
        plan = self.plan()
        for action in plan.actions:
            if action.action != "mark_retryable_failed":
                continue
            status = self.store.read_status(action.round_id, action.step_id, action.step_name)
            if status and status.state == "inflight":
                self.store.mark_failed(status, retryable=True, error={"type": "StaleInflight", "message": action.reason})
        return self.plan()

    def _is_stale(self, status: StepStatus) -> bool:
        try:
            updated = datetime.fromisoformat(status.updated_at)
        except ValueError:
            return True
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - updated).total_seconds() > self.inflight_timeout_seconds


def _action(action: str, status: StepStatus, reason: str) -> ResumeAction:
    return ResumeAction(action=action, round_id=status.round_id, step_id=status.step_id, step_name=status.step_name, reason=reason, status=status.state)


__all__ = ["ResumeAction", "ResumePlan", "ResumePlanner"]
