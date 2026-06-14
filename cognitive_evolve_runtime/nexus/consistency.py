"""Runtime consistency predicates for checkpoints, events, and job status."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TERMINAL_STATUSES = {
    "completed",
    "solved",
    "needs_continuation",
    "best_current_route",
    "route_incomplete",
    "failed_verification",
    "failed",
    "interrupted_checkpointed",
    "paused_quota",
}

KNOWN_STATUSES = TERMINAL_STATUSES | {"running"}


@dataclass(frozen=True)
class RuntimeConsistencyResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checkpoint_round: int = 0
    latest_event_round: int = 0
    completion_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def runtime_consistency_predicate(
    *,
    checkpoint: dict[str, Any] | Any,
    events: list[dict[str, Any]] | None = None,
    nexus_data: dict[str, Any] | None = None,
    job_status: str | None = None,
) -> RuntimeConsistencyResult:
    """Check the durable round/status contract written by NexusRuntime.

    This predicate is deliberately small and deterministic: the checkpoint round,
    the latest evolution progress event round, and the budget's current round
    must describe the same point in the run.  This catches the confusing
    ``progress=53`` while ``budget.current_round=0`` class of failures before
    artifacts are treated as trustworthy.
    """

    data = checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else dict(checkpoint or {})
    errors: list[str] = []
    warnings: list[str] = []
    checkpoint_round = _int(data.get("round"))
    progress_event = data.get("progress_event") if isinstance(data.get("progress_event"), dict) else {}
    progress_round = _int(progress_event.get("round"))
    if progress_event and progress_round != checkpoint_round:
        errors.append(f"checkpoint_progress_round_mismatch:{checkpoint_round}!={progress_round}")

    evolution_events = [event for event in (events or []) if isinstance(event, dict) and event.get("type") == "evolution_progress"]
    latest_event_round = _int(evolution_events[-1].get("round")) if evolution_events else progress_round
    if evolution_events and latest_event_round != checkpoint_round:
        errors.append(f"checkpoint_event_round_mismatch:{checkpoint_round}!={latest_event_round}")

    budget = data.get("budget") if isinstance(data.get("budget"), dict) else {}
    budget_round = budget.get("current_round")
    if budget_round is not None and _int(budget_round) != checkpoint_round:
        errors.append(f"checkpoint_budget_round_mismatch:{checkpoint_round}!={_int(budget_round)}")

    checkpoint_max = _int(data.get("max_rounds"))
    event_max = _int(progress_event.get("max_rounds")) if progress_event else checkpoint_max
    if progress_event and event_max and checkpoint_max and event_max != checkpoint_max:
        errors.append(f"checkpoint_event_max_rounds_mismatch:{checkpoint_max}!={event_max}")

    completion_status = _completion_status(nexus_data, budget)
    if completion_status and completion_status not in KNOWN_STATUSES:
        errors.append(f"unknown_completion_status:{completion_status}")
    if job_status:
        normalized_job = str(job_status)
        if normalized_job in TERMINAL_STATUSES and completion_status and normalized_job != completion_status:
            errors.append(f"job_status_completion_mismatch:{normalized_job}!={completion_status}")
        elif normalized_job == "running" and completion_status in TERMINAL_STATUSES:
            warnings.append(f"job_status_still_running_after_terminal_completion:{completion_status}")

    return RuntimeConsistencyResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        checkpoint_round=checkpoint_round,
        latest_event_round=latest_event_round,
        completion_status=completion_status,
    )


def assert_runtime_consistency(**kwargs: Any) -> RuntimeConsistencyResult:
    result = runtime_consistency_predicate(**kwargs)
    if not result.passed:
        raise ValueError("runtime consistency failed: " + "; ".join(result.errors))
    return result


def _completion_status(nexus_data: dict[str, Any] | None, budget: dict[str, Any]) -> str:
    if isinstance(nexus_data, dict):
        evolution = nexus_data.get("evolution")
        if isinstance(evolution, dict):
            status = evolution.get("completion_status")
            if status:
                return str(status)
        status = nexus_data.get("completion_status")
        if status:
            return str(status)
    status = budget.get("completion_status")
    return str(status or "")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["RuntimeConsistencyResult", "assert_runtime_consistency", "runtime_consistency_predicate"]
