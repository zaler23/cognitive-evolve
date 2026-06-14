"""Cleanup/reaper helpers for stale durable steps."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .resume_planner import ResumePlanner


class DurableReaper:
    def __init__(self, root: Path | str, *, inflight_timeout_seconds: float = 900.0) -> None:
        self.planner = ResumePlanner(root, inflight_timeout_seconds=inflight_timeout_seconds)

    def reap(self) -> dict[str, Any]:
        before = self.planner.plan()
        after = self.planner.apply_stale_inflight()
        return {
            "status": after.status,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "policy": "stale_inflight_steps_become_retryable_failed_not_success",
        }


__all__ = ["DurableReaper"]
