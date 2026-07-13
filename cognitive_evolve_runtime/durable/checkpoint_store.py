"""Atomic checkpoint storage for durable steps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .file_lock import atomic_write_json
from .idempotency import stable_hash
from .step_state import StepStatus, utc_now


class CheckpointStore:
    """Round/step checkpoint layout with tmp-write validation and atomic rename."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def step_dir(self, round_id: str, step_id: str, step_name: str | None = None) -> Path:
        safe_round = _safe_component(round_id)
        safe_step = _safe_component(step_id)
        safe_name = _safe_component(step_name or step_id)
        return self.root / ".rounds" / safe_round / "steps" / f"{safe_step}_{safe_name}"

    def write_input(self, round_id: str, step_id: str, step_name: str, input_data: dict[str, Any], *, run_id: str = "run") -> StepStatus:
        directory = self.step_dir(round_id, step_id, step_name)
        directory.mkdir(parents=True, exist_ok=True)
        self.atomic_write_json(directory / "input.json", input_data)
        status = self.read_status(round_id, step_id, step_name) or StepStatus(
            run_id=run_id,
            round_id=round_id,
            step_id=step_id,
            step_name=step_name,
            state="created",
            input_hash=stable_hash(input_data),
        )
        if not status.input_hash:
            status.input_hash = stable_hash(input_data)
        status.updated_at = utc_now()
        self.write_status(status)
        return status

    def read_input(self, round_id: str, step_id: str, step_name: str) -> dict[str, Any] | None:
        return self.read_json(self.step_dir(round_id, step_id, step_name) / "input.json")

    def write_status(self, status: StepStatus) -> None:
        self.atomic_write_json(self.step_dir(status.round_id, status.step_id, status.step_name) / "status.json", status.to_dict())

    def read_status(self, round_id: str, step_id: str, step_name: str) -> StepStatus | None:
        data = self.read_json(self.step_dir(round_id, step_id, step_name) / "status.json")
        return StepStatus.from_dict(data) if isinstance(data, dict) and data else None

    def write_inflight_status(self, status: StepStatus) -> StepStatus:
        status.attempt += 1
        status.transition("inflight", reason="step_started")
        self.write_status(status)
        return status

    def commit_output(self, status: StepStatus, output_data: dict[str, Any]) -> StepStatus:
        directory = self.step_dir(status.round_id, status.step_id, status.step_name)
        directory.mkdir(parents=True, exist_ok=True)
        self.atomic_write_json(directory / "output.json", output_data)
        status.output_hash = stable_hash(output_data)
        status.transition("committed", reason="output_validated_and_atomically_committed")
        self.write_status(status)
        return status

    def read_output(self, round_id: str, step_id: str, step_name: str) -> dict[str, Any] | None:
        # Never read output.tmp.json as success.
        return self.read_json(self.step_dir(round_id, step_id, step_name) / "output.json")

    def mark_failed(self, status: StepStatus, *, retryable: bool, error: dict[str, Any]) -> StepStatus:
        status.transition("retryable_failed" if retryable else "terminal_failed", reason=error.get("message", "step_failed"), error=error)
        self.write_status(status)
        return status

    def iter_statuses(self) -> list[StepStatus]:
        statuses: list[StepStatus] = []
        for path in sorted(self.root.glob(".rounds/*/steps/*/status.json")):
            data = self.read_json(path)
            if isinstance(data, dict) and data:
                statuses.append(StepStatus.from_dict(data))
        return statuses

    def atomic_write_json(self, path: Path, data: dict[str, Any]) -> None:
        atomic_write_json(path, data, sort_keys=True, allow_cycles=True)

    def read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None


def _safe_component(value: str | int | None) -> str:
    raw = str(value or "x")
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw.strip())
    return out[:80] or "x"


__all__ = ["CheckpointStore"]
