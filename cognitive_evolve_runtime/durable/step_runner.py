"""Idempotent durable step runner."""
from __future__ import annotations

from typing import Any, Callable

from .checkpoint_store import CheckpointStore
from .event_log import EventLog
from .idempotency import stable_hash
from .step_state import StepStatus


class StepRunner:
    def __init__(self, store: CheckpointStore, *, event_log: EventLog | None = None, run_id: str = "run") -> None:
        self.store = store
        self.event_log = event_log
        self.run_id = run_id

    def run_step(
        self,
        *,
        round_id: str,
        step_id: str,
        step_name: str,
        input_data: dict[str, Any],
        fn: Callable[[], dict[str, Any]],
        retry_exceptions: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, RuntimeError),
    ) -> dict[str, Any]:
        current_hash = stable_hash(input_data)
        status = self.store.read_status(round_id, step_id, step_name)
        output = self.store.read_output(round_id, step_id, step_name)
        if status and status.state == "committed" and status.input_hash == current_hash and isinstance(output, dict):
            self._event("step_resumed_committed", status, {"input_hash": current_hash})
            return output

        if status and status.state == "committed" and status.input_hash != current_hash:
            status = StepStatus(run_id=self.run_id, round_id=round_id, step_id=step_id, step_name=step_name, state="created", input_hash=current_hash, metadata={"previous_input_hash": status.input_hash})
            self.store.write_status(status)
        else:
            status = self.store.write_input(round_id, step_id, step_name, input_data, run_id=self.run_id)
            status.input_hash = current_hash
            self.store.write_status(status)
        if status.state in {"retryable_failed", "resumed"}:
            status.transition("ready", reason="retrying_retryable_or_resumed_step")
            self.store.write_status(status)
        elif status.state == "created":
            status.transition("ready", reason="input_checkpointed")
            self.store.write_status(status)
        self.store.write_inflight_status(status)
        self._event("step_started", status, {"attempt": status.attempt})
        try:
            result = fn()
            if not isinstance(result, dict):
                raise TypeError("durable step output must be a JSON object/dict")
        except retry_exceptions as exc:
            self.store.mark_failed(status, retryable=True, error={"type": exc.__class__.__name__, "message": str(exc)})
            self._event("step_retryable_failed", status, {"error": str(exc)})
            raise
        except BaseException as exc:
            self.store.mark_failed(status, retryable=False, error={"type": exc.__class__.__name__, "message": str(exc)})
            self._event("step_terminal_failed", status, {"error": str(exc)})
            raise
        self.store.commit_output(status, result)
        self._event("step_committed", status, {"output_hash": status.output_hash})
        return result

    def _event(self, event_type: str, status: StepStatus, payload: dict[str, Any]) -> None:
        if self.event_log is not None:
            self.event_log.append(event_type, {"state": status.state, **payload}, run_id=status.run_id, round_id=status.round_id, step_id=status.step_id)


__all__ = ["StepRunner"]
