from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from ..artifacts.store import _write_json
from ..core.redaction import redact
from ..durable import ResumePlanner
from .config import get_service_config
from .models import ChatCompletionRequest
from .prompting import build_one_shot_prompt

logger = logging.getLogger(__name__)

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_FUTURES_LOCK = threading.Lock()
_JOB_FUTURES: dict[str, Future[Any]] = {}
_TERMINAL_JOB_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "rejected",
    "interrupted",
    "interrupted_checkpointed",
    "needs_continuation",
    "failed_verification",
    "paused_quota",
}
_LEGACY_COMPLETED_STATUSES = {"best" + "_current" + "_route", "route" + "_incomplete"}


def _normalize_public_job_status(status: Any) -> str:
    text = str(status or "unknown")
    return "completed" if text in _LEGACY_COMPLETED_STATUSES else text


class JobQueue:
    """Stateless facade over the API job registry.

    This is a visibility and testability surface, not a second runtime
    authority.  Instances do not own state: writes go through ``_set_job``,
    reads go through ``_get_job``, and snapshots copy the existing ``_JOBS``
    registry under ``_JOBS_LOCK``.  The facade intentionally avoids destructive
    pop/delete semantics so status polling and artifact rehydration keep their
    existing lifecycle.
    """

    def push(self, job: dict[str, Any]) -> dict[str, Any]:
        """Create or update a job using the existing registry authority."""

        if not isinstance(job, dict):
            raise TypeError("JobQueue.push requires a job mapping")
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            raise ValueError("JobQueue.push requires a non-empty job id")
        updates = dict(job)
        updates.pop("id", None)
        logger.debug("JobQueue.push id=%s keys=%s", job_id, sorted(updates))
        return _set_job(job_id, **updates)

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return the current job snapshot through the existing read path."""

        normalized = str(job_id or "").strip()
        if not normalized:
            return None
        logger.debug("JobQueue.get id=%s", normalized)
        return _get_job(normalized)

    def snapshot(self) -> list[dict[str, Any]]:
        """Copy all in-memory job snapshots without exposing mutable state."""

        logger.debug("JobQueue.snapshot")
        with _JOBS_LOCK:
            return [dict(job) for job in _JOBS.values()]




def _job_has_result_payload(job: dict[str, Any]) -> bool:
    if not isinstance(job, dict):
        return False
    status = _normalize_public_job_status(job.get("status"))
    terminal = status in _TERMINAL_JOB_STATUSES or status in {"checkpointed"}
    if not terminal:
        return False
    if str(job.get("answer") or "").strip():
        return True
    data = job.get("nexus_data") if isinstance(job.get("nexus_data"), dict) else {}
    synthesis = dict(dict(data.get("evolution") or {}).get("synthesis") or {}) if isinstance(data, dict) else {}
    return bool(str(synthesis.get("final_answer") or "").strip())

def _job_public(job: dict[str, Any], *, include_answer: bool = True) -> dict[str, Any]:
    durable_plan = _durable_resume_snapshot(job)
    payload = {
        "id": job.get("id"),
        "object": "cogev.job",
        "status": _normalize_public_job_status(job.get("status")),
        "created": job.get("created"),
        "updated": job.get("updated"),
        "heartbeat": job.get("heartbeat"),
        "model": job.get("model"),
        "task_dir": job.get("task_dir"),
        "artifact_root": job.get("artifact_root"),
        "error": job.get("error"),
        "cancellation_requested": bool(job.get("cancellation_requested")),
        "executor_tracked": _job_future_tracked(str(job.get("id") or "")),
        "durable_resume_plan": durable_plan,
        "resume_available": (durable_plan or {}).get("status") in {"resume_available", "all_committed"} if isinstance(durable_plan, dict) else False,
    }
    if include_answer and _job_has_result_payload(job):
        payload["answer"] = job.get("answer", "")
        from .payloads import _nexus_actual_rounds, _nexus_answer_produced, _nexus_completion_status, _nexus_objective_solved, _nexus_verification_passed

        data = job.get("nexus_data") if isinstance(job.get("nexus_data"), dict) else {}
        payload["cognitive_evolve"] = {
            "runtime_path": "nexus",
            "actual_rounds": _nexus_actual_rounds(data),
            "verification_passed": _nexus_verification_passed(data),
            "objective_solved": _nexus_objective_solved(data),
            "objective_solved_semantics": "not_claimed_without_user_or_external_verification",
            "answer_produced": _nexus_answer_produced(data),
            "completion_status": _nexus_completion_status(data),
        }
    return redact(payload)


def _durable_resume_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Return an honest, read-only resume capability snapshot.

    The API used to surface ``ResumePlanner`` output as if an operator could
    resume any job from the public endpoint.  Until a full API resume command is
    implemented, expose only artifact/checkpoint discoverability and make the
    limitation explicit.
    """

    root_raw = job.get("artifact_root") or job.get("task_dir")
    if not root_raw:
        return {"status": "not_available", "actions": [], "api_resume_supported": False}
    try:
        root = Path(str(root_raw))
        planner = ResumePlanner(root).plan().to_dict()
    except Exception:
        planner = {"status": "unknown", "actions": []}
    return {
        "status": "snapshot_only",
        "api_resume_supported": True,
        "reason": "POST /v1/cogev/jobs/{id}/resume can continue from a persisted Nexus checkpoint when one exists",
        "artifact_root": str(root_raw),
        "planner_status": planner.get("status"),
        "planner_actions": list(planner.get("actions") or [])[:8] if isinstance(planner, dict) else [],
    }



def _set_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _JOBS_LOCK:
        _prune_jobs_locked()
        logger.debug("job_registry.set id=%s keys=%s", job_id, sorted(updates))
        job = _JOBS.setdefault(job_id, {"id": job_id, "created": _now()})
        job.update(updates)
        job["updated"] = _now()
        if job.get("status") in {"queued", "running", "cancellation_requested"}:
            job["heartbeat"] = job["updated"]
        snapshot = dict(job)
    root_raw = snapshot.get("artifact_root") or snapshot.get("task_dir")
    if root_raw:
        try:
            _write_json(Path(str(root_raw)) / "job-status.json", _job_public(snapshot))
        except Exception:
            logger.warning("job_registry.persist_failed id=%s", job_id, exc_info=True)
    return snapshot



def _get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        _prune_jobs_locked()
        job = _JOBS.get(job_id)
        logger.debug("job_registry.get id=%s hit=%s", job_id, bool(job))
        if job:
            return dict(job)
    rehydrated = _rehydrate_job_from_artifact(job_id)
    if rehydrated:
        with _JOBS_LOCK:
            _JOBS.setdefault(job_id, dict(rehydrated))
        return rehydrated
    return None



def _register_job_future(job_id: str, future: Future[Any]) -> None:
    with _FUTURES_LOCK:
        _JOB_FUTURES[job_id] = future


def _pop_job_future(job_id: str) -> Future[Any] | None:
    with _FUTURES_LOCK:
        return _JOB_FUTURES.pop(job_id, None)


def _cancel_job_future(job_id: str) -> bool:
    with _FUTURES_LOCK:
        future = _JOB_FUTURES.get(job_id)
    if future is None:
        return False
    return future.cancel()


def _job_future_tracked(job_id: str) -> bool:
    if not job_id:
        return False
    with _FUTURES_LOCK:
        return job_id in _JOB_FUTURES



def _rehydrate_job_from_artifact(job_id: str) -> dict[str, Any] | None:
    """Read persisted job status after an API process restart.

    Long-running jobs write ``job-status.json`` in their task directory on every
    status transition.  If the in-memory registry is empty after a restart, the
    status endpoint can still serve a read-only snapshot from disk instead of
    returning a false 404 while artifacts remain available.
    """

    if not job_id or Path(job_id).name != job_id:
        return None
    try:
        root = get_service_config().api_task_root / job_id
        status_path = root / "job-status.json"
        if not status_path.exists():
            return None
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(status, dict):
        return None
    job: dict[str, Any] = {
        "id": status.get("id") or job_id,
        "status": status.get("status", "unknown"),
        "created": status.get("created"),
        "updated": status.get("updated"),
        "model": status.get("model"),
        "task_dir": status.get("task_dir") or str(root),
        "artifact_root": status.get("artifact_root") or str(root),
        "error": status.get("error"),
        "cancellation_requested": bool(status.get("cancellation_requested")),
        "answer": status.get("answer", ""),
        "rehydrated_from_artifact": True,
        "rehydrated_readonly": True,
    }
    if job["status"] in {"queued", "running", "cancellation_requested"}:
        job["status"] = "interrupted"
        job["error"] = job.get("error") or "API process restarted before this job reached a terminal state."
        job["cancellation_requested"] = False
    nexus_path = root / "nexus-runtime" / "run-result.json"
    if nexus_path.exists():
        try:
            nexus_data = json.loads(nexus_path.read_text(encoding="utf-8"))
            if isinstance(nexus_data, dict):
                job["nexus_data"] = nexus_data
                job["status"] = _status_from_nexus_data(nexus_data, fallback=str(job.get("status") or "unknown"))
                job["answer"] = job.get("answer") or str(dict(dict(nexus_data.get("evolution") or {}).get("synthesis") or {}).get("final_answer") or "")
        except Exception:
            pass
    request_path = root / "api-request.json"
    if request_path.exists():
        try:
            request_data = json.loads(request_path.read_text(encoding="utf-8"))
            if isinstance(request_data, dict):
                job["raw_request"] = request_data.get("request")
                raw_request = request_data.get("request") if isinstance(request_data.get("request"), dict) else {}
                if raw_request:
                    try:
                        job["prompt"] = build_one_shot_prompt(ChatCompletionRequest(**raw_request).messages)
                    except Exception:
                        pass
        except Exception:
            pass
    return job



def _now() -> int:
    return int(time.time())



def _task_dir_for_request(root: Path, request_id: str) -> Path:
    task_dir = root / request_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _status_from_nexus_data(nexus_data: dict[str, Any], *, fallback: str = "completed") -> str:
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    completion = str(evolution.get("completion_status") or synthesis.get("completion_status") or "").strip()
    if completion in _LEGACY_COMPLETED_STATUSES:
        return "completed"
    if completion in {"needs_continuation", "failed_verification", "failed", "interrupted_checkpointed", "paused_quota"}:
        return completion
    if evolution.get("interrupted"):
        return "interrupted"
    return fallback or "completed"


def _prune_jobs_locked(*, now: int | None = None) -> int:
    """Remove old terminal in-memory job snapshots.

    Durable ``job-status.json`` and run artifacts remain on disk; this only
    bounds the process-local registry so long-lived API servers do not retain
    every completed job forever.
    """

    config = get_service_config()
    now = int(now or _now())
    ttl = int(config.job_ttl_seconds)
    max_jobs = int(config.max_tracked_jobs)
    removed = 0
    for job_id, job in list(_JOBS.items()):
        status = str(job.get("status") or "")
        updated = int(job.get("updated") or job.get("created") or now)
        if status in _TERMINAL_JOB_STATUSES and now - updated > ttl:
            _JOBS.pop(job_id, None)
            removed += 1
    overflow = max(0, len(_JOBS) - max_jobs)
    if overflow:
        candidates = sorted(
            (
                (int(job.get("updated") or job.get("created") or 0), job_id)
                for job_id, job in _JOBS.items()
                if str(job.get("status") or "") in _TERMINAL_JOB_STATUSES
            ),
            key=lambda item: item[0],
        )
        for _, job_id in candidates[:overflow]:
            _JOBS.pop(job_id, None)
            removed += 1
    return removed


__all__ = [
    'JobQueue',
    '_job_public',
    '_job_has_result_payload',
    '_durable_resume_snapshot',
    '_set_job',
    '_get_job',
    '_register_job_future',
    '_pop_job_future',
    '_cancel_job_future',
    '_rehydrate_job_from_artifact',
    '_now',
    '_task_dir_for_request',
    '_status_from_nexus_data',
    '_prune_jobs_locked',
]
