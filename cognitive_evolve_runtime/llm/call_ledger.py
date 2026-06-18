"""Durable LLM call ledger.

The existing inflight registry is process-local.  This ledger records boundary
states as JSONL so resume logic can explain unattached completed calls without
counting them as rounds by accident.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.durable import append_jsonl, stable_hash
from cognitive_evolve_runtime.llm.journal import journal_dir

CALL_LEDGER_ENV = "COGEV_LLM_CALL_LEDGER"
_LOCK = threading.RLock()


def call_ledger_path() -> Path | None:
    configured = str(os.environ.get(CALL_LEDGER_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    directory = journal_dir()
    return (directory / "llm-call-ledger.jsonl") if directory is not None else None


def record_call_state(status: str, *, call_id: str, request_type: str = "", request_hash: str = "", round_id: str = "", step_id: str = "", extra: dict[str, Any] | None = None) -> None:
    path = call_ledger_path()
    if path is None:
        return
    event_time = time.time()
    if extra and "event_time" in extra:
        try:
            event_time = float(extra["event_time"])
        except (TypeError, ValueError):
            event_time = time.time()
    record = {
        "status": str(status),
        "call_id": str(call_id),
        "request_type": str(request_type or ""),
        "request_hash": str(request_hash or ""),
        "round_id": str(round_id or ""),
        "step_id": str(step_id or ""),
        "event_time": event_time,
    }
    if extra:
        for k, v in extra.items():
            key = str(k)
            if key == "event_time":
                continue
            if key in {"status", "call_id"}:
                record[f"extra_{key}"] = v
            else:
                record[key] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        append_jsonl(path, record)


def ledger_summary(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else call_ledger_path()
    if target is None or not target.exists():
        return {"status_counts": {}, "unattached_completed_count": 0, "discarded_after_stop_count": 0, "ledger_digest": ""}
    counts: dict[str, int] = {}
    last_attached = ""
    discarded = 0
    last_status_by_call: dict[str, str] = {}
    started_at_by_call: dict[str, float] = {}
    terminal_at_by_call: dict[str, float] = {}
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for raw in fh:
            digest.update(raw)
            try:
                import json

                item = json.loads(raw.decode("utf-8"))
            except Exception:
                counts["corrupt"] = counts.get("corrupt", 0) + 1
                continue
            status = str(item.get("status") or "")
            call = str(item.get("call_id") or "")
            try:
                event_time = float(item.get("event_time"))
            except (TypeError, ValueError):
                event_time = 0.0
            counts[status] = counts.get(status, 0) + 1
            if call:
                last_status_by_call[call] = status
                if status == "started" and event_time > 0:
                    started_at_by_call[call] = event_time
                elif status in {"completed", "failed", "provider_unavailable", "retryable_failed", "discarded_after_stop"} and event_time > 0:
                    terminal_at_by_call[call] = event_time
            if status == "attached_to_round":
                last_attached = call or last_attached
            elif status == "discarded_after_stop":
                discarded += 1
    unattached = sum(1 for status in last_status_by_call.values() if status == "completed")
    max_overlap, interval_count = _max_overlap(started_at_by_call, terminal_at_by_call)
    return {
        "status_counts": counts,
        "last_attached_call_id": last_attached,
        "unattached_completed_count": unattached,
        "discarded_after_stop_count": discarded,
        "max_observed_concurrent_calls": max_overlap,
        "completed_interval_count": interval_count,
        "ledger_digest": "sha256:" + digest.hexdigest(),
        "ledger_key": stable_hash({"counts": counts, "last_attached": last_attached})[:16],
    }


def _max_overlap(started_at_by_call: dict[str, float], terminal_at_by_call: dict[str, float]) -> tuple[int, int]:
    events: list[tuple[float, int]] = []
    for call_id, started in started_at_by_call.items():
        ended = terminal_at_by_call.get(call_id)
        if ended is None or ended < started:
            continue
        events.append((started, 1))
        events.append((ended, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active = max(0, active + delta)
        max_active = max(max_active, active)
    return max_active, len(events) // 2


__all__ = ["CALL_LEDGER_ENV", "call_ledger_path", "ledger_summary", "record_call_state"]
