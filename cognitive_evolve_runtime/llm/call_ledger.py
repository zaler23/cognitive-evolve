"""Durable LLM call ledger.

The existing inflight registry is process-local.  This ledger records boundary
states as JSONL so resume logic can explain unattached completed calls without
counting them as rounds by accident.
"""
from __future__ import annotations

import hashlib
import os
import threading
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
    record = {
        "status": str(status),
        "call_id": str(call_id),
        "request_type": str(request_type or ""),
        "request_hash": str(request_hash or ""),
        "round_id": str(round_id or ""),
        "step_id": str(step_id or ""),
    }
    if extra:
        for k, v in extra.items():
            key = str(k)
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
            counts[status] = counts.get(status, 0) + 1
            if call:
                last_status_by_call[call] = status
            if status == "attached_to_round":
                last_attached = call or last_attached
            elif status == "discarded_after_stop":
                discarded += 1
    unattached = sum(1 for status in last_status_by_call.values() if status == "completed")
    return {
        "status_counts": counts,
        "last_attached_call_id": last_attached,
        "unattached_completed_count": unattached,
        "discarded_after_stop_count": discarded,
        "ledger_digest": "sha256:" + digest.hexdigest(),
        "ledger_key": stable_hash({"counts": counts, "last_attached": last_attached})[:16],
    }


__all__ = ["CALL_LEDGER_ENV", "call_ledger_path", "ledger_summary", "record_call_state"]
