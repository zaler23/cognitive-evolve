from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from ..durable import append_jsonl
from ..core.redaction import redact
from .session import current_llm_session

_JOURNAL_LOCK = threading.RLock()


def journal_dir() -> Path | None:
    session = current_llm_session()
    raw = session.journal_dir or os.environ.get("COGEV_LLM_JOURNAL_DIR", "").strip()
    return Path(raw) if raw else None


def write_llm_journal(record: dict[str, Any], *, raw_response: Any | None = None, parsed_response: dict[str, Any] | None = None) -> None:
    directory = journal_dir()
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    call_id = str(record.get("call_id") or uuid.uuid4())
    record = dict(record)
    with _JOURNAL_LOCK:
        if raw_response is not None:
            raw_path = directory / "raw" / f"{call_id}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(redact(safe_json(raw_response)), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            record["raw_response_path"] = str(raw_path)
        if parsed_response is not None:
            parsed_path = directory / "parsed" / f"{call_id}.json"
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_path.write_text(json.dumps(redact(parsed_response), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            record["parsed_response_path"] = str(parsed_path)
        append_jsonl(directory / "llm-calls.jsonl", redact(record))


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


__all__ = ["journal_dir", "safe_json", "write_llm_journal"]
