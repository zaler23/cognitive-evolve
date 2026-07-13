"""Optional runtime prompt audit sink."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
PROMPT_AUDIT_ENV = "COGEV_PROMPT_AUDIT_PATH"


def maybe_record_prompt_audit(request_type: str, prompt_view: Any, *, metadata: dict[str, Any] | None = None) -> None:
    path = _audit_path(metadata)
    if path is None:
        return
    payload = getattr(prompt_view, "payload", {})
    view_metadata = getattr(prompt_view, "metadata", {})
    record = {
        "request_type": str(request_type),
        "prompt_metadata": dict(view_metadata) if isinstance(view_metadata, dict) else {},
        "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
        "sent_payload_chars": int(view_metadata.get("sent_payload_chars", 0)) if isinstance(view_metadata, dict) else 0,
        "verification_regime_present": bool(isinstance(payload, dict) and payload.get("verification_regime") is not None),
        "forbidden_strength_shortcuts_present": _contains_forbidden_strength_shortcut(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    with _LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _audit_path(metadata: dict[str, Any] | None) -> Path | None:
    configured = ""
    if isinstance(metadata, dict):
        configured = str(metadata.get("prompt_audit_path") or "")
    configured = configured or str(os.environ.get(PROMPT_AUDIT_ENV) or "")
    if not configured:
        return None
    return Path(configured).expanduser()


def _contains_forbidden_strength_shortcut(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"strength_contribution", "legacy_strength", "measured_strength", "measured_strength_value"}:
                return True
            if key == "replayable" and any(k in value for k in ("strength", "strength_value", "measured_strength")):
                return True
            if _contains_forbidden_strength_shortcut(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_strength_shortcut(item) for item in value)
    return False


__all__ = ["PROMPT_AUDIT_ENV", "maybe_record_prompt_audit"]
