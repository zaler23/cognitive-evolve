from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_CIRCULAR_REF = "[CIRCULAR]"


def json_ready(value: Any) -> Any:
    return _json_ready(value, seen=set())


def _json_ready(value: Any, *, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            return {field.name: _json_ready(getattr(value, field.name), seen=seen) for field in fields(value)}
        finally:
            seen.remove(obj_id)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            return {str(k): _json_ready(v, seen=seen) for k, v in value.items()}
        finally:
            seen.remove(obj_id)
    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            return [_json_ready(v, seen=seen) for v in value]
        finally:
            seen.remove(obj_id)
    if isinstance(value, set):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            items = [_json_ready(v, seen=seen) for v in value]
            return sorted(items, key=_stable_sort_key)
        finally:
            seen.remove(obj_id)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_ready(value.to_dict(), seen=seen)
        except Exception:
            pass
    return str(value)


def _stable_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    # codeql[py/weak-sensitive-data-hashing] This is deterministic content addressing for JSON state, not credential storage.
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
