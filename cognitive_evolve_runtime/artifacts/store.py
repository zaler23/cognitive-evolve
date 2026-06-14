#!/usr/bin/env python3
"""Small IO, formatting, parsing, and task-store helpers."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..core import TASKS
from ..core.redaction import redact
from ..durable.file_lock import atomic_write_json


def ensure_dirs() -> None:
    TASKS.mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "task"


def latest_task_dir() -> Path | None:
    latest = TASKS / "latest"
    if latest.exists():
        return latest.resolve() if latest.is_symlink() else latest
    pointer = TASKS / "LATEST.txt"
    if pointer.exists():
        return TASKS / pointer.read_text(encoding="utf-8").strip()
    return None


def _first_nonempty_line(text: str, default: str = "") -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("# "):
            return clean[2:].strip() or default
    for line in text.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            return clean
    return default


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def _bullet_lines(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)


def _format_contract_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _section_presence(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _ok(condition: bool, message: str) -> bool:
    prefix = "OK" if condition else "FAIL"
    print(f"[{prefix}] {message}")
    return condition


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _task_dir(path: str | None) -> Path:
    return Path(path) if path else (latest_task_dir() or TASKS / "latest")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    atomic_write_json(path, data, sort_keys=False)


def _append_trace(task_dir: Path, event: str, payload: dict) -> None:
    trace = task_dir / "trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    record = redact({"time": _now(), "event": event, **payload})
    with trace.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _toml_array_block(text: str, key: str) -> str:
    lines = text.splitlines()
    prefix = f"{key} = ["
    for index, line in enumerate(lines):
        if line.strip() == prefix:
            block: list[str] = []
            for item in lines[index + 1 :]:
                if item.strip() == "]":
                    break
                block.append(item)
            return "\n".join(block)
        if line.strip().startswith(prefix) and line.strip().endswith("]"):
            return line.strip()[len(prefix) : -1]
    return ""


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _declares_command(command: str, entries: list[str]) -> bool:
    return any(command == entry or command.endswith(entry) or entry.endswith(command) for entry in entries)


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _normalize_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def ok(condition: bool, message: str) -> bool:
    return _ok(condition, message)
