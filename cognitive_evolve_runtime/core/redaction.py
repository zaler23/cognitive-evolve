"""Runtime redaction helpers for persisted diagnostics.

The runtime writes provider errors, checkpoints, events, and journals to disk.
Those records are valuable evidence, but they must not become an accidental
secret sink.  This module keeps the policy deliberately small and
domain-neutral: redact values when either the key name is secret-shaped or the
string contains a secret-shaped environment value.
"""
from __future__ import annotations

import os
import re
from typing import Any

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|token|secret|password|passwd|cookie|session[_-]?id|auth[_-]?code|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE_ENV_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|token|secret|password|passwd|cookie|auth[_-]?code|private[_-]?key)",
    re.IGNORECASE,
)
_INLINE_SECRET_RE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}|(sk-[A-Za-z0-9._-]{6,})|([A-Za-z0-9_-]{24,}\.[A-Za-z0-9._-]{12,})"
)
_NON_SECRET_DIAGNOSTIC_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "max_tokens",
    "estimated_tokens",
    "credential_configured",
    "credential_placeholder",
}


_CIRCULAR_REF = "[CIRCULAR]"


def redact(value: Any) -> Any:
    """Return a JSON-compatible copy with obvious secrets removed."""

    secrets = _secret_values_from_env()
    return _redact(value, secrets=secrets, parent_key="", seen=set())


def redact_text(text: str) -> str:
    """Redact secret-shaped substrings from text."""

    redacted = str(text)
    for secret in _secret_values_from_env():
        if secret and secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _INLINE_SECRET_RE.sub(lambda m: (m.group(1) or "") + "[REDACTED]", redacted)
    return redacted


def _redact(value: Any, *, secrets: tuple[str, ...], parent_key: str, seen: set[int]) -> Any:
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        out: dict[str, Any] = {}
        try:
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if key in _NON_SECRET_DIAGNOSTIC_KEYS:
                    out[key] = _redact(raw_value, secrets=secrets, parent_key=key, seen=seen)
                elif _SENSITIVE_KEY_RE.search(key):
                    out[key] = _redacted_sensitive_value(raw_value)
                else:
                    out[key] = _redact(raw_value, secrets=secrets, parent_key=key, seen=seen)
            return out
        finally:
            seen.remove(obj_id)
    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            return [_redact(item, secrets=secrets, parent_key=parent_key, seen=seen) for item in value]
        finally:
            seen.remove(obj_id)
    if isinstance(value, tuple):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            return [_redact(item, secrets=secrets, parent_key=parent_key, seen=seen) for item in value]
        finally:
            seen.remove(obj_id)
    if isinstance(value, set):
        obj_id = id(value)
        if obj_id in seen:
            return _CIRCULAR_REF
        seen.add(obj_id)
        try:
            redacted_items = [_redact(item, secrets=secrets, parent_key=parent_key, seen=seen) for item in value]
            return sorted(redacted_items, key=lambda item: repr(item))
        finally:
            seen.remove(obj_id)
    if isinstance(value, str):
        text = value
        for secret in secrets:
            if secret and secret in text:
                text = text.replace(secret, "[REDACTED]")
        return _INLINE_SECRET_RE.sub(lambda m: (m.group(1) or "") + "[REDACTED]", text)
    return value


def _redacted_sensitive_value(value: Any) -> Any:
    """Redact non-empty values under a secret-shaped key without inspecting them.

    Sensitive-key payloads can be nested dictionaries/lists from provider or
    ledger metadata.  The redaction decision must therefore be type-safe: only
    preserve explicit empty sentinels, and never compare arbitrary containers
    against a set.
    """

    if value is None or value == "":
        return value
    return "[REDACTED]"


def _secret_values_from_env() -> tuple[str, ...]:
    values: list[str] = []
    for key, value in os.environ.items():
        if not value or len(value) < 6:
            continue
        if _SECRET_VALUE_ENV_RE.search(key):
            values.append(value)
    # Longest first avoids partial replacement artifacts.
    return tuple(sorted(set(values), key=len, reverse=True))


__all__ = ["redact", "redact_text"]
