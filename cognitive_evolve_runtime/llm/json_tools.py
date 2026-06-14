from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from .env import LLM_MAX_PROMPT_CHARS_ENV, LLMResponseError


def usage_dict(result: Any) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    if usage is None and isinstance(result, dict):
        usage = result.get("usage")
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = {key: getattr(usage, key, 0) for key in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_REFUSAL_RE = re.compile(
    r"\b(i\s+cannot|i\s+can't|cannot\s+fulfill|can't\s+fulfill|unable\s+to\s+comply|sorry|as\s+an\s+ai)\b",
    re.IGNORECASE,
)


def extract_json_from_text(text: str) -> dict[str, Any]:
    cleaned = str(text or "").lstrip("\ufeff").strip()
    if not cleaned:
        raise LLMResponseError("LLM response was not valid JSON: parse_hint=refusal_or_empty: empty assistant content")

    candidates = _json_candidates(cleaned)
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(decoded, dict):
            raise LLMResponseError("LLM response must be a JSON object.")
        return decoded

    hint = "refusal_or_empty" if _REFUSAL_RE.search(cleaned) else "no_json_object_found"
    detail = f"{last_error}: " if last_error is not None else ""
    raise LLMResponseError(f"LLM response was not valid JSON: parse_hint={hint}: {detail}{cleaned[:500]}")


def _json_candidates(cleaned: str) -> list[str]:
    candidates: list[str] = []
    candidates.append(cleaned)
    for match in _FENCED_JSON_RE.finditer(cleaned):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        candidate = cleaned[index : index + end].strip()
        if candidate:
            candidates.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def bounded_prompt_for_provider(prompt: str) -> tuple[str, dict[str, Any]]:
    raw_limit = os.environ.get(LLM_MAX_PROMPT_CHARS_ENV, "120000").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 120000
    original_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if limit <= 0 or len(prompt) <= limit:
        return prompt, {
            "truncated": False,
            "original_chars": len(prompt),
            "sent_chars": len(prompt),
            "max_prompt_chars": limit,
            "original_sha256": original_sha,
        }
    marker = "\n\n...[COGEV_PROMPT_MIDDLE_TRUNCATED]...\n\n"
    if limit <= len(marker) + 2:
        sent = (prompt[: max(1, limit // 2)] + prompt[-max(1, limit - max(1, limit // 2)) :])[:limit]
    else:
        head_chars = max(1, (limit - len(marker)) // 2)
        tail_chars = max(1, limit - len(marker) - head_chars)
        sent = prompt[:head_chars] + marker + prompt[-tail_chars:]
    return sent, {
        "truncated": True,
        "original_chars": len(prompt),
        "sent_chars": len(sent),
        "max_prompt_chars": limit,
        "original_sha256": original_sha,
        "sent_sha256": hashlib.sha256(sent.encode("utf-8")).hexdigest(),
        "policy": "preserve_head_and_tail_drop_middle",
    }
