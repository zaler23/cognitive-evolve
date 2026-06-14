from __future__ import annotations

import random
import re
import time
from typing import Any

from .env import (
    LLM_EMPTY_RETRY_PROMPT_CHARS_ENV,
    LLM_RETRY_ATTEMPTS_ENV,
    LLM_RETRY_BASE_SLEEP_ENV,
    LLM_RETRY_JITTER_ENV,
    LLM_RETRY_MAX_SLEEP_ENV,
    LLM_RETRY_MAX_TOKENS_ENV,
    LLM_TIMEOUT_ENV,
    LLMResponseError,
    env_float,
    env_int,
)
from .governor import estimate_request_tokens, llm_governor, llm_governor_status
from .session import _LAST_RETRY_HISTORY


def retry_attempts() -> int:
    return max(1, env_int(LLM_RETRY_ATTEMPTS_ENV, 5))


def timeout_seconds() -> float:
    return max(1.0, env_float(LLM_TIMEOUT_ENV, 60.0))


def retry_after_seconds(exc: Exception) -> float | None:
    candidates: list[Any] = []
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        candidates.extend([headers.get("Retry-After"), headers.get("retry-after")])
    candidates.extend([getattr(exc, "retry_after", None), getattr(exc, "retry_after_seconds", None)])
    text = str(exc)
    match = re.search(r"retry[- ]after[:= ]+([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if match:
        candidates.append(match.group(1))
    for value in candidates:
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def provider_error_category(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        code = None
    if (
        "empty_assistant_content" in text
        or "empty assistant content" in text
        or "empty assistant message" in text
        or "empty_content" in text
    ):
        return "empty_assistant_content"
    if (
        "truncated" in text
        or "finish_reason=length" in text
        or "finish reason length" in text
        or "finish_reason length" in text
    ):
        return "truncated_response"
    if (
        code == 429
        and (
            "quota" in text
            or "resource_exhausted" in text
            or "resource exhausted" in text
            or "insufficient_quota" in text
            or "billing" in text
        )
    ):
        return "quota_exhausted"
    if code == 429 or "429" in text or "rate limit" in text or "ratelimit" in name:
        return "rate_limit_429"
    if code is not None and 500 <= code <= 599:
        return "provider_5xx"
    if "timeout" in text or "timeout" in name:
        return "timeout"
    if "connection" in text or "network" in text or "temporar" in text:
        return "network_or_transient"
    if (
        "budget" in text
        and ("exhausted" in text or "exceeded" in text or "already exhausted" in text)
    ) or "cost budget" in text or "stage budget" in text:
        return "budget_exhausted"
    if "authentication" in text or "api key" in text or "permission" in text or "forbidden" in text:
        return "configuration_or_auth"
    if "badrequest" in name or "invalid request" in text or (code is not None and 400 <= code <= 499):
        return "non_retryable_request"
    if isinstance(exc, LLMResponseError):
        return "response_json_or_contract_error"
    return "unknown"


def is_retryable_provider_error(exc: Exception) -> bool:
    return provider_error_category(exc) in {
        "rate_limit_429",
        "timeout",
        "network_or_transient",
        "provider_5xx",
        "empty_assistant_content",
        "truncated_response",
        "unknown",
        "response_json_or_contract_error",
    }


def retry_sleep_seconds(exc: Exception, attempts: int) -> float:
    retry_after = retry_after_seconds(exc)
    max_sleep = max(1.0, env_float(LLM_RETRY_MAX_SLEEP_ENV, 60.0))
    base_sleep = max(0.0, env_float(LLM_RETRY_BASE_SLEEP_ENV, 1.0))
    jitter = max(0.0, env_float(LLM_RETRY_JITTER_ENV, 0.0))
    base = retry_after if retry_after is not None else min(max_sleep, base_sleep * (2 ** max(0, attempts - 1)))
    if jitter and base > 0:
        base += random.uniform(0.0, min(jitter, max(0.0, base * 0.25)))
    return min(max_sleep, max(0.0, base))


def completion_with_retry(completion_fn: Any, **kwargs: Any) -> tuple[Any, int]:
    max_attempts = max(1, int(kwargs.pop("_retry_max_attempts", retry_attempts()) or 1))
    attempt_offset = max(0, int(kwargs.pop("_retry_attempt_offset", 0) or 0))
    last_category = "unknown"
    retry_history: list[dict[str, Any]] = []
    _LAST_RETRY_HISTORY.set(retry_history)
    for attempts in range(1, max_attempts + 1):
        governor_snapshot: dict[str, Any] | None = None
        try:
            estimated_tokens = estimate_request_tokens(kwargs)
            with llm_governor().acquire(estimated_tokens=estimated_tokens) as governor_snapshot:
                return completion_fn(**kwargs), attempts
        except Exception as exc:
            last_category = provider_error_category(exc)
            retryable = is_retryable_provider_error(exc)
            global_attempt = attempt_offset + attempts
            record = {
                "attempt": global_attempt,
                "local_attempt": attempts,
                "category": last_category,
                "retryable": retryable,
                "retry_after_seconds": retry_after_seconds(exc),
                "message": str(exc)[:500],
                "governor": governor_snapshot or llm_governor_status(),
            }
            if attempts >= max_attempts or not retryable:
                retry_history.append({**record, "slept_seconds": 0.0, "final": True})
                _LAST_RETRY_HISTORY.set(retry_history)
                raise
            retry_mutation = _mutate_retry_kwargs(kwargs, category=last_category)
            slept = retry_sleep_seconds(exc, global_attempt)
            retry_history.append({**record, "slept_seconds": round(slept, 3), "final": False, "retry_mutation": retry_mutation})
            _LAST_RETRY_HISTORY.set(retry_history)
            if slept > 0:
                time.sleep(slept)
    raise LLMResponseError(f"LLM provider call exhausted retry policy without returning; last_error_category={last_category}.")


def _mutate_retry_kwargs(kwargs: dict[str, Any], *, category: str) -> dict[str, Any]:
    """Change the next retry shape when the previous attempt returned no usable text.

    Replaying the identical request after an empty assistant response or a
    length-truncated response tends to reproduce the same failure on long Nexus
    calls.  The retry keeps the same semantic task, but gives the provider more
    output room and, when the prompt is oversized, sends a compact excerpt rather
    than the exact same giant message.
    """

    if category not in {"empty_assistant_content", "truncated_response"}:
        return {}
    mutation: dict[str, Any] = {}
    cap = max(1, env_int(LLM_RETRY_MAX_TOKENS_ENV, 32768))
    current = _positive_int(kwargs.get("max_tokens"), default=4096)
    next_tokens = min(cap, max(current + max(1024, current), 8192))
    if next_tokens > current:
        kwargs["max_tokens"] = next_tokens
        mutation["max_tokens"] = next_tokens
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        max_prompt_chars = max(1024, env_int(LLM_EMPTY_RETRY_PROMPT_CHARS_ENV, 24000))
        shrunk, changed = _shrink_retry_messages(messages, max_chars=max_prompt_chars)
        if changed:
            kwargs["messages"] = shrunk
            mutation["prompt_chars"] = max_prompt_chars
    return mutation


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _shrink_retry_messages(messages: list[Any], *, max_chars: int) -> tuple[list[Any], bool]:
    total = sum(len(str(item.get("content") or "")) for item in messages if isinstance(item, dict))
    if total <= max_chars:
        return messages, False
    out: list[Any] = []
    remaining = max_chars
    changed = False
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            out.append(item)
            continue
        fixed = dict(item)
        content = str(fixed.get("content") or "")
        # Preserve short system/developer instructions first; trim bulky user
        # payloads from the middle while keeping both beginning and ending
        # context for JSON/schema repairs.
        budget = max(256, remaining // max(1, len(messages) - index))
        if len(content) > budget:
            head = content[: max(128, budget // 2)]
            tail = content[-max(128, budget // 3) :]
            fixed["content"] = head + "\n\n[retry prompt excerpted after empty/truncated provider response]\n\n" + tail
            changed = True
        remaining = max(0, remaining - len(str(fixed.get("content") or "")))
        out.append(fixed)
    return out, changed
