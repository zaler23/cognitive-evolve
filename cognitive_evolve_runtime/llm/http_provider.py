"""Direct OpenAI-compatible HTTP provider.

This provider intentionally avoids LiteLLM and the OpenAI Python SDK.  Some
OpenAI-compatible endpoints are stable for plain HTTP requests but reject
or mishandle SDK/LiteLLM-shaped requests.  The runtime only needs the small
``choices[0].message.content`` and ``usage`` surface, so a direct transport is
both simpler and more predictable for those endpoints.
"""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .env import LLM_API_BASE_ENV, LLM_API_KEY_ENV, LLM_BASE_URL_ENV, LLMConfigurationError
from .provider_interface import LLMProviderResult
from .retry import completion_with_retry


class DirectHTTPProviderError(RuntimeError):
    """HTTP provider error with a status code for retry classification."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DirectHTTPSettings:
    api_base: str
    api_key: str


def direct_http_settings(*, api_base_override: str | None = None) -> DirectHTTPSettings:
    api_base = str(api_base_override or "").strip() or os.environ.get(LLM_API_BASE_ENV, "").strip() or os.environ.get(LLM_BASE_URL_ENV, "").strip()
    if not api_base:
        raise LLMConfigurationError(f"direct_http provider requires {LLM_API_BASE_ENV}=<openai-compatible-base-url>.")
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    return DirectHTTPSettings(api_base=api_base.rstrip("/"), api_key=api_key)


def normalize_direct_http_model(model: str) -> str:
    """Return the model id expected by an OpenAI-compatible HTTP endpoint.

    Operators often use ``openai/<model>`` to force LiteLLM's OpenAI transport.
    A direct OpenAI-compatible endpoint usually expects the raw model id, so we
    strip only that routing prefix while preserving all other ids verbatim.
    """

    model = str(model or "").strip()
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _to_response_object(raw: dict[str, Any]) -> Any:
    choices = []
    for choice in raw.get("choices") or []:
        message = choice.get("message") or {}
        content = _choice_message_content(choice)
        choices.append(
            SimpleNamespace(
                index=choice.get("index"),
                finish_reason=choice.get("finish_reason"),
                message=SimpleNamespace(
                    role=message.get("role"),
                    content=content,
                ),
            )
        )
    return SimpleNamespace(
        id=raw.get("id"),
        object=raw.get("object"),
        created=raw.get("created"),
        model=raw.get("model"),
        choices=choices,
        usage=raw.get("usage") or {},
        raw=raw,
    )


def _message_content(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices") or []
    if not choices:
        return None
    first = choices[0] if isinstance(choices[0], dict) else {}
    return _choice_message_content(first)


def _finish_reason(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "").strip().lower()


def _choice_message_content(choice: dict[str, Any]) -> str | None:
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if isinstance(reasoning, dict):
        nested_content = reasoning.get("content") or reasoning.get("text")
        if isinstance(nested_content, str) and nested_content.strip():
            return nested_content
        if reasoning:
            return json.dumps(reasoning, ensure_ascii=False, default=str)
    return content if isinstance(content, str) else None


class DirectHTTPProvider:
    provider_id = "direct_http"

    def complete_json(self, **kwargs: Any) -> LLMProviderResult:
        result, attempts = completion_with_retry(self._completion, **kwargs)
        return LLMProviderResult(response=result, attempts=attempts, estimated_cost_usd=None)

    def _completion(self, **kwargs: Any) -> Any:
        settings = direct_http_settings(api_base_override=str(kwargs.get("api_base") or "").strip() or None)
        url = f"{settings.api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": normalize_direct_http_model(str(kwargs.get("model") or "")),
            "messages": kwargs.get("messages") or [],
        }
        for key in ("temperature", "max_tokens", "response_format", "stream", "seed"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(kwargs.get("timeout") or 28.0)) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise DirectHTTPProviderError(f"HTTP {exc.code} from direct_http provider: {detail}", status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise DirectHTTPProviderError(f"Network error from direct_http provider at {url}: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise DirectHTTPProviderError(f"Timeout from direct_http provider at {url}") from exc
        except Exception as exc:
            raise DirectHTTPProviderError(f"direct_http provider failed: {exc}") from exc
        if _finish_reason(raw) == "length":
            raise DirectHTTPProviderError(
                "TRUNCATED assistant content from direct_http provider; finish_reason=length",
                status_code=None,
            )
        if not (_message_content(raw) or "").strip():
            # Some OpenAI-compatible endpoints occasionally return HTTP 200 with an empty
            # assistant message.  Treat that as a transient transport failure,
            # not as a semantic JSON failure, so completion_with_retry can
            # issue a fresh request before the higher-level JSON repair loop
            # spends an attempt on an empty string.
            raise DirectHTTPProviderError("EMPTY_ASSISTANT_CONTENT from direct_http provider", status_code=502)
        return _to_response_object(raw)


__all__ = [
    "DirectHTTPProvider",
    "DirectHTTPProviderError",
    "DirectHTTPSettings",
    "direct_http_settings",
    "_message_content",
    "normalize_direct_http_model",
]
