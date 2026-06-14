"""LiteLLM provider implementation for the LLM interface."""
from __future__ import annotations

import os
from typing import Any

from .env import LLM_API_BASE_ENV, LLM_API_KEY_ENV, LLM_BASE_URL_ENV, LLMConfigurationError
from .provider_interface import LLMProviderResult
from .retry import completion_with_retry


def litellm_provider_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    api_base = os.environ.get(LLM_API_BASE_ENV, "").strip() or os.environ.get(LLM_BASE_URL_ENV, "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


class LiteLLMProvider:
    provider_id = "litellm"

    def complete_json(self, **kwargs: Any) -> LLMProviderResult:
        try:
            from litellm import completion, completion_cost
        except Exception as exc:  # pragma: no cover - only hit when dependency missing in a real install
            raise LLMConfigurationError("litellm is required for LiteLLM execution. Install project dependencies first.") from exc

        result, attempts = completion_with_retry(completion, **kwargs, **litellm_provider_kwargs())
        try:
            estimated_cost = float(completion_cost(completion_response=result) or 0.0)
        except Exception:
            estimated_cost = None
        return LLMProviderResult(response=result, attempts=attempts, estimated_cost_usd=estimated_cost)


__all__ = ["LiteLLMProvider", "litellm_provider_kwargs"]
