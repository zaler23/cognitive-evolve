from __future__ import annotations

import os
from typing import Any

from ..configuration import load_layered_config

LLM_PROVIDER_ENV = "COGEV_LLM_PROVIDER"
LLM_MODEL_ENV = "COGEV_LLM_MODEL"
LLM_FIXTURE_ENV = "COGEV_LLM_FIXTURE"
LLM_TEMPERATURE_ENV = "COGEV_LLM_TEMPERATURE"
LLM_MAX_TOKENS_ENV = "COGEV_LLM_MAX_TOKENS"
LLM_LIGHT_MAX_TOKENS_ENV = "COGEV_LLM_LIGHT_MAX_TOKENS"
LLM_LONG_MAX_TOKENS_ENV = "COGEV_LLM_LONG_MAX_TOKENS"
LLM_RETRY_MAX_TOKENS_ENV = "COGEV_LLM_RETRY_MAX_TOKENS"
LLM_EMPTY_RETRY_PROMPT_CHARS_ENV = "COGEV_LLM_EMPTY_RETRY_PROMPT_CHARS"
LLM_TIMEOUT_ENV = "COGEV_LLM_TIMEOUT"
LLM_RETRY_ATTEMPTS_ENV = "COGEV_LLM_RETRY_ATTEMPTS"
LLM_BUDGET_USD_ENV = "COGEV_LLM_BUDGET_USD"
LLM_API_KEY_ENV = "COGEV_LLM_API_KEY"
LLM_API_BASE_ENV = "COGEV_LLM_API_BASE"
LLM_BASE_URL_ENV = "COGEV_LLM_BASE_URL"
LLM_REQUIRED_MODEL_ENV = "COGEV_LLM_REQUIRED_MODEL"
LLM_MAX_CONCURRENT_ENV = "COGEV_LLM_MAX_CONCURRENT"
LLM_RPM_ENV = "COGEV_LLM_RPM"
LLM_TPM_ENV = "COGEV_LLM_TPM"
LLM_RETRY_MAX_SLEEP_ENV = "COGEV_LLM_RETRY_MAX_SLEEP"
LLM_RETRY_BASE_SLEEP_ENV = "COGEV_LLM_RETRY_BASE_SLEEP"
LLM_RETRY_JITTER_ENV = "COGEV_LLM_RETRY_JITTER"
LLM_JSON_RETRY_ATTEMPTS_ENV = "COGEV_LLM_JSON_RETRY_ATTEMPTS"
LLM_MAX_PROMPT_CHARS_ENV = "COGEV_LLM_MAX_PROMPT_CHARS"


class LLMConfigurationError(RuntimeError):
    """Raised when no mandatory LLM provider is configured."""


class LLMResponseError(RuntimeError):
    """Raised when the configured LLM provider returns unusable JSON."""


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def looks_like_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return (
        not lowered
        or "replace-with" in lowered
        or "change-me" in lowered
        or lowered.startswith("your-")
        or lowered.startswith("<")
        or lowered.endswith(">")
    )


def llm_status() -> dict[str, Any]:
    load_layered_config(override=False)
    provider = os.environ.get(LLM_PROVIDER_ENV, "litellm").strip().lower()
    fixture = os.environ.get(LLM_FIXTURE_ENV, "").strip()
    model = os.environ.get(LLM_MODEL_ENV, "").strip()
    if provider == "fixture":
        return {
            "configured": bool(fixture),
            "provider": "fixture",
            "model": "fixture",
            "fixture": fixture,
            "requires_real_llm": False,
            "test_provider_only": True,
        }
    api_base = os.environ.get(LLM_API_BASE_ENV, "").strip() or os.environ.get(LLM_BASE_URL_ENV, "").strip()
    raw_api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    api_key_placeholder = bool(raw_api_key) and looks_like_placeholder_secret(raw_api_key)
    api_key_configured = bool(raw_api_key) and not api_key_placeholder
    return {
        "configured": bool(model),
        "provider": provider or "litellm",
        "model": model,
        "requires_real_llm": True,
        "test_provider_only": False,
        "api_base": api_base,
        "api_key_configured": api_key_configured,
        "api_key_placeholder": api_key_placeholder,
    }


def llm_public_status(status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return LLM status safe for logs, reports, and CLI output.

    ``llm_status`` intentionally keeps legacy boolean names such as
    ``api_key_configured`` for in-process compatibility.  Public diagnostics do
    not need secret-shaped keys, and avoiding those names prevents downstream
    log/report sinks from becoming accidental credential carriers.
    """

    source = dict(status if status is not None else llm_status())
    public: dict[str, Any] = {}
    for key, value in source.items():
        if key == "api_key":
            continue
        if key == "api_key_configured":
            public["credential_configured"] = bool(value)
            continue
        if key == "api_key_placeholder":
            public["credential_placeholder"] = bool(value)
            continue
        public[key] = value
    return public


def require_llm_config() -> dict[str, Any]:
    status = llm_status()
    if not status["configured"]:
        if status["provider"] == "fixture":
            raise LLMConfigurationError(f"{LLM_PROVIDER_ENV}=fixture requires {LLM_FIXTURE_ENV}=<fixture.json>.")
        raise LLMConfigurationError(
            "CognitiveEvolve is LLM-first and has no no-LLM fallback. "
            f"Set {LLM_MODEL_ENV}=<provider/model> for LiteLLM, or set "
            f"{LLM_PROVIDER_ENV}=fixture and {LLM_FIXTURE_ENV}=<fixture.json> for tests."
        )
    if status.get("requires_real_llm") and status.get("api_key_placeholder"):
        raise LLMConfigurationError(
            f"{LLM_API_KEY_ENV} is still a placeholder from an example configuration. "
            "Replace it with a real upstream model API key, or use "
            f"{LLM_PROVIDER_ENV}=fixture with {LLM_FIXTURE_ENV}=<fixture.json> for tests."
        )
    required = _required_model_set()
    if status.get("requires_real_llm") and required:
        model = str(status.get("model") or "").strip()
        if model not in required:
            allowed = ", ".join(sorted(required))
            raise LLMConfigurationError(
                f"{LLM_REQUIRED_MODEL_ENV} requires model {allowed}; current {LLM_MODEL_ENV}={model or '<unset>'}."
            )
    return status


def _required_model_set() -> set[str]:
    raw = os.environ.get(LLM_REQUIRED_MODEL_ENV, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


# Current private aliases used by older tests/operators.
_env_float = env_float
_env_int = env_int
_looks_like_placeholder_secret = looks_like_placeholder_secret
