from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from .budget import enforce_budget
from .env import (
    LLM_API_BASE_ENV,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_JSON_RETRY_ATTEMPTS_ENV,
    LLM_LIGHT_MAX_TOKENS_ENV,
    LLM_LONG_MAX_TOKENS_ENV,
    LLM_MAX_TOKENS_ENV,
    LLM_TEMPERATURE_ENV,
    LLMConfigurationError,
    LLMResponseError,
    env_float,
    env_int,
    require_llm_config,
)
from .fixtures import load_fixture_response
from .governor import llm_governor_status
from .json_tools import bounded_prompt_for_provider, extract_json_from_text, usage_dict
from .retry import provider_error_category, timeout_seconds
from .retry import retry_attempts as configured_retry_attempts
from ..durable import llm_idempotency_key, stable_hash
from ..durable.provider_circuit_breaker import ProviderUnavailableError, default_provider_circuit_breaker
from .call_ledger import record_call_state
from .call_identity import identity_from_status
from .journal import safe_json, write_llm_journal
from .http_provider import DirectHTTPProvider
from .litellm_provider import LiteLLMProvider, litellm_provider_kwargs
from .provider_interface import LLMProviderInterface
from .model_spec import LLMModelSpec
from .request_policy import LLMRequestPolicy
from .session import _LAST_RETRY_HISTORY
from .telemetry import record_event


def max_tokens_for_request(request_type: str, request_policy: LLMRequestPolicy | None = None) -> int:
    """Return output budget from explicit policy or generic env defaults.

    Transport no longer knows Nexus request classes.  Nexus/model adapters pass
    ``LLMRequestPolicy`` for long-context calls; other callers keep the generic
    environment-driven behavior.
    """

    if request_policy is not None and request_policy.max_output_tokens:
        return max(1, int(request_policy.max_output_tokens))
    if LLM_MAX_TOKENS_ENV in os.environ:
        return max(1, env_int(LLM_MAX_TOKENS_ENV, 4096))
    if request_policy is not None and request_policy.long_context:
        return max(1, env_int(LLM_LONG_MAX_TOKENS_ENV, 32768))
    return max(1, env_int(LLM_LIGHT_MAX_TOKENS_ENV, 4096))



def _default_provider_for_status(status: dict[str, Any]) -> LLMProviderInterface:
    provider_id = str(status.get("provider") or "").strip().lower()
    if provider_id in {"direct_http", "http", "openai_http"}:
        return DirectHTTPProvider()
    return LiteLLMProvider()


def _truncated_transport_content(*, request_type: str, schema_hint: dict[str, Any], prompt_bounds: dict[str, Any], bounded_request_text: str) -> str:
    """Return valid JSON content that respects the configured prompt cap."""

    limit = int(prompt_bounds.get("max_prompt_chars") or 0)
    excerpt = bounded_request_text
    while True:
        content = json.dumps(
            {
                "request_type": request_type,
                "schema_hint": schema_hint,
                "payload": {
                    "_transport_prompt_truncated": True,
                    "_prompt_bounds": prompt_bounds,
                    "bounded_request_excerpt": excerpt,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if limit <= 0 or len(content) <= limit:
            return content
        over = len(content) - limit
        if len(excerpt) <= max(16, over + 16):
            excerpt = excerpt[: max(0, len(excerpt) - over - 16)]
            # If metadata alone exceeds the limit, return the smallest valid
            # request object we can build.
            if not excerpt:
                return json.dumps(
                    {
                        "request_type": request_type,
                        "schema_hint": schema_hint if len(json.dumps(schema_hint, ensure_ascii=False, default=str)) < max(64, limit // 2) else {},
                        "payload": {"_transport_prompt_truncated": True, "original_chars": prompt_bounds.get("original_chars")},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
        else:
            excerpt = excerpt[: max(0, len(excerpt) - over - 32)]


def _result_message_content(result: Any) -> str:
    choice = result.choices[0]  # type: ignore[attr-defined]
    message = choice.message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if isinstance(reasoning, dict):
        nested = reasoning.get("content") or reasoning.get("text")
        if isinstance(nested, str) and nested.strip():
            return nested
        if reasoning:
            return json.dumps(reasoning, ensure_ascii=False, default=str)
    return content if isinstance(content, str) else ""


def llm_json(request_type: str, payload: dict[str, Any], *, system: str, schema_hint: dict[str, Any], provider: LLMProviderInterface | None = None, model_spec: LLMModelSpec | None = None, request_policy: LLMRequestPolicy | None = None) -> dict[str, Any]:
    status = require_llm_config()
    if model_spec is not None:
        status = model_spec.apply_to_status(status)
    call_identity = identity_from_status(status, request_type=request_type)
    status["model_profile_id"] = call_identity.profile_id
    status["llm_call_identity"] = call_identity.to_dict()
    enforce_budget(preflight=True)
    call_id = f"llm-{uuid.uuid4().hex}"
    started = time.time()
    request_hash = stable_hash({"request_type": request_type, "payload": payload, "schema_hint": schema_hint, "system": system, "model_spec": model_spec.to_dict() if model_spec is not None else {}})
    idem_key = llm_idempotency_key(
        provider=call_identity.breaker_key,
        model=str(status.get("model") or status.get("fixture") or ""),
        prompt={"system": system, "payload": payload, "request_type": request_type},
        schema=schema_hint,
        temperature=os.environ.get(LLM_TEMPERATURE_ENV),
        seed=os.environ.get("COGEV_LLM_SEED"),
        contract_version=str((payload.get("contract") or payload.get("evaluation_contract") or {}).get("version", "unknown")) if isinstance(payload, dict) else "unknown",
    )
    record_call_state(
        "started",
        call_id=call_id,
        request_type=request_type,
        request_hash=request_hash,
        round_id=os.environ.get("COGEV_ROUND_ID", "runtime"),
        step_id=os.environ.get("COGEV_STEP_ID", request_type),
        extra={"idempotency_key": idem_key, "provider": status.get("provider"), "model": status.get("model") or status.get("fixture"), "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id},
    )
    write_llm_journal({
        "call_id": call_id,
        "run_id": os.environ.get("COGEV_RUN_ID", "run"),
        "round_id": os.environ.get("COGEV_ROUND_ID", "runtime"),
        "step_id": os.environ.get("COGEV_STEP_ID", request_type),
        "idempotency_key": idem_key,
        "provider": status.get("provider"),
        "model": status.get("model") or status.get("fixture"),
        "model_profile_id": call_identity.profile_id,
        "llm_call_identity": call_identity.to_dict(),
        "request_hash": request_hash,
        "request_type": request_type,
        "status": "inflight",
        "attempt": 0,
        "started_at": started,
    })
    if status["provider"] == "fixture":
        response = load_fixture_response(request_type, payload, str(status["fixture"]))
        response.setdefault("provider", "fixture")
        response.setdefault("model", "fixture")
        record_event(request_type, response, status, attempts=1, governor=llm_governor_status())
        enforce_budget(preflight=False)
        write_llm_journal({
            "call_id": call_id,
            "run_id": os.environ.get("COGEV_RUN_ID", "run"),
            "round_id": os.environ.get("COGEV_ROUND_ID", "runtime"),
            "step_id": os.environ.get("COGEV_STEP_ID", request_type),
            "idempotency_key": idem_key,
            "provider": status.get("provider"),
            "model": status.get("model") or status.get("fixture"),
            "model_profile_id": call_identity.profile_id,
            "llm_call_identity": call_identity.to_dict(),
            "request_hash": request_hash,
            "request_type": request_type,
            "status": "ok",
            "attempt": 1,
            "started_at": started,
            "ended_at": time.time(),
            "usage": {},
            "estimated_cost_usd": 0.0,
        }, parsed_response=response)
        record_call_state("completed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=os.environ.get("COGEV_ROUND_ID", "runtime"), step_id=os.environ.get("COGEV_STEP_ID", request_type), extra={"attempt": 1, "usage": {}, "estimated_cost_usd": 0.0, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id})
        return response
    provider = provider or _default_provider_for_status(status)
    breaker = default_provider_circuit_breaker()
    try:
        breaker.before_call(call_identity.breaker_key)
    except ProviderUnavailableError as exc:
        write_llm_journal({
            "call_id": call_id,
            "run_id": os.environ.get("COGEV_RUN_ID", "run"),
            "round_id": os.environ.get("COGEV_ROUND_ID", "runtime"),
            "step_id": os.environ.get("COGEV_STEP_ID", request_type),
            "idempotency_key": idem_key,
            "provider": status.get("provider"),
            "model": status.get("model"),
            "model_profile_id": call_identity.profile_id,
            "llm_call_identity": call_identity.to_dict(),
            "request_hash": request_hash,
            "request_type": request_type,
            "status": "provider_unavailable",
            "attempt": 0,
            "started_at": started,
            "ended_at": time.time(),
            "error": str(exc),
        })
        record_call_state("failed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=os.environ.get("COGEV_ROUND_ID", "runtime"), step_id=os.environ.get("COGEV_STEP_ID", request_type), extra={"error": str(exc), "status": "provider_unavailable", "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id})
        raise LLMResponseError(str(exc)) from exc

    request = {"request_type": request_type, "schema_hint": schema_hint, "payload": payload}
    request_text = json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
    bounded_request_text, prompt_bounds = bounded_prompt_for_provider(request_text)
    if prompt_bounds.get("truncated"):
        # Last-resort transport guard.  Nexus should normally send a compact
        # prompt view well below this limit; if another caller accidentally
        # builds a huge payload, keep the provider request bounded and valid
        # JSON instead of silently sending a multi-megabyte prompt.
        user_content = _truncated_transport_content(
            request_type=request_type,
            schema_hint=schema_hint,
            prompt_bounds=prompt_bounds,
            bounded_request_text=bounded_request_text,
        )
    else:
        user_content = bounded_request_text
    messages = [
        {"role": "system", "content": system + "\nReturn only valid JSON."},
        {"role": "user", "content": user_content},
    ]
    total_attempt_budget = max(1, configured_retry_attempts())
    json_attempts = max(1, env_int(LLM_JSON_RETRY_ATTEMPTS_ENV, total_attempt_budget))
    attempts = 0
    estimated_cost: float | None = None
    result: Any = None
    response: dict[str, Any] | None = None
    parse_error: Exception | None = None
    active_messages = list(messages)
    json_attempt = 0
    while json_attempt < json_attempts and attempts < total_attempt_budget:
        json_attempt += 1
        try:
            remaining_attempts = max(1, total_attempt_budget - attempts)
            provider_result = provider.complete_json(
                model=str(status["model"]),
                messages=active_messages,
                api_base=str(status.get("api_base") or ""),
                temperature=env_float(LLM_TEMPERATURE_ENV, 0.2),
                max_tokens=max_tokens_for_request(request_type, request_policy),
                response_format={"type": "json_object"},
                timeout=float(request_policy.timeout_seconds) if request_policy is not None and request_policy.timeout_seconds else timeout_seconds(),
                _retry_max_attempts=remaining_attempts,
                _retry_attempt_offset=attempts,
            )
            result = provider_result.response
            estimated_cost = provider_result.estimated_cost_usd
            attempts += provider_result.attempts
        except Exception as exc:
            category = provider_error_category(exc)
            retry_history = _LAST_RETRY_HISTORY.get([])
            if retry_history:
                attempts = max(attempts, max(int(item.get("attempt") or 0) for item in retry_history if isinstance(item, dict)))
            circuit_state = breaker.record_failure(call_identity.breaker_key, exc)
            write_llm_journal({
                "call_id": call_id,
                "run_id": os.environ.get("COGEV_RUN_ID", "run"),
                "round_id": os.environ.get("COGEV_ROUND_ID", "runtime"),
                "step_id": os.environ.get("COGEV_STEP_ID", request_type),
                "idempotency_key": idem_key,
                "provider": status.get("provider"),
                "model": status.get("model"),
                "model_profile_id": call_identity.profile_id,
                "llm_call_identity": call_identity.to_dict(),
                "request_hash": request_hash,
                "request_type": request_type,
                "status": "provider_unavailable" if circuit_state.state == "open" else "retryable_failed",
                "attempt": attempts,
                "prompt_bounds": prompt_bounds,
                "started_at": started,
                "ended_at": time.time(),
                "error": str(exc),
                "category": category,
                "circuit_breaker": circuit_state.to_dict(),
            })
            record_call_state("failed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=os.environ.get("COGEV_ROUND_ID", "runtime"), step_id=os.environ.get("COGEV_STEP_ID", request_type), extra={"error": str(exc), "category": category, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id})
            raise LLMResponseError(f"LLM provider call failed after retry policy ({category}): {exc}") from exc
        try:
            content = _result_message_content(result)
        except Exception as exc:
            raise LLMResponseError(f"LLM response had no message content: {result}") from exc
        try:
            response = extract_json_from_text(content or "")
            break
        except LLMResponseError as exc:
            parse_error = exc
            if attempts >= total_attempt_budget or json_attempt >= json_attempts:
                raise
            active_messages = messages + [
                {"role": "assistant", "content": str(content or "")[:1200]},
                {
                    "role": "user",
                    "content": (
                        "The previous answer was not valid JSON for the requested schema. "
                        "Return exactly one non-empty valid JSON object. Do not return prose, "
                        "markdown fences, or an empty response."
                    ),
                },
            ]
    if response is None:
        raise LLMResponseError(f"LLM response was not valid JSON after retry: {parse_error}")
    response.setdefault("provider", status["provider"])
    response.setdefault("model", status["model"])
    usage = usage_dict(result)
    breaker.record_success(call_identity.breaker_key)
    record_event(
        request_type,
        response,
        status,
        usage=usage,
        estimated_cost_usd=estimated_cost,
        attempts=attempts,
        retry_history=_LAST_RETRY_HISTORY.get([]),
        governor=llm_governor_status(),
    )
    write_llm_journal({
        "call_id": call_id,
        "run_id": os.environ.get("COGEV_RUN_ID", "run"),
        "round_id": os.environ.get("COGEV_ROUND_ID", "runtime"),
        "step_id": os.environ.get("COGEV_STEP_ID", request_type),
        "idempotency_key": idem_key,
        "provider": status.get("provider"),
        "model": status.get("model"),
        "model_profile_id": call_identity.profile_id,
        "llm_call_identity": call_identity.to_dict(),
        "request_hash": request_hash,
        "request_type": request_type,
        "status": "ok",
        "attempt": attempts,
        "prompt_bounds": prompt_bounds,
        "started_at": started,
        "ended_at": time.time(),
        "usage": usage,
        "estimated_cost_usd": estimated_cost,
    }, raw_response=safe_json(result), parsed_response=response)
    record_call_state("completed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=os.environ.get("COGEV_ROUND_ID", "runtime"), step_id=os.environ.get("COGEV_STEP_ID", request_type), extra={"attempt": attempts, "usage": usage, "estimated_cost_usd": estimated_cost, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id})
    enforce_budget(preflight=False)
    return response
