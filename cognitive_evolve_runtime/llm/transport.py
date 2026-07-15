from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any

from .budget import budget_reservation, enforce_budget
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
from .response_cache import load_response, response_signature, store_response
from .session import _LAST_RETRY_HISTORY, current_llm_round, current_llm_session, current_logical_llm_call
from .telemetry import record_event, transport_cost_attribution
from ..core.redaction import public_error_message


def max_tokens_for_request(request_type: str, request_policy: LLMRequestPolicy | None = None) -> int:
    """Return output budget from explicit policy or generic env defaults.

    Transport no longer knows Nexus request classes.  Nexus/model adapters pass
    ``LLMRequestPolicy`` for long-context calls; other callers keep the generic
    environment-driven behavior.
    """

    if request_policy is not None and request_policy.max_output_tokens:
        return max(1, int(request_policy.max_output_tokens))
    if LLM_MAX_TOKENS_ENV in os.environ:
        return max(1, env_int(LLM_MAX_TOKENS_ENV, 16384))
    if request_policy is not None and request_policy.long_context:
        return max(1, env_int(LLM_LONG_MAX_TOKENS_ENV, 65536))
    return max(1, env_int(LLM_LIGHT_MAX_TOKENS_ENV, 16384))


def _resolved_sampling(
    request_policy: LLMRequestPolicy | None,
    logical_context: tuple[str, str, LLMRequestPolicy | None] | None,
) -> dict[str, float | int | None]:
    call_policy = logical_context[2] if logical_context is not None else None

    def _value(name: str) -> Any:
        call_value = getattr(call_policy, name, None)
        if call_value is not None:
            return call_value
        return getattr(request_policy, name, None)

    temperature = _value("temperature")
    top_p = _value("top_p")
    seed = _value("seed")
    if seed is None:
        configured_seed = os.environ.get("COGEV_LLM_SEED")
        if configured_seed is not None and configured_seed.strip():
            try:
                seed = int(configured_seed)
            except ValueError as exc:
                raise LLMConfigurationError("COGEV_LLM_SEED must be an integer") from exc
    return {
        "temperature": float(temperature) if temperature is not None else env_float(LLM_TEMPERATURE_ENV, 0.2),
        "top_p": float(top_p) if top_p is not None else None,
        "seed": int(seed) if seed is not None else None,
    }



def _default_provider_for_status(status: dict[str, Any]) -> LLMProviderInterface:
    provider_id = str(status.get("provider") or "").strip().lower()
    if provider_id in {"direct_http", "http", "openai_http"}:
        return DirectHTTPProvider()
    return LiteLLMProvider()


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


def _provider_response_error(result: Any) -> LLMResponseError | None:
    """Detect provider-neutral semantic failures before JSON repair."""

    try:
        choice = result.choices[0]  # type: ignore[attr-defined]
    except Exception:
        return LLMResponseError("EMPTY_ASSISTANT_CONTENT: provider response had no choices.")
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason is None and isinstance(choice, dict):
        finish_reason = choice.get("finish_reason")
    if str(finish_reason or "").strip().lower() in {"length", "max_tokens"}:
        return LLMResponseError(f"TRUNCATED_RESPONSE: finish_reason={finish_reason}.")
    try:
        content = _result_message_content(result)
    except Exception:
        content = ""
    if not str(content or "").strip():
        return LLMResponseError("EMPTY_ASSISTANT_CONTENT: provider returned no usable assistant text.")
    return None


def _expanded_output_tokens(current: int) -> int:
    cap = max(1, env_int("COGEV_LLM_RETRY_MAX_TOKENS", 65536))
    return min(cap, max(current * 2, current + 1024, 8192))


def llm_json(request_type: str, payload: dict[str, Any], *, system: str, schema_hint: dict[str, Any], provider: LLMProviderInterface | None = None, model_spec: LLMModelSpec | None = None, request_policy: LLMRequestPolicy | None = None) -> dict[str, Any]:
    require_llm_config()
    if request_policy is not None and request_policy.structured_prompt:
        request_text = json.dumps(
            {"request_type": request_type, "schema_hint": schema_hint, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        _, prompt_bounds = bounded_prompt_for_provider(request_text, max_chars=request_policy.max_prompt_chars)
        if prompt_bounds.get("truncated"):
            source_context = payload.get("source_context") if isinstance(payload.get("source_context"), dict) else {}
            inherited_entries = source_context.get("inherited_gene_entries") if isinstance(source_context, dict) else None
            inherited_detail = ""
            if isinstance(inherited_entries, list):
                selected_ids = [str(item.get("candidate_id") or "") for item in inherited_entries if isinstance(item, dict)]
                inherited_chars = len(json.dumps(inherited_entries, ensure_ascii=False, sort_keys=True, default=str))
                inherited_detail = (
                    f" inherited handoff selected_ids={selected_ids!r} inherited_chars={inherited_chars} "
                    f"cap={prompt_bounds.get('max_prompt_chars')};"
                )
            raise LLMResponseError(
                f"{request_type} structured prompt exceeded effective cap of {prompt_bounds.get('max_prompt_chars')} characters; "
                + inherited_detail
                + " refusing lossy transport excerpt replacement."
            )
    return _budgeted_llm_json(request_type, payload, system=system, schema_hint=schema_hint, provider=provider, model_spec=model_spec, request_policy=request_policy)


@budget_reservation()
def _budgeted_llm_json(request_type: str, payload: dict[str, Any], *, system: str, schema_hint: dict[str, Any], provider: LLMProviderInterface | None = None, model_spec: LLMModelSpec | None = None, request_policy: LLMRequestPolicy | None = None) -> dict[str, Any]:
    status = require_llm_config()
    if model_spec is not None:
        status = model_spec.apply_to_status(status)
    call_identity = identity_from_status(status, request_type=request_type)
    status["model_profile_id"] = call_identity.profile_id
    status["llm_call_identity"] = call_identity.to_dict()
    status["reasoning_effort"] = call_identity.reasoning_effort
    status["reasoning_effort_configured"] = call_identity.reasoning_effort is not None
    enforce_budget(preflight=True)
    call_id = f"llm-{uuid.uuid4().hex}"
    started = time.time()
    logical_context = current_logical_llm_call()
    resolved_sampling = _resolved_sampling(request_policy, logical_context)
    resolved_temperature = resolved_sampling["temperature"]
    resolved_top_p = resolved_sampling["top_p"]
    resolved_seed = resolved_sampling["seed"]
    resolved_max_tokens = max_tokens_for_request(request_type, request_policy)
    request_hash = stable_hash(
        {
            "request_type": request_type,
            "payload": payload,
            "schema_hint": schema_hint,
            "system": system,
            "model_spec": model_spec.to_dict() if model_spec is not None else {},
            "reasoning_effort": call_identity.reasoning_effort,
            **resolved_sampling,
        }
    )
    idem_key = llm_idempotency_key(
        provider=call_identity.breaker_key,
        model=str(status.get("model") or status.get("fixture") or ""),
        prompt={"system": system, "payload": payload, "request_type": request_type},
        schema=schema_hint,
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        seed=resolved_seed,
        contract_version=str((payload.get("contract") or payload.get("evaluation_contract") or {}).get("version", "unknown")) if isinstance(payload, dict) else "unknown",
        reasoning_effort=call_identity.reasoning_effort,
    )
    run_id = str(current_llm_session().run_id or os.environ.get("COGEV_RUN_ID") or "run")
    round_id = str(current_llm_round() or os.environ.get("COGEV_ROUND_ID") or "0")
    step_id = str(os.environ.get("COGEV_STEP_ID") or request_type)
    logical_call_id = logical_context[0] if logical_context is not None else "/".join(
        (
            run_id,
            round_id,
            str(os.environ.get("COGEV_STEP_ID") or request_type),
            request_hash,
        )
    )
    template_version = logical_context[1] if logical_context is not None else ""
    telemetry_identity = {
        "logical_call_id": logical_call_id,
        "request_hash": request_hash,
        "idempotency_key": idem_key,
        "run_id": run_id,
        "round_id": round_id,
        "step_id": step_id,
    }
    signature = response_signature(
        {
            "logical_call_id": logical_call_id,
            "run_id": run_id,
            "round_id": round_id,
            "provider": call_identity.provider,
            "model": call_identity.model,
            "request_type": request_type,
            "system": system,
            "payload": payload,
            "schema_hint": schema_hint,
            "temperature": resolved_temperature,
            "top_p": resolved_top_p,
            "seed": resolved_seed,
            "reasoning_effort": call_identity.reasoning_effort,
            "max_tokens": resolved_max_tokens,
            "request_policy": asdict(request_policy) if request_policy is not None else {},
            "template_version": template_version,
        }
    )
    cached = load_response(signature) if logical_context is not None else None
    if cached is not None:
        response = dict(cached["parsed_response"])
        usage = dict(cached.get("usage") or {})
        estimated_cost = cached.get("estimated_cost_usd")
        physical_call_id = str(cached.get("physical_call_id") or "")
        record_event(
            request_type,
            response,
            status,
            usage=usage,
            usage_provenance="run_local_replay",
            estimated_cost_usd=estimated_cost,
            attempts=0,
            governor=llm_governor_status(),
            cache_replayed=True,
            physical_call_id=physical_call_id,
            sampling=resolved_sampling,
            **telemetry_identity,
            **transport_cost_attribution(payload, response),
        )
        write_llm_journal(
            {
                "call_id": call_id,
                "physical_call_id": physical_call_id,
                "logical_call_id": logical_call_id,
                "response_signature": signature,
                "run_id": run_id,
                "round_id": round_id,
                "step_id": step_id,
                "provider": status.get("provider"),
                "model": status.get("model") or status.get("fixture"),
                "request_hash": request_hash,
                **resolved_sampling,
                "request_type": request_type,
                "status": "cache_replayed",
                "attempt": 0,
                "started_at": started,
                "ended_at": time.time(),
                "usage": usage,
                "estimated_cost_usd": estimated_cost,
            },
            parsed_response=response,
        )
        record_call_state(
            "completed",
            call_id=call_id,
            request_type=request_type,
            request_hash=request_hash,
            round_id=round_id,
            step_id=step_id,
            extra={"cache_replayed": True, "physical_call_id": physical_call_id, "logical_call_id": logical_call_id, **resolved_sampling},
        )
        enforce_budget(preflight=False)
        return response
    record_call_state(
        "started",
        call_id=call_id,
        request_type=request_type,
        request_hash=request_hash,
        round_id=round_id,
        step_id=step_id,
        extra={"idempotency_key": idem_key, "provider": status.get("provider"), "model": status.get("model") or status.get("fixture"), "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling},
    )
    write_llm_journal({
        "call_id": call_id,
        "run_id": run_id,
        "round_id": round_id,
        "step_id": step_id,
        "idempotency_key": idem_key,
        "provider": status.get("provider"),
        "model": status.get("model") or status.get("fixture"),
        "reasoning_effort": call_identity.reasoning_effort,
        "model_profile_id": call_identity.profile_id,
        "llm_call_identity": call_identity.to_dict(),
        "request_hash": request_hash,
        **resolved_sampling,
        "request_type": request_type,
        "status": "inflight",
        "attempt": 0,
        "started_at": started,
    })
    if status["provider"] == "fixture":
        try:
            response = load_fixture_response(request_type, payload, str(status["fixture"]))
        except (LLMConfigurationError, LLMResponseError) as exc:
            safe_error = public_error_message(exc)
            record_event(
                request_type,
                {},
                status,
                attempts=0,
                error_type=provider_error_category(exc),
                sampling=resolved_sampling,
                **telemetry_identity,
                **transport_cost_attribution(payload, {}),
            )
            record_call_state("failed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=round_id, step_id=step_id, extra={"attempt": 1, "error": safe_error, "category": provider_error_category(exc), "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling})
            raise
        response.setdefault("provider", "fixture")
        response.setdefault("model", "fixture")
        if logical_context is not None:
            store_response(
                signature,
                {
                    "logical_call_id": logical_call_id,
                    "physical_call_id": call_id,
                    "request_hash": request_hash,
                    "sampling": resolved_sampling,
                    "parsed_response": response,
                    "raw_response": response,
                    "response_digest": stable_hash(response),
                    "usage": {},
                    "estimated_cost_usd": 0.0,
                },
            )
        record_event(
            request_type,
            response,
            status,
            attempts=1,
            usage_provenance="fixture",
            estimated_cost_usd=0.0,
            governor=llm_governor_status(),
            physical_call_id=call_id,
            sampling=resolved_sampling,
            **telemetry_identity,
            **transport_cost_attribution(payload, response),
        )
        enforce_budget(preflight=False)
        write_llm_journal({
            "call_id": call_id,
            "run_id": run_id,
            "round_id": round_id,
            "step_id": step_id,
            "idempotency_key": idem_key,
            "provider": status.get("provider"),
            "model": status.get("model") or status.get("fixture"),
            "reasoning_effort": call_identity.reasoning_effort,
            "model_profile_id": call_identity.profile_id,
            "llm_call_identity": call_identity.to_dict(),
            "request_hash": request_hash,
            **resolved_sampling,
            "request_type": request_type,
            "status": "ok",
            "attempt": 1,
            "started_at": started,
            "ended_at": time.time(),
            "usage": {},
            "estimated_cost_usd": 0.0,
        }, parsed_response=response)
        record_call_state("completed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=round_id, step_id=step_id, extra={"attempt": 1, "usage": {}, "estimated_cost_usd": 0.0, "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling})
        return response
    provider = provider or _default_provider_for_status(status)
    breaker = default_provider_circuit_breaker()
    try:
        breaker.before_call(call_identity.breaker_key)
    except ProviderUnavailableError as exc:
        safe_error = public_error_message(exc)
        write_llm_journal({
            "call_id": call_id,
            "run_id": run_id,
            "round_id": round_id,
            "step_id": step_id,
            "idempotency_key": idem_key,
            "provider": status.get("provider"),
            "model": status.get("model"),
            "reasoning_effort": call_identity.reasoning_effort,
            "model_profile_id": call_identity.profile_id,
            "llm_call_identity": call_identity.to_dict(),
            "request_hash": request_hash,
            **resolved_sampling,
            "request_type": request_type,
            "status": "provider_unavailable",
            "attempt": 0,
            "started_at": started,
            "ended_at": time.time(),
            "error": safe_error,
        })
        record_event(
            request_type,
            {},
            status,
            attempts=0,
            error_type="provider_unavailable",
            sampling=resolved_sampling,
            **telemetry_identity,
            **transport_cost_attribution(payload, {}),
        )
        record_call_state("failed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=round_id, step_id=step_id, extra={"error": safe_error, "status": "provider_unavailable", "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling})
        raise LLMResponseError(safe_error) from exc

    request = {"request_type": request_type, "schema_hint": schema_hint, "payload": payload}
    request_text = json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
    _, prompt_bounds = bounded_prompt_for_provider(
        request_text,
        max_chars=request_policy.max_prompt_chars if request_policy is not None else None,
    )
    user_content = request_text
    if prompt_bounds.get("truncated"):
        prompt_bounds["transport_action"] = "sent_full_request"
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
    active_max_tokens = resolved_max_tokens
    json_attempt = 0
    retry_history: list[dict[str, Any]] = []
    _LAST_RETRY_HISTORY.set([])
    while attempts < total_attempt_budget:
        try:
            remaining_attempts = max(1, total_attempt_budget - attempts)
            provider_kwargs: dict[str, Any] = {
                "model": str(status["model"]),
                "messages": active_messages,
                "api_base": str(status.get("api_base") or ""),
                "temperature": resolved_temperature,
                "max_tokens": active_max_tokens,
                "response_format": {"type": "json_object"},
                "timeout": float(request_policy.timeout_seconds)
                if request_policy is not None and request_policy.timeout_seconds
                else timeout_seconds(),
                "_retry_max_attempts": remaining_attempts,
                "_retry_attempt_offset": attempts,
                "_retry_allow_prompt_shrink": not bool(request_policy and request_policy.structured_prompt),
            }
            if resolved_top_p is not None:
                provider_kwargs["top_p"] = resolved_top_p
            if resolved_seed is not None:
                provider_kwargs["seed"] = resolved_seed
            if call_identity.reasoning_effort is not None:
                provider_kwargs["reasoning_effort"] = call_identity.reasoning_effort
            provider_result = provider.complete_json(**provider_kwargs)
            result = provider_result.response
            estimated_cost = provider_result.estimated_cost_usd
            attempts += max(1, int(provider_result.attempts or 1))
            provider_history = _LAST_RETRY_HISTORY.get([])
            if provider_history and provider_history is not retry_history:
                provider_history_snapshot = list(provider_history)
                retry_history.extend(dict(item) for item in provider_history_snapshot if isinstance(item, dict))
        except Exception as exc:
            category = provider_error_category(exc)
            safe_error = public_error_message(exc)
            provider_history = _LAST_RETRY_HISTORY.get([])
            if provider_history and provider_history is not retry_history:
                provider_history_snapshot = [dict(item) for item in list(provider_history) if isinstance(item, dict)]
                retry_history.extend(provider_history_snapshot)
                attempts = max(attempts, max(int(item.get("attempt") or 0) for item in provider_history_snapshot))
            _LAST_RETRY_HISTORY.set(retry_history)
            circuit_state = breaker.record_failure(call_identity.breaker_key, exc)
            write_llm_journal({
                "call_id": call_id,
                "run_id": run_id,
                "round_id": round_id,
                "step_id": step_id,
                "idempotency_key": idem_key,
                "provider": status.get("provider"),
                "model": status.get("model"),
                "reasoning_effort": call_identity.reasoning_effort,
                "model_profile_id": call_identity.profile_id,
                "llm_call_identity": call_identity.to_dict(),
                "request_hash": request_hash,
                **resolved_sampling,
                "request_type": request_type,
                "status": "provider_unavailable" if circuit_state.state == "open" else "retryable_failed",
                "attempt": attempts,
                "prompt_bounds": prompt_bounds,
                "started_at": started,
                "ended_at": time.time(),
                "error": safe_error,
                "category": category,
                "circuit_breaker": circuit_state.to_dict(),
            })
            attempts = max(1, attempts)
            record_event(
                request_type,
                {},
                status,
                attempts=attempts,
                retry_history=retry_history,
                governor=llm_governor_status(),
                error_type=category,
                physical_call_id=call_id,
                sampling=resolved_sampling,
                **telemetry_identity,
                **transport_cost_attribution(payload, {}),
            )
            record_call_state("failed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=round_id, step_id=step_id, extra={"error": safe_error, "category": category, "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling})
            raise LLMResponseError(f"LLM provider call failed after retry policy ({category}): {safe_error}") from exc
        semantic_error = _provider_response_error(result)
        if semantic_error is not None:
            if attempts >= total_attempt_budget:
                raise semantic_error
            previous_max_tokens = active_max_tokens
            active_max_tokens = _expanded_output_tokens(active_max_tokens)
            retry_history.append(
                {
                    "attempt": attempts,
                    "category": provider_error_category(semantic_error),
                    "retryable": True,
                    "slept_seconds": 0.0,
                    "final": False,
                    "retry_mutation": {"max_tokens": active_max_tokens} if active_max_tokens > previous_max_tokens else {},
                }
            )
            _LAST_RETRY_HISTORY.set(retry_history)
            active_messages = list(messages)
            continue
        try:
            content = _result_message_content(result)
        except Exception as exc:
            raise LLMResponseError("LLM response had no message content.") from exc
        try:
            response = extract_json_from_text(content or "")
            break
        except LLMResponseError as exc:
            parse_error = exc
            json_attempt += 1
            if attempts >= total_attempt_budget or json_attempt >= json_attempts:
                raise
            active_messages = messages + [
                {"role": "assistant", "content": str(content or "")},
                {
                    "role": "user",
                    "content": (
                        "The previous answer was not valid JSON for the requested schema. "
                        "Return exactly one non-empty valid JSON object. Do not return prose, "
                        "markdown fences, or an empty response."
                    ),
                },
            ]
    _LAST_RETRY_HISTORY.set(retry_history)
    if response is None:
        raise LLMResponseError(
            f"LLM response was not valid JSON after retry: {public_error_message(parse_error or 'unknown parse error')}"
        )
    response.setdefault("provider", status["provider"])
    response.setdefault("model", status["model"])
    usage = usage_dict(result)
    if logical_context is not None:
        store_response(
            signature,
            {
                "logical_call_id": logical_call_id,
                "physical_call_id": call_id,
                "request_hash": request_hash,
                "sampling": resolved_sampling,
                "parsed_response": response,
                "raw_response": safe_json(result),
                "response_digest": stable_hash(response),
                "usage": usage,
                "estimated_cost_usd": estimated_cost,
            },
        )
    breaker.record_success(call_identity.breaker_key)
    record_event(
        request_type,
        response,
        status,
        usage=usage,
        usage_provenance="provider_reported" if usage else "unavailable",
        estimated_cost_usd=estimated_cost,
        attempts=attempts,
        retry_history=_LAST_RETRY_HISTORY.get([]),
        governor=llm_governor_status(),
        physical_call_id=call_id,
        sampling=resolved_sampling,
        **telemetry_identity,
        **transport_cost_attribution(payload, response),
    )
    write_llm_journal({
        "call_id": call_id,
        "run_id": run_id,
        "round_id": round_id,
        "step_id": step_id,
        "idempotency_key": idem_key,
        "provider": status.get("provider"),
        "model": status.get("model"),
        "reasoning_effort": call_identity.reasoning_effort,
        "model_profile_id": call_identity.profile_id,
        "llm_call_identity": call_identity.to_dict(),
        "request_hash": request_hash,
        **resolved_sampling,
        "request_type": request_type,
        "status": "ok",
        "attempt": attempts,
        "prompt_bounds": prompt_bounds,
        "started_at": started,
        "ended_at": time.time(),
        "usage": usage,
        "estimated_cost_usd": estimated_cost,
    }, raw_response=safe_json(result), parsed_response=response)
    record_call_state("completed", call_id=call_id, request_type=request_type, request_hash=request_hash, round_id=round_id, step_id=step_id, extra={"attempt": attempts, "usage": usage, "estimated_cost_usd": estimated_cost, "reasoning_effort": call_identity.reasoning_effort, "llm_call_identity": call_identity.to_dict(), "model_profile_id": call_identity.profile_id, **resolved_sampling})
    enforce_budget(preflight=False)
    return response
