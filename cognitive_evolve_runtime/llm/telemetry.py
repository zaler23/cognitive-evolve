from __future__ import annotations

from typing import Any

from ..nexus.request_context import get_llm_stage
from .session import current_llm_session
from .utils import now_iso


def record_event(
    request_type: str,
    response: dict[str, Any],
    status: dict[str, Any],
    *,
    usage: dict[str, int] | None = None,
    usage_provenance: str | None = None,
    estimated_cost_usd: float | None = None,
    attempts: int = 1,
    retry_history: list[dict[str, Any]] | None = None,
    governor: dict[str, Any] | None = None,
    error_type: str | None = None,
    cache_replayed: bool = False,
    physical_call_id: str = "",
) -> None:
    usage = usage or {}
    event = {
        "time": now_iso(),
        "request_type": request_type,
        "stage": get_llm_stage() or "unscoped",
        "provider": status.get("provider"),
        "model": status.get("model"),
        "model_profile_id": status.get("model_profile_id"),
        "llm_call_identity": status.get("llm_call_identity"),
        "test_provider_only": status.get("test_provider_only", False),
        "confidence": response.get("confidence"),
        "attempts": attempts,
        "usage": usage,
        "usage_provenance": usage_provenance or ("unspecified" if usage else "unavailable"),
        "estimated_cost_usd": estimated_cost_usd,
    }
    if retry_history:
        event["retry_history"] = retry_history
    if governor:
        event["governor"] = governor
    if error_type:
        event["error_type"] = error_type
    if physical_call_id:
        event["physical_call_id"] = physical_call_id
    if cache_replayed:
        event["cache_replayed"] = True
    current_llm_session().record(event)
