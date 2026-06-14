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
    estimated_cost_usd: float | None = None,
    attempts: int = 1,
    retry_history: list[dict[str, Any]] | None = None,
    governor: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> None:
    usage = usage or {}
    event = {
        "time": now_iso(),
        "request_type": request_type,
        "stage": get_llm_stage() or "unscoped",
        "provider": status.get("provider"),
        "model": status.get("model"),
        "test_provider_only": status.get("test_provider_only", False),
        "confidence": response.get("confidence"),
        "attempts": attempts,
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
    }
    if retry_history:
        event["retry_history"] = retry_history
    if governor:
        event["governor"] = governor
    if error_type:
        event["error_type"] = error_type
    current_llm_session().record(event)
