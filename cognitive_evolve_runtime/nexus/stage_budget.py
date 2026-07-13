"""Shared Nexus budget policy for LLM stage spending and telemetry."""
from __future__ import annotations

import os
from typing import Any

from .request_context import llm_stage_budget_percentages as _stage_percentages
from .request_context import llm_stage_groups as _stage_groups


def total_llm_budget_usd() -> float | None:
    raw_budget = os.environ.get("COGEV_LLM_BUDGET_USD", "").strip()
    if not raw_budget:
        return None
    try:
        return max(0.0, float(raw_budget))
    except ValueError:
        return None


def stage_budget_policy(task_type: str | None = None) -> dict[str, Any]:
    percentages = _stage_percentages(task_type)
    total_budget = total_llm_budget_usd()
    per_stage_usd = {name: round(total_budget * pct, 6) for name, pct in percentages.items()} if total_budget is not None else {}
    return {
        "mode": "advisory_unless_strict_env_enabled",
        "strict_env": "COGEV_LLM_STAGE_BUDGET_STRICT",
        "total_budget_env": "COGEV_LLM_BUDGET_USD",
        "total_budget_usd": total_budget,
        "profile": task_type or "default",
        "percentages": percentages,
        "per_stage_budget_usd": per_stage_usd,
        "stage_groups": _stage_groups(),
        "coordination_source": "cognitive_evolve_runtime.nexus.stage_budget.stage_budget_policy",
        "performance_first_default": True,
    }


__all__ = ["stage_budget_policy", "total_llm_budget_usd"]
