from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from ..nexus.stage_budget import stage_budget_policy, total_llm_budget_usd
from ..nexus.request_context import get_llm_stage, llm_stage_group
from .env import LLM_BUDGET_USD_ENV, LLMResponseError
from .session import current_llm_session


def budget_usd() -> float | None:
    return total_llm_budget_usd()


def total_estimated_cost_usd() -> float:
    return current_llm_session().total_estimated_cost_usd()


@contextmanager
def budget_reservation() -> Iterator[None]:
    """Serialize budgeted calls and reserve observed per-call headroom.

    Providers report dollar cost only after completion, so this guard
    uses the largest completed call as the next-call estimate. It closes the
    concurrent check-then-act race without pretending that estimate is a
    provider hard limit: the first or a larger call may still cross remaining
    headroom, matching the existing serial-call semantics.
    """

    budget = budget_usd()
    if budget is None:
        yield
        return
    session = current_llm_session()
    # Generic providers expose cost only after a response, so budgeted calls in
    # one session must be serialized.  The largest observed call is the only
    # defensible pre-call estimate; the first call may still cross the remaining
    # budget, matching the existing serial semantics.
    with session.budget_call_lock:
        with session.lock:
            costs = [float(event.get("estimated_cost_usd") or 0.0) for event in session.events]
            completed = sum(costs)
            reservation = max(costs, default=0.0)
            committed = completed + reservation
            if completed >= budget or (reservation > 0 and committed > budget):
                raise LLMResponseError(
                    f"LLM cost budget already exhausted or lacks observed per-call headroom: estimated ${round(committed, 6)} "
                    f"> ${budget}. Increase {LLM_BUDGET_USD_ENV} or reduce rounds/candidates."
                )
            session.budget_reservation_usd = reservation
        try:
            yield
        finally:
            with session.lock:
                session.budget_reservation_usd = 0.0


def stage_budget_strict() -> bool:
    return os.environ.get("COGEV_LLM_STAGE_BUDGET_STRICT", "").strip().lower() in {"1", "true", "yes"}


def stage_group_for(stage: str | None) -> str | None:
    return llm_stage_group(stage)


def stage_budget_percentages() -> dict[str, float]:
    return dict(stage_budget_policy().get("percentages", {}))


def enforce_stage_budget(*, preflight: bool = False) -> None:
    if not stage_budget_strict():
        return
    total_budget = budget_usd()
    if total_budget is None:
        return
    current_group = stage_group_for(get_llm_stage())
    if current_group is None:
        return
    group_cost = 0.0
    for event in current_llm_session().snapshot():
        if stage_group_for(str(event.get("stage") or "")) == current_group:
            group_cost += float(event.get("estimated_cost_usd") or 0.0)
    allowance = total_budget * stage_budget_percentages()[current_group]
    over_budget = group_cost >= allowance if preflight else group_cost > allowance
    if over_budget:
        phase = "already exhausted before next call" if preflight else "exceeded after last call"
        raise LLMResponseError(
            f"LLM stage budget {phase} for {current_group}: estimated ${round(group_cost, 6)} "
            f">= ${round(allowance, 6)}. "
            "Unset COGEV_LLM_STAGE_BUDGET_STRICT or increase COGEV_LLM_BUDGET_USD for performance-first runs."
        )


def enforce_budget(*, preflight: bool = False) -> None:
    budget = budget_usd()
    total_cost = total_estimated_cost_usd()
    if budget is not None:
        over_budget = total_cost >= budget if preflight else total_cost > budget
        if over_budget:
            phase = "already exhausted before next call" if preflight else "exceeded after last call"
            raise LLMResponseError(
                f"LLM cost budget {phase}: estimated ${total_cost} >= ${budget}. "
                f"Increase {LLM_BUDGET_USD_ENV} or reduce rounds/candidates."
            )
    enforce_stage_budget(preflight=preflight)
