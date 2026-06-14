"""Canonical construction helpers for Nexus evolution budgets."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from cognitive_evolve_runtime.nexus.budgeting import NexusRoundBudget
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus._shared import positive_int_or_default


def route_incomplete_round_budget(round_budget: NexusRoundBudget) -> NexusRoundBudget:
    """Return the single-diagnostic budget used for route-incomplete requests."""

    return replace(
        round_budget,
        max_rounds=1,
        source="route_incomplete_diagnostic",
        stop_policy="route_incomplete_single_diagnostic",
        min_rounds_before_stop=1,
        adaptive=False,
        round_safety_limit=0,
        completion_requires_stop_signal=False,
    )


def evolution_budget_from_round_budget(round_budget: NexusRoundBudget) -> EvolutionBudget:
    return EvolutionBudget(
        max_rounds=positive_int_or_default(round_budget.max_rounds, default=1),
        branch_factor=max(0, int(round_budget.mutation_branches_per_round or 0)),
        initial_candidate_count=max(0, int(round_budget.initial_candidate_count or 0)),
        stop_policy=round_budget.stop_policy,
        min_rounds_before_stop=positive_int_or_default(round_budget.min_rounds_before_stop, default=1),
        adaptive=bool(round_budget.adaptive),
        round_safety_limit=max(0, int(round_budget.round_safety_limit or 0)),
        completion_requires_stop_signal=bool(round_budget.completion_requires_stop_signal),
    )


def evolution_budget_from_params(
    *,
    max_rounds: Any = 1,
    branch_factor: Any = 0,
    initial_candidate_count: Any = 0,
    stop_policy: str = "llm_after_minimum",
    min_rounds_before_stop: Any = 1,
) -> EvolutionBudget:
    return EvolutionBudget(
        max_rounds=positive_int_or_default(max_rounds, default=1),
        branch_factor=max(0, int(branch_factor or 0)),
        initial_candidate_count=max(0, int(initial_candidate_count or 0)),
        stop_policy=str(stop_policy or "llm_after_minimum"),
        min_rounds_before_stop=positive_int_or_default(min_rounds_before_stop, default=1),
    )


def resume_evolution_budget(*, checkpoint_round: Any, checkpoint_max_rounds: Any, budget_data: dict[str, Any], max_rounds: Any | None = None) -> EvolutionBudget:
    current_round = int(checkpoint_round or 0)
    adaptive_resume = bool(budget_data.get("adaptive"))
    if max_rounds is not None:
        target_rounds = max(int(max_rounds), current_round + 1)
    elif adaptive_resume:
        previous_limit = int(budget_data.get("round_safety_limit") or checkpoint_max_rounds or 1)
        target_rounds = current_round + max(1, previous_limit)
    else:
        target_rounds = max(int(checkpoint_max_rounds or current_round or 1), current_round)
    return EvolutionBudget(
        max_rounds=target_rounds,
        history=[],
        current_round=current_round,
        branch_factor=max(0, int(budget_data.get("branch_factor") or 0)),
        initial_candidate_count=max(0, int(budget_data.get("initial_candidate_count") or 0)),
        stop_policy=str(budget_data.get("stop_policy") or "llm_after_minimum"),
        min_rounds_before_stop=positive_int_or_default(budget_data.get("min_rounds_before_stop"), default=1),
        adaptive=adaptive_resume,
        round_safety_limit=target_rounds if adaptive_resume else max(0, int(budget_data.get("round_safety_limit") or target_rounds)),
        completion_requires_stop_signal=bool(budget_data.get("completion_requires_stop_signal")),
    )


__all__ = [
    "evolution_budget_from_params",
    "evolution_budget_from_round_budget",
    "resume_evolution_budget",
    "route_incomplete_round_budget",
]
