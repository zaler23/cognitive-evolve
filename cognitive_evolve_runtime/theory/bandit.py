"""Bandit-style budget suggestions for M6.6.

The first landing only emits suggestions.  It does not control or mutate runtime
budgets.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatorArmStats:
    arm_id: str
    pulls: int = 0
    reward_sum: float = 0.0
    risk_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / max(1, self.pulls)

    @property
    def mean_risk(self) -> float:
        return self.risk_sum / max(1, self.pulls)

    def __post_init__(self) -> None:
        if not str(self.arm_id).strip():
            raise ValueError("arm_id is required")
        if self.pulls < 0:
            raise ValueError("pulls must be non-negative")
        for name in ("reward_sum", "risk_sum"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetSuggestion:
    arm_id: str
    suggestion_score: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    advisory_only: bool = True

    def __post_init__(self) -> None:
        if not str(self.arm_id).strip():
            raise ValueError("arm_id is required")
        score = float(self.suggestion_score)
        if not math.isfinite(score):
            raise ValueError("suggestion_score must be finite")
        if self.advisory_only is not True:
            raise ValueError("budget suggestions must be advisory_only=True")
        object.__setattr__(self, "suggestion_score", score)
        object.__setattr__(self, "reason_codes", tuple(str(item) for item in self.reason_codes if str(item)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "suggestion_score": self.suggestion_score,
            "reason_codes": list(self.reason_codes),
            "advisory_only": True,
        }


def suggest_budget_allocation(arms: tuple[OperatorArmStats, ...], *, exploration_weight: float = 1.0) -> tuple[BudgetSuggestion, ...]:
    total_pulls = sum(max(0, arm.pulls) for arm in arms)
    suggestions: list[BudgetSuggestion] = []
    for arm in arms:
        exploration = math.sqrt(math.log(max(2, total_pulls + 1)) / max(1, arm.pulls))
        score = arm.mean_reward + exploration_weight * exploration - arm.mean_risk
        reasons = ["bandit_advisory_ucb"]
        if arm.pulls == 0:
            reasons.append("unexplored_arm")
        suggestions.append(BudgetSuggestion(arm_id=arm.arm_id, suggestion_score=score, reason_codes=tuple(reasons)))
    return tuple(sorted(suggestions, key=lambda item: (item.suggestion_score, item.arm_id), reverse=True))


__all__ = ["BudgetSuggestion", "OperatorArmStats", "suggest_budget_allocation"]
