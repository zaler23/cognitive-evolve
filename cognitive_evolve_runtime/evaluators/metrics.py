"""Evaluator metric helpers."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score


def frontier_score(metrics: dict[str, Any]) -> float:
    return bounded_score(metrics.get("frontier_score", metrics.get("score", metrics.get("challenge_pass_rate", 0.0))))


__all__ = ["frontier_score"]
