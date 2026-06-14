"""Metric helpers for external evaluator scores."""
from __future__ import annotations

from typing import Any


def normalized_metric(value: Any, *, direction: str = "maximize") -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if direction == "minimize":
        return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, parsed))))
    return max(0.0, min(1.0, parsed))


__all__ = ["normalized_metric"]
