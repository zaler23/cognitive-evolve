"""Metric parsing helpers for stage policy."""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

_METRIC_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)

def parse_metric_value(value: Any) -> float | None:
    """Parse model/runtime metric values into a bounded finite score.

    This accepts normal decimal strings and scientific notation without routing
    through ``float`` first, so values such as ``1e-300`` or ``1e309`` do not
    raise or silently become a non-finite float.  Malformed strings and NaN/Inf
    are rejected as ``None``; valid numeric values are clamped into ``[0, 1]``.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int):
        return _bound_decimal_score(Decimal(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return _bound_decimal_score(Decimal(str(value)))
    if isinstance(value, Decimal):
        return _bound_decimal_score(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or not _METRIC_NUMBER_PATTERN.fullmatch(text):
            return None
        try:
            return _bound_decimal_score(Decimal(text))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, dict):
        for key in ("score", "value", "mean", "rating"):
            if key in value:
                parsed = parse_metric_value(value.get(key))
                if parsed is not None:
                    return parsed
        numeric = [parse_metric_value(item) for item in value.values()]
        numeric = [item for item in numeric if item is not None]
        if numeric:
            return _bound_decimal_score(Decimal(str(sum(numeric) / len(numeric))))
        return None
    if isinstance(value, (list, tuple, set)):
        numeric = [parse_metric_value(item) for item in value]
        numeric = [item for item in numeric if item is not None]
        if numeric:
            return _bound_decimal_score(Decimal(str(sum(numeric) / len(numeric))))
        return None
    return None

def _bound_decimal_score(value: Decimal) -> float | None:
    if not value.is_finite():
        return None
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(value)

__all__ = ["parse_metric_value"]
