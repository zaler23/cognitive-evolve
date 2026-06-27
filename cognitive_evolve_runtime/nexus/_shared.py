"""Shared Nexus primitives for small cross-cutting logic.

Keep this module dependency-light: it centralizes duplicated parsing and model
boundary helpers without becoming a new runtime layer.
"""
from __future__ import annotations

import math
import inspect
from collections.abc import Callable
from typing import Any, TypeAlias

from cognitive_evolve_runtime.llm.env import LLMConfigurationError, LLMResponseError
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError

MODEL_BOUNDARY_ERRORS: tuple[type[Exception], ...] = (LLMConfigurationError, LLMResponseError, ModelResponseSchemaError)


def positive_int(value: Any) -> int | None:
    """Return a positive integer or ``None`` for absent/invalid/non-positive input."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def positive_int_or_default(value: Any, *, default: int = 0) -> int:
    parsed = positive_int(value)
    return parsed if parsed is not None else default


def bounded_score(value: Any, *, default: float = 0.0) -> float:
    """Coerce a numeric score to the canonical [0.0, 1.0] Nexus range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(1.0, parsed))


def bounded_score_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def classify_with_fallback(prompt: str, *, model: object | None = None) -> object:
    """Call the semantic classifier while tolerating legacy fixtures without a model kwarg."""

    from cognitive_evolve_runtime.nexus.semantics import classify

    if model is not None:
        try:
            return classify(prompt, model=model)
        except TypeError as exc:
            if "model" not in str(exc):
                raise
    return classify(prompt)


def call_with_optional_context(method: Callable[..., Any], /, *, provided_context: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Call a model method with ``provided_context`` only when it accepts it."""

    if provided_context is not None and _accepts_kwarg(method, "provided_context"):
        kwargs["provided_context"] = provided_context
    return method(**kwargs)


def _accepts_kwarg(method: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


__all__ = [
    "MODEL_BOUNDARY_ERRORS",
    "bounded_score",
    "bounded_score_or_none",
    "call_with_optional_context",
    "classify_with_fallback",
    "positive_int",
    "positive_int_or_default",
]
