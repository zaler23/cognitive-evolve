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

MODEL_RUNTIME_CONTROL_METADATA_KEYS = (
    "challenge_failures",
    "created_in_round",
    "evidence_records",
    "evidence_state",
    "evaluator",
    "failure_classification",
    "hard_reject_reason",
    "offspring_verification",
    "patch_result",
    "progressive_evidence",
    "repair_value",
    "resolved_challenge_ids",
    "safety_blocked",
    "secret_leak",
    "sensitive_leak",
    "stage_eligibility",
    "structural_failure",
    "target_challenge_ids",
    "terminal_failure",
    "terminal_reject",
    "terminal_reject_reason",
    "terminal_structural_failure",
    "verification_results",
)

MODEL_RUNTIME_CONTROL_PAYLOAD_KEYS = (
    "created_at",
    "current_fate",
    "patch_application_result",
    "preliminary_result",
    "verification_result",
    "verification_trace",
)


def demote_model_runtime_metadata(metadata: Any) -> dict[str, Any]:
    """Move evaluator/runtime-owned model claims out of executable control fields."""

    cleaned = dict(metadata or {}) if isinstance(metadata, dict) else {}
    raw_controls = cleaned.pop("model_claimed_runtime_controls", {})
    controls = dict(raw_controls) if isinstance(raw_controls, dict) else {}
    for key in MODEL_RUNTIME_CONTROL_METADATA_KEYS:
        if key in cleaned:
            controls[key] = cleaned.pop(key)
    if controls:
        cleaned["model_claimed_runtime_controls"] = controls
    return cleaned


def demote_model_candidate_runtime_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize one model candidate payload before runtime evaluation."""

    cleaned = dict(data)
    metadata = demote_model_runtime_metadata(cleaned.get("metadata"))
    controls = dict(metadata.pop("model_claimed_runtime_controls", {}))
    for key in MODEL_RUNTIME_CONTROL_PAYLOAD_KEYS:
        if key in cleaned:
            value = cleaned.pop(key)
            if value not in (None, "", {}, []):
                controls[key] = value
    if controls:
        metadata["model_claimed_runtime_controls"] = controls
    cleaned["metadata"] = metadata
    cleaned["current_fate"] = "Active"
    return cleaned


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
    "MODEL_RUNTIME_CONTROL_METADATA_KEYS",
    "MODEL_RUNTIME_CONTROL_PAYLOAD_KEYS",
    "bounded_score",
    "bounded_score_or_none",
    "call_with_optional_context",
    "classify_with_fallback",
    "demote_model_candidate_runtime_payload",
    "demote_model_runtime_metadata",
    "positive_int",
    "positive_int_or_default",
]
