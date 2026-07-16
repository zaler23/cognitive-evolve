"""Canonical Nexus stop reasons and terminal-status mapping helpers."""
from __future__ import annotations

CANDIDATE_READY_FOR_EXTERNAL_REVIEW = "candidate_ready_for_external_review"
DIMINISHING_RETURNS_CHECKPOINT = "diminishing_returns_checkpoint"
GOAL_REACHED = "goal_reached"
STAGNATION_EXHAUSTED = "stagnation_exhausted"

EXTERNAL_REVIEW_STOP_REASONS = frozenset(
    {
        CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
        DIMINISHING_RETURNS_CHECKPOINT,
    }
)

SOLVED_STOP_REASONS = frozenset(
    {
        "objective_solved",
        "verified_solution",
        "self_observed_convergence",
        "self_observed_stable_best_candidate",
    }
)


def normalize_external_review_stop_reason(value: object) -> str:
    """Return a canonical allowed early-stop reason or an empty string."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ready_for_external_review": CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
        "candidate_ready": CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
        "external_review_ready": CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
        "human_review_ready": CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
        "diminishing_returns": DIMINISHING_RETURNS_CHECKPOINT,
        "low_marginal_gain_checkpoint": DIMINISHING_RETURNS_CHECKPOINT,
        "low_expected_gain_checkpoint": DIMINISHING_RETURNS_CHECKPOINT,
    }
    text = aliases.get(text, text)
    return text if text in EXTERNAL_REVIEW_STOP_REASONS else ""


def is_external_review_stop_reason(value: object) -> bool:
    return bool(normalize_external_review_stop_reason(value))


def is_solved_stop_reason(value: object) -> bool:
    return str(value or "").strip().lower() in SOLVED_STOP_REASONS


def stop_reason_class(value: object) -> str:
    reason = str(value or "").strip().lower()
    if reason == CANDIDATE_READY_FOR_EXTERNAL_REVIEW:
        return GOAL_REACHED
    if reason in {DIMINISHING_RETURNS_CHECKPOINT, "adaptive_safety_checkpoint", "max_rounds"}:
        return STAGNATION_EXHAUSTED
    return ""


__all__ = [
    "CANDIDATE_READY_FOR_EXTERNAL_REVIEW",
    "DIMINISHING_RETURNS_CHECKPOINT",
    "GOAL_REACHED",
    "STAGNATION_EXHAUSTED",
    "EXTERNAL_REVIEW_STOP_REASONS",
    "SOLVED_STOP_REASONS",
    "is_external_review_stop_reason",
    "is_solved_stop_reason",
    "normalize_external_review_stop_reason",
    "stop_reason_class",
]
