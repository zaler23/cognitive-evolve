"""Incubation metadata helpers for stage policy."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.adaptive_signals import adaptive_attempt_limit
from cognitive_evolve_runtime.nexus.policy import DEFAULT_ELIGIBILITY_POLICY

from .metrics import parse_metric_value
from .stages import _positive_int
from .types import EligibilityDecision

def _incubation_started_round(metadata: dict[str, Any], *, current_round: int) -> int:
    for key in ("incubation_started_round",):
        try:
            parsed = int(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return max(0, int(current_round or 0))

def _max_incubation_attempts(metadata: dict[str, Any], policy_config: dict[str, Any] | None = None, diagnostics: set[str] | None = None) -> int:
    try:
        parsed = int(metadata.get("max_incubation_attempts"))
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        parsed = _positive_int(coerce_dict(policy_config).get("max_incubation_attempts"))
    if parsed <= 0:
        parsed = _positive_int(DEFAULT_ELIGIBILITY_POLICY.get("max_incubation_attempts"))
    if parsed > 0:
        return parsed
    repair = coerce_dict(metadata.get("repair_required"))
    blockers = set(str(item) for item in repair.get("blockers", []) or [] if item)
    blockers.update(str(item) for item in diagnostics or set() if item)
    return adaptive_attempt_limit(
        population_size=_positive_int(metadata.get("observed_repair_population_size")),
        distinct_blockers=len(blockers),
        configured=coerce_dict(policy_config).get("max_incubation_attempts"),
        fallback=0,
    )

def _max_incubation_age(metadata: dict[str, Any], *, round_limit: int, policy_config: dict[str, Any] | None = None) -> int:
    try:
        parsed = int(metadata.get("max_incubation_age"))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    policy = coerce_dict(policy_config)
    configured_rounds = _positive_int(policy.get("max_incubation_age_rounds"))
    if configured_rounds:
        return configured_rounds
    fraction = parse_metric_value(policy.get("max_incubation_age_fraction"))
    if fraction is None or fraction <= 0:
        fraction = parse_metric_value(DEFAULT_ELIGIBILITY_POLICY.get("max_incubation_age_fraction"))
    if fraction is None or fraction <= 0:
        fraction = 1.0
    return max(_min_incubation_age_rounds(policy), int(max(1, round_limit or 1) * min(1.0, fraction)))

def _min_incubation_age_rounds(policy_config: dict[str, Any] | None = None) -> int:
    parsed = _positive_int(coerce_dict(policy_config).get("min_incubation_age_rounds"))
    if parsed <= 0:
        parsed = _positive_int(DEFAULT_ELIGIBILITY_POLICY.get("min_incubation_age_rounds"))
    return parsed if parsed > 0 else 1

def _max_repeated_repair_blockers(metadata: dict[str, Any], policy_config: dict[str, Any] | None = None) -> int:
    try:
        parsed = int(metadata.get("max_repeated_repair_blockers"))
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        parsed = _positive_int(coerce_dict(policy_config).get("max_repeated_repair_blockers"))
    if parsed <= 0:
        parsed = _positive_int(DEFAULT_ELIGIBILITY_POLICY.get("max_repeated_repair_blockers"))
    return parsed if parsed > 0 else 1

def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)

def _score(value: Any) -> float:
    parsed = parse_metric_value(value)
    return 0.0 if parsed is None else parsed

def _update_incubation_metadata(candidate: CandidateGenome, decision: EligibilityDecision, *, current_round: int) -> None:
    if not decision.repair_required:
        return
    metadata = candidate.metadata
    metadata.setdefault("incubation_started_round", decision.incubation_started_round)
    metadata.setdefault("repair_attempts", decision.repair_attempts)
    metadata.setdefault("max_incubation_attempts", decision.max_incubation_attempts)
    metadata.setdefault("max_incubation_age", decision.max_incubation_age)
    current_blockers = list(decision.repair_blockers)
    previous = metadata.get("last_repair_blockers")
    previous_list = [str(item) for item in previous] if isinstance(previous, list) else []
    last_round = _nonnegative_int(metadata.get("last_repair_check_round"))
    round_index = max(0, int(current_round or 0))
    if current_blockers == previous_list and round_index != last_round:
        metadata["repair_blocker_repeat_count"] = _nonnegative_int(metadata.get("repair_blocker_repeat_count")) + 1
    elif current_blockers != previous_list:
        metadata["repair_blocker_repeat_count"] = 1 if current_blockers else 0
    metadata["last_repair_blockers"] = current_blockers
    metadata["last_repair_check_round"] = round_index
    if decision.state_transition_reason:
        metadata["state_transition_reason"] = decision.state_transition_reason
    if decision.reactivation_condition:
        metadata["reactivation_condition"] = decision.reactivation_condition

__all__ = []
