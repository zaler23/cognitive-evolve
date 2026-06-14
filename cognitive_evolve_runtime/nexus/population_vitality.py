"""Population-vitality helpers for Nexus lifecycle routing.

This module deliberately does not decide final-answer validity.  It only keeps
repairable, relevant candidates alive in the non-final search pool and records
why dormant candidates are parked.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.stage_policy import EligibilityDecision

DORMANT_HARD_REJECT = "hard_reject"
DORMANT_DUPLICATE = "duplicate"
DORMANT_REPAIRABLE = "repairable"
DORMANT_REPAIR_EXHAUSTED = "repair_exhausted"
DORMANT_NOVELTY_RESERVE = "novelty_reserve"
DORMANT_LOW_PRIORITY = "low_priority_reserve"

DORMANT_KINDS = {
    DORMANT_HARD_REJECT,
    DORMANT_DUPLICATE,
    DORMANT_REPAIRABLE,
    DORMANT_REPAIR_EXHAUSTED,
    DORMANT_NOVELTY_RESERVE,
    DORMANT_LOW_PRIORITY,
}

NON_TERMINAL_FATES = {
    CandidateFate.ACTIVE.value,
    CandidateFate.ELITE.value,
    CandidateFate.INCUBATING.value,
    CandidateFate.DORMANT.value,
    CandidateFate.AUXILIARY.value,
}

TERMINAL_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}


@dataclass(frozen=True)
class VitalitySnapshot:
    """Small per-round snapshot emitted into progress metadata."""

    active: int = 0
    elite: int = 0
    incubating: int = 0
    dormant: int = 0
    auxiliary: int = 0
    terminal: int = 0
    viable_non_terminal: int = 0
    target_active_floor: int = 0
    active_floor_promotions: int = 0
    dormant_kinds: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_active_floor(
    *,
    non_terminal_count: int,
    viable_count: int,
    branch_factor: int = 0,
    final_stage: bool = False,
    branch_multiplier: float | str | None = None,
    minimum: int | str | None = None,
    enabled: bool = True,
) -> int:
    """Dynamic Active floor; no fixed round or candidate count is embedded."""

    return _target_active_floor(
        non_terminal_count=non_terminal_count,
        viable_count=viable_count,
        branch_factor=branch_factor,
        final_stage=final_stage,
        branch_multiplier=branch_multiplier,
        minimum=minimum,
        enabled=enabled,
    )


def _target_active_floor(
    *,
    non_terminal_count: int,
    viable_count: int,
    branch_factor: int = 0,
    final_stage: bool = False,
    branch_multiplier: float | str | None = None,
    minimum: int | str | None = None,
    enabled: bool = True,
) -> int:
    viable = max(0, int(viable_count or 0))
    if not enabled or viable <= 0 or final_stage:
        return 0
    non_terminal = max(1, int(non_terminal_count or 1))
    branch = max(0, int(branch_factor or 0))
    multiplier = _optional_positive_float(branch_multiplier)
    branch_floor = int(math.ceil(multiplier * branch)) if branch and multiplier is not None else branch
    minimum_floor = _optional_positive_int(minimum)
    if minimum_floor is None:
        minimum_floor = 1 if branch <= 0 else 0
    target = max(int(math.ceil(math.sqrt(non_terminal))), branch_floor, minimum_floor)
    return min(viable, target)


def repair_slot_count(
    *,
    target: int,
    primary_count: int,
    incubating_count: int,
    max_parent_fraction: float | str | None = None,
    enabled: bool = True,
) -> int:
    """Reserve bounded parent-selection capacity for Incubating repair lanes."""

    target = max(0, int(target or 0))
    primary_count = max(0, int(primary_count or 0))
    incubating_count = max(0, int(incubating_count or 0))
    if not enabled or target <= 0 or incubating_count <= 0:
        return 0
    if primary_count <= 0:
        return min(target, incubating_count)
    base = max(1, int(math.ceil(math.sqrt(incubating_count))))
    fraction = _optional_positive_float(max_parent_fraction)
    if fraction is None:
        max_slots = 1 if primary_count >= target else max(1, target - primary_count)
    else:
        max_slots = max(1, int(target * min(1.0, fraction)))
    bounded_base = min(base, max_slots)
    active_deficit = max(0, target - primary_count)
    slots = max(1, bounded_base + active_deficit)
    return min(target - 1, incubating_count, max_slots, slots)


def classify_dormant_kind(candidate: CandidateGenome, decision: EligibilityDecision | None = None) -> str:
    """Classify a dormant candidate so reactivation is explicit, not accidental."""

    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    blockers = list(getattr(decision, "repair_blockers", []) or [])
    diagnostics = _diagnostics(candidate)
    blocker_set = {str(item) for item in blockers}.union(diagnostics)
    duplicate_reason = str(metadata.get("rejected_offspring_reason") or metadata.get("duplicate_reason") or "")
    if "duplicate" in duplicate_reason or "duplicate_formal_signature" in blocker_set:
        return DORMANT_DUPLICATE
    if decision is not None and decision.hard_reject_reason:
        return DORMANT_HARD_REJECT
    if metadata.get("hard_reject_reason") or metadata.get("terminal_reject_reason"):
        return DORMANT_HARD_REJECT
    if decision is not None and decision.repair_required:
        return DORMANT_REPAIR_EXHAUSTED if decision.repair_exhausted else DORMANT_REPAIRABLE
    if metadata.get("repair_required"):
        return DORMANT_REPAIRABLE
    if candidate.edge_knowledge_seeds or candidate.novelty_descriptors:
        return DORMANT_NOVELTY_RESERVE
    scores = coerce_dict(getattr(candidate, "multihead_scores", {}))
    if _score(scores.get("novelty")) >= 0.45 or _score(scores.get("rarity")) >= 0.45:
        return DORMANT_NOVELTY_RESERVE
    return DORMANT_LOW_PRIORITY


def reactivation_condition_for_kind(kind: str, decision: EligibilityDecision | None = None) -> str:
    kind = kind if kind in DORMANT_KINDS else DORMANT_LOW_PRIORITY
    if decision is not None and decision.reactivation_condition:
        return decision.reactivation_condition
    if kind == DORMANT_REPAIRABLE:
        return "reactivate_when_new_evidence_delta_source_binding_or_formal_artifact_can_be_attempted"
    if kind == DORMANT_REPAIR_EXHAUSTED:
        return "reactivate_only_after_new_repair_strategy_or_external_evidence_changes_blocker"
    if kind == DORMANT_NOVELTY_RESERVE:
        return "reactivate_when_novelty_deficit_or_niche_monoculture_is_detected"
    if kind == DORMANT_DUPLICATE:
        return "reactivate_only_if_signature_changes_materially"
    if kind == DORMANT_HARD_REJECT:
        return "do_not_reactivate_without_explicit_hard_reject_override"
    return "reactivate_when_search_needs_diversity_or_complementarity"


def vitality_snapshot(candidates: list[CandidateGenome], *, branch_factor: int = 0) -> VitalitySnapshot:
    counts = {
        CandidateFate.ACTIVE.value: 0,
        CandidateFate.ELITE.value: 0,
        CandidateFate.INCUBATING.value: 0,
        CandidateFate.DORMANT.value: 0,
        CandidateFate.AUXILIARY.value: 0,
    }
    terminal = 0
    dormant_kinds: dict[str, int] = {}
    active_floor_promotions = 0
    viable_non_terminal = 0
    for candidate in candidates:
        fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
        if fate in counts:
            counts[fate] += 1
        if fate in TERMINAL_FATES:
            terminal += 1
            continue
        if fate in NON_TERMINAL_FATES:
            viable_non_terminal += 1
        metadata = coerce_dict(getattr(candidate, "metadata", {}))
        if metadata.get("active_repair_floor"):
            active_floor_promotions += 1
        kind = str(metadata.get("dormant_kind") or "")
        if fate == CandidateFate.DORMANT.value:
            kind = kind if kind in DORMANT_KINDS else classify_dormant_kind(candidate)
            dormant_kinds[kind] = dormant_kinds.get(kind, 0) + 1
    return VitalitySnapshot(
        active=counts[CandidateFate.ACTIVE.value],
        elite=counts[CandidateFate.ELITE.value],
        incubating=counts[CandidateFate.INCUBATING.value],
        dormant=counts[CandidateFate.DORMANT.value],
        auxiliary=counts[CandidateFate.AUXILIARY.value],
        terminal=terminal,
        viable_non_terminal=viable_non_terminal,
        target_active_floor=target_active_floor(
            non_terminal_count=viable_non_terminal,
            viable_count=viable_non_terminal,
            branch_factor=branch_factor,
        ),
        active_floor_promotions=active_floor_promotions,
        dormant_kinds=dormant_kinds,
    )


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _diagnostics(candidate: CandidateGenome) -> set[str]:
    result = getattr(candidate, "verification_result", {}) or {}
    diagnostics: set[str] = set()
    if isinstance(result, dict):
        diagnostics.update(str(item) for item in result.get("diagnostics", []) if item)
        for section in ("proof_progress", "evidence_obligation"):
            payload = result.get(section)
            if isinstance(payload, dict):
                diagnostics.update(str(item) for item in payload.get("diagnostics", []) if item)
    return diagnostics


def _score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))


__all__ = [
    "DORMANT_HARD_REJECT",
    "DORMANT_DUPLICATE",
    "DORMANT_REPAIRABLE",
    "DORMANT_REPAIR_EXHAUSTED",
    "DORMANT_NOVELTY_RESERVE",
    "DORMANT_LOW_PRIORITY",
    "DORMANT_KINDS",
    "VitalitySnapshot",
    "classify_dormant_kind",
    "reactivation_condition_for_kind",
    "repair_slot_count",
    "target_active_floor",
    "vitality_snapshot",
]
