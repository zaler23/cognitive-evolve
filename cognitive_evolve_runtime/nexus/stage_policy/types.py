"""Data contracts for stage eligibility decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import EARLY_STAGE

@dataclass
class EligibilityDecision:
    """Candidate-stage decision used by archive routing and parent selection."""

    candidate_id: str
    stage: str
    global_stage: str
    candidate_age_stage: str
    candidate_claim_stage: str
    current_round: int = 0
    round_limit: int = 0
    created_in_round: int = 0
    candidate_age: int = 0
    incubation_started_round: int = 0
    repair_attempts: int = 0
    max_incubation_attempts: int = 0
    max_incubation_age: int = 0
    exploration_eligible: bool = False
    parent_eligible: bool = False
    final_eligible: bool = False
    strict_rank_eligible: bool = False
    strict_final_eligible: bool = False
    repair_required: bool = False
    incubating: bool = False
    repair_exhausted: bool = False
    repair_blockers: list[str] = field(default_factory=list)
    hard_reject_reason: str = ""
    state_transition_reason: str = ""
    reactivation_condition: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class RepairRequirement:
    """Compact repair target exposed to mutation planning."""

    blockers: list[str]
    evidence_needed: list[str] = field(default_factory=list)
    source_bindings: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    stage: str = EARLY_STAGE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

__all__ = ["EligibilityDecision", "RepairRequirement"]
