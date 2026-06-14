"""Stage-adaptive eligibility policy for Nexus candidates.

This package preserves the historical ``nexus.stage_policy`` import boundary
while splitting metric parsing, stage classification, repair requirements, and
eligibility decisions into focused modules.
"""
from __future__ import annotations

from .constants import EARLY_STAGE, FINAL_STAGE, LATE_STAGE, MIDDLE_STAGE
from .eligibility import annotate_stage_eligibility, stage_eligibility, strict_final_eligible, strict_rank_eligible
from .metrics import parse_metric_value
from .repair import repair_requirement
from .stages import candidate_claim_maturity_stage, candidate_created_in_round, candidate_diagnostics, stage_for_candidate, stage_for_candidate_age, stage_for_round
from .types import EligibilityDecision, RepairRequirement

__all__ = [
    "EARLY_STAGE",
    "MIDDLE_STAGE",
    "LATE_STAGE",
    "FINAL_STAGE",
    "EligibilityDecision",
    "RepairRequirement",
    "annotate_stage_eligibility",
    "candidate_created_in_round",
    "candidate_diagnostics",
    "candidate_claim_maturity_stage",
    "parse_metric_value",
    "repair_requirement",
    "stage_eligibility",
    "stage_for_candidate",
    "stage_for_candidate_age",
    "stage_for_round",
    "strict_final_eligible",
    "strict_rank_eligible",
]
