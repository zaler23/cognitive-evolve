"""Stage-adaptive eligibility decisions."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict

from .constants import (
    EARLY_STAGE, FINAL_STAGE, LATE_STAGE, MIDDLE_STAGE, REPAIRABLE_DIAGNOSTICS, TERMINAL_FATES,
)
from .metadata import (
    _incubation_started_round, _max_incubation_age, _max_incubation_attempts,
    _max_repeated_repair_blockers, _nonnegative_int, _update_incubation_metadata,
)
from .repair import (
    _candidate_nonempty, _hard_reject_reason, _repair_guidance, repair_requirement,
)
from .stages import _has_evidence_progress, candidate_diagnostics, stage_for_candidate
from .types import EligibilityDecision

def strict_rank_eligible(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
    if fate in TERMINAL_FATES:
        return False
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata.get("terminal_failure") or metadata.get("terminal_reject"):
        return False
    return _candidate_nonempty(candidate)

def strict_final_eligible(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
    if fate in TERMINAL_FATES:
        return False
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata.get("terminal_failure") or metadata.get("terminal_reject"):
        return False
    return _candidate_nonempty(candidate)

def stage_eligibility(
    candidate: CandidateGenome,
    *,
    current_round: int = 0,
    round_limit: int = 0,
    policy_config: dict[str, Any] | None = None,
) -> EligibilityDecision:
    stage, global_stage, age_stage, claim_stage, created, age = stage_for_candidate(
        candidate,
        current_round=current_round,
        round_limit=round_limit,
        policy_config=policy_config,
    )
    diagnostics = candidate_diagnostics(candidate)
    repair_blockers = sorted(diagnostics.intersection(REPAIRABLE_DIAGNOSTICS))
    hard_reject = _hard_reject_reason(candidate, diagnostics, stage=stage)
    strict_rank = strict_rank_eligible(candidate)
    strict_final = strict_final_eligible(candidate)
    notes: list[str] = []
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    incubation_started_round = _incubation_started_round(metadata, current_round=current_round)
    repair_attempts = _nonnegative_int(metadata.get("repair_attempts"))
    max_attempts = _max_incubation_attempts(metadata, policy_config=policy_config, diagnostics=diagnostics)
    max_age = _max_incubation_age(metadata, round_limit=round_limit, policy_config=policy_config)
    incubation_age = max(0, max(0, int(current_round or 0)) - incubation_started_round)
    repeated_blockers = _nonnegative_int(metadata.get("repair_blocker_repeat_count"))
    max_repeated_blockers = _max_repeated_repair_blockers(metadata, policy_config=policy_config)
    has_any_progress_signal = bool(_has_evidence_progress(candidate) or candidate.formal_artifacts or candidate.source_bindings or candidate.evidence_refs)
    repair_deprioritized = bool((repair_attempts > 0 or repeated_blockers > 0) and not has_any_progress_signal)
    repair_exhausted = bool(
        (repair_attempts >= max_attempts or incubation_age > max_age or repeated_blockers >= max_repeated_blockers)
        and not has_any_progress_signal
    )
    reactivation_condition = "reactivate_only_if_new_evidence_delta_or_source_binding_or_formal_artifact_exists"

    if hard_reject:
        return EligibilityDecision(
            candidate_id=candidate.id,
            stage=stage,
            global_stage=global_stage,
            candidate_age_stage=age_stage,
            candidate_claim_stage=claim_stage,
            current_round=max(0, int(current_round or 0)),
            round_limit=max(1, int(round_limit or 1)),
            created_in_round=created,
            candidate_age=age,
            incubation_started_round=incubation_started_round,
            repair_attempts=repair_attempts,
            max_incubation_attempts=max_attempts,
            max_incubation_age=max_age,
            strict_rank_eligible=strict_rank,
            strict_final_eligible=strict_final,
            final_eligible=False,
            repair_exhausted=repair_exhausted,
            hard_reject_reason=hard_reject,
            repair_blockers=repair_blockers,
            state_transition_reason="hard_reject_not_incubated",
            reactivation_condition=reactivation_condition,
            notes=["hard_reject_not_incubated"],
        )

    has_repair_target = bool(repair_blockers or _repair_guidance(candidate))
    has_evidence_progress = _has_evidence_progress(candidate)
    has_formal_or_source_progress = bool(candidate.formal_artifacts or candidate.source_bindings or candidate.evidence_refs or has_evidence_progress)
    nonempty = _candidate_nonempty(candidate)

    exploration_eligible = False
    parent_eligible = False
    repair_required = False

    exploration_eligible = nonempty and strict_rank
    parent_eligible = nonempty and strict_rank

    if repair_exhausted:
        notes.append("incubation_repair_budget_exhausted")
    elif repair_deprioritized:
        notes.append("repair_candidate_deprioritized_until_new_delta")
    elif has_any_progress_signal and (repair_attempts >= max_attempts or repeated_blockers >= max_repeated_blockers):
        notes.append("incubation_extended_by_observed_progress_signal")
    if has_repair_target:
        notes.append("verification_diagnostics_advisory_only")

    incubating = bool(repair_required and exploration_eligible and parent_eligible and not strict_final and not repair_exhausted)
    transition_reason = ""
    if incubating:
        transition_reason = "repairable_incomplete_candidate_kept_in_incubating_lane"
    elif repair_required and repair_exhausted:
        transition_reason = "incubation_budget_exhausted_demote_to_dormant"
    return EligibilityDecision(
        candidate_id=candidate.id,
        stage=stage,
        global_stage=global_stage,
        candidate_age_stage=age_stage,
        candidate_claim_stage=claim_stage,
        current_round=max(0, int(current_round or 0)),
        round_limit=max(1, int(round_limit or 1)),
        created_in_round=created,
        candidate_age=age,
        incubation_started_round=incubation_started_round,
        repair_attempts=repair_attempts,
        max_incubation_attempts=max_attempts,
        max_incubation_age=max_age,
        exploration_eligible=exploration_eligible,
        parent_eligible=parent_eligible,
        final_eligible=strict_final,
        strict_rank_eligible=strict_rank,
        strict_final_eligible=strict_final,
        repair_required=repair_required,
        incubating=incubating,
        repair_exhausted=repair_exhausted,
        repair_blockers=repair_blockers,
        state_transition_reason=transition_reason,
        reactivation_condition=reactivation_condition,
        notes=notes,
    )

def annotate_stage_eligibility(
    candidate: CandidateGenome,
    *,
    current_round: int = 0,
    round_limit: int = 0,
    policy_config: dict[str, Any] | None = None,
) -> EligibilityDecision:
    decision = stage_eligibility(candidate, current_round=current_round, round_limit=round_limit, policy_config=policy_config)
    _update_incubation_metadata(candidate, decision, current_round=current_round)
    candidate.metadata["stage_eligibility"] = decision.to_dict()
    if "repair_candidate_deprioritized_until_new_delta" in decision.notes:
        candidate.metadata["selection_deprioritized_until_new_delta"] = True
    else:
        candidate.metadata.pop("selection_deprioritized_until_new_delta", None)
    if decision.repair_required:
        candidate.metadata["repair_required"] = repair_requirement(candidate, decision=decision).to_dict()
    else:
        candidate.metadata.pop("repair_required", None)
    return decision

__all__ = [
    "annotate_stage_eligibility", "stage_eligibility",
    "strict_rank_eligible", "strict_final_eligible",
]
