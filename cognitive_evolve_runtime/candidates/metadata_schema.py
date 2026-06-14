"""Candidate metadata schema helpers.

The runtime keeps ``metadata`` extensible, but known keys are centralized here so
new writes can be audited and tested instead of becoming an invisible schema.
"""
from __future__ import annotations

from typing import Any, Mapping

KNOWN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "active_repair_floor",
        "bootstrap_entry_survival",
        "claim_maturity_stage",
        "created_in_round",
        "dedupe_signature",
        "dormant_kind",
        "dormant_recovery_reject",
        "dormant_repair_reactivation",
        "evidence_need",
        "exploration_source",
        "failure_archive_reseed",
        "failure_classification",
        "failure_micro_guidance",
        "fate_transition_history",
        "final_answer_blocked_until_repaired",
        "generation_plan_fate",
        "generation_plan_id",
        "generation_plan_round",
        "generation_plan_source",
        "hard_reject_reason",
        "incubation_started_round",
        "latent_pareto_frontier",
        "latent_ranking",
        "max_incubation_age",
        "max_incubation_attempts",
        "model_critique_degraded",
        "model_offspring_degraded",
        "model_seed_batch",
        "model_seed_error",
        "ranking_schema_repair_error",
        "reactivated_in_round",
        "reactivation_condition",
        "repair_attempts",
        "repair_context",
        "repair_required",
        "repair_seed",
        "required_evidence_kinds",
        "score_source",
        "search_seed_not_final",
        "seed_type",
        "selection_deprioritized_until_new_delta",
        "selection_pressure",
        "source_grounding_required",
        "stage_eligibility",
        "state_transition_reason",
        "target_obligation_ids",
        "offspring_repair_lane",
        "offspring_verification",
    }
)


def unknown_metadata_keys(metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    return tuple(sorted(str(key) for key in metadata if str(key) not in KNOWN_METADATA_KEYS))


def metadata_audit(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    unknown = unknown_metadata_keys(metadata)
    return {"known_key_count": len(KNOWN_METADATA_KEYS), "unknown_keys": list(unknown), "has_unknown_keys": bool(unknown)}


__all__ = ["KNOWN_METADATA_KEYS", "metadata_audit", "unknown_metadata_keys"]
