"""Concept authority contracts for the v2 research bus."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


class AuthorityLevel(IntEnum):
    OBSERVE = 0
    ADVISE = 1
    SHAPE = 2
    GATE = 3
    VERIFY = 4


CHANNEL_AUTHORITY: dict[str, int] = {
    "metrics": int(AuthorityLevel.OBSERVE),
    "warnings": int(AuthorityLevel.OBSERVE),
    "selection_advisory": int(AuthorityLevel.ADVISE),
    "search_pressures": int(AuthorityLevel.SHAPE),
    "archive_directives": int(AuthorityLevel.SHAPE),
    "budget_directives": int(AuthorityLevel.SHAPE),
    "context_transforms": int(AuthorityLevel.SHAPE),
    "candidate_transforms": int(AuthorityLevel.SHAPE),
    # Evidence records alter candidate evidence state and therefore sit behind
    # the same gate-level permission as final-blocking directives.
    "evidence_records": int(AuthorityLevel.GATE),
    "final_gate_directives": int(AuthorityLevel.GATE),
    "verification_obligations": int(AuthorityLevel.VERIFY),
    # Proposal channels are deliberately out of the linear comparison.
    "contract_delta_proposals": -1,
}

PROPOSAL_CHANNELS = frozenset({"contract_delta_proposals"})
_BASE_PRODUCES = frozenset({"metrics", "warnings"})


@dataclass(frozen=True)
class ConceptContract:
    concept_id: str
    consumes: frozenset[str]
    produces: frozenset[str]
    max_authority: AuthorityLevel
    replay_required: bool
    ablation_metric: str
    falsification_metric: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["consumes"] = sorted(self.consumes)
        data["produces"] = sorted(self.produces)
        data["max_authority"] = self.max_authority.name
        data["max_authority_value"] = int(self.max_authority)
        return data


def _contract(
    concept_id: str,
    *,
    authority: AuthorityLevel,
    produces: set[str] | frozenset[str] = frozenset(),
    consumes: set[str] | frozenset[str] = frozenset(),
    replay_required: bool = False,
    ablation_metric: str = "decision_changed_per_cost",
    falsification_metric: str = "authority_violation_count",
) -> ConceptContract:
    return ConceptContract(
        concept_id=concept_id,
        consumes=frozenset(consumes),
        produces=frozenset(_BASE_PRODUCES | frozenset(produces)),
        max_authority=authority,
        replay_required=bool(replay_required),
        ablation_metric=ablation_metric,
        falsification_metric=falsification_metric,
    )


CONTRACTS: dict[str, ConceptContract] = {
    "noop": _contract("noop", authority=AuthorityLevel.OBSERVE, ablation_metric="noop_cost"),
    "immune_necropsy": _contract(
        "immune_necropsy",
        authority=AuthorityLevel.VERIFY,
        produces={"selection_advisory", "search_pressures", "final_gate_directives", "verification_obligations"},
        replay_required=True,
        ablation_metric="repeat_failure_recurrence_down",
        falsification_metric="regression_replay_failure_count",
    ),
    "parameter_sweep": _contract(
        "parameter_sweep",
        authority=AuthorityLevel.GATE,
        produces={"selection_advisory", "search_pressures", "evidence_records", "candidate_transforms", "final_gate_directives"},
        replay_required=True,
        ablation_metric="uncollapsed_candidate_block_count",
        falsification_metric="sweep_replay_divergence_count",
    ),
    "bft_quorum": _contract(
        "bft_quorum",
        authority=AuthorityLevel.GATE,
        produces={"final_gate_directives"},
        replay_required=True,
        ablation_metric="overclaim_intercept_count",
        falsification_metric="quorum_disagreement_rate",
    ),
    "chaos": _contract(
        "chaos",
        authority=AuthorityLevel.GATE,
        produces={"final_gate_directives"},
        replay_required=True,
        ablation_metric="flaky_detection_precision",
        falsification_metric="chaos_replay_instability_count",
    ),
    "mdl_compression": _contract(
        "mdl_compression",
        authority=AuthorityLevel.GATE,
        produces={"selection_advisory", "search_pressures", "candidate_transforms", "final_gate_directives"},
        replay_required=False,
        ablation_metric="description_length_down_without_score_regression",
        falsification_metric="compression_score_regression_count",
    ),
    "budget_backpressure": _contract(
        "budget_backpressure",
        authority=AuthorityLevel.SHAPE,
        produces={"selection_advisory", "search_pressures", "budget_directives"},
        ablation_metric="frontier_gain_per_compute",
        falsification_metric="negative_roi_shift_count",
    ),
    "context_pruning": _contract(
        "context_pruning",
        authority=AuthorityLevel.SHAPE,
        produces={"search_pressures", "context_transforms"},
        replay_required=True,
        ablation_metric="critical_counterevidence_retention_rate",
        falsification_metric="protected_ref_drop_count",
    ),
    "pattern_memory": _contract(
        "pattern_memory",
        authority=AuthorityLevel.SHAPE,
        produces={"selection_advisory", "search_pressures", "archive_directives"},
        ablation_metric="verified_motif_transfer_gain",
        falsification_metric="quarantined_pattern_reuse_count",
    ),
    "spatial_selection": _contract(
        "spatial_selection",
        authority=AuthorityLevel.SHAPE,
        produces={"selection_advisory", "archive_directives"},
        ablation_metric="occupied_cells_times_best_score",
        falsification_metric="diversity_collapse_count",
    ),
    "latent_outcomes": _contract(
        "latent_outcomes",
        authority=AuthorityLevel.SHAPE,
        produces={"archive_directives"},
        ablation_metric="latent_axis_future_failure_predictiveness",
        falsification_metric="correlation_marked_as_causal_count",
    ),
    "contract_refinement": _contract(
        "contract_refinement",
        authority=AuthorityLevel.OBSERVE,
        produces={"contract_delta_proposals"},
        replay_required=True,
        ablation_metric="silent_objective_mutation_count",
        falsification_metric="proposal_direct_apply_count",
    ),
    "verification_synthesizer": _contract(
        "verification_synthesizer",
        authority=AuthorityLevel.VERIFY,
        produces={"verification_obligations"},
        replay_required=True,
        ablation_metric="verification_strength_gain_per_cost",
        falsification_metric="unreplayable_strength_claim_count",
    ),
}


def contract_for(concept_id: str) -> ConceptContract:
    normalized = str(concept_id or "").strip()
    if normalized in CONTRACTS:
        return CONTRACTS[normalized]
    return _contract(normalized or "unknown", authority=AuthorityLevel.OBSERVE, falsification_metric="unknown_concept_signal_count")


__all__ = [
    "AuthorityLevel",
    "CHANNEL_AUTHORITY",
    "CONTRACTS",
    "ConceptContract",
    "PROPOSAL_CHANNELS",
    "contract_for",
]
