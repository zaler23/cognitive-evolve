"""Single-authority logical island allocation for reproduction slots."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import stable_hash

from .branch_allocator import (
    ProductiveBranchAllocation,
    allocate_productive_branches,
    count_observed_mechanism_families,
    lineage_root,
)


@dataclass(frozen=True)
class LogicalIslandAllocation:
    branches: ProductiveBranchAllocation
    candidate_islands: dict[str, int] = field(default_factory=dict)
    slot_islands: dict[str, int] = field(default_factory=dict)
    borrowed_parent_ids: dict[int, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": "single_nexus_global_rank_archive_finality",
            "island_count": len(set(self.candidate_islands.values())),
            "candidate_islands": dict(self.candidate_islands),
            "slot_islands": dict(self.slot_islands),
            "borrowed_parent_ids": {str(key): list(value) for key, value in self.borrowed_parent_ids.items()},
        }


def derive_island_count(*, total_slots: int, lineage_count: int, configured: Any = "auto") -> int:
    if total_slots <= 0 or lineage_count <= 0:
        return 0
    if configured in {None, "", "auto"}:
        requested = math.ceil(math.sqrt(lineage_count))
    else:
        try:
            requested = int(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError("policy.metadata['islands']['count'] must be a positive integer or 'auto'") from exc
        if requested <= 0:
            raise ValueError("policy.metadata['islands']['count'] must be positive")
    return min(total_slots, lineage_count, requested)


def assign_candidate_islands(
    candidates: Iterable[CandidateGenome],
    *,
    island_count: int,
) -> dict[str, int]:
    candidate_list = list(candidates)
    if island_count <= 1:
        return {candidate.id: 0 for candidate in candidate_list}
    roots = sorted({lineage_root(candidate) for candidate in candidate_list})
    root_islands = {root: index % island_count for index, root in enumerate(roots)}
    return {candidate.id: root_islands[lineage_root(candidate)] for candidate in candidate_list}


def allocate_logical_islands(
    *,
    parents: list[CandidateGenome],
    candidates: Iterable[CandidateGenome],
    total_slots: int,
    budget_history: Iterable[dict[str, Any]] = (),
    metric_directions: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    current_round: int = 0,
) -> LogicalIslandAllocation:
    candidate_list = list(candidates)
    observed_family_counts = count_observed_mechanism_families(candidate_list)
    parent_roots = sorted({lineage_root(parent) for parent in parents})
    island_count = derive_island_count(
        total_slots=total_slots,
        lineage_count=len(parent_roots),
        configured=dict(config or {}).get("count", "auto"),
    )
    if island_count <= 1:
        branches = allocate_productive_branches(
            parents=parents,
            candidates=candidate_list,
            budget_history=budget_history,
            metric_directions=metric_directions,
            total_slots=total_slots,
            observed_family_counts=observed_family_counts,
        )
        candidate_islands = {candidate.id: 0 for candidate in candidate_list}
        return LogicalIslandAllocation(
            branches=branches,
            candidate_islands=candidate_islands,
            slot_islands={slot.slot_id: 0 for slot in branches.slots},
        )

    preset_islands = {
        candidate.id: int(candidate.metadata["island_id"])
        for candidate in [*candidate_list, *parents]
        if isinstance(candidate.metadata, dict)
        and isinstance(candidate.metadata.get("island_id"), int)
        and 0 <= int(candidate.metadata["island_id"]) < island_count
    }
    root_islands = {root: index % island_count for index, root in enumerate(parent_roots)}
    candidate_islands = {}
    for candidate in candidate_list:
        island_id = preset_islands.get(candidate.id)
        if island_id is None:
            island_id = root_islands.get(
                lineage_root(candidate),
                int(stable_hash({"lineage_root": lineage_root(candidate)})[:16], 16) % island_count,
            )
        candidate_islands[candidate.id] = island_id
    parent_islands = {
        parent.id: preset_islands.get(parent.id, root_islands[lineage_root(parent)])
        for parent in parents
    }
    parent_groups = {
        island_id: [parent for parent in parents if parent_islands[parent.id] == island_id]
        for island_id in range(island_count)
    }
    candidate_groups = {
        island_id: [candidate for candidate in candidate_list if candidate_islands[candidate.id] == island_id]
        for island_id in range(island_count)
    }
    borrowed: dict[int, tuple[str, ...]] = {}
    migration_interval = _positive_int(dict(config or {}).get("migration_interval"))
    if migration_interval is not None and current_round > 0 and current_round % migration_interval == 0:
        for island_id in range(island_count):
            donor_id = (island_id - 1) % island_count
            if parent_groups[donor_id]:
                donor = max(parent_groups[donor_id], key=_grounded_parent_order)
                parent_groups[island_id] = [*parent_groups[island_id], donor]
                borrowed[island_id] = (donor.id,)

    base_slots, extra_slots = divmod(total_slots, island_count)
    branches = []
    arms = []
    credit: dict[str, int] = {}
    credited_transfer_artifact_hashes: set[str] = set()
    slot_islands: dict[str, int] = {}
    for island_id in range(island_count):
        island_slots = base_slots + int(island_id < extra_slots)
        allocation = allocate_productive_branches(
            parents=parent_groups[island_id],
            candidates=candidate_groups[island_id],
            budget_history=budget_history,
            metric_directions=metric_directions,
            total_slots=island_slots,
            observed_family_counts=observed_family_counts,
            credited_transfer_artifact_hashes=credited_transfer_artifact_hashes,
        )
        island_branches = [
            replace(
                slot,
                slot_id="branch-" + stable_hash({"island_id": island_id, "slot_id": slot.slot_id})[:16],
            )
            for slot in allocation.slots
        ]
        branches.extend(island_branches)
        arms.extend(allocation.arms)
        for key, value in allocation.credit_summary.items():
            credit[key] = credit.get(key, 0) + int(value)
        credited_transfer_artifact_hashes.update(allocation.credited_transfer_artifact_hashes)
        slot_islands.update({slot.slot_id: island_id for slot in island_branches})
    return LogicalIslandAllocation(
        branches=ProductiveBranchAllocation(
            slots=tuple(branches),
            arms=tuple(arms),
            credit_summary=credit,
            observed_family_counts=observed_family_counts,
            credited_transfer_artifact_hashes=tuple(sorted(credited_transfer_artifact_hashes)),
        ),
        candidate_islands=candidate_islands,
        slot_islands=slot_islands,
        borrowed_parent_ids=borrowed,
    )


def _grounded_parent_order(candidate: CandidateGenome) -> tuple[int, float, str]:
    evaluator = candidate.metadata.get("evaluator") if isinstance(candidate.metadata, dict) and isinstance(candidate.metadata.get("evaluator"), dict) else {}
    status = str(evaluator.get("status") or "").lower()
    tier = 2 if evaluator.get("passed") is True or status == "passed" else 0 if evaluator.get("passed") is False or status == "failed" else 1
    metrics = evaluator.get("metrics") if isinstance(evaluator.get("metrics"), dict) else {}
    value = metrics.get("score", candidate.multihead_scores.get("objective_score", 0.0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return tier, score if score == score else 0.0, candidate.id


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = ["LogicalIslandAllocation", "allocate_logical_islands", "assign_candidate_islands", "derive_island_count"]
