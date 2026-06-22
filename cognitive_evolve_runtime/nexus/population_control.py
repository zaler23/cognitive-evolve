"""Adaptive live-population compaction for Nexus runs.

The runtime keeps durable archives and tombstones, but the live population used
for ranking, novelty, lineage saturation, and reproduction should not retain
every terminal clone forever.  This module deliberately avoids a fixed global
candidate cap: live size may grow with occupied quality-diversity bins, while
overfull clone-heavy bins are compacted into archives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment, TERMINAL_FAILURE_FATES
from cognitive_evolve_runtime.archives.quality_diversity import (
    candidate_bin_key,
    descriptor_cell_distribution,
    descriptor_population_entropy,
    quality_diversity_survivors,
)
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.nextgen import record_candidate_budget_decision, structurally_blocked
from cognitive_evolve_runtime.nexus.v23_theory_config import V23TheoryRuntimeConfig


@dataclass
class PopulationCompactionResult:
    removed_terminal_ids: list[str]
    compacted_clone_ids: list[str]
    live_population_size: int
    tombstone_count: int
    bin_capacity: int
    rare_reserve_per_bin: int
    reason: str = "v23_entropy_quality_diversity_live_compaction"
    population_entropy_before: float = 0.0
    population_entropy_after: float = 0.0
    descriptor_cell_count_before: int = 0
    descriptor_cell_count_after: int = 0
    v23_theory_config_hash: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.removed_terminal_ids or self.compacted_clone_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact_live_population(
    population: CandidatePopulation,
    archives: ArchiveManager,
    policy: EvolutionPolicy,
    *,
    branch_factor: int,
    round_index: int = 0,
) -> PopulationCompactionResult:
    """Move terminal/overfull candidates out of the live pool.

    Terminal candidates are tombstoned through ``ArchiveManager.update`` before
    removal.  Non-terminal compaction is per quality-diversity bin, so a run can
    still grow when it discovers genuinely new mechanisms or evidence shapes.
    """

    bin_capacity = _bin_capacity(policy=policy, branch_factor=branch_factor)
    rare_reserve = _rare_reserve(policy)
    terminal: list[CandidateGenome] = []
    live: list[CandidateGenome] = []
    for candidate in population.candidates:
        fate = CandidateFate.normalize(archives.fates.get(candidate.id, candidate.current_fate))
        if fate in TERMINAL_FAILURE_FATES and structurally_blocked(candidate):
            candidate.mark_fate(fate)
            terminal.append(candidate)
        else:
            if fate in TERMINAL_FAILURE_FATES:
                candidate.mark_fate(CandidateFate.DORMANT.value)
                record_candidate_budget_decision(candidate, source="live_compaction", reason="legacy_terminal_fate_reopened_as_dormant", action="soft_retain")
            live.append(candidate)
    if terminal:
        archives.update(
            [
                FateAssignment(
                    candidate_id=candidate.id,
                    fate=CandidateFate.normalize(candidate.current_fate),
                    failure_signature=str(candidate.metadata.get("live_compaction_failure_signature") or ""),
                )
                for candidate in terminal
            ],
            candidates=terminal,
        )

    best_answer = archives.best_answer_candidate(live)
    protected_ids = {best_answer.id} if best_answer is not None else set()
    protected: list[CandidateGenome] = []
    qd_input: list[CandidateGenome] = []
    for candidate in live:
        fate = CandidateFate.normalize(candidate.current_fate)
        if candidate.id in protected_ids or (fate == CandidateFate.DORMANT.value and archives.is_final_answer_eligible(candidate)):
            protected.append(candidate)
        else:
            qd_input.append(candidate)
    v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, branch_factor=branch_factor, population_size=len(qd_input))
    entropy_before = descriptor_population_entropy(qd_input)
    cells_before = len(descriptor_cell_distribution(qd_input))
    survivors, compacted = quality_diversity_survivors(
        qd_input,
        bin_capacity=bin_capacity,
        rare_reserve_per_bin=rare_reserve,
        config=v23_config,
    )
    entropy_after = descriptor_population_entropy(survivors)
    cells_after = len(descriptor_cell_distribution(survivors))
    if compacted:
        assignments: list[FateAssignment] = []
        for candidate in compacted:
            candidate.mark_fate(CandidateFate.DORMANT.value)
            lesson = "compacted_from_live_population_by_quality_diversity_bin_capacity"
            if lesson not in candidate.failure_lessons:
                candidate.failure_lessons.append(lesson)
            candidate.metadata["live_compaction"] = {
                "round": int(round_index or 0),
                "bin_capacity": bin_capacity,
                "bin_key": candidate_bin_key(candidate),
                "reason": lesson,
            }
            assignments.append(
                FateAssignment(
                    candidate_id=candidate.id,
                    fate=CandidateFate.DORMANT.value,
                    failure_signature=lesson,
                    future_reactivation_condition="reactivate_via_nextgen_budget_reserve",
                )
            )
            record_candidate_budget_decision(candidate, source="live_compaction", reason="quality_diversity_capacity_reserve", action="soft_reserve")
        archives.update(assignments, candidates=compacted)

    keep_ids = {candidate.id for candidate in protected}
    keep_ids.update(candidate.id for candidate in survivors)
    population.candidates = [candidate for candidate in population.candidates if candidate.id in keep_ids]
    return PopulationCompactionResult(
        removed_terminal_ids=[candidate.id for candidate in terminal],
        compacted_clone_ids=[candidate.id for candidate in compacted],
        live_population_size=len(population.candidates),
        tombstone_count=len(archives.terminal_tombstones),
        bin_capacity=bin_capacity,
        rare_reserve_per_bin=rare_reserve,
        population_entropy_before=entropy_before,
        population_entropy_after=entropy_after,
        descriptor_cell_count_before=cells_before,
        descriptor_cell_count_after=cells_after,
        v23_theory_config_hash=v23_config.config_hash,
    )


def _bin_capacity(*, policy: EvolutionPolicy, branch_factor: int) -> int:
    metadata = policy.metadata or {}
    for key in ("quality_diversity_bin_capacity", "live_bin_capacity"):
        try:
            value = int(metadata.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return max(1, value)
    return max(4, 2 * max(1, int(branch_factor or 1)))


def _rare_reserve(policy: EvolutionPolicy) -> int:
    try:
        value = int((policy.metadata or {}).get("quality_diversity_rare_reserve_per_bin", 1))
    except (TypeError, ValueError):
        value = 1
    return max(0, value)


__all__ = ["PopulationCompactionResult", "compact_live_population"]
