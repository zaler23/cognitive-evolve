"""Quality-diversity helpers for Nexus genomes.

This module deliberately separates *adaptive diversity pressure* from a fixed
global population cap.  A Nexus run may keep growing across genuinely different
niches, but clone-heavy bins should not keep every low-value live genome forever.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.adaptive_signals import in_top_band, score
from cognitive_evolve_runtime.core.scalars import bounded_score


def candidate_quality(candidate: CandidateGenome) -> float:
    return candidate_final_quality(candidate)


def candidate_final_quality(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    axes = [
        "final_verification",
        "final_confidence",
        "objective_alignment",
        "answer_likelihood",
        "core_mechanism_strength",
        "verifiability",
        "tool_progress",
        "proof_progress",
        "evidence_progress",
        "robustness",
    ]
    if not scores:
        return 0.0
    return sum(_score(scores.get(axis, 0.0)) for axis in axes) / max(1, len(axes))


def candidate_search_quality(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    axes = [
        "frontier_score",
        "challenge_pass_rate",
        "challenge_resolution",
        "schema_cleanliness",
        "continuation_value",
        "novelty",
        "rarity",
        "repair_value",
    ]
    if not scores:
        return candidate_final_quality(candidate)
    return sum(_score(scores.get(axis, 0.0)) for axis in axes) / max(1, len(axes))


def candidate_bin_key(candidate: CandidateGenome) -> str:
    """Return the MAP-Elites-style live bin for a candidate.

    The key is intentionally coarse and semantic.  It avoids creating a new bin
    for every tiny wording change while still preserving rare mechanisms,
    evidence-bearing variants, and project/source-grounded candidates.
    """

    primary = (
        (candidate.niche_memberships[0] if candidate.niche_memberships else "")
        or (candidate.novelty_descriptors[0] if candidate.novelty_descriptors else "")
        or candidate.core_mechanism
        or candidate.artifact_type
        or "general"
    )
    evidence_shape = "evidence" if (candidate.evidence_delta or candidate.evidence_refs or candidate.source_bindings) else "proposal"
    rare_shape = "rare" if (candidate.edge_knowledge_seeds or score(candidate, "rarity") > 0 and score(candidate, "rarity") >= score(candidate, "novelty")) else "common"
    return "|".join(_token(item) for item in (primary, evidence_shape, rare_shape) if item)


def live_reproductive_candidates(candidates: list[CandidateGenome]) -> list[CandidateGenome]:
    """Candidates allowed to influence live novelty, saturation, and parents."""

    allowed = {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
    return [candidate for candidate in candidates if CandidateFate.normalize(candidate.current_fate) in allowed]


def pareto_frontier_ids(candidates: list[CandidateGenome], *, max_axes: int = 3) -> set[str]:
    """Return a conservative low-dimensional Pareto frontier.

    High-dimensional Pareto ranking makes almost everything non-dominated.  The
    runtime therefore uses at most three aggregate axes and treats the frontier
    as a small parent/archive diversity signal, not as final-answer proof.
    """

    axes = _pareto_vectors(candidates, max_axes=max_axes)
    frontier: set[str] = set()
    for candidate in candidates:
        vector = axes.get(candidate.id)
        if vector is None:
            continue
        dominated = False
        for other in candidates:
            if other.id == candidate.id:
                continue
            other_vector = axes.get(other.id)
            if other_vector is None:
                continue
            if _dominates(other_vector, vector):
                dominated = True
                break
        if not dominated:
            frontier.add(candidate.id)
    return frontier


def quality_diversity_survivors(
    candidates: list[CandidateGenome],
    *,
    bin_capacity: int,
    rare_reserve_per_bin: int = 1,
) -> tuple[list[CandidateGenome], list[CandidateGenome]]:
    """Select live survivors by niche bin and return ``(survivors, compacted)``.

    This is not a fixed total population cap: the number of live candidates can
    grow with the number of occupied bins.  Inside an overfull bin, it keeps the
    strongest candidates, a small rare reserve, and low-dimensional Pareto
    frontier members.
    """

    capacity = max(1, int(bin_capacity or 1))
    rare_reserve = max(0, int(rare_reserve_per_bin or 0))
    bins: dict[str, list[CandidateGenome]] = {}
    for candidate in candidates:
        bins.setdefault(candidate_bin_key(candidate), []).append(candidate)
    frontier_ids = pareto_frontier_ids(candidates)
    survivors: list[CandidateGenome] = []
    compacted: list[CandidateGenome] = []
    seen: set[str] = set()
    for bin_candidates in bins.values():
        if len(bin_candidates) <= capacity:
            chosen = list(bin_candidates)
        else:
            chosen = sorted(bin_candidates, key=candidate_quality, reverse=True)[:capacity]
            rare = [
                candidate
                for candidate in sorted(bin_candidates, key=lambda c: c.multihead_scores.get("rarity", 0.0), reverse=True)
                if (candidate.edge_knowledge_seeds or (score(candidate, "rarity") > 0 and in_top_band(candidate, bin_candidates, "rarity")))
            ][:rare_reserve]
            frontier = sorted(
                [candidate for candidate in bin_candidates if candidate.id in frontier_ids],
                key=candidate_quality,
                reverse=True,
            )[:rare_reserve]
            chosen = _dedupe_candidates(chosen + rare + frontier)
        chosen_ids = {candidate.id for candidate in chosen}
        for candidate in bin_candidates:
            if candidate.id in chosen_ids:
                if candidate.id not in seen:
                    survivors.append(candidate)
                    seen.add(candidate.id)
            else:
                compacted.append(candidate)
    return survivors, compacted


@dataclass
class QualityDiversityArchive:
    elites_by_niche: dict[str, dict[str, Any]] = field(default_factory=dict)

    def update(self, candidate: CandidateGenome) -> None:
        niches = candidate.niche_memberships or candidate.novelty_descriptors or [candidate.core_mechanism or "general"]
        final_quality = candidate_final_quality(candidate)
        search_quality = candidate_search_quality(candidate)
        quality = final_quality
        for niche in niches:
            current = self.elites_by_niche.get(niche)
            current_score = max(float(current.get("quality", -1.0)), float(current.get("search_quality", -1.0))) if current else -1.0
            if current is None or max(final_quality, search_quality) >= current_score:
                self.elites_by_niche[niche] = {
                    "candidate_id": candidate.id,
                    "quality": quality,
                    "search_quality": search_quality,
                    "final_quality": final_quality,
                    "bin_key": candidate_bin_key(candidate),
                    "candidate": candidate.to_dict(),
                }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityDiversityArchive":
        return cls(elites_by_niche=dict(data.get("elites_by_niche") or {}))


def _score(value: Any) -> float:
    return bounded_score(value)


def _token(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("|", " ").split()) or "general"


def _pareto_vectors(candidates: list[CandidateGenome], *, max_axes: int) -> dict[str, tuple[float, ...]]:
    axis_count = max(1, min(3, int(max_axes or 3)))
    vectors: dict[str, tuple[float, ...]] = {}
    for candidate in candidates:
        scores = candidate.multihead_scores
        solution_quality = (_score(scores.get("objective_alignment")) + _score(scores.get("answer_likelihood"))) / 2.0
        evidence_progress = max(_score(scores.get("evidence_progress")), _score(scores.get("proof_progress")), _score(scores.get("tool_progress")))
        diversity = max(_score(scores.get("novelty")), _score(scores.get("rarity")), 1.0 if candidate.edge_knowledge_seeds else 0.0)
        vectors[candidate.id] = (evidence_progress, solution_quality, diversity)[:axis_count]
    return vectors


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(left >= right for left, right in zip(a, b)) and any(left > right for left, right in zip(a, b))


def _dedupe_candidates(candidates: list[CandidateGenome]) -> list[CandidateGenome]:
    out: list[CandidateGenome] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        out.append(candidate)
    return out


__all__ = [
    "QualityDiversityArchive",
    "candidate_bin_key",
    "candidate_final_quality",
    "candidate_quality",
    "candidate_search_quality",
    "live_reproductive_candidates",
    "pareto_frontier_ids",
    "quality_diversity_survivors",
]
