"""Constrained MMR selector for parent selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from .descriptor_cells import descriptor_cell_key
from .fingerprints import candidate_fingerprint
from .math_model import mmr_score, similarity
from .relevance import relevance_score


@dataclass
class DiverseSelectionTrace:
    selected_ids: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    lane_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"selected_ids": list(self.selected_ids), "rejected": list(self.rejected), "lane_counts": dict(self.lane_counts)}


def select_diverse(
    candidates: Iterable[CandidateGenome],
    *,
    limit: int,
    quality_fn: Callable[[CandidateGenome], float] | None = None,
    archives: Any | None = None,
    advisory_features: Mapping[str, Any] | None = None,
    eligibility_policy: Mapping[str, Any] | None = None,
) -> tuple[list[CandidateGenome], DiverseSelectionTrace]:
    pool = list(candidates)
    target = max(0, int(limit or 0))
    trace = DiverseSelectionTrace()
    if target <= 0 or not pool:
        return [], trace
    policy = coerce_dict(eligibility_policy)
    max_per_lineage = max(1, int(policy.get("max_per_lineage") or max(1, (target + 1) // 2)))
    max_per_cell = max(1, int(policy.get("max_per_descriptor_cell") or max(1, (target + 2) // 2)))
    selected: list[CandidateGenome] = []
    remaining: list[CandidateGenome] = []
    seen_ids: set[str] = set()
    for candidate in pool:
        if candidate.id in seen_ids:
            continue
        seen_ids.add(candidate.id)
        remaining.append(candidate)
    while remaining and len(selected) < target:
        best: CandidateGenome | None = None
        best_score = float("-inf")
        for candidate in remaining:
            reason = _constraint_reason(candidate, selected, max_per_lineage=max_per_lineage, max_per_cell=max_per_cell)
            if reason:
                continue
            q = _quality(candidate, quality_fn=quality_fn, advisory_features=advisory_features, archives=archives)
            rel = relevance_score(candidate)
            max_sim = max((similarity(candidate, other) for other in selected), default=0.0)
            score = mmr_score(quality=q, relevance=rel, max_similarity=max_sim)
            if _is_sparse_directive_target(candidate, archives):
                score += 0.08
            if score > best_score or (score == best_score and candidate.id < (best.id if best else "~")):
                best = candidate
                best_score = score
        if best is None:
            break
        selected.append(best)
        trace.selected_ids.append(best.id)
        cell = descriptor_cell_key(best)
        lineage = candidate_fingerprint(best).lineage_root
        trace.lane_counts[cell] = trace.lane_counts.get(cell, 0) + 1
        trace.lane_counts[f"lineage:{lineage}"] = trace.lane_counts.get(f"lineage:{lineage}", 0) + 1
        remaining = [candidate for candidate in remaining if candidate.id != best.id]
    if len(selected) < target:
        for candidate in remaining:
            if len(selected) >= target:
                break
            if candidate.id in {item.id for item in selected}:
                continue
            selected.append(candidate)
            trace.selected_ids.append(candidate.id)
            trace.rejected.append({"candidate_id": candidate.id, "reason": "constraint_relaxed_fill"})
    selected_ids = {candidate.id for candidate in selected}
    for candidate in pool:
        if candidate.id not in selected_ids:
            trace.rejected.append({"candidate_id": candidate.id, "reason": _constraint_reason(candidate, selected, max_per_lineage=max_per_lineage, max_per_cell=max_per_cell) or "lower_mmr_score"})
    return selected[:target], trace


def _quality(candidate: CandidateGenome, *, quality_fn: Callable[[CandidateGenome], float] | None, advisory_features: Mapping[str, Any] | None, archives: Any | None) -> float:
    base = quality_fn(candidate) if quality_fn is not None else _score_from_candidate(candidate)
    if advisory_features and candidate.id in advisory_features:
        feature = advisory_features[candidate.id]
        if isinstance(feature, Mapping):
            base += 0.05 * _bounded(feature.get("diversity")) + 0.05 * _bounded(feature.get("plan_value")) - 0.05 * _bounded(feature.get("risk"))
    qd = getattr(archives, "quality_diversity", None)
    if qd is not None and hasattr(qd, "directive_boost"):
        try:
            base += float(qd.directive_boost(candidate))
        except Exception:
            pass
    return max(0.0, min(1.0, float(base)))


def _score_from_candidate(candidate: CandidateGenome) -> float:
    axes = ("objective_alignment", "answer_likelihood", "verifiability", "frontier_score", "novelty", "rarity")
    if not candidate.multihead_scores:
        return 0.0
    return sum(_bounded(candidate.multihead_scores.get(axis)) for axis in axes) / len(axes)


def _constraint_reason(candidate: CandidateGenome, selected: list[CandidateGenome], *, max_per_lineage: int, max_per_cell: int) -> str:
    fp = candidate_fingerprint(candidate)
    cell = descriptor_cell_key(candidate)
    lineage_count = sum(1 for item in selected if candidate_fingerprint(item).lineage_root == fp.lineage_root)
    if lineage_count >= max_per_lineage:
        return "lineage_cap"
    cell_count = sum(1 for item in selected if descriptor_cell_key(item) == cell)
    if cell_count >= max_per_cell:
        return "descriptor_cell_cap"
    return ""


def _is_sparse_directive_target(candidate: CandidateGenome, archives: Any | None) -> bool:
    qd = getattr(archives, "quality_diversity", None)
    if qd is None:
        return False
    requests = getattr(qd, "rebalance_requests", []) or []
    if not requests:
        return False
    tokens = set(candidate_fingerprint(candidate).descriptor_tokens)
    cell = descriptor_cell_key(candidate).lower()
    for request in requests:
        if not isinstance(request, dict):
            continue
        descriptor = str(request.get("descriptor") or "").lower()
        if descriptor and (descriptor in cell or any(part and part in tokens for part in descriptor.replace("|", ":").split(":"))):
            return True
    return False


def _bounded(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = ["DiverseSelectionTrace", "select_diverse"]
