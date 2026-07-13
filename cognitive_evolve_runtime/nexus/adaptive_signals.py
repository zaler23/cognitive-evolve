"""Self-observed relative signals for Nexus runtime decisions."""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome


def score(candidate: CandidateGenome, axis: str, *, default: float = 0.0) -> float:
    try:
        value = float(candidate.multihead_scores.get(axis, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        return float(default)
    return max(0.0, min(1.0, value))


def score_values(candidates: Iterable[CandidateGenome], axis: str) -> list[float]:
    return [score(candidate, axis) for candidate in candidates]


def percentile_rank(candidate: CandidateGenome, population: list[CandidateGenome], axis: str) -> float:
    if not population:
        return score(candidate, axis)
    values = sorted(score_values(population, axis))
    value = score(candidate, axis)
    if len(values) <= 1:
        return value
    below = sum(1 for item in values if item < value)
    equal = sum(1 for item in values if item == value)
    return (below + (equal - 1) / 2) / max(1, len(values) - 1)


def mean_percentile(candidate: CandidateGenome, population: list[CandidateGenome], axes: Iterable[str]) -> float:
    signals = [percentile_rank(candidate, population, axis) for axis in axes]
    return sum(signals) / len(signals) if signals else 0.0


def top_band_cutoff(candidates: list[CandidateGenome], axis: str) -> float | None:
    values = sorted(score_values(candidates, axis))
    if not values:
        return None
    if len(values) == 1 or values[0] == values[-1]:
        return values[-1]
    upper = values[len(values) // 2 :]
    return float(median(upper))


def bottom_band_cutoff(candidates: list[CandidateGenome], axis: str) -> float | None:
    values = sorted(score_values(candidates, axis))
    if not values:
        return None
    if len(values) == 1 or values[0] == values[-1]:
        return values[0]
    lower = values[: max(1, (len(values) + 1) // 2)]
    return float(median(lower))


def in_top_band(candidate: CandidateGenome, population: list[CandidateGenome], axis: str) -> bool:
    cutoff = top_band_cutoff(population, axis)
    return False if cutoff is None else score(candidate, axis) >= cutoff


def in_bottom_band(candidate: CandidateGenome, population: list[CandidateGenome], axis: str) -> bool:
    cutoff = bottom_band_cutoff(population, axis)
    return False if cutoff is None else score(candidate, axis) <= cutoff


def observed_majority(part: int, whole: int) -> bool:
    return whole > 0 and part > (whole - part)


def observed_frontier_signal(candidate: CandidateGenome, population: list[CandidateGenome]) -> bool:
    return bool(
        candidate.edge_knowledge_seeds
        or candidate.formal_artifacts
        or candidate.obligation_delta
        or candidate.evidence_delta
        or candidate.evidence_refs
        or candidate.source_bindings
        or (score(candidate, "rarity") > 0 and in_top_band(candidate, population, "rarity"))
        or (score(candidate, "novelty") > 0 and in_top_band(candidate, population, "novelty"))
    )


def adaptive_attempt_limit(*, population_size: int = 0, distinct_blockers: int = 0, configured: Any = None, fallback: int = 0) -> int:
    try:
        parsed = int(configured)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    breadth = max(1, int(population_size or 1))
    blockers = max(1, int(distinct_blockers or 1))
    observed = 1 + int(math.log2(breadth + blockers))
    if fallback and fallback > 0:
        return max(1, max(observed, int(fallback)))
    return max(1, observed)


__all__ = [
    "adaptive_attempt_limit",
    "bottom_band_cutoff",
    "in_bottom_band",
    "in_top_band",
    "mean_percentile",
    "observed_frontier_signal",
    "observed_majority",
    "percentile_rank",
    "score",
    "score_values",
    "top_band_cutoff",
]
