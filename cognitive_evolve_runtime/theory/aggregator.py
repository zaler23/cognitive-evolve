"""Deterministic aggregation of advisory-only theory signals."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .config import TheoryConfig
from .signals import AdvisoryRankingFeatures, TheorySignal


def aggregate_advisory_features(signals: Iterable[TheorySignal], config: TheoryConfig | None = None) -> dict[str, AdvisoryRankingFeatures]:
    cfg = config or TheoryConfig()
    grouped: dict[str, list[TheorySignal]] = defaultdict(list)
    for signal in signals:
        if signal.target_type == "candidate":
            grouped[signal.target_id].append(signal)
    result: dict[str, AdvisoryRankingFeatures] = {}
    for candidate_id in sorted(grouped):
        rank_prior = 0.0
        plan_value = 0.0
        risk = 0.0
        diversity = 0.0
        provenance: list[str] = []
        for signal in sorted(grouped[candidate_id], key=lambda item: (item.source, item.kind, item.target_id, item.value, item.confidence)):
            weighted = _clamp(signal.value * signal.confidence * cfg.weight_for(signal.source), cfg.clamp_min, cfg.clamp_max)
            if signal.kind == "rank_prior":
                rank_prior += weighted
            elif signal.kind == "plan_value":
                plan_value += weighted
            elif signal.kind == "risk":
                risk += weighted
            elif signal.kind == "diversity":
                diversity += weighted
            provenance.extend(signal.provenance or (f"theory:{signal.source}:{signal.kind}",))
        result[candidate_id] = AdvisoryRankingFeatures(
            candidate_id=candidate_id,
            rank_prior=_clamp(rank_prior, cfg.clamp_min, cfg.clamp_max),
            plan_value=_clamp(plan_value, cfg.clamp_min, cfg.clamp_max),
            risk=_clamp(risk, cfg.clamp_min, cfg.clamp_max),
            diversity=_clamp(diversity, cfg.clamp_min, cfg.clamp_max),
            provenance=tuple(dict.fromkeys(provenance)),
        )
    return result


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    lower, upper = (lo, hi) if lo <= hi else (hi, lo)
    return max(lower, min(upper, float(value)))


__all__ = ["aggregate_advisory_features"]
