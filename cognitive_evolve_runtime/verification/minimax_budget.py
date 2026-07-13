"""Minimax-style adversarial budget allocation for verifier obligations."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus.v23_theory_config import MinimaxBudgetConfig
from cognitive_evolve_runtime.verification.strength import candidate_verification_strength


def _allocate_adversarial_budget(
    candidates: list[Any],
    base_budget: int,
    config: MinimaxBudgetConfig | None = None,
    total_override: int | None = None,
) -> dict[str, int]:
    cfg = config or MinimaxBudgetConfig()
    ordered = [candidate for candidate in candidates if getattr(candidate, "id", "")]
    if not ordered:
        return {}
    floor = max(0, int(cfg.min_budget_per_candidate or 0))
    total = int(total_override) if total_override is not None else max(floor, int(base_budget or 0)) * len(ordered)
    total = max(total, floor * len(ordered))
    scores = {candidate.id: int(candidate_verification_strength(candidate)) for candidate in ordered}
    if not any(scores.values()):
        return _uniform_budget([candidate.id for candidate in ordered], total=total, floor=floor)
    base_total = floor * len(ordered)
    remaining = max(0, total - base_total)
    score_sum = sum(scores.values())
    raw: dict[str, float] = {cid: (remaining * value / score_sum if score_sum else 0.0) for cid, value in scores.items()}
    out = {cid: floor + int(raw[cid]) for cid in scores}
    residue = total - sum(out.values())
    order = sorted(scores, key=lambda cid: (raw[cid] - int(raw[cid]), scores[cid], cid), reverse=True)
    for cid in order[:residue]:
        out[cid] += 1
    return out


def _uniform_budget(candidate_ids: list[str], *, total: int, floor: int) -> dict[str, int]:
    if not candidate_ids:
        return {}
    base = max(floor, total // len(candidate_ids))
    out = {cid: base for cid in candidate_ids}
    residue = max(0, total - sum(out.values()))
    for cid in sorted(candidate_ids)[:residue]:
        out[cid] += 1
    return out


def allocation_summary(allocation: dict[str, int], *, base_budget: int, total_override: int | None = None) -> dict[str, Any]:
    return {
        "base_budget": int(base_budget or 0),
        "total_override": total_override,
        "allocated_total": sum(int(v) for v in allocation.values()),
        "candidate_count": len(allocation),
        "candidate_budgets": dict(allocation),
    }


__all__ = ["_allocate_adversarial_budget", "allocation_summary"]
