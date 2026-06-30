"""Seed coverage helpers for wide self-bootstrap runs.

Metadata-only: these functions describe coverage and continuation pressure; they
never verify, solve, or gate candidates.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json

SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY = "_seed_reservoir_sidecar_payload"


def assess_seed_coverage(
    accepted: Iterable[CandidateGenome],
    *,
    reservoir: Iterable[CandidateGenome] | None = None,
    rejected: Iterable[dict[str, Any]] | None = None,
    harvest_summary: dict[str, Any] | None = None,
    previous: dict[str, Any] | None = None,
    contract: Any | None = None,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Return an open, numeric coverage snapshot; no finite family taxonomy."""

    accepted_items = list(accepted or [])
    reservoir_items = list(reservoir or [])
    families = Counter(_family_key(candidate) for candidate in accepted_items if _family_key(candidate))
    niches = Counter(niche for candidate in accepted_items for niche in (candidate.niche_memberships or [])[:4] if str(niche).strip())
    origins = Counter(str(coerce_dict(candidate.metadata).get("origin_model_index") or coerce_dict(candidate.metadata).get("origin_model") or "unknown") for candidate in accepted_items)
    target = _positive_int(coerce_dict(getattr(policy, "metadata", {})).get("initial_candidate_count")) or len(accepted_items)
    accepted_count = len(accepted_items)
    family_count = len(families)
    singleton_count = sum(1 for count in families.values() if count == 1)
    top_counts = [count for _, count in families.most_common(3)]
    top1_share = (top_counts[0] / max(1, accepted_count)) if top_counts else 0.0
    top3_share = (sum(top_counts) / max(1, accepted_count)) if top_counts else 0.0
    reservoir_count = len(reservoir_items)
    rejected_count = len([item for item in rejected or [] if isinstance(item, dict)])
    claim_counts = Counter(_claim_key(candidate) for candidate in accepted_items if _claim_key(candidate))
    exact_duplicate_excess = sum(max(0, count - 1) for count in claim_counts.values())
    previous_families = set()
    previous = coerce_dict(previous)
    if previous:
        previous_families.update(str(item[0] if isinstance(item, (list, tuple)) and item else item) for item in previous.get("top_families") or [])
    new_family_count = len(set(families) - previous_families) if previous_families else family_count
    summary = coerce_dict(harvest_summary)
    stop_reason = str(summary.get("stopped_reason") or "")
    family_floor = max(3, min(12, accepted_count // 4 or 1))
    broad_enough = bool(
        accepted_count >= max(1, min(target, 16))
        and family_count >= family_floor
        and (singleton_count >= max(1, family_count // 5) or family_count >= 8)
    )
    status = "broad" if broad_enough else "thin"
    if accepted_count == 0:
        status = "empty"
    undercovered_family_signals = _undercovered_family_signals(families)
    return {
        "schema": "seed_coverage.v1",
        "candidate_count": accepted_count,
        "accepted_count": accepted_count,
        "reservoir_count": reservoir_count,
        "rejected_count": rejected_count,
        "family_count": family_count,
        "singleton_family_count": singleton_count,
        "top1_family_share": round(top1_share, 4),
        "top3_family_share": round(top3_share, 4),
        "exact_claim_duplicate_excess": exact_duplicate_excess,
        "new_family_count": new_family_count,
        "top_families": families.most_common(16),
        "top_niches": niches.most_common(12),
        "origin_model_counts": dict(origins),
        "stopped_reason": stop_reason,
        "partial_failure_count": len(summary.get("failed_batch_ids") or []),
        "status": status,
        "coverage_status": status,
        "needs_more_seed": status != "broad" and stop_reason != "fatal_model_error",
        "needs_more_seed_reason": "coverage_thin" if status != "broad" and stop_reason != "fatal_model_error" else "",
        "needs_target_perturb": status != "broad" or top1_share >= 0.35 or top3_share >= 0.60,
        "reasons": _coverage_reasons(status=status, top1_share=top1_share, top3_share=top3_share, duplicate_excess=exact_duplicate_excess),
        "undercovered_family_signals": undercovered_family_signals,
        "novelty_debt": _novelty_debt(families, accepted_count=accepted_count, family_floor=family_floor, undercovered_family_signals=undercovered_family_signals),
        "fingerprint": stable_hash({"families": families.most_common(), "accepted_count": accepted_count}),
        "policy": "descriptive_only_no_seed_cap_gate",
    }


def target_perturb_seed_judgment(
    candidates: Iterable[CandidateGenome],
    *,
    coverage: dict[str, Any] | None = None,
    baseline_family_count: int = 0,
    baseline_seed_count: int = 0,
    current_round: int = 0,
    diagnosis: Any | None = None,
    best_current_history: Iterable[Any] | None = None,
    generation_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend, but never run, a target-perturb seed continuation."""

    items = list(candidates or [])
    families = {_family_key(candidate) for candidate in items if _family_key(candidate)}
    new_generations = [candidate for candidate in items if int(getattr(candidate, "generation", 0) or 0) > 0]
    best_ids = [candidate.id for candidate in items if coerce_dict(candidate.metadata).get("best_current_direction")]
    diag_text = str(getattr(diagnosis, "stagnation_type", "") or coerce_dict(diagnosis).get("stagnation_type") or "").lower()
    loop_hint = bool(diag_text and diag_text not in {"none", "healthy", "not_needed"})
    family_expansion = len(families) - int(baseline_family_count or 0)
    novelty_expansion = sum(1 for candidate in new_generations if candidate.novelty_descriptors or candidate.edge_knowledge_seeds)
    coverage = coerce_dict(coverage)
    history = [str(item or "") for item in best_current_history or [] if str(item or "").strip()]
    best_stuck = len(history) >= 3 and len(set(history[-3:])) == 1
    stats = coerce_dict(generation_stats)
    top_share = _float(coverage.get("top1_family_share"), 0.0) or _float(stats.get("top1_family_share"), 0.0)
    thin = str(coverage.get("coverage_status") or coverage.get("status") or "").lower() in {"thin", "undercovered", "watch"}
    novelty_decline = _float(stats.get("new_generation_novelty"), 1.0) <= 0.05
    stuck = int(current_round or 0) >= 10 and (family_expansion <= max(1, int(baseline_family_count or 0) // 10) or best_stuck or top_share >= 0.35 or novelty_decline)
    reasons = []
    if thin:
        reasons.append("seed_coverage_not_broad")
    if best_stuck:
        reasons.append("best_current_stuck")
    if top_share >= 0.35:
        reasons.append("top_family_concentration")
    if novelty_decline:
        reasons.append("generation_novelty_decline")
    if loop_hint:
        reasons.append("diagnosis_loop_or_collapse")
    judgment = "trigger_recommended" if stuck and (loop_hint or thin or best_stuck or top_share >= 0.35) else ("watch" if stuck or loop_hint or thin else "not_needed")
    return {
        "schema": "target_perturb_seed_judgment.v1",
        "judgment": judgment,
        "current_round": int(current_round or 0),
        "baseline_family_count": int(baseline_family_count or 0),
        "baseline_seed_count": int(baseline_seed_count or 0),
        "current_family_count": len(families),
        "family_expansion": family_expansion,
        "new_generation_count": len(new_generations),
        "novelty_expansion_count": novelty_expansion,
        "best_direction_markers": best_ids[:5],
        "diagnosis_hint": diag_text[:160],
        "reasons": reasons,
        "evidence": {
            "coverage_status": coverage.get("coverage_status") or coverage.get("status"),
            "top1_family_share": top_share,
            "best_current_stuck": best_stuck,
            "novelty_decline": novelty_decline,
        },
        "suggested_prompt_delta": "Generate target-perturb seeds that directly answer the frozen goal while avoiding the current dominant basin and reusing only useful loser-pool factors.",
        "policy": "recommend_only_resume_from_latest_checkpoint",
    }


def seed_reservoir_sidecar_payload(reservoir: Iterable[CandidateGenome]) -> list[dict[str, Any]]:
    return [candidate.to_dict() for candidate in reservoir or []]


def persist_seed_reservoir_sidecar(output_dir: str | Path, payload: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    items = [dict(item) for item in payload or [] if isinstance(item, dict)]
    if not items:
        return {}
    digest = stable_hash({"seed_reservoir": items})
    path = Path(output_dir) / f"seed-reservoir-{digest[:16]}.json"
    atomic_write_json(path, {"schema": "seed_reservoir_sidecar.v1", "digest": digest, "count": len(items), "candidates": items}, sort_keys=True)
    return {"sidecar_schema": "seed_reservoir_sidecar.v1", "path": str(path), "digest": digest, "count": len(items)}


def _family_key(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    nextgen = coerce_dict(metadata.get("nextgen"))
    search_space = coerce_dict(metadata.get("search_space"))
    for value in (
        nextgen.get("canonical_mechanism_family_id"),
        nextgen.get("mechanism_family_id"),
        search_space.get("family_id"),
        search_space.get("plane_id"),
        candidate.niche_memberships[0] if candidate.niche_memberships else "",
        candidate.lineage[0] if candidate.lineage else "",
        candidate.core_mechanism,
    ):
        text = str(value or "").strip()
        if text:
            return text[:160]
    return ""


def _claim_key(candidate: CandidateGenome) -> str:
    return " ".join(str(candidate.concise_claim or candidate.core_mechanism or candidate.artifact or "").lower().split())[:240]


def _coverage_reasons(*, status: str, top1_share: float, top3_share: float, duplicate_excess: int) -> list[str]:
    out: list[str] = []
    if status != "broad":
        out.append("coverage_not_broad")
    if top1_share >= 0.35:
        out.append("top_family_concentration")
    if top3_share >= 0.60:
        out.append("top3_family_concentration")
    if duplicate_excess:
        out.append("exact_claim_duplicates")
    return out


def _undercovered_family_signals(families: Counter[str]) -> list[dict[str, Any]]:
    if not families:
        return []
    medianish = sorted(families.values())[len(families) // 2]
    return [{"family": family, "count": count, "reason": "singleton_or_below_median"} for family, count in families.items() if count <= max(1, medianish)][:16]


def _novelty_debt(
    families: Counter[str],
    *,
    accepted_count: int,
    family_floor: int,
    undercovered_family_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    overrepresented = []
    for family, count in families.most_common(16):
        share = count / max(1, accepted_count)
        if share >= 0.35:
            overrepresented.append(
                {
                    "family": family,
                    "count": count,
                    "share": round(share, 4),
                    "excess_share": round(share - 0.35, 4),
                    "reason": "top_family_concentration",
                }
            )
    missing_family_count = max(0, int(family_floor or 0) - len(families))
    score = round(sum(item["excess_share"] for item in overrepresented) + missing_family_count / max(1, family_floor), 4)
    return {
        "schema": "seed_coverage.novelty_debt.v1",
        "status": "watch" if score > 0 else "clear",
        "score": score,
        "overrepresented_families": overrepresented,
        "missing_family_count": missing_family_count,
        "undercovered_family_signals": list(undercovered_family_signals[:16]) if missing_family_count else [],
        "policy": "advisory_metadata_only_no_gate",
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY",
    "assess_seed_coverage",
    "persist_seed_reservoir_sidecar",
    "seed_reservoir_sidecar_payload",
    "target_perturb_seed_judgment",
]
