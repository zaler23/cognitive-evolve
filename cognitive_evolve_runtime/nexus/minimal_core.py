"""Deterministic minimal-core ablation helpers.

These helpers compare open strategy profiles on the same candidate pool.  They
produce advisory evidence only; they do not verify, solve, or add a second final
authority.
"""
from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.nextgen import structurally_blocked

ABLATION_PROFILES = ("score_only", "Nexus_QD_failure_replay", "minimal_active_core", "full_fusion")


def run_core_ablation(
    candidates: Iterable[CandidateGenome],
    *,
    archives: Any | None = None,
    policy: Any | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    items = [candidate for candidate in candidates or [] if _has_claim(candidate) and not structurally_blocked(candidate)]
    if not items:
        return {"schema": "minimal_core_ablation.v1", "profiles": {}, "recommendation": "no_candidate", "policy": "advisory_not_verified"}
    factors = extract_failure_theorems(items)
    profile_results = {profile: _profile_result(profile, items, factors=factors, limit=limit) for profile in ABLATION_PROFILES}
    minimal = profile_results["minimal_active_core"].get("best_score", 0.0)
    fusion = profile_results["full_fusion"].get("best_score", 0.0)
    recommendation = "full_fusion_has_incremental_signal" if fusion > minimal + 0.08 else "minimal_core_first"
    families = Counter(_family_key(candidate) for candidate in items)
    return {
        "schema": "minimal_core_ablation.v1",
        "profiles": profile_results,
        "failure_theorems": factors[:16],
        "population_metrics": {
            "candidate_count": len(items),
            "family_count": len(families),
            "top_family_share": round(max(families.values()) / max(1, len(items)), 4) if families else 0.0,
        },
        "efficiency_metrics": {
            "algorithm": "same_pool_profile_replay_no_provider_calls",
            "provider_calls_added": 0,
            "profiles_compared": len(ABLATION_PROFILES),
            "policy": "efficiency_observed_without_reducing_search_breadth",
        },
        "recommendation": recommendation,
        "verification_status": "advisory",
        "policy": "compare_profiles_without_hardcoding_project_direction",
    }


def select_seed_active_frontier(candidates: Iterable[CandidateGenome], *, limit: int = 64) -> list[str]:
    items = [candidate for candidate in candidates or [] if _has_claim(candidate)]
    cap = max(1, int(limit or 1))
    if len(items) <= cap:
        return [candidate.id for candidate in items]
    by_family: dict[str, list[CandidateGenome]] = {}
    for candidate in items:
        by_family.setdefault(_family_key(candidate), []).append(candidate)
    selected: list[CandidateGenome] = []
    for family in sorted(by_family):
        selected.append(max(by_family[family], key=lambda c: (_frontier_score(c), c.id)))
        if len(selected) >= cap:
            return [candidate.id for candidate in selected]
    selected_ids = {candidate.id for candidate in selected}
    for candidate in sorted(items, key=lambda c: (_frontier_score(c), c.id), reverse=True):
        if candidate.id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.id)
        if len(selected) >= cap:
            break
    return [candidate.id for candidate in selected]


def apply_seed_active_frontier(candidates: Iterable[CandidateGenome], *, limit: int = 64) -> dict[str, Any]:
    items = list(candidates or [])
    selected_ids = set(select_seed_active_frontier(items, limit=limit))
    if len(items) <= max(1, int(limit or 1)):
        return {"schema": "seed_active_frontier.v1", "selected_ids": list(selected_ids), "dormant_count": 0, "policy": "pool_fits_frontier"}
    for candidate in items:
        meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        candidate.metadata = meta
        if candidate.id in selected_ids:
            meta["seed_active_frontier"] = {"selected": True, "limit": int(limit or 1)}
            continue
        candidate.current_fate = CandidateFate.DORMANT.value
        meta["seed_active_frontier"] = {"selected": False, "limit": int(limit or 1), "reason": "large_seed_pool_reserve"}
    return {
        "schema": "seed_active_frontier.v1",
        "selected_ids": sorted(selected_ids),
        "dormant_count": len([candidate for candidate in items if candidate.id not in selected_ids]),
        "limit": int(limit or 1),
        "policy": "large_seed_pool_small_active_frontier",
    }


def estimate_reproduction_pressure(candidate: CandidateGenome, population: Iterable[CandidateGenome] | None = None, *, factor_count: int = 0) -> dict[str, Any]:
    family = _family_key(candidate)
    items = list(population or [])
    same_family = sum(1 for item in items if _family_key(item) == family)
    crowding = same_family / max(1, len(items)) if items else 0.0
    quality = _base_score(candidate)
    rarity = _score(candidate, "rarity", "novelty")
    edge = 1.0 if candidate.edge_knowledge_seeds or candidate.novelty_descriptors else 0.0
    theorem = 1.0 if extract_failure_theorem(candidate) else 0.0
    r_eff = max(0.0, min(3.0, 0.25 + 1.25 * quality + 0.45 * rarity + 0.25 * edge + 0.10 * min(3, factor_count) + 0.15 * theorem - 0.65 * crowding))
    return {
        "schema": "r_eff.v1",
        "candidate_id": candidate.id,
        "R_eff": round(r_eff, 4),
        "local_quality": round(quality, 4),
        "rarity_signal": round(rarity, 4),
        "family_crowding": round(crowding, 4),
        "failure_theorem_signal": bool(theorem),
        "policy": "advisory_reproduction_pressure_not_verified",
    }


def extract_failure_theorem(candidate: CandidateGenome) -> dict[str, Any] | None:
    texts = [*candidate.failure_lessons, *candidate.missing_parts]
    diagnostics = coerce_dict(candidate.verification_result).get("diagnostics") or []
    texts.extend(str(item) for item in diagnostics if str(item).strip())
    if not texts and CandidateFate.normalize(candidate.current_fate, default="") in {CandidateFate.FAILED.value, CandidateFate.CULLED.value, CandidateFate.DORMANT.value}:
        texts.append(candidate.concise_claim or candidate.core_mechanism)
    theorem = " ".join(str(item).strip() for item in texts if str(item).strip())
    if not theorem:
        return None
    mechanism = candidate.core_mechanism or candidate.concise_claim or candidate.id
    payload = {
        "schema": "failure_theorem.v1",
        "source_candidate_id": candidate.id,
        "mechanism": str(mechanism)[:240],
        "theorem_claim": theorem[:500],
        "repair_route": "change the next offspring along the failing pressure dimension before reusing this factor",
        "verification_status": "advisory",
    }
    payload["fingerprint"] = stable_hash(payload)
    return payload


def extract_failure_theorems(candidates: Iterable[CandidateGenome], *, limit: int = 32) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        theorem = extract_failure_theorem(candidate)
        if not theorem:
            continue
        key = str(theorem.get("fingerprint") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(theorem)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def single_promotion_gate(candidate: CandidateGenome) -> dict[str, Any]:
    blocked = structurally_blocked(candidate)
    has_artifact = _has_claim(candidate)
    verification = coerce_dict(candidate.verification_result)
    verified = bool(verification.get("passed") is True)
    return {
        "schema": "single_promotion_gate.v1",
        "candidate_id": candidate.id,
        "promotion_eligible": bool(has_artifact and not blocked),
        "verified_claim_allowed": verified,
        "blocked_reason": "structural_or_safety" if blocked else ("empty_artifact" if not has_artifact else ""),
        "policy": "one_gate_for_promotion_not_one_gate_for_truth",
    }


def _profile_result(profile: str, candidates: list[CandidateGenome], *, factors: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    scored = sorted(((_profile_score(profile, candidate, candidates, factors=factors), candidate) for candidate in candidates), key=lambda item: (item[0], item[1].id), reverse=True)
    best_score, best = scored[0]
    return {
        "profile": profile,
        "best_candidate_id": best.id,
        "best_score": round(best_score, 4),
        "top_candidates": [{"candidate_id": candidate.id, "score": round(score, 4)} for score, candidate in scored[: max(1, int(limit or 1))]],
        "moving_parts": _profile_moving_parts(profile),
        "policy": "advisory_same_pool_ablation",
    }


def _profile_score(profile: str, candidate: CandidateGenome, population: list[CandidateGenome], *, factors: list[dict[str, Any]]) -> float:
    base = _base_score(candidate)
    qd = 0.45 * max(_score(candidate, "rarity"), _score(candidate, "novelty")) + 0.10 / sqrt(max(1, sum(1 for item in population if _family_key(item) == _family_key(candidate))))
    factor_bonus = 0.18 if candidate.failure_lessons or any(item.get("source_candidate_id") == candidate.id for item in factors) else 0.0
    if profile == "score_only":
        return base
    if profile == "Nexus_QD_failure_replay":
        return base + qd + factor_bonus
    pressure = estimate_reproduction_pressure(candidate, population, factor_count=len(factors))["R_eff"] / 3.0
    gate = single_promotion_gate(candidate)
    gate_bonus = 0.12 if gate.get("promotion_eligible") else -0.35
    minimal = base + qd + 0.35 * pressure + factor_bonus + gate_bonus
    if profile == "minimal_active_core":
        return minimal
    optional = _optional_layer_signal(candidate)
    return minimal + optional - 0.06 * max(0, _profile_moving_parts("full_fusion") - _profile_moving_parts("minimal_active_core"))


def _profile_moving_parts(profile: str) -> int:
    return {"score_only": 1, "Nexus_QD_failure_replay": 3, "minimal_active_core": 5, "full_fusion": 9}.get(profile, 0)


def _optional_layer_signal(candidate: CandidateGenome) -> float:
    signal = 0.0
    if candidate.formal_artifacts or candidate.proof_obligations:
        signal += 0.08
    if candidate.source_bindings or candidate.evidence_refs:
        signal += 0.06
    metadata = coerce_dict(candidate.metadata)
    for key in ("causal_signal", "epistemic_signal", "adversarial_signal", "proof_adapter_signal"):
        signal += min(0.05, max(0.0, _float(coerce_dict(metadata.get(key)).get("score"), 0.0)))
    return min(0.22, signal)


def _base_score(candidate: CandidateGenome) -> float:
    return max(
        _score(candidate, "objective_alignment"),
        0.5 * _score(candidate, "answer_likelihood") + 0.5 * _score(candidate, "core_mechanism_strength"),
        0.4 * _score(candidate, "frontier_score") + 0.3 * _score(candidate, "novelty") + 0.3 * _score(candidate, "rarity"),
    )


def _frontier_score(candidate: CandidateGenome) -> float:
    return _base_score(candidate) + 0.25 * max(_score(candidate, "rarity"), _score(candidate, "novelty")) + 0.10 * bool(candidate.edge_knowledge_seeds)


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
        candidate.core_mechanism,
        candidate.id,
    ):
        text = str(value or "").strip()
        if text:
            return text[:160]
    return candidate.id


def _has_claim(candidate: CandidateGenome) -> bool:
    return bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())


def _score(candidate: CandidateGenome, *keys: str) -> float:
    values = []
    for key in keys:
        values.append(_float(candidate.multihead_scores.get(key), 0.0))
    return max(values) if values else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


__all__ = [
    "ABLATION_PROFILES",
    "apply_seed_active_frontier",
    "estimate_reproduction_pressure",
    "extract_failure_theorem",
    "extract_failure_theorems",
    "run_core_ablation",
    "select_seed_active_frontier",
    "single_promotion_gate",
]
