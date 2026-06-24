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

ABLATION_PROFILES = (
    "score_only",
    "Nexus_QD_failure_replay",
    "minimal_active_core",
    "minimal_core_plus_useful_attachments",
    "current_project_incumbent",
    "full_fusion",
)


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
    incumbent_inventory = effective_project_attachment_inventory(items, factors=factors, archives=archives, policy=policy)
    profile_results = {
        profile: _profile_result(profile, items, factors=factors, limit=limit, incumbent_inventory=incumbent_inventory)
        for profile in ABLATION_PROFILES
    }
    minimal = profile_results["minimal_active_core"].get("best_score", 0.0)
    stacked = profile_results["minimal_core_plus_useful_attachments"].get("best_score", 0.0)
    incumbent = profile_results["current_project_incumbent"].get("best_score", 0.0)
    fusion = profile_results["full_fusion"].get("best_score", 0.0)
    coverage = _effective_attachment_coverage(profile_results["minimal_core_plus_useful_attachments"], incumbent_inventory)
    if coverage.get("coverage_complete") and stacked > incumbent + 0.02:
        recommendation = "minimal_core_with_all_effective_attachments_beats_current_project"
    elif not coverage.get("coverage_complete"):
        recommendation = "current_project_incumbent_until_stacked_core_covers_effective_attachments"
    elif fusion > max(minimal, stacked, incumbent) + 0.08:
        recommendation = "full_fusion_has_incremental_signal"
    elif stacked > minimal + 0.02:
        recommendation = "minimal_core_with_useful_attachments_needs_current_project_margin"
    else:
        recommendation = "minimal_core_first"
    families = Counter(_family_key(candidate) for candidate in items)
    return {
        "schema": "minimal_core_ablation.v1",
        "profiles": profile_results,
        "failure_theorems": factors[:16],
        "current_project_comparison": {
            "stacked_score": round(float(stacked or 0.0), 4),
            "current_project_incumbent_score": round(float(incumbent or 0.0), 4),
            "score_margin": round(float(stacked or 0.0) - float(incumbent or 0.0), 4),
            "effective_attachment_coverage": coverage,
            "policy": "advisory_same_pool_current_project_baseline_not_verified",
        },
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


def _profile_result(
    profile: str,
    candidates: list[CandidateGenome],
    *,
    factors: list[dict[str, Any]],
    limit: int,
    incumbent_inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inventory = incumbent_inventory or []
    scored = sorted(
        _dedupe_carrier_targets(profile, candidates, factors=factors, incumbent_inventory=inventory),
        key=lambda item: (item[0], item[1].id),
        reverse=True,
    )
    best_score, best, best_carriers = scored[0]
    result = {
        "profile": profile,
        "best_candidate_id": best.id,
        "best_score": round(best_score, 4),
        "top_candidates": [{"candidate_id": candidate.id, "score": round(score, 4)} for score, candidate, _ in scored[: max(1, int(limit or 1))]],
        "moving_parts": _profile_moving_parts(profile),
        "policy": "advisory_same_pool_ablation",
    }
    if profile == "minimal_core_plus_useful_attachments":
        stack = useful_attachment_stack(best, factors=factors, carriers=best_carriers)
        result["active_support_stack"] = stack
        result["moving_parts"] = _profile_moving_parts("minimal_active_core") + len(stack)
        result["policy"] = "advisory_same_pool_ablation_all_positive_attachments"
    if profile == "current_project_incumbent":
        result["effective_attachment_inventory"] = inventory
        result["moving_parts"] = _profile_moving_parts("minimal_active_core") + len(inventory)
        result["policy"] = "advisory_same_pool_current_project_incumbent"
    return result


def _dedupe_carrier_targets(
    profile: str,
    candidates: list[CandidateGenome],
    *,
    factors: list[dict[str, Any]],
    incumbent_inventory: list[dict[str, Any]] | None = None,
) -> list[tuple[float, CandidateGenome, list[CandidateGenome]]]:
    by_target: dict[str, tuple[float, CandidateGenome, list[CandidateGenome]]] = {}
    for candidate in candidates:
        target = _best_current_carrier_target(candidate, candidates) or candidate
        carriers = [candidate] if target is not candidate else []
        score = _profile_score(profile, target, candidates, factors=factors, incumbent_inventory=incumbent_inventory or [])
        if carriers and profile == "minimal_core_plus_useful_attachments":
            score += _carrier_attachment_signal(candidate)
        previous = by_target.get(target.id)
        if previous is None or score > previous[0]:
            by_target[target.id] = (score, target, carriers)
        elif carriers:
            previous[2].extend(carriers)
    return list(by_target.values())


def _profile_score(
    profile: str,
    candidate: CandidateGenome,
    population: list[CandidateGenome],
    *,
    factors: list[dict[str, Any]],
    incumbent_inventory: list[dict[str, Any]] | None = None,
) -> float:
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
    if profile == "minimal_core_plus_useful_attachments":
        return minimal + _useful_attachment_signal(candidate, factors=factors)
    if profile == "current_project_incumbent":
        inventory = incumbent_inventory or []
        return minimal + _inventory_attachment_signal(inventory) - _incumbent_coordination_drag(inventory)
    optional = _optional_layer_signal(candidate)
    return minimal + optional - 0.06 * max(0, _profile_moving_parts("full_fusion") - _profile_moving_parts("minimal_active_core"))


def _profile_moving_parts(profile: str) -> int:
    return {
        "score_only": 1,
        "Nexus_QD_failure_replay": 3,
        "minimal_active_core": 5,
        "minimal_core_plus_useful_attachments": 5,
        "current_project_incumbent": 5,
        "full_fusion": 9,
    }.get(profile, 0)


def useful_attachment_stack(candidate: CandidateGenome, *, factors: list[dict[str, Any]] | None = None, carriers: list[CandidateGenome] | None = None) -> list[dict[str, Any]]:
    """Return every positive same-pool support signal for the candidate.

    The entries name source fields rather than project-domain categories.  They
    are advisory attachments: they can explain why a minimal active core should
    carry extra support material, but they do not verify or gate the candidate.
    """

    stack = [
        item
        for item in [*_attachment_signals(candidate, factors=factors or []), *(_carrier_attachment_entries(carriers or []))]
        if float(item.get("marginal_score") or 0.0) > 0.0
    ]
    stack.sort(key=lambda item: (float(item.get("marginal_score") or 0.0), str(item.get("source") or "")), reverse=True)
    return stack


def effective_project_attachment_inventory(
    candidates: Iterable[CandidateGenome],
    *,
    factors: list[dict[str, Any]] | None = None,
    archives: Any | None = None,
    policy: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect currently effective same-pool support attachments.

    This is an open evidence inventory, not a fixed taxonomy.  Entries are keyed
    by source field path plus evidence fingerprint so future components can add
    support signals through their own metadata without central enum changes.
    """

    inventory: dict[str, dict[str, Any]] = {}

    def remember(item: dict[str, Any]) -> None:
        if float(item.get("marginal_score") or 0.0) <= 0.0:
            return
        key = stable_hash({"source": item.get("source"), "evidence": item.get("evidence")})
        existing = inventory.get(key)
        if existing is None or float(item.get("marginal_score") or 0.0) > float(existing.get("marginal_score") or 0.0):
            payload = dict(item)
            payload["inventory_key"] = key
            inventory[key] = payload

    factor_items = factors or []
    candidate_items = list(candidates or [])
    for candidate in candidate_items:
        target = _best_current_carrier_target(candidate, candidate_items) or candidate
        carriers = [candidate] if target is not candidate else []
        for item in useful_attachment_stack(target, factors=factor_items, carriers=carriers):
            remember(item)
    for item in _open_metadata_inventory("policy.metadata", coerce_dict(getattr(policy, "metadata", None))):
        remember(item)
    archive_payload = archives.to_dict() if hasattr(archives, "to_dict") else archives
    for item in _open_metadata_inventory("archives", coerce_dict(archive_payload)):
        remember(item)
    ordered = sorted(inventory.values(), key=lambda item: (float(item.get("marginal_score") or 0.0), str(item.get("source") or "")), reverse=True)
    return ordered


def _useful_attachment_signal(candidate: CandidateGenome, *, factors: list[dict[str, Any]]) -> float:
    return min(0.35, sum(float(item.get("marginal_score") or 0.0) for item in useful_attachment_stack(candidate, factors=factors)))


def _inventory_attachment_signal(inventory: list[dict[str, Any]]) -> float:
    return min(0.35, sum(float(item.get("marginal_score") or 0.0) for item in inventory))


def _incumbent_coordination_drag(inventory: list[dict[str, Any]]) -> float:
    # Advisory complexity drag only; it prevents treating all accumulated support
    # material as free, while still letting a fully covered smaller stack win.
    return min(0.08, 0.01 * len(inventory))


def _effective_attachment_coverage(stacked_profile: dict[str, Any], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    active = stacked_profile.get("active_support_stack") if isinstance(stacked_profile, dict) else []
    active_keys = {_attachment_match_key(item) for item in active or []}
    missing = [item for item in inventory if _attachment_match_key(item) not in active_keys]
    return {
        "inventory_count": len(inventory),
        "covered_count": max(0, len(inventory) - len(missing)),
        "missing_effective_attachment_count": len(missing),
        "coverage_complete": not missing,
        "missing_effective_attachments": [
            {"source": item.get("source"), "evidence": item.get("evidence"), "marginal_score": item.get("marginal_score")}
            for item in missing[:16]
        ],
        "policy": "advisory_open_field_path_coverage",
    }


def _attachment_match_key(item: dict[str, Any]) -> str:
    return stable_hash({"source": item.get("source"), "evidence": item.get("evidence")})


def _open_metadata_inventory(prefix: str, metadata: dict[str, Any], *, limit: int = 32) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, value in sorted((metadata or {}).items(), key=lambda item: str(item[0])):
        if len(out) >= max(1, int(limit or 1)):
            break
        if value in (None, "", [], {}):
            continue
        source = f"{prefix}.{key}"
        payload = coerce_dict(value)
        score = _float(payload.get("score"), 0.0)
        if score <= 0.0 and (payload or isinstance(value, (list, tuple, set))):
            score = 0.03
        if score <= 0.0:
            continue
        evidence = payload.get("rationale") or payload.get("reason") or payload.get("status") or stable_hash({"source": source, "value": value})
        out.append(
            {
                "source": source,
                "marginal_score": round(min(0.06, score), 4),
                "policy": "advisory_project_metadata_attachment",
                "evidence": str(evidence)[:220],
            }
        )
    return out


def _best_current_carrier_target(candidate: CandidateGenome, candidates: list[CandidateGenome]) -> CandidateGenome | None:
    if not isinstance(candidate.artifact, dict):
        return None
    best = candidate.artifact.get("best_current_direction")
    if not isinstance(best, dict):
        return None
    target_id = str(best.get("candidate_id") or "").strip()
    if not target_id or target_id == candidate.id:
        return None
    for item in candidates:
        if item.id == target_id and _has_claim(item) and not structurally_blocked(item):
            return item
    return None


def _carrier_attachment_signal(candidate: CandidateGenome) -> float:
    return min(0.12, 0.04 + 0.02 * len(_carrier_attachment_entries([candidate])))


def _carrier_attachment_entries(carriers: list[CandidateGenome]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for carrier in carriers:
        if not isinstance(carrier.artifact, dict):
            continue
        entries.append(
            {
                "source": "artifact.best_current_direction.carrier",
                "marginal_score": _carrier_attachment_signal_without_entries(carrier),
                "policy": "advisory_support_attachment",
                "evidence": carrier.id,
            }
        )
    return entries


def _carrier_attachment_signal_without_entries(candidate: CandidateGenome) -> float:
    score = 0.04
    artifact = coerce_dict(candidate.artifact)
    if coerce_dict(artifact.get("claim_permissions")):
        score += 0.02
    if artifact.get("smallest_next_proof_object"):
        score += 0.02
    if coerce_dict(artifact.get("comparison_summary")):
        score += 0.02
    return round(min(0.12, score), 4)


def _attachment_signals(candidate: CandidateGenome, *, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = coerce_dict(candidate.metadata)
    signals: list[dict[str, Any]] = []

    def add(source: str, score: float, evidence: Any = None) -> None:
        bounded = max(0.0, min(0.12, float(score or 0.0)))
        if bounded <= 0.0:
            return
        payload: dict[str, Any] = {"source": source, "marginal_score": round(bounded, 4), "policy": "advisory_support_attachment"}
        if evidence not in (None, "", [], {}):
            payload["evidence"] = str(evidence)[:220]
        signals.append(payload)

    add("candidate.failure_lessons", 0.10 if candidate.failure_lessons else 0.0, candidate.failure_lessons[:2])
    add("candidate.failure_theorem", 0.08 if any(item.get("source_candidate_id") == candidate.id for item in factors) else 0.0)
    add("candidate.formal_artifacts", 0.08 if candidate.formal_artifacts else 0.0, candidate.formal_artifacts[:2])
    add("candidate.proof_obligations", 0.07 if candidate.proof_obligations else 0.0, candidate.proof_obligations[:2])
    add("candidate.source_bindings", 0.06 if candidate.source_bindings else 0.0, candidate.source_bindings[:2])
    add("candidate.evidence_refs", 0.06 if candidate.evidence_refs else 0.0, candidate.evidence_refs[:2])
    add("candidate.edge_knowledge_seeds", 0.05 if candidate.edge_knowledge_seeds else 0.0, candidate.edge_knowledge_seeds[:2])
    add("candidate.novelty_descriptors", 0.04 if candidate.novelty_descriptors else 0.0, candidate.novelty_descriptors[:2])

    intent = coerce_dict(metadata.get("intent_binding"))
    add("metadata.intent_binding.direct_answer_score", 0.10 * _float(intent.get("direct_answer_score"), 0.0), intent.get("alignment_rationale"))
    if metadata.get("resurrection_lane") or metadata.get("resurrection_reason") or metadata.get("resurrection_score") is not None:
        add("metadata.resurrection", 0.10 * _float(metadata.get("resurrection_score"), 0.5), metadata.get("resurrection_reason"))
    for key, value in metadata.items():
        if not str(key).endswith("_signal"):
            continue
        signal = coerce_dict(value)
        add(f"metadata.{key}", 0.06 * _float(signal.get("score"), 0.0), signal.get("rationale") or signal.get("reason"))
    return signals


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
    "effective_project_attachment_inventory",
    "estimate_reproduction_pressure",
    "extract_failure_theorem",
    "extract_failure_theorems",
    "run_core_ablation",
    "select_seed_active_frontier",
    "single_promotion_gate",
    "useful_attachment_stack",
]
