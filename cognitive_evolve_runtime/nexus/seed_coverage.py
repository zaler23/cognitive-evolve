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
    niches = Counter(niche for candidate in accepted_items for niche in candidate.niche_memberships or [] if str(niche).strip())
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
    portfolio_coverage = _seed_portfolio_coverage(accepted_items, policy=policy)
    family_floor = max(3, min(12, accepted_count // 4 or 1))
    broad_enough = bool(
        accepted_count >= max(1, min(target, 16))
        and family_count >= family_floor
        and top1_share < 0.35
        and top3_share < 0.60
    )
    status = "broad" if broad_enough else "thin"
    if accepted_count == 0:
        status = "empty"
    contract_status = str(portfolio_coverage.get("contract_coverage_status") or "not_planned")
    coverage_status = status
    fatal_model_error = stop_reason in {"fatal_model_error", "model_error"}
    needs_more_seed = (status != "broad" or contract_status == "shortfall") and not fatal_model_error
    if needs_more_seed:
        needs_more_seed_reason = "seed_portfolio_shortfall" if contract_status == "shortfall" else "coverage_thin"
    else:
        needs_more_seed_reason = ""
    undercovered_family_signals = _undercovered_family_signals(families)
    reasons = _coverage_reasons(status=status, top1_share=top1_share, top3_share=top3_share, duplicate_excess=exact_duplicate_excess)
    if contract_status == "shortfall":
        reasons.insert(0, "seed_portfolio_shortfall")
    return {
        "schema": "seed_coverage.v2",
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
        "top_families": families.most_common(),
        "top_niches": niches.most_common(),
        "origin_model_counts": dict(origins),
        "stopped_reason": stop_reason,
        "partial_failure_count": len(summary.get("failed_batch_ids") or []),
        "status": status,
        "coverage_status": coverage_status,
        "needs_more_seed": needs_more_seed,
        "needs_more_seed_reason": needs_more_seed_reason,
        "needs_target_perturb": status != "broad" or contract_status == "shortfall" or top1_share >= 0.35 or top3_share >= 0.60,
        "reasons": reasons,
        "undercovered_family_signals": undercovered_family_signals,
        "novelty_debt": _novelty_debt(families, accepted_count=accepted_count, family_floor=family_floor, undercovered_family_signals=undercovered_family_signals),
        "fingerprint": stable_hash({"families": families.most_common(), "accepted_count": accepted_count, "portfolio": portfolio_coverage}),
        "policy": "descriptive_only_no_seed_cap_gate",
        **portfolio_coverage,
    }


def _seed_portfolio_coverage(candidates: list[CandidateGenome], *, policy: Any | None) -> dict[str, Any]:
    policy_data = coerce_dict(policy.to_dict()) if hasattr(policy, "to_dict") else coerce_dict(policy)
    metadata = coerce_dict(getattr(policy, "metadata", None)) or coerce_dict(policy_data.get("metadata"))
    raw_slots = metadata.get("seed_portfolio") if isinstance(metadata.get("seed_portfolio"), list) else []
    slots = [
        {
            "slot_id": str(item.get("slot_id") or "").strip(),
            "family_id": str(item.get("family_id") or "").strip(),
            "seed_axis": str(item.get("seed_axis") or "").strip(),
        }
        for item in raw_slots
        if isinstance(item, dict)
        and str(item.get("slot_id") or "").strip()
        and str(item.get("family_id") or "").strip()
        and str(item.get("seed_axis") or "").strip()
    ]
    declarations = [_seed_contract_declaration(candidate) for candidate in candidates]
    edge_signatures = {
        " ".join(str(seed).lower().split())
        for candidate in candidates
        for seed in candidate.edge_knowledge_seeds or []
        if str(seed).strip()
    }
    lens_signatures = {
        " ".join(str(lens).lower().split())
        for candidate in candidates
        for lens in candidate.niche_memberships or []
        if str(lens).strip()
    }
    if not slots:
        return {
            "contract_coverage_status": "not_planned",
            "required_slot_count": 0,
            "covered_slot_count": 0,
            "covered_slot_ids": [],
            "missing_slot_ids": [],
            "missing_edge_slot_ids": [],
            "redundant_edge_slot_ids": [],
            "missing_lens_slot_ids": [],
            "redundant_lens_slot_ids": [],
            "missing_outcome_slot_ids": [],
            "missing_axis_declaration_candidate_ids": [],
            "missing_axis_specific_claim_candidate_ids": [],
            "contract_receipt_count": 0,
            "axis_claim_candidate_count": sum(bool(item["seed_axis"] and item["seed_axis_claim"]) for item in declarations),
            "edge_seed_candidate_count": sum(bool(candidate.edge_knowledge_seeds) for candidate in candidates),
            "distinct_edge_seed_count": len(edge_signatures),
            "lens_signature_count": len(lens_signatures),
            "outcome_ready_count": sum(_candidate_outcome_ready(candidate) for candidate in candidates),
            "capability_status": "pending_pba_outcome",
            "seed_labels_are_contract_receipts": True,
        }

    planned_slot_ids = {slot["slot_id"] for slot in slots}
    planned_families = {slot["family_id"] for slot in slots}
    unused = set(range(len(candidates)))
    covered_slot_ids: list[str] = []
    missing_slot_ids: list[str] = []
    missing_edge_slot_ids: list[str] = []
    redundant_edge_slot_ids: list[str] = []
    missing_lens_slot_ids: list[str] = []
    redundant_lens_slot_ids: list[str] = []
    missing_outcome_slot_ids: list[str] = []
    missing_axis_specific_claim_candidate_ids: list[str] = []
    seen_edge_bundles: set[tuple[str, ...]] = set()
    seen_lens_bundles: set[tuple[str, ...]] = set()
    for slot in slots:
        match_index = next(
            (
                index
                for index in sorted(unused)
                if declarations[index]["slot_id"] == slot["slot_id"]
                and declarations[index]["family_id"] == slot["family_id"]
                and declarations[index]["seed_axis"] == slot["seed_axis"]
            ),
            None,
        )
        if match_index is None:
            missing_slot_ids.append(slot["slot_id"])
            continue
        unused.remove(match_index)
        candidate = candidates[match_index]
        receipt = seed_axis_contract_receipt(candidate, slot)
        if not receipt["complete"]:
            missing_slot_ids.append(slot["slot_id"])
            missing_axis_specific_claim_candidate_ids.append(candidate.id)
            if slot["seed_axis"] == "edge_knowledge" and not candidate.edge_knowledge_seeds:
                missing_edge_slot_ids.append(slot["slot_id"])
            continue
        covered_slot_ids.append(slot["slot_id"])
        if slot["seed_axis"] == "edge_knowledge":
            edge_bundle = tuple(sorted({" ".join(str(seed).lower().split()) for seed in candidate.edge_knowledge_seeds if str(seed).strip()}))
            if edge_bundle in seen_edge_bundles:
                redundant_edge_slot_ids.append(slot["slot_id"])
            seen_edge_bundles.add(edge_bundle)
        lens_bundle = tuple(sorted({" ".join(str(lens).lower().split()) for lens in candidate.niche_memberships if str(lens).strip()}))
        if not lens_bundle:
            missing_lens_slot_ids.append(slot["slot_id"])
        elif lens_bundle in seen_lens_bundles:
            redundant_lens_slot_ids.append(slot["slot_id"])
        seen_lens_bundles.add(lens_bundle)
        if not _candidate_outcome_ready(candidate):
            missing_outcome_slot_ids.append(slot["slot_id"])

    missing_axis_declaration_candidate_ids = [
        candidates[index].id
        for index, declaration in enumerate(declarations)
        if declaration["family_id"] in planned_families
        and (
            not declaration["seed_axis"]
            or (declaration["seed_axis"] == "direct_mainstream" and not declaration["seed_axis_claim"])
        )
    ]
    covered_axes = sorted(
        {
            slot["seed_axis"]
            for slot in slots
            if any(seed_axis_contract_receipt(candidate, slot)["complete"] for candidate in candidates)
        }
    )
    required_axes = sorted({slot["seed_axis"] for slot in slots})
    complete = not any(
        (
            missing_slot_ids,
            missing_edge_slot_ids,
            redundant_edge_slot_ids,
            missing_lens_slot_ids,
            redundant_lens_slot_ids,
            missing_outcome_slot_ids,
        )
    )
    return {
        "contract_coverage_status": "complete" if complete else "shortfall",
        "required_slot_count": len(slots),
        "covered_slot_count": len(covered_slot_ids),
        "covered_slot_ids": covered_slot_ids,
        "missing_slot_ids": missing_slot_ids,
        "missing_edge_slot_ids": missing_edge_slot_ids,
        "redundant_edge_slot_ids": redundant_edge_slot_ids,
        "missing_lens_slot_ids": missing_lens_slot_ids,
        "redundant_lens_slot_ids": redundant_lens_slot_ids,
        "missing_outcome_slot_ids": missing_outcome_slot_ids,
        "missing_axis_declaration_candidate_ids": missing_axis_declaration_candidate_ids,
        "missing_axis_specific_claim_candidate_ids": list(dict.fromkeys(missing_axis_specific_claim_candidate_ids)),
        "contract_receipt_count": sum(declaration["slot_id"] in planned_slot_ids for declaration in declarations),
        "axis_claim_candidate_count": sum(bool(item["seed_axis"] and item["seed_axis_claim"]) for item in declarations),
        "required_seed_axes": required_axes,
        "covered_seed_axes": covered_axes,
        "missing_seed_axes": sorted(set(required_axes) - set(covered_axes)),
        "edge_seed_candidate_count": sum(bool(candidate.edge_knowledge_seeds) for candidate in candidates),
        "distinct_edge_seed_count": len(edge_signatures),
        "lens_signature_count": len(lens_signatures),
        "outcome_ready_count": sum(_candidate_outcome_ready(candidate) for candidate in candidates),
        "capability_status": "pending_pba_outcome",
        "seed_labels_are_contract_receipts": True,
        "capability_evidence_rule": "Productive capability requires later PBA grounded outcome; seed labels and declarations are not reward.",
    }


def _seed_contract_declaration(candidate: CandidateGenome) -> dict[str, str]:
    metadata = coerce_dict(candidate.metadata)
    search_space = coerce_dict(metadata.get("search_space"))
    return {
        "slot_id": str(metadata.get("seed_type") or "").strip(),
        "family_id": str(search_space.get("family_id") or search_space.get("plane_id") or "").strip(),
        "seed_axis": str(search_space.get("seed_axis") or "").strip(),
        "seed_axis_claim": str(search_space.get("seed_axis_claim") or "").strip(),
    }


def seed_axis_contract_receipt(
    candidate: CandidateGenome,
    slot: dict[str, Any],
    *,
    require_slot_id: bool = True,
) -> dict[str, Any]:
    """Check one seed slot without treating its label as capability evidence."""

    metadata = coerce_dict(candidate.metadata)
    search_space = coerce_dict(metadata.get("search_space"))
    axis = str(slot.get("seed_axis") or "").strip()
    missing: list[str] = []
    if require_slot_id and str(metadata.get("seed_type") or "").strip() != str(slot.get("slot_id") or "").strip():
        missing.append("metadata.seed_type")
    if str(search_space.get("family_id") or search_space.get("plane_id") or "").strip() != str(slot.get("family_id") or "").strip():
        missing.append("metadata.search_space.family_id")
    if str(search_space.get("seed_axis") or "").strip() != axis:
        missing.append("metadata.search_space.seed_axis")
    if axis == "direct_mainstream" and not str(search_space.get("seed_axis_claim") or "").strip():
        missing.append("metadata.search_space.seed_axis_claim")
    if axis == "cross_domain_transfer" and not str(search_space.get("transfer_source_domain") or "").strip():
        missing.append("metadata.search_space.transfer_source_domain")
    elif axis == "edge_knowledge" and not candidate.edge_knowledge_seeds:
        missing.append("edge_knowledge_seeds")
    elif axis == "counterexample_probe" and not str(search_space.get("probe_target_assumption") or "").strip():
        missing.append("metadata.search_space.probe_target_assumption")
    elif axis == "representation_shift":
        shift = coerce_dict(search_space.get("representation_shift"))
        if not str(shift.get("from") or "").strip():
            missing.append("metadata.search_space.representation_shift.from")
        if not str(shift.get("to") or "").strip():
            missing.append("metadata.search_space.representation_shift.to")
    elif axis == "tool_probe" and not _nonempty_claim(search_space.get("tool_probe_plan")):
        missing.append("metadata.search_space.tool_probe_plan")
    return {"complete": not missing, "missing_fields": missing}


def _nonempty_claim(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_nonempty_claim(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonempty_claim(item) for item in value)
    return value is not None


def _candidate_outcome_ready(candidate: CandidateGenome) -> bool:
    artifact = candidate.artifact
    has_artifact = artifact not in (None, "", {}, [])
    metadata = coerce_dict(candidate.metadata)
    structured = coerce_dict(metadata.get("structured_output_fields"))
    dimensions = structured.get("evaluation_dimensions") or metadata.get("evaluation_dimensions") or []
    return bool(has_artifact and isinstance(dimensions, list) and any(str(item).strip() for item in dimensions))


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
    thin = str(coverage.get("coverage_status") or coverage.get("status") or "").lower() in {"thin", "undercovered", "watch", "shortfall"}
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
        "best_direction_markers": best_ids,
        "diagnosis_hint": diag_text,
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
            return text
    return ""


def _claim_key(candidate: CandidateGenome) -> str:
    return " ".join(str(candidate.concise_claim or candidate.core_mechanism or candidate.artifact or "").lower().split())


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
    return [{"family": family, "count": count, "reason": "singleton_or_below_median"} for family, count in families.items() if count <= max(1, medianish)]


def _novelty_debt(
    families: Counter[str],
    *,
    accepted_count: int,
    family_floor: int,
    undercovered_family_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    overrepresented = []
    for family, count in families.most_common():
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
        "undercovered_family_signals": list(undercovered_family_signals) if missing_family_count else [],
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
    "seed_axis_contract_receipt",
    "seed_reservoir_sidecar_payload",
    "target_perturb_seed_judgment",
]
