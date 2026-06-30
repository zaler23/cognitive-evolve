"""Small NextGen helpers for answer-first exploration.

These helpers deliberately stay metadata-only: they can bias budget traces and
parent ordering, but they are not verification authority and only hard-exclude
structural/safety-broken candidates.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import base_mechanism_family, normalize_token

NEXTGEN_METADATA_KEY = "nextgen"
INTENT_BINDING_KEY = "intent_binding"
CANONICAL_FAMILY_VERSION = "bounded-mechanism-v2"
CANONICAL_BUCKETS_PER_DECLARED_FAMILY = 8
_CANONICAL_TOKEN_NOISE = {
    "candidate",
    "mechanism",
    "artifact",
    "patch",
    "test",
    "tests",
    "file",
    "files",
    "code",
    "runtime",
    "nexus",
    "cognitive",
    "evolve",
    "evolution",
    "with",
    "from",
    "that",
    "this",
}


def candidate_answer_text(candidate: CandidateGenome | None) -> str:
    return _text(candidate)


def bind_candidate_intent(candidate: CandidateGenome, *, contract: Any | None = None) -> dict[str, Any]:
    """Attach a free-text intent binding; no target enums or domain word lists."""

    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    candidate.metadata = metadata
    existing = coerce_dict(metadata.get(INTENT_BINDING_KEY))
    contract_intent = _contract_search_intent(contract)
    existing_intent = str(existing.get("search_intent") or "").strip()
    search_intent = str(contract_intent or existing_intent or "").strip()
    main_claim = str(existing.get("candidate_main_claim") or _main_claim(candidate) or "").strip()
    stale_no_contract_binding = bool(
        contract_intent
        and not existing_intent
        and str(existing.get("alignment_rationale") or "").startswith("no frozen search intent supplied")
    )
    stale_different_goal_binding = bool(contract_intent and existing_intent and existing_intent != contract_intent)
    stale_binding = stale_no_contract_binding or stale_different_goal_binding
    score = None if stale_binding else _explicit_intent_score(candidate, existing)
    if score is None:
        score = _direct_answer_score(search_intent, main_claim, _text(candidate))
    supporting = existing.get("supporting_claims")
    if not isinstance(supporting, list):
        supporting = _supporting_claims(candidate, main_claim)
    rationale = "" if stale_binding else str(existing.get("alignment_rationale") or "")
    payload = {
        "search_intent": search_intent,
        "candidate_main_claim": main_claim,
        "direct_answer_score": round(_bounded(score), 4),
        "supporting_claims": [str(item)[:600] for item in supporting if str(item or "").strip()][:6],
        "alignment_rationale": str(rationale or _alignment_rationale(search_intent, main_claim, float(score or 0.0)))[:600],
    }
    metadata[INTENT_BINDING_KEY] = payload
    ensure_nextgen_identity(candidate)[INTENT_BINDING_KEY] = dict(payload)
    return payload


def candidate_verification_status(candidate: CandidateGenome) -> str:
    """Candidate-local status; final user-facing verified needs graded evidence."""

    metadata = coerce_dict(candidate.metadata)
    result = coerce_dict(candidate.verification_result)
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
    if (
        fate in {CandidateFate.FAILED.value, CandidateFate.CULLED.value}
        or metadata.get("terminal_failure")
        or metadata.get("terminal_reject")
        or metadata.get("terminal_reject_reason")
        or result.get("passed") is False
    ):
        return "failed"
    if bool(result.get("passed")):
        return "verified"
    advisory = coerce_dict(metadata.get("final_answer_advisory"))
    if advisory.get("final_eligible") is False or metadata.get("advisory_final_blocked"):
        return "advisory"
    evidence_meta = coerce_dict(metadata.get("evidence_state"))
    if evidence_meta.get("final_blocked"):
        return "advisory"
    return "unverified"


def user_facing_verification_status(
    candidate: CandidateGenome,
    *,
    final_certificate: dict[str, Any] | None = None,
    graded_output: Any | None = None,
) -> str:
    """Return the final-display status without treating local metadata as proof."""

    if _graded_verified(graded_output) or _certificate_verified(final_certificate, candidate_id=candidate.id):
        return "verified"
    status = candidate_verification_status(candidate)
    return "advisory" if status == "verified" else status


def blocked_from_verified_claim_reason(candidate: CandidateGenome, *, user_status: str | None = None) -> str:
    metadata = coerce_dict(candidate.metadata)
    result = coerce_dict(candidate.verification_result)
    if structurally_blocked(candidate):
        return "structural_or_safety_blocked"
    status = user_status or candidate_verification_status(candidate)
    if status == "verified":
        return ""
    if candidate_verification_status(candidate) == "verified" and status != "verified":
        return "graded_verified_result_absent"
    for key in ("terminal_reject_reason", "hard_reject_reason", "model_seed_error"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value[:240]
    diagnostics = [str(item) for item in result.get("diagnostics", []) or [] if item]
    if diagnostics:
        return "; ".join(diagnostics[:3])[:240]
    advisory = coerce_dict(metadata.get("final_answer_advisory"))
    diagnostics = [str(item) for item in advisory.get("diagnostics", []) or [] if item]
    if diagnostics:
        return "; ".join(diagnostics[:3])[:240]
    return f"verification_status:{status}"


def best_current_direction_payload(
    candidate: CandidateGenome,
    *,
    route: str = "best_current",
    why_best: str = "highest intent-bound answer signal",
    contract: Any | None = None,
    final_certificate: dict[str, Any] | None = None,
    graded_output: Any | None = None,
) -> dict[str, Any]:
    binding = bind_candidate_intent(candidate, contract=contract)
    verification_status = user_facing_verification_status(candidate, final_certificate=final_certificate, graded_output=graded_output)
    return {
        "candidate_id": candidate.id,
        "route": route,
        "mechanism_summary": binding.get("candidate_main_claim") or _summary_text(candidate),
        "candidate_main_claim": binding.get("candidate_main_claim") or _summary_text(candidate),
        "supporting_claims": list(binding.get("supporting_claims") or [])[:6],
        "intent_alignment_rationale": str(binding.get("alignment_rationale") or ""),
        "direct_answer_score": float(binding.get("direct_answer_score") or 0.0),
        "why_best": why_best,
        "verification_status": verification_status,
        "blocked_from_verified_claim_reason": blocked_from_verified_claim_reason(candidate, user_status=verification_status),
    }


def select_best_current_direction(candidates: Iterable[CandidateGenome], *, contract: Any | None = None) -> CandidateGenome | None:
    displayable = [candidate for candidate in candidates if not structurally_blocked(candidate) and candidate_answer_text(candidate).strip()]
    if not displayable:
        return None
    context = list(displayable)
    return max(displayable, key=lambda candidate: (best_current_direction_score(candidate, context, contract=contract), candidate.id))


def best_current_direction_score(candidate: CandidateGenome, context: Iterable[CandidateGenome] | None = None, *, contract: Any | None = None) -> float:
    if structurally_blocked(candidate) or not candidate_answer_text(candidate).strip():
        return -1000000.0
    context_items = list(context or [candidate])
    scores = candidate.multihead_scores or {}
    status = candidate_verification_status(candidate)
    intent = float(bind_candidate_intent(candidate, contract=contract).get("direct_answer_score") or 0.0)
    novelty = _bounded(scores.get("novelty"))
    rarity = _bounded(scores.get("rarity"))
    edge = 1.0 if candidate.edge_knowledge_seeds else 0.0
    singleton = 1.0 if _is_singleton_family(candidate, context_items) else 0.0
    formal = 1.0 if candidate.formal_artifacts or candidate.proof_obligations else 0.0
    answer = min(1.0, len(candidate_answer_text(candidate)) / 1200.0)
    quality = max(
        _bounded(scores.get("frontier_score")),
        _bounded(scores.get("answer_likelihood")),
        _bounded(scores.get("core_mechanism_strength")),
        _bounded(scores.get("objective_alignment")),
    )
    observation = coerce_dict(ensure_nextgen_identity(candidate).get("productive_child_observation"))
    reskin = 1.0 if observation.get("near_verbatim_reskin") else 0.0
    failed_penalty = 0.15 if status == "failed" else 0.0
    return (
        0.38 * intent
        + 0.18 * quality
        + 0.12 * max(novelty, rarity)
        + 0.10 * edge
        + 0.08 * singleton
        + 0.06 * formal
        + 0.08 * answer
        - 0.10 * reskin
        - failed_penalty
    )


def resurrection_quota(branch_factor: int | None, *, pool_size: int | None = None, pressure: float | None = None) -> int:
    try:
        value = int(branch_factor or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return 1
    base = max(1, (value + 3) // 4)
    try:
        pool = int(pool_size or 0)
    except (TypeError, ValueError):
        pool = 0
    try:
        pressure_bonus = max(0, int(float(pressure or 0.0) * value))
    except (TypeError, ValueError):
        pressure_bonus = 0
    pool_bonus = 0
    if pool >= value * 4:
        pool_bonus += 1
    if pool >= value * 10:
        pool_bonus += max(1, value // 4)
    return max(1, min(value, base + pool_bonus + pressure_bonus))


def resurrection_score(candidate: CandidateGenome, context: Iterable[CandidateGenome] | None = None, *, contract: Any | None = None) -> float:
    score = best_current_direction_score(candidate, context, contract=contract)
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
    if fate in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}:
        score -= 0.35
    return score


def mark_resurrection_candidate(candidate: CandidateGenome, *, round_index: int = 0, source_pool: str = "") -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    candidate.metadata = metadata
    pool = source_pool or _candidate_source_pool(candidate)
    score = resurrection_score(candidate, [candidate])
    intent_score = float(bind_candidate_intent(candidate).get("direct_answer_score") or 0.0)
    reason = "intent_aligned_resurrection" if intent_score >= 0.5 else "loser_pool_soft_reentry"
    metadata["resurrection_lane"] = True
    metadata["resurrection_score"] = round(float(score), 4)
    metadata["resurrection_reason"] = reason
    metadata["resurrection_round"] = int(round_index or 0)
    metadata["source_pool"] = pool
    ensure_nextgen_identity(candidate)["resurrection_trace"] = {
        "round": int(round_index or 0),
        "source_pool": pool,
        "score": metadata["resurrection_score"],
        "reason": reason,
    }
    record_candidate_budget_decision(candidate, source="resurrection_lane", reason=reason, action="soft_boost", details={"source_pool": pool, "score": metadata["resurrection_score"]})
    return metadata


@dataclass(frozen=True)
class ProductiveChildObservation:
    """Observation only; never a reproduction eligibility decision."""

    transition_signature: str
    family_id: str
    canonical_family_id: str
    novelty_delta: float = 0.0
    obligation_bonus: float = 0.0
    near_verbatim_reskin: bool = False
    engineering_noise: bool = False
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nextgen_metadata(candidate: CandidateGenome) -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    candidate.metadata = metadata
    payload = metadata.setdefault(NEXTGEN_METADATA_KEY, {})
    if not isinstance(payload, dict):
        payload = {}
        metadata[NEXTGEN_METADATA_KEY] = payload
    return payload


def ensure_nextgen_identity(
    candidate: CandidateGenome,
    *,
    origin_model: str | None = None,
    model_profile_id: str | None = None,
) -> dict[str, Any]:
    payload = nextgen_metadata(candidate)
    family_id = str(_declared_family(candidate) or base_mechanism_family(candidate))
    canonical = _canonical_mechanism_family_id(candidate)
    old_version = str(payload.get("canonical_mechanism_family_version") or "")
    old_canonical = str(payload.get("canonical_mechanism_family_id") or "")
    if old_version != CANONICAL_FAMILY_VERSION:
        payload["mechanism_family_id"] = family_id
        if old_canonical and old_canonical != canonical:
            payload["canonical_mechanism_migration"] = {
                "from_version": old_version,
                "from_canonical_mechanism_family_id": old_canonical,
                "to_version": CANONICAL_FAMILY_VERSION,
                "to_canonical_mechanism_family_id": canonical,
            }
        payload["canonical_mechanism_family_id"] = canonical
        payload["canonical_mechanism_family_version"] = CANONICAL_FAMILY_VERSION
    else:
        payload.setdefault("mechanism_family_id", family_id)
        payload.setdefault("canonical_mechanism_family_id", canonical)
    payload.setdefault("family_signature", family_signature(candidate))
    payload.setdefault("transition_signature", transition_signature(candidate))
    if origin_model:
        payload.setdefault("origin_model", origin_model)
    if model_profile_id:
        payload.setdefault("model_profile_id", model_profile_id)
    return payload


def _canonical_mechanism_family_id(candidate: CandidateGenome) -> str:
    base = base_mechanism_family(candidate)
    tokens = sorted(
        {
            token
            for token in (normalize_token(item) for item in _tokens(_text(candidate)[:1200]))
            if token and len(token) >= 3 and token not in _CANONICAL_TOKEN_NOISE and not any(char.isdigit() for char in token)
        }
    )[:48]
    coarse_sig = {"niche": list(candidate.niche_memberships[:2]), "tokens": tokens}
    bucket = int(stable_hash(coarse_sig)[:12], 16) % CANONICAL_BUCKETS_PER_DECLARED_FAMILY
    return f"{base}#m{bucket}"


def family_signature(candidate: CandidateGenome) -> str:
    return "family-" + stable_hash(
        {
            "declared": _declared_family(candidate),
            "mechanism": _text(candidate)[:1200],
            "niches": list(candidate.niche_memberships[:6]),
        }
    )[:16]


def transition_signature(candidate: CandidateGenome) -> str:
    return "transition-" + stable_hash(
        {
            "parents": sorted(str(item) for item in candidate.parent_ids),
            "family": _declared_family(candidate) or _lineage_family(candidate),
            "mechanism": _text(candidate)[:1600],
            "obligation_delta": candidate.obligation_delta,
            "evidence_delta": candidate.evidence_delta,
        }
    )[:16]


def observe_productive_child(parent: CandidateGenome | None, child: CandidateGenome) -> ProductiveChildObservation:
    child_meta = ensure_nextgen_identity(child)
    parent_text = _text(parent) if parent is not None else ""
    child_text = _text(child)
    novelty_delta = 1.0 if parent is None else _rough_text_delta(parent_text, child_text)
    obligation_bonus = 1.0 if (child.obligation_delta or child.evidence_delta) else 0.0
    near_reskin = parent is not None and novelty_delta < 0.08
    engineering_noise = _looks_like_engineering_noise(child_text)
    signals: list[str] = []
    if novelty_delta > 0.35:
        signals.append("novel_delta")
    if obligation_bonus:
        signals.append("obligation_or_evidence_progress")
    if near_reskin:
        signals.append("near_verbatim_reskin")
    if engineering_noise:
        signals.append("engineering_noise")
    observation = ProductiveChildObservation(
        transition_signature=transition_signature(child),
        family_id=str(child_meta["mechanism_family_id"]),
        canonical_family_id=str(child_meta["canonical_mechanism_family_id"]),
        novelty_delta=round(novelty_delta, 4),
        obligation_bonus=obligation_bonus,
        near_verbatim_reskin=near_reskin,
        engineering_noise=engineering_noise,
        signals=signals,
    )
    child_meta["productive_child_observation"] = observation.to_dict()
    return observation


def record_candidate_budget_decision(
    candidate: CandidateGenome,
    *,
    source: str,
    reason: str,
    action: str = "soft_retain",
    hard_gate: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_nextgen_identity(candidate)
    event = {
        "at": utc_now(),
        "source": str(source),
        "reason": str(reason),
        "action": str(action),
        "hard_gate": bool(hard_gate),
        "details": dict(details or {}),
    }
    decisions = candidate.metadata.setdefault("candidate_budget_decisions", [])
    if not isinstance(decisions, list):
        decisions = []
        candidate.metadata["candidate_budget_decisions"] = decisions
    decisions.append(event)
    del decisions[:-100]
    candidate.metadata["candidate_budget_decision"] = event
    return event


def structurally_blocked(candidate: CandidateGenome) -> bool:
    metadata = coerce_dict(candidate.metadata)
    sensitive_flag = "sec" + "ret_leak"
    if any(metadata.get(key) for key in ("structural_failure", "safety_blocked", "sensitive_leak", sensitive_flag, "terminal_structural_failure")):
        return True
    verdict = coerce_dict(metadata.get("failure_classification"))
    category = str(verdict.get("category") or "").lower()
    sensitive_token = "sec" + "ret"
    structural_tokens = (
        "unsafe",
        "credential",
        sensitive_token,
        "symlink",
        "escape",
        "structural",
        "missing_project_path",
        "missing_existing_project_path",
        "second_runtime_or_ranking_authority",
        "second runtime",
        "parallel runtime",
        "new ranking authority",
        "hidden_fallback",
        "hidden fallback",
        "fallback router",
        "path_escape",
    )
    if any(token in category for token in structural_tokens):
        return True
    stage = coerce_dict(metadata.get("stage_eligibility"))
    reason_text = " ".join(
        str(value or "").lower()
        for value in (
            metadata.get("hard_reject_reason"),
            metadata.get("terminal_reject_reason"),
            stage.get("hard_reject_reason"),
        )
    )
    if any(token in reason_text for token in structural_tokens):
        return True
    diagnostics = " ".join(str(item).lower() for item in candidate.failure_lessons[:12])
    return any(token in diagnostics for token in ("unsafe", "credential", sensitive_token, "symlink", "path_escape"))


def budget_eligible_candidates(candidates: Iterable[CandidateGenome]) -> list[CandidateGenome]:
    out: list[CandidateGenome] = []
    for candidate in candidates:
        ensure_nextgen_identity(candidate)
        if structurally_blocked(candidate):
            record_candidate_budget_decision(candidate, source="budget_eligible_candidates", reason="structural_or_safety_blocked", action="hard_exclude", hard_gate=True)
            continue
        if not _text(candidate) and not candidate.multihead_scores:
            record_candidate_budget_decision(candidate, source="budget_eligible_candidates", reason="empty_candidate_payload", action="soft_deprioritize")
        else:
            record_candidate_budget_decision(candidate, source="budget_eligible_candidates", reason="eligible_for_exploration_budget")
        out.append(candidate)
    return out


def cbt_soft_budget_adjustment(candidate: CandidateGenome, population: Iterable[CandidateGenome]) -> float:
    meta = ensure_nextgen_identity(candidate)
    family = str(meta.get("canonical_mechanism_family_id") or candidate.id)
    family_counts: dict[str, int] = {}
    total = 0
    for item in population:
        item_meta = ensure_nextgen_identity(item)
        key = str(item_meta.get("canonical_mechanism_family_id") or item.id)
        family_counts[key] = family_counts.get(key, 0) + 1
        total += 1
    count = family_counts.get(family, 1)
    rarity = float(candidate.multihead_scores.get("rarity", 0.0) or 0.0)
    novelty = float(candidate.multihead_scores.get("novelty", 0.0) or 0.0)
    observation = coerce_dict(meta.get("productive_child_observation"))
    near_reskin = bool(observation.get("near_verbatim_reskin"))
    low_sample = count <= 1
    protected = low_sample or rarity >= 0.5 or novelty >= 0.5 or bool(candidate.edge_knowledge_seeds)
    overrepresented = total > 0 and count / max(1, total) > 0.45
    throttled = bool(overrepresented and near_reskin and not protected)
    signal = {
        "family_id": family,
        "family_count": count,
        "population_count": total,
        "would_protect": protected,
        "would_throttle": throttled,
        "floor": 1,
        "mode": "soft_quota_trace",
    }
    meta["cbt_soft_quota"] = signal
    if throttled:
        record_candidate_budget_decision(candidate, source="cbt_soft_quota", reason="persistent_near_verbatim_reskin_family", action="soft_downweight", details=signal)
        return -0.12
    if protected:
        record_candidate_budget_decision(candidate, source="cbt_soft_quota", reason="low_sample_or_novelty_protected", action="soft_boost", details=signal)
        return 0.12
    return 0.0


def false_cull_monitor(candidates: Iterable[CandidateGenome]) -> dict[str, Any]:
    items = list(candidates)
    blocked: dict[str, int] = {}
    low_similarity_or_singleton = 0
    high_intent_nonactive = 0
    for candidate in items:
        meta = ensure_nextgen_identity(candidate)
        decisions = candidate.metadata.get("candidate_budget_decisions", [])
        if isinstance(decisions, list):
            for item in decisions:
                if isinstance(item, dict):
                    reason = str(item.get("reason") or "")
                    blocked[reason] = blocked.get(reason, 0) + 1
        cbt = coerce_dict(meta.get("cbt_soft_quota"))
        if cbt.get("would_protect"):
            low_similarity_or_singleton += 1
        intent = bind_candidate_intent(candidate)
        fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
        if fate not in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value} and float(intent.get("direct_answer_score") or 0.0) >= 0.55:
            high_intent_nonactive += 1
    total = max(1, len(items))
    return {
        "blocked_reason_histogram": blocked,
        "protected_branch_count": low_similarity_or_singleton,
        "high_intent_nonactive_count": high_intent_nonactive,
        "high_intent_nonactive_share": round(high_intent_nonactive / total, 4),
        "policy": "defang_throttle_if_high_intent_candidates_leave_active_lane",
    }


def _declared_family(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    search_space = coerce_dict(metadata.get("search_space"))
    return str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()


def _lineage_family(candidate: CandidateGenome) -> str:
    return str(candidate.lineage[0]) if candidate.lineage else ""


def _text(candidate: CandidateGenome | None) -> str:
    if candidate is None:
        return ""
    artifact = candidate.artifact
    if isinstance(artifact, dict):
        artifact_text = " ".join(str(v) for v in artifact.values())
    else:
        artifact_text = str(artifact or "")
    return " ".join(str(item or "") for item in (candidate.core_mechanism, candidate.concise_claim, artifact_text)).strip()


def _rough_text_delta(a: str, b: str) -> float:
    a_terms = set(a.lower().split())
    b_terms = set(b.lower().split())
    if not a_terms and not b_terms:
        return 0.0
    return 1.0 - (len(a_terms & b_terms) / max(1, len(a_terms | b_terms)))


def _looks_like_engineering_noise(text: str) -> bool:
    return False


def _summary_text(candidate: CandidateGenome) -> str:
    text = candidate.core_mechanism or candidate.concise_claim or candidate_answer_text(candidate)
    if isinstance(text, (dict, list)):
        text = str(text)
    return " ".join(str(text or "").split())[:600]


def _contract_search_intent(contract: Any | None) -> str:
    if contract is None:
        return ""
    if hasattr(contract, "normalized_goal") or hasattr(contract, "original_user_goal"):
        return str(getattr(contract, "normalized_goal", "") or getattr(contract, "original_user_goal", "") or "").strip()
    data = coerce_dict(contract)
    return str(data.get("normalized_goal") or data.get("original_user_goal") or data.get("objective") or "").strip()


def _main_claim(candidate: CandidateGenome) -> str:
    for value in (candidate.core_mechanism, candidate.concise_claim):
        text = " ".join(str(value or "").split())
        if text:
            return text[:900]
    artifact = candidate.artifact
    if isinstance(artifact, dict):
        for key in ("mechanism", "core_mechanism", "claim", "summary", "description", "content"):
            text = " ".join(str(artifact.get(key) or "").split())
            if text:
                return text[:900]
    return " ".join(str(artifact or "").split())[:900]


def _supporting_claims(candidate: CandidateGenome, main_claim: str) -> list[str]:
    seen = {_normalize_text(main_claim)}
    out: list[str] = []
    for value in (candidate.concise_claim, candidate.artifact):
        if isinstance(value, dict):
            parts = [str(item) for item in value.values() if isinstance(item, (str, int, float))]
            text = " ".join(parts)
        else:
            text = str(value or "")
        text = " ".join(text.split())[:900]
        key = _normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _explicit_intent_score(candidate: CandidateGenome, existing: dict[str, Any]) -> float | None:
    for source in (
        existing,
        coerce_dict(candidate.metadata).get(INTENT_BINDING_KEY) if isinstance(coerce_dict(candidate.metadata).get(INTENT_BINDING_KEY), dict) else {},
        ensure_nextgen_identity(candidate).get(INTENT_BINDING_KEY) if isinstance(ensure_nextgen_identity(candidate).get(INTENT_BINDING_KEY), dict) else {},
        candidate.multihead_scores,
    ):
        if not isinstance(source, dict):
            continue
        for key in ("direct_answer_score", "intent_directness", "objective_directness"):
            if key in source:
                try:
                    return _bounded(source.get(key))
                except (TypeError, ValueError):
                    return None
    return None


def _direct_answer_score(search_intent: str, main_claim: str, full_text: str) -> float:
    if not str(main_claim or "").strip():
        return 0.0
    if not str(search_intent or "").strip():
        return 0.5
    intent_terms = set(_tokens(search_intent))
    claim_terms = set(_tokens(main_claim))
    full_terms = set(_tokens(full_text))
    if not intent_terms or not claim_terms:
        return 0.5
    overlap = intent_terms & claim_terms
    main_recall = len(overlap) / max(1, len(intent_terms))
    main_precision = len(overlap) / max(1, len(claim_terms))
    full_recall = len(intent_terms & full_terms) / max(1, len(intent_terms))
    return _bounded(0.62 * main_recall + 0.25 * main_precision + 0.13 * full_recall)


def _alignment_rationale(search_intent: str, main_claim: str, score: float) -> str:
    if not search_intent:
        return "no frozen search intent supplied; used candidate answer signal fallback"
    shared = sorted(set(_tokens(search_intent)) & set(_tokens(main_claim)))[:10]
    return f"free_text_intent_binding score={round(_bounded(score), 4)} shared_terms={shared}"


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[\w\-]{3,}", str(text or ""), flags=re.UNICODE)]


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _graded_verified(graded_output: Any | None) -> bool:
    if graded_output is None:
        return False
    data = graded_output.to_dict() if hasattr(graded_output, "to_dict") else coerce_dict(graded_output)
    if str(data.get("mode") or "").lower() != "verified_result":
        return False
    result = coerce_dict(data.get("result"))
    replay = coerce_dict(data.get("replay_certificate"))
    return bool(result or replay)


def _certificate_verified(final_certificate: dict[str, Any] | None, *, candidate_id: str) -> bool:
    cert = coerce_dict(final_certificate)
    if not cert:
        return False
    if candidate_id and str(cert.get("candidate_id") or "") not in {"", candidate_id}:
        return False
    replay = coerce_dict(cert.get("replay_certificate") or cert.get("verified_replay_certificate"))
    return bool(cert.get("objective_solved") and replay)


def _is_singleton_family(candidate: CandidateGenome, context: Iterable[CandidateGenome]) -> bool:
    meta = ensure_nextgen_identity(candidate)
    family = str(meta.get("canonical_mechanism_family_id") or candidate.id)
    count = 0
    for item in context:
        item_meta = ensure_nextgen_identity(item)
        if str(item_meta.get("canonical_mechanism_family_id") or item.id) == family:
            count += 1
    return count <= 1


def _candidate_source_pool(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    if metadata.get("seed_reservoir") or metadata.get("seed_reservoir_reason"):
        return "reservoir"
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="").lower()
    if fate == CandidateFate.DORMANT.value.lower():
        return "dormant"
    if fate == CandidateFate.FAILED.value.lower():
        return "failed"
    if fate == CandidateFate.CULLED.value.lower():
        return "reserve"
    return fate or "reserve"


def _bounded(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "CANONICAL_BUCKETS_PER_DECLARED_FAMILY",
    "CANONICAL_FAMILY_VERSION",
    "NEXTGEN_METADATA_KEY",
    "ProductiveChildObservation",
    "select_best_current_direction",
    "resurrection_score",
    "resurrection_quota",
    "mark_resurrection_candidate",
    "bind_candidate_intent",
    "candidate_verification_status",
    "user_facing_verification_status",
    "candidate_answer_text",
    "blocked_from_verified_claim_reason",
    "best_current_direction_score",
    "best_current_direction_payload",
    "budget_eligible_candidates",
    "cbt_soft_budget_adjustment",
    "ensure_nextgen_identity",
    "false_cull_monitor",
    "family_signature",
    "nextgen_metadata",
    "observe_productive_child",
    "record_candidate_budget_decision",
    "structurally_blocked",
    "transition_signature",
]
