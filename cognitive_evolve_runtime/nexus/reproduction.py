"""Reproduction-stage helpers for the Nexus evolution loop.

This module keeps parent fallback, offspring verification, repair-lane marking,
and elite-gap merge mechanics out of the round controller.  The loop should
sequence the stage; this module owns the mechanics.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.failure_classifier import FailureVerdict, classify_candidate_failure
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def verify_offspring(offspring: list[CandidateGenome], verifier: Callable[[list[CandidateGenome]], list[Any]] | None) -> list[dict[str, Any]]:
    if verifier is None or not offspring:
        return []
    raw = verifier(offspring)
    summaries: list[dict[str, Any]] = []
    by_id = {candidate.id: candidate for candidate in offspring}
    for item in raw or []:
        payload = item.to_dict() if hasattr(item, "to_dict") else dict(item) if isinstance(item, dict) else {}
        if not payload:
            continue
        summaries.append(payload)
        candidate_id = str(payload.get("candidate_id") or "")
        candidate = by_id.get(candidate_id)
        if candidate is not None:
            candidate.metadata.setdefault("offspring_verification", payload)
            if payload.get("passed") is False:
                verdict = classify_candidate_failure(candidate, payload)
                candidate.metadata["failure_classification"] = _failure_verdict_with_candidate_id(verdict, candidate.id)
                if verdict.repairable:
                    _mark_offspring_repair_incubating(candidate, payload, verdict)
                else:
                    candidate.mark_fate(CandidateFate.FAILED.value)
                    candidate.failure_lessons.append("project_offspring_failed_sandbox_verification")
    return summaries


def offspring_failure_is_repairable(candidate: CandidateGenome, payload: dict[str, Any]) -> bool:
    """Return whether failed offspring should stay in the repair lane."""

    return classify_candidate_failure(candidate, payload).repairable


def dedupe_offspring_against_population(offspring: list[CandidateGenome], population: CandidatePopulation) -> list[CandidateGenome]:
    deduper = CandidateDeduper(list(population.candidates))
    unique: list[CandidateGenome] = []
    for candidate in offspring:
        if deduper.add(candidate):
            unique.append(candidate)
        else:
            candidate.metadata["rejected_offspring_reason"] = "duplicate_semantic_signature"
    return unique


def ranked_repair_fallback_parents(
    candidates: list[CandidateGenome],
    *,
    rankings: RelativeRankingResult,
    diagnosis: SearchDiagnosis,
    limit: int,
    current_round: int,
) -> list[CandidateGenome]:
    """Use ranked repair material as parents when strict live selection is empty."""

    target = max(0, int(limit or 0))
    if target <= 0 or not candidates or not _diagnosis_requests_repair_parent_fallback(diagnosis):
        return []
    by_id = {candidate.id: candidate for candidate in candidates}
    ranked_ids = list(
        dict.fromkeys(
            [
                *rankings.mutation_worthy_ids,
                *rankings.preserve_incomplete_ids,
                rankings.strongest_mechanism_id,
                *rankings.dormant_ids,
                *rankings.auxiliary_ids,
            ]
        )
    )
    ordered = [by_id[candidate_id] for candidate_id in ranked_ids if candidate_id in by_id]
    if not ordered:
        ordered = list(candidates)
    out: list[CandidateGenome] = []
    for candidate in ordered:
        if not _repair_fallback_parent_allowed(candidate):
            continue
        candidate.metadata["no_parent_repair_fallback"] = {
            "round": int(current_round or 0),
            "reason": "ranked_repair_parent_used_after_strict_parent_selection_empty",
            "final_answer_blocked": True,
        }
        candidate.metadata["final_answer_blocked_until_repaired"] = True
        out.append(candidate)
        if len(out) >= target:
            break
    return out


def sync_repair_parent_attempts_to_dormant_archive(archives: ArchiveManager, parents: list[CandidateGenome]) -> None:
    """Persist repair-attempt metadata for dormant archive recovery parents."""

    dormant = getattr(archives, "dormant_archive", None)
    store = getattr(dormant, "candidates", None) if dormant is not None else None
    if not isinstance(store, dict):
        return
    for parent in parents:
        if parent.id not in store:
            continue
        metadata = parent.metadata if isinstance(parent.metadata, dict) else {}
        if not (metadata.get("repair_seed") or metadata.get("dormant_repair_reactivation")):
            continue
        data = dict(store.get(parent.id) or {})
        data["metadata"] = dict(metadata)
        data["failure_lessons"] = list(parent.failure_lessons)
        data["mutation_history"] = list(parent.mutation_history)
        data["current_fate"] = CandidateFate.DORMANT.value
        store[parent.id] = data


def parents_for_crossover(parents: list[CandidateGenome], pair: tuple[str, str]) -> tuple[CandidateGenome, CandidateGenome]:
    by_id = {parent.id: parent for parent in parents}
    first = by_id.get(pair[0]) or parents[0]
    second = by_id.get(pair[1]) or (parents[1] if len(parents) > 1 else parents[0])
    return first, second


def elite_gap_merge_offspring(
    population: list[CandidateGenome],
    *,
    archives: ArchiveManager,
    policy: EvolutionPolicy,
    branch_factor: int,
) -> list[CandidateGenome]:
    config = _elite_gap_merge_policy(policy)
    if config.get("enabled", True) is False:
        return []
    elites = [candidate for candidate in population if CandidateFate.normalize(candidate.current_fate) == CandidateFate.ELITE.value]
    best = archives.best_answer_candidate(population)
    if best is not None and all(candidate.id != best.id for candidate in elites):
        elites.insert(0, best)
    if not elites:
        return []
    elite = max(elites, key=_answer_like_score)
    elite_gaps = _candidate_gap_terms(elite)
    if not elite_gaps:
        return []
    donors: list[tuple[float, CandidateGenome]] = []
    for candidate in population:
        if candidate.id == elite.id:
            continue
        fate = CandidateFate.normalize(candidate.current_fate)
        if fate in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}:
            continue
        donor_terms = _candidate_gap_terms(candidate) | _candidate_evidence_terms(candidate)
        overlap = elite_gaps.intersection(donor_terms)
        if not overlap and not _candidate_has_merge_material(candidate):
            continue
        score = len(overlap) + _answer_like_score(candidate) + (0.4 if _candidate_has_merge_material(candidate) else 0.0)
        donors.append((score, candidate))
    if not donors:
        return []
    donors.sort(key=lambda item: item[0], reverse=True)
    max_fraction = _float_config(config.get("max_fraction_of_branch_factor"), default=0.5)
    limit = max(1, int(max(1, branch_factor or 1) * max(0.1, min(1.0, max_fraction))))
    offspring: list[CandidateGenome] = []
    for _, donor in donors[:limit]:
        child = crossover(elite, donor, instruction="elite_gap_merge: transplant donor evidence/repair material into elite missing parts")
        child.metadata["elite_gap_merge"] = {
            "elite_parent_id": elite.id,
            "donor_parent_id": donor.id,
            "elite_gap_terms": sorted(elite_gaps)[:12],
            "donor_terms": sorted(_candidate_gap_terms(donor) | _candidate_evidence_terms(donor))[:12],
            "policy_source": str(config.get("source") or "evolution_policy"),
        }
        child.metadata["merged_from"] = [elite.id, donor.id]
        child.metadata["created_by_operator"] = "elite_gap_merge"
        child.metadata["final_answer_blocked_until_reverified"] = True
        child.verification_result = {
            "passed": False,
            "rank_eligible": True,
            "final_eligible": False,
            "diagnostics": ["elite_gap_merge_requires_verification"],
        }
        offspring.append(child)
    return offspring


def _offspring_failure_terms(candidate: CandidateGenome, payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(str(item or "") for item in candidate.failure_lessons)
    verification_result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
    terms.extend(str(item or "") for item in verification_result.get("diagnostics", []) or [])
    for key in ("diagnostics", "failed_files", "failure_lessons"):
        terms.extend(str(item or "") for item in payload.get(key, []) or [])
    patch_result = payload.get("patch_result")
    if isinstance(patch_result, dict):
        for key in ("diagnostics", "failed_files", "applied_files"):
            terms.extend(str(item or "") for item in patch_result.get(key, []) or [])
    for feedback in payload.get("tool_feedback", []) or []:
        if not isinstance(feedback, dict):
            continue
        for key in ("diagnostics", "failed_fragments"):
            terms.extend(str(item or "") for item in feedback.get(key, []) or [])
    return terms


def _failure_verdict_with_candidate_id(verdict: FailureVerdict, candidate_id: str) -> dict[str, Any]:
    payload = verdict.to_dict()
    guidance: list[dict[str, Any]] = []
    for item in payload.get("failure_guidance", []) or []:
        if isinstance(item, dict):
            fixed = dict(item)
            fixed["candidate_id"] = candidate_id
            guidance.append(fixed)
    payload["failure_guidance"] = guidance
    return payload


def _mark_offspring_repair_incubating(candidate: CandidateGenome, payload: dict[str, Any], verdict: FailureVerdict | None = None) -> None:
    verdict = verdict or classify_candidate_failure(candidate, payload)
    terms = [term for term in _offspring_failure_terms(candidate, payload) if term]
    blockers = list(dict.fromkeys([*verdict.blockers, *(str(term)[:160] for term in terms)]))[:8]
    repair_targets = list(dict.fromkeys(verdict.repair_targets))[:8]
    if "project_offspring_failed_sandbox_verification" not in candidate.failure_lessons:
        candidate.failure_lessons.append("project_offspring_failed_sandbox_verification")
    if "offspring_failed_but_repairable" not in candidate.failure_lessons:
        candidate.failure_lessons.append("offspring_failed_but_repairable")
    candidate.mark_fate(CandidateFate.INCUBATING.value)
    candidate.metadata["final_answer_blocked_until_repaired"] = True
    try:
        repair_attempts = int(candidate.metadata.get("repair_attempts") or candidate.metadata.get("repair_attempt") or 0)
    except (TypeError, ValueError):
        repair_attempts = 0
    candidate.metadata["repair_attempts"] = max(0, repair_attempts)
    candidate.metadata["repair_context"] = {
        "category": verdict.category,
        "reason": verdict.reason,
        "failure_signature": verdict.failure_signature,
        "repair_targets": repair_targets,
        "diagnostics": list(verdict.diagnostics)[:12],
    }
    candidate.metadata["offspring_repair_lane"] = {
        "reason": "failed_offspring_kept_for_bounded_repair",
        "passed": False,
        "blockers": blockers,
        "repair_targets": repair_targets,
        "failure_signature": verdict.failure_signature,
    }
    existing_guidance = candidate.metadata.get("failure_micro_guidance") if isinstance(candidate.metadata, dict) else None
    guidance: list[dict[str, Any]] = [dict(item) for item in existing_guidance or [] if isinstance(item, dict)]
    for item in verdict.failure_guidance:
        fixed = dict(item)
        fixed["candidate_id"] = candidate.id
        guidance.append(fixed)
    if guidance:
        candidate.metadata["failure_micro_guidance"] = _dedupe_guidance(guidance)[:5]
    candidate.metadata["repair_required"] = {
        "blockers": blockers,
        "evidence_needed": ["valid_unified_diff_or_patch_set", "source_binding", "post_pass_local_verification"],
        "source_bindings": [{"path": path, "kind": "source_file", "source": "offspring_failure_classifier"} for path in repair_targets[:5]],
        "next_actions": [
            "rewrite the patch against exact project-relative source context",
            "preserve the candidate's useful repair mechanism while fixing sandbox diagnostics",
        ],
        "failure_signature": verdict.failure_signature,
        "source": "offspring_verification_repair_lane",
    }


def _dedupe_guidance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = "|".join([str(item.get("blocker") or ""), str(item.get("next_action") or "")])
        if not key.strip("|") or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _diagnosis_requests_repair_parent_fallback(diagnosis: SearchDiagnosis) -> bool:
    actions = " ".join(str(item or "").lower() for item in diagnosis.recommended_actions)
    stagnation = str(diagnosis.stagnation_type or "").lower()
    notes = str(diagnosis.notes or "").lower()
    text = " ".join([actions, stagnation, notes])
    return bool(
        "reactivate" in text
        or "repair" in text
        or "verification" in text
        or "obligation" in text
        or "proofobjectabsence" in text
        or "proof_object" in text
        or "patch_no_effect" in text
        or "no_parents" in text
        or "pool_dormancy" in text
        or "routeincomplete" in text
        or "route_incomplete" in text
        or "docs_only" in text
        or "documentation_only" in text
        or "markdown_note" in text
        or "seed_note" in text
        or "runtime_code_change" in text
        or "concrete_code_patch" in text
        or "implementation patch" in text
        or "patch_application" in text
        or "malformed patch" in text
        or "unexpected eof" in text
        or "unexpected end of file" in text
        or "dry_run" in text
        or "dry run" in text
    )


def _repair_fallback_parent_allowed(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(candidate.current_fate)
    if fate == CandidateFate.CULLED.value:
        return False
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    decision = metadata.get("stage_eligibility")
    if isinstance(decision, dict) and decision.get("repair_exhausted") is True:
        return False
    if fate == CandidateFate.FAILED.value and not _failed_candidate_repairable(candidate):
        return False
    if metadata.get("semantic_drift") or metadata.get("unrelated_drift"):
        return False
    if not _has_repair_signal(candidate):
        return False
    return True


def _failed_candidate_repairable(candidate: CandidateGenome) -> bool:
    return classify_candidate_failure(candidate).repairable


def _has_repair_signal(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    if candidate.missing_parts or candidate.failure_lessons:
        return True
    if metadata.get("failure_micro_guidance") or metadata.get("repair_required"):
        return True
    result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
    if result.get("diagnostics") or result.get("failure_guidance"):
        return True
    return False


def _elite_gap_merge_policy(policy: EvolutionPolicy) -> dict[str, Any]:
    eligibility = _eligibility_policy(policy)
    raw = eligibility.get("elite_gap_merge") if isinstance(eligibility.get("elite_gap_merge"), dict) else {}
    if not raw and isinstance(policy.metadata, dict):
        raw = policy.metadata.get("elite_gap_merge") if isinstance(policy.metadata.get("elite_gap_merge"), dict) else {}
    return dict(raw or {"enabled": True, "max_fraction_of_branch_factor": 0.5, "source": "offline_fallback_model_overridable"})


def _eligibility_policy(policy: EvolutionPolicy) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("eligibility_policy") or metadata.get("stage_policy")
    return dict(raw) if isinstance(raw, dict) else {}


def _candidate_gap_terms(candidate: CandidateGenome) -> set[str]:
    terms: set[str] = set()
    for item in list(candidate.missing_parts) + list(candidate.failure_lessons):
        terms.update(_term_tokens(item))
    repair = candidate.metadata.get("repair_required") if isinstance(candidate.metadata, dict) else None
    if isinstance(repair, dict):
        for key in ("blockers", "evidence_needed", "acceptance_criteria", "next_actions"):
            for item in repair.get(key, []) or []:
                terms.update(_term_tokens(item))
    return {term for term in terms if term}


def _candidate_evidence_terms(candidate: CandidateGenome) -> set[str]:
    terms: set[str] = set()
    for payload in list(candidate.formal_artifacts) + list(candidate.evidence_refs) + list(candidate.source_bindings):
        if isinstance(payload, dict):
            for value in payload.values():
                terms.update(_term_tokens(value))
    for payload in (candidate.obligation_delta, candidate.evidence_delta):
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    for item in value:
                        terms.update(_term_tokens(item))
                else:
                    terms.update(_term_tokens(value))
    return {term for term in terms if term}


def _candidate_has_merge_material(candidate: CandidateGenome) -> bool:
    return bool(candidate.formal_artifacts or candidate.evidence_refs or candidate.source_bindings or candidate.obligation_delta or candidate.evidence_delta)


def _term_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text) if token not in {"the", "and", "for", "with", "this", "that"}}


def _answer_like_score(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    return float(scores.get("objective_alignment", 0.0) or 0.0) + float(scores.get("answer_likelihood", 0.0) or 0.0) + 0.5 * float(scores.get("verifiability", 0.0) or 0.0)


def _float_config(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


__all__ = [
    "dedupe_offspring_against_population",
    "elite_gap_merge_offspring",
    "offspring_failure_is_repairable",
    "parents_for_crossover",
    "ranked_repair_fallback_parents",
    "sync_repair_parent_attempts_to_dormant_archive",
    "verify_offspring",
]
