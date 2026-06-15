"""Parent selection based on reproductive value, not winner-only score."""
from __future__ import annotations

from typing import Any, Mapping

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.archives.quality_diversity import live_reproductive_candidates, pareto_frontier_ids
from cognitive_evolve_runtime.nexus.adaptive_signals import mean_percentile, percentile_rank
from cognitive_evolve_runtime.nexus.obligations import candidate_has_obligation_or_evidence_delta
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.population_vitality import repair_slot_count
from cognitive_evolve_runtime.nexus.stage_policy import stage_eligibility
from .novelty import population_novelty
from .lineage_saturation import detect_lineage_saturation


def reproductive_value(
    candidate: CandidateGenome,
    population: list[CandidateGenome],
    archives: object | None = None,
    *,
    advisory_features: Mapping[str, Any] | None = None,
) -> float:
    live_context = live_reproductive_candidates(population)
    fate = CandidateFate.normalize(candidate.current_fate)
    if fate not in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}:
        return -1.0
    if fate == CandidateFate.INCUBATING.value and not _incubating_parent_allowed(candidate):
        return -1.0
    context = live_context or population or [candidate]
    relative_quality = mean_percentile(
        candidate,
        context,
        ["objective_alignment", "answer_likelihood", "core_mechanism_strength", "verifiability"],
    )
    evidence_signal = mean_percentile(candidate, context, ["tool_progress", "proof_progress", "evidence_progress"])
    diversity_signal = max(
        population_novelty(candidate, live_context),
        mean_percentile(candidate, context, ["novelty", "rarity", "transfer_potential"]),
        1.0 if candidate.edge_knowledge_seeds else 0.0,
    )
    elo_signal = percentile_rank(candidate, context, "elo_reproductive_signal") if any("elo_reproductive_signal" in c.multihead_scores for c in context) else 0.0
    latent_signal = percentile_rank(candidate, context, "latent_reproductive_signal") if any("latent_reproductive_signal" in c.multihead_scores for c in context) else 0.0
    latent_frontier_signal = 1.0 if bool((candidate.metadata if isinstance(candidate.metadata, dict) else {}).get("latent_pareto_frontier")) else 0.0
    repair_signal = 1.0 if _repair_target_candidate(candidate) else 0.0
    niche_signal = 1.0 if candidate.niche_memberships else 0.0
    uncertainty_signal = 1.0 if candidate.uncertainty_notes else 0.0
    complementarity_signal = 1.0 if candidate.missing_parts and (candidate.edge_knowledge_seeds or candidate.core_mechanism) else 0.0
    lineage_report = detect_lineage_saturation(live_context)
    family = candidate.lineage[0] if candidate.lineage else candidate.id
    lineage_penalty = diversity_signal if family in lineage_report.saturated_families and not candidate_has_obligation_or_evidence_delta(candidate) else 0.0
    frontier_signal = 1.0 if candidate.id in pareto_frontier_ids(live_context) else 0.0
    constraint_penalty = _archive_constraint_penalty(candidate, archives)
    repeated_failure_penalty = len(candidate.failure_lessons) / max(1, len(candidate.failure_lessons) + len(context))
    auxiliary_penalty = 1.0 if candidate.current_fate == CandidateFate.AUXILIARY else 0.0
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    incubation_penalty = 0.0 if repair_signal else (1.0 if fate == CandidateFate.INCUBATING.value else 0.0)
    deprioritized_penalty = repair_signal if metadata.get("selection_deprioritized_until_new_delta") and not candidate_has_obligation_or_evidence_delta(candidate) else 0.0
    positive = [
        relative_quality,
        evidence_signal,
        diversity_signal,
        elo_signal,
        latent_signal,
        latent_frontier_signal,
        repair_signal,
        niche_signal,
        uncertainty_signal,
        complementarity_signal,
        frontier_signal,
    ]
    value = sum(positive) / max(1, len(positive))
    return (
        value
        + _advisory_selection_adjustment(candidate, advisory_features)
        - lineage_penalty
        - constraint_penalty
        - repeated_failure_penalty
        - auxiliary_penalty
        - incubation_penalty
        - deprioritized_penalty
    )


def _archive_constraint_penalty(candidate: CandidateGenome, archives: object | None) -> float:
    records = getattr(archives, "constraint_records", None)
    if not isinstance(records, list) or candidate_has_obligation_or_evidence_delta(candidate):
        return 0.0
    if CandidateFate.normalize(candidate.current_fate) == CandidateFate.INCUBATING.value and _incubating_parent_allowed(candidate):
        # Verification constraints on Incubating candidates are the repair target,
        # not a reason to block the bounded repair lane.  Final/rank eligibility is
        # still strict elsewhere; this only keeps repairable parents selectable.
        return 0.0
    targets = {candidate.id, candidate.core_mechanism, candidate.concise_claim}
    if candidate.lineage:
        targets.add(candidate.lineage[0])
    penalty = 0.0
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("target") or "") not in targets:
            continue
        kind = str(record.get("kind") or "")
        if kind == "lineage_freeze":
            penalty = max(penalty, 0.8)
        elif kind == "verification_constraint":
            penalty = max(penalty, 0.6)
        elif kind == "failure_lesson_constraint":
            penalty = max(penalty, 0.2)
    return penalty


class ParentSelector:
    def select(
        self,
        population: list[CandidateGenome],
        archives: object | None = None,
        *,
        limit: int = 2,
        eligibility_policy: dict[str, object] | None = None,
        advisory_features: Mapping[str, Any] | None = None,
    ) -> list[CandidateGenome]:
        viable = live_reproductive_candidates(population)
        selection_pressure = coerce_dict(coerce_dict(eligibility_policy).get("selection_pressure"))
        base_values = {candidate.id: reproductive_value(candidate, population, archives) + _selection_pressure_adjustment(candidate, selection_pressure) for candidate in viable}
        by_value = sorted(
            viable,
            key=lambda candidate: (
                base_values.get(candidate.id, -1.0) + _advisory_selection_adjustment(candidate, advisory_features),
                base_values.get(candidate.id, -1.0),
                candidate.id,
            ),
            reverse=True,
        )
        # Advisory features are never eligibility gates: they may reorder viable
        # parents, but the >=0 reproductive threshold remains the original base
        # runtime value.
        ranked = [candidate for candidate in by_value if base_values.get(candidate.id, -1.0) >= 0.0]
        primary = [candidate for candidate in ranked if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}]
        incubating = [candidate for candidate in ranked if CandidateFate.normalize(candidate.current_fate) == CandidateFate.INCUBATING.value and _incubating_parent_allowed(candidate)]
        target = max(0, limit)
        if target <= 0:
            return []
        if primary:
            repair_policy = coerce_dict(coerce_dict(eligibility_policy).get("repair_selection"))
            repair_slots = repair_slot_count(
                target=target,
                primary_count=len(primary),
                incubating_count=len(incubating),
                max_parent_fraction=_float(repair_policy.get("max_parent_fraction"), default=None),
                enabled=repair_policy.get("enabled", True) is not False,
            )
            selected = primary[: max(0, target - repair_slots)]
            selected.extend(incubating[: max(0, target - len(selected))])
            return selected[:target]
        # If the run has temporarily lost all Active/Elite candidates, keep a
        # small repair lane alive instead of declaring no parents available.
        if incubating:
            return incubating[:target]
        # A current Elite/Active candidate can still have a negative numeric
        # reproductive value after conservative penalties for repeated failure
        # lessons, lineage constraints, or zero final-answer scores.  That
        # should lower its priority, not collapse an unfinished run into
        # ``no_parents_available`` when it is the only live parent left.  Keep a
        # tiny floor for candidates that the stage policy still marks parent
        # eligible, while preserving hard-reject exclusions.
        primary_floor = [
            candidate
            for candidate in by_value
            if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}
            and _stage_parent_eligible(candidate)
        ]
        if primary_floor:
            return primary_floor[:target]
        # A conservative verifier can assign negative reproductive value to all
        # live candidates because they carry failure lessons, zero final-answer
        # scores, or archive constraints.  That should block final synthesis,
        # not kill the only repairable parents.  Keep a bounded repair fallback
        # so targeted mutation can add the missing evidence/formal/source delta.
        repairable = [
            candidate
            for candidate in by_value
            if _repair_target_candidate(candidate) and CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.INCUBATING.value}
        ]
        return repairable[:target]


def _incubating_parent_allowed(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    repair = metadata.get("repair_required")
    decision_payload = metadata.get("stage_eligibility")
    if isinstance(decision_payload, dict):
        if decision_payload.get("parent_eligible") is False:
            return False
        if decision_payload.get("repair_required") is False:
            return False
        if decision_payload.get("repair_exhausted") is True:
            return False
    elif not stage_eligibility(candidate).parent_eligible:
        return False
    else:
        decision = stage_eligibility(candidate)
        if decision.repair_exhausted:
            return False
    if isinstance(repair, dict) and repair.get("blockers"):
        return True
    if isinstance(repair, list) and repair:
        return True
    guidance = metadata.get("failure_micro_guidance")
    return bool(guidance)


def _repair_target_candidate(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    decision_payload = metadata.get("stage_eligibility")
    if isinstance(decision_payload, dict) and decision_payload.get("repair_exhausted") is True:
        return False
    repair = metadata.get("repair_required")
    if isinstance(repair, dict) and repair.get("blockers"):
        return True
    if isinstance(repair, list) and repair:
        return True
    guidance = metadata.get("failure_micro_guidance")
    return bool(guidance)


def _stage_parent_eligible(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    decision_payload = metadata.get("stage_eligibility")
    if isinstance(decision_payload, dict):
        if decision_payload.get("parent_eligible") is False:
            return False
        if str(decision_payload.get("hard_reject_reason") or "").strip():
            return False
    hard_reject = metadata.get("hard_reject_reason") or metadata.get("terminal_reject_reason") or metadata.get("hard_reject_class")
    return not bool(str(hard_reject or "").strip())


def _selection_pressure_adjustment(candidate: CandidateGenome, pressure: dict[str, object] | None) -> float:
    data = coerce_dict(pressure)
    if not data:
        return 0.0
    family_terms = _candidate_family_terms(candidate)
    over = {str(item).strip().lower() for item in data.get("over_explored_families", []) or [] if str(item).strip()}
    under = {str(item).strip().lower() for item in data.get("under_explored_families", []) or [] if str(item).strip()}
    prematurely_culled = {str(item).strip().lower() for item in data.get("prematurely_culled_genes", []) or [] if str(item).strip()}
    adjustment = 0.0
    over_penalty = _bounded_float(data.get("over_explored_penalty"), default=1.0)
    under_bonus = _bounded_float(data.get("under_explored_bonus"), default=1.0)
    if family_terms.intersection(over):
        adjustment -= over_penalty
        _selection_pressure_metadata(candidate)["over_explored_penalty"] = sorted(family_terms.intersection(over))
    if family_terms.intersection(under | prematurely_culled):
        adjustment += under_bonus
        _selection_pressure_metadata(candidate)["under_explored_bonus"] = sorted(family_terms.intersection(under | prematurely_culled))
    return adjustment


def _advisory_selection_adjustment(candidate: CandidateGenome, advisory_features: Mapping[str, Any] | None) -> float:
    if not advisory_features:
        return 0.0
    feature = advisory_features.get(candidate.id)
    if feature is None:
        return 0.0
    rank_prior = _bounded_float(_feature_value(feature, "rank_prior"), default=0.0)
    plan_value = _bounded_float(_feature_value(feature, "plan_value"), default=0.0)
    diversity = _bounded_float(_feature_value(feature, "diversity"), default=0.0)
    risk = _bounded_float(_feature_value(feature, "risk"), default=0.0)
    return ((rank_prior + plan_value + diversity) / 3.0) - risk


def _feature_value(feature: Any, key: str) -> Any:
    if isinstance(feature, Mapping):
        return feature.get(key, 0.0)
    return getattr(feature, key, 0.0)


def _selection_pressure_metadata(candidate: CandidateGenome) -> dict[str, object]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    raw = metadata.get("selection_pressure")
    if not isinstance(raw, dict):
        raw = {}
        metadata["selection_pressure"] = raw
        candidate.metadata = metadata
    return raw


def _candidate_family_terms(candidate: CandidateGenome) -> set[str]:
    values = [
        candidate.core_mechanism,
        candidate.concise_claim,
        *(candidate.niche_memberships or []),
        *(candidate.edge_knowledge_seeds or []),
    ]
    if candidate.lineage:
        values.append(candidate.lineage[0])
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for key in ("seed_type", "exploration_source"):
        values.append(str(metadata.get(key) or ""))
    terms: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text:
            terms.add(text)
    return terms


def _float(value: object, *, default: float | None) -> float | None:
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_float(value: object, *, default: float) -> float:
    parsed = _float(value, default=default)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))


__all__ = ["ParentSelector", "reproductive_value"]
