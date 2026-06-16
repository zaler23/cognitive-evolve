"""Relative ranking schemas and deterministic fake rater."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.adaptive_signals import in_top_band
from cognitive_evolve_runtime.nexus.policy import DEFAULT_FITNESS_AXES
from cognitive_evolve_runtime.nexus._serde import coerce_str_list
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.fallbacks import record_fallback
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus.stage_policy import parse_metric_value, stage_eligibility


@dataclass
class RelativeRankingResult:
    best_final_answer_id: str = ""
    strongest_mechanism_id: str = ""
    mutation_worthy_ids: list[str] = field(default_factory=list)
    edge_value_ids: list[str] = field(default_factory=list)
    auxiliary_ids: list[str] = field(default_factory=list)
    dormant_ids: list[str] = field(default_factory=list)
    dominated_pairs: list[tuple[str, str]] = field(default_factory=list)
    crossover_pairs: list[tuple[str, str]] = field(default_factory=list)
    preserve_incomplete_ids: list[str] = field(default_factory=list)
    pairwise_preferences: list[dict[str, Any]] = field(default_factory=list)
    multihead_observations: dict[str, dict[str, float]] = field(default_factory=dict)
    raw_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dominated_pairs"] = [list(pair) for pair in self.dominated_pairs]
        data["crossover_pairs"] = [list(pair) for pair in self.crossover_pairs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelativeRankingResult":
        observations, observation_warnings = _coerce_multihead_observations(data.get("multihead_observations"))
        raw_notes = str(data.get("raw_notes") or "")
        if observation_warnings:
            raw_notes = (raw_notes + "; " if raw_notes else "") + "ranking_schema_repair:" + ",".join(observation_warnings[:8])
        return cls(
            best_final_answer_id=str(data.get("best_final_answer_id") or ""),
            strongest_mechanism_id=str(data.get("strongest_mechanism_id") or ""),
            mutation_worthy_ids=coerce_str_list(data.get("mutation_worthy_ids")),
            edge_value_ids=coerce_str_list(data.get("edge_value_ids")),
            auxiliary_ids=coerce_str_list(data.get("auxiliary_ids")),
            dormant_ids=coerce_str_list(data.get("dormant_ids")),
            dominated_pairs=_pairs(data.get("dominated_pairs")),
            crossover_pairs=_pairs(data.get("crossover_pairs")),
            preserve_incomplete_ids=coerce_str_list(data.get("preserve_incomplete_ids")),
            pairwise_preferences=[dict(item) for item in data.get("pairwise_preferences", []) if isinstance(item, dict)],
            multihead_observations=observations,
            raw_notes=raw_notes,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "RelativeRankingResult":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("relative ranking JSON must decode to an object")
        return cls.from_dict(data)


def relative_rater_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "best_final_answer_id",
            "strongest_mechanism_id",
            "mutation_worthy_ids",
            "edge_value_ids",
            "auxiliary_ids",
            "dormant_ids",
            "dominated_pairs",
            "crossover_pairs",
            "preserve_incomplete_ids",
        ],
        "properties": {
            "best_final_answer_id": {"type": "string", "description": "Candidate most likely to become final answer, excluding auxiliary-only scaffolds by default."},
            "strongest_mechanism_id": {"type": "string"},
            "mutation_worthy_ids": {"type": "array", "items": {"type": "string"}},
            "edge_value_ids": {"type": "array", "items": {"type": "string"}},
            "auxiliary_ids": {"type": "array", "items": {"type": "string"}},
            "dormant_ids": {"type": "array", "items": {"type": "string"}},
            "dominated_pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "crossover_pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "preserve_incomplete_ids": {"type": "array", "items": {"type": "string"}},
            "pairwise_preferences": {"type": "array", "items": {"type": "object"}},
            "multihead_observations": {"type": "object"},
            "raw_notes": {"type": "string"},
        },
    }


class RelativeRater:
    """Adapter that can call a model, with deterministic fallback for tests."""

    def __init__(self, model: NexusModelLike | None = None) -> None:
        self.model = model

    def schema(self) -> dict[str, Any]:
        return relative_rater_schema()

    def rank(self, *, candidates: list[CandidateGenome], contract: Any | None = None, policy: Any | None = None, archives: Any | None = None) -> RelativeRankingResult:
        if self.model is not None and hasattr(self.model, "relative_rank"):
            try:
                primary = _ranking_from_raw(
                    self.model.relative_rank(candidates=candidates, contract=contract, policy=policy, archives=archives)
                )
                # Model judges are sensitive to presentation order.  For real
                # multi-candidate rankings, ask for the same comparison in the
                # reverse order and merge by candidate id.  This is not a second
                # authority; it is a cheap position-bias diagnostic on the same
                # judge surface.  Quota/provider errors still propagate from the
                # first call so runs pause truthfully instead of silently
                # degrading.
                if len(candidates) > 1:
                    try:
                        reverse = _ranking_from_raw(
                            self.model.relative_rank(candidates=list(reversed(candidates)), contract=contract, policy=policy, archives=archives)
                        )
                        return _sanitize_ranking(_merge_ab_rankings(primary, reverse, candidates), candidates)
                    except Exception as reverse_exc:
                        if is_quota_error(reverse_exc):
                            raise
                        record_fallback(stage="relative_ranking_reverse_pass", reason=reverse_exc.__class__.__name__, detail=str(reverse_exc))
                        primary.raw_notes = (
                            primary.raw_notes + "; " if primary.raw_notes else ""
                        ) + f"ab_order_reverse_pass_unavailable:{reverse_exc.__class__.__name__}"
                return _sanitize_ranking(primary, candidates)
            except Exception as exc:
                if is_quota_error(exc):
                    raise
                record_fallback(stage="relative_ranking", reason=exc.__class__.__name__, detail=str(exc))
                repaired = _deterministic_rank(candidates, raw_notes=f"model_relative_rank_schema_error_repaired:{exc.__class__.__name__}:{exc}; evidence_degraded_no_final_answer_claim")
                repaired.best_final_answer_id = ""
                for candidate in candidates:
                    candidate.metadata["ranking_schema_repair_error"] = f"{exc.__class__.__name__}: {exc}"
                    candidate.metadata["evidence_degraded"] = True
                    candidate.metadata["final_output_blocked_reason"] = "model_relative_rank_unavailable"
                return repaired
        if not candidates:
            return RelativeRankingResult(raw_notes="no candidates")
        return _deterministic_rank(candidates, raw_notes="deterministic relative fallback; verifier-ineligible candidates cannot become best_final_answer")


def _ranking_from_raw(raw: Any) -> RelativeRankingResult:
    if isinstance(raw, RelativeRankingResult):
        return raw
    if isinstance(raw, dict):
        return RelativeRankingResult.from_dict(raw)
    raise ValueError("relative_rank model response must be a RelativeRankingResult or dict")


def _merge_ab_rankings(a: RelativeRankingResult, b: RelativeRankingResult, candidates: list[CandidateGenome]) -> RelativeRankingResult:
    ids = {candidate.id for candidate in candidates}
    by_id = {candidate.id: candidate for candidate in candidates}
    best = _merge_best_id(a.best_final_answer_id, b.best_final_answer_id, candidates, axis="answer_likelihood")
    mechanism = _merge_best_id(a.strongest_mechanism_id, b.strongest_mechanism_id, candidates, axis="core_mechanism_strength")
    notes = "; ".join(
        item
        for item in [
            a.raw_notes,
            b.raw_notes,
            "ab_order_bias_mitigation:original_plus_reversed_order; verbosity_bias_guard=model_prompt",
        ]
        if item
    )
    return RelativeRankingResult(
        best_final_answer_id=best,
        strongest_mechanism_id=mechanism,
        mutation_worthy_ids=_ordered_union(candidates, a.mutation_worthy_ids, b.mutation_worthy_ids),
        edge_value_ids=_ordered_union(candidates, a.edge_value_ids, b.edge_value_ids),
        auxiliary_ids=_ordered_union(candidates, a.auxiliary_ids, b.auxiliary_ids),
        dormant_ids=_ordered_union(candidates, a.dormant_ids, b.dormant_ids),
        dominated_pairs=_pair_union(a.dominated_pairs, b.dominated_pairs, ids),
        crossover_pairs=_pair_union(a.crossover_pairs, b.crossover_pairs, ids),
        preserve_incomplete_ids=_ordered_union(candidates, a.preserve_incomplete_ids, b.preserve_incomplete_ids),
        pairwise_preferences=[*a.pairwise_preferences, *b.pairwise_preferences],
        multihead_observations=_merge_multihead_observations(a.multihead_observations, b.multihead_observations, by_id),
        raw_notes=notes,
    )


def _merge_best_id(first: str, second: str, candidates: list[CandidateGenome], *, axis: str) -> str:
    ids = {candidate.id for candidate in candidates}
    first = first if first in ids else ""
    second = second if second in ids else ""
    if first and first == second:
        return first
    contenders = [candidate for candidate in candidates if candidate.id in {first, second}]
    eligible = [candidate for candidate in contenders if _rank_eligible(candidate)]
    if eligible:
        return max(eligible, key=lambda candidate: (_candidate_score(candidate, axis), _main_answer_score(candidate), candidate.id)).id
    eligible_all = [candidate for candidate in candidates if _rank_eligible(candidate)]
    if eligible_all:
        return max(eligible_all, key=lambda candidate: (_candidate_score(candidate, axis), _main_answer_score(candidate), candidate.id)).id
    return ""


def _ordered_union(candidates: list[CandidateGenome], *groups: list[str]) -> list[str]:
    allowed = {candidate.id for candidate in candidates}
    ranked_ids = []
    for group in groups:
        ranked_ids.extend(str(item) for item in group or [] if str(item) in allowed)
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate.id in ranked_ids and candidate.id not in seen:
            out.append(candidate.id)
            seen.add(candidate.id)
    return out


def _pair_union(a: list[tuple[str, str]], b: list[tuple[str, str]], ids: set[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for winner, loser in [*a, *b]:
        pair = (str(winner), str(loser))
        if pair[0] in ids and pair[1] in ids and pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out[:40]


def _merge_multihead_observations(
    a: dict[str, dict[str, float]],
    b: dict[str, dict[str, float]],
    by_id: dict[str, CandidateGenome],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for candidate_id, candidate in by_id.items():
        axes = set(a.get(candidate_id, {})) | set(b.get(candidate_id, {})) | set(candidate.multihead_scores)
        merged: dict[str, float] = {}
        for axis in axes:
            values: list[float] = []
            if axis in a.get(candidate_id, {}):
                values.append(a[candidate_id][axis])
            if axis in b.get(candidate_id, {}):
                values.append(b[candidate_id][axis])
            if not values:
                values.append(_candidate_score(candidate, axis))
            merged[str(axis)] = _bound_score(sum(values) / len(values))
        out[candidate_id] = merged
    return out


def _deterministic_rank(candidates: list[CandidateGenome], *, raw_notes: str) -> RelativeRankingResult:
    if not candidates:
        return RelativeRankingResult(raw_notes=raw_notes or "no candidates")
    eligible = [candidate for candidate in candidates if _rank_eligible(candidate)]
    answer_candidates = [c for c in eligible if _main_answer_score(c) >= _auxiliary_score(c)]
    best = max(answer_candidates, key=_main_answer_score) if answer_candidates else None
    mechanism_pool = eligible or candidates
    mechanism = max(mechanism_pool, key=lambda c: _candidate_score(c, "core_mechanism_strength"))
    repair_pool = [candidate for candidate in candidates if _parent_eligible(candidate)]
    mutation_worthy_pool = eligible or repair_pool or candidates
    mutation_worthy = sorted(mutation_worthy_pool, key=_reproductive_hint, reverse=True)[: max(1, min(3, len(mutation_worthy_pool)))]
    edge_ids = [c.id for c in candidates if c.edge_knowledge_seeds or _candidate_score(c, "rarity") > 0.3]
    auxiliary_ids = [c.id for c in candidates if _auxiliary_score(c) > _main_answer_score(c)]
    dormant_ids = [
        c.id
        for c in candidates
        if (c.missing_parts or not _rank_eligible(c)) and c.id not in auxiliary_ids and (best is None or c.id != best.id)
    ]
    preserve = [c.id for c in candidates if c.missing_parts and (c.edge_knowledge_seeds or (_candidate_score(c, "novelty") > 0 and in_top_band(c, candidates, "novelty")))]
    crossover_pairs: list[tuple[str, str]] = []
    if len(mutation_worthy) >= 2:
        crossover_pairs.append((mutation_worthy[0].id, mutation_worthy[1].id))
    dominated: list[tuple[str, str]] = []
    for a in candidates:
        for b in candidates:
            if a.id != b.id and _dominates(a, b):
                dominated.append((a.id, b.id))
    return RelativeRankingResult(
        best_final_answer_id=best.id if best is not None else "",
        strongest_mechanism_id=mechanism.id,
        mutation_worthy_ids=[c.id for c in mutation_worthy],
        edge_value_ids=edge_ids,
        auxiliary_ids=auxiliary_ids,
        dormant_ids=dormant_ids,
        dominated_pairs=dominated[:20],
        crossover_pairs=crossover_pairs,
        preserve_incomplete_ids=preserve,
        pairwise_preferences=_deterministic_pairwise_preferences(candidates),
        multihead_observations={c.id: {axis: _candidate_score(c, axis) for axis in DEFAULT_FITNESS_AXES} for c in candidates},
        raw_notes=raw_notes,
    )


def _pairs(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def _coerce_multihead_observations(value: Any) -> tuple[dict[str, dict[str, float]], list[str]]:
    observations: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    if not isinstance(value, dict):
        if value not in (None, "", []):
            warnings.append("multihead_observations_not_object")
        return observations, warnings
    for candidate_id, raw_axes in value.items():
        if not isinstance(raw_axes, dict):
            warnings.append(f"{candidate_id}:axes_not_object")
            continue
        axes: dict[str, float] = {}
        for axis, raw_score in raw_axes.items():
            score = _coerce_score(raw_score)
            if score is None:
                warnings.append(f"{candidate_id}.{axis}:score_dropped")
                continue
            axes[str(axis)] = score
        observations[str(candidate_id)] = axes
    return observations, warnings


def _coerce_score(value: Any) -> float | None:
    return parse_metric_value(value)


def _bound_score(value: float) -> float:
    return bounded_score(value)


def _candidate_score(candidate: CandidateGenome, axis: str, default: float = 0.0) -> float:
    parsed = _coerce_score(candidate.multihead_scores.get(axis, default))
    return float(default) if parsed is None else parsed


def _main_answer_score(candidate: CandidateGenome) -> float:
    verifier = _verification_score(candidate)
    axes = [
        _candidate_score(candidate, "objective_alignment"),
        _candidate_score(candidate, "answer_likelihood"),
        _candidate_score(candidate, "verifiability"),
        verifier,
    ]
    return sum(axes) / len(axes)


def _auxiliary_score(candidate: CandidateGenome) -> float:
    return _candidate_score(candidate, "auxiliary_value")


def _reproductive_hint(candidate: CandidateGenome) -> float:
    hard_failure_penalty = 1.0 if not _rank_eligible(candidate) else 0.0
    repair_bonus = 1.0 if _parent_eligible(candidate) and not _rank_eligible(candidate) else 0.0
    positive = [
        _main_answer_score(candidate),
        _candidate_score(candidate, "novelty"),
        _candidate_score(candidate, "rarity"),
        _candidate_score(candidate, "tool_progress"),
        repair_bonus,
    ]
    return sum(positive) / len(positive) - _candidate_score(candidate, "deferral_risk") - hard_failure_penalty


def _dominates(a: CandidateGenome, b: CandidateGenome) -> bool:
    axes = ["objective_alignment", "answer_likelihood", "core_mechanism_strength", "verifiability", "robustness"]
    return all(_candidate_score(a, axis) >= _candidate_score(b, axis) for axis in axes) and any(_candidate_score(a, axis) > _candidate_score(b, axis) for axis in axes)


def _deterministic_pairwise_preferences(candidates: list[CandidateGenome]) -> list[dict[str, Any]]:
    """Create local pairwise comparisons without a best-vs-everyone star.

    The deterministic fallback is not a judge; it is a schema-safe pressure
    source when the model is absent.  Adjacent comparisons along each axis keep
    Elo connected while avoiding a topology where the first best candidate
    dominates every later rating update.
    """

    preferences: list[dict[str, Any]] = []
    axes = ["objective_alignment", "answer_likelihood", "core_mechanism_strength", "verifiability", "novelty"]
    for axis in axes:
        ordered = sorted(candidates, key=lambda c: _candidate_score(c, axis), reverse=True)
        for winner, loser in zip(ordered, ordered[1:]):
            if winner.id == loser.id:
                continue
            winner_score = _candidate_score(winner, axis)
            loser_score = _candidate_score(loser, axis)
            if winner_score <= loser_score:
                continue
            preferences.append({"winner": winner.id, "loser": loser.id, "axis": axis, "weight": max(0.05, min(1.0, winner_score - loser_score))})
    return preferences[: max(0, min(64, len(preferences)))]


def _rank_eligible(candidate: CandidateGenome) -> bool:
    result = getattr(candidate, "verification_result", {}) or {}
    if not isinstance(result, dict) or not result:
        return True
    if result.get("passed") is False:
        return False
    if result.get("rank_eligible") is False:
        return False
    diagnostics = set(str(item) for item in result.get("diagnostics", []) if item)
    hard = {
        "proof_object_absent",
        "proof_object_structurally_weak",
        "ledger_non_progressing",
        "duplicate_formal_signature",
        "blocking_obligation_not_targeted",
        "obligation_delta_absent",
        "evidence_ref_absent",
        "evidence_ref_unverified",
        "source_binding_absent",
        "source_binding_missing_path",
        "patch_target_missing",
    }
    return not diagnostics.intersection(hard)


def _parent_eligible(candidate: CandidateGenome) -> bool:
    metadata = getattr(candidate, "metadata", {}) or {}
    decision = metadata.get("stage_eligibility") if isinstance(metadata, dict) else None
    if isinstance(decision, dict):
        return bool(decision.get("parent_eligible")) and bool(decision.get("repair_required"))
    computed = stage_eligibility(candidate)
    return computed.parent_eligible and computed.repair_required


def _verification_score(candidate: CandidateGenome) -> float:
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict):
        component_scores: list[float] = []
        proof = result.get("proof_progress")
        if isinstance(proof, dict) and proof.get("score") is not None:
            parsed = parse_metric_value(proof.get("score"))
            if parsed is not None:
                component_scores.append(parsed)
        evidence = result.get("evidence_obligation")
        if isinstance(evidence, dict) and evidence.get("score") is not None:
            parsed = parse_metric_value(evidence.get("score"))
            if parsed is not None:
                component_scores.append(parsed)
        if component_scores:
            return sum(component_scores) / len(component_scores)
        return 1.0 if result.get("passed") is True else 0.0 if result.get("passed") is False else 0.5
    return 0.5


def _sanitize_ranking(ranking: RelativeRankingResult, candidates: list[CandidateGenome]) -> RelativeRankingResult:
    ids = {candidate.id for candidate in candidates}
    eligible = [candidate for candidate in candidates if _rank_eligible(candidate)]
    eligible_ids = {candidate.id for candidate in eligible}
    if ranking.best_final_answer_id and ranking.best_final_answer_id not in eligible_ids:
        fallback = max(eligible, key=_main_answer_score).id if eligible else ""
        ranking.best_final_answer_id = fallback
        ranking.raw_notes = (ranking.raw_notes + "; " if ranking.raw_notes else "") + "runtime_replaced_verifier_ineligible_best_candidate"
    if ranking.strongest_mechanism_id not in ids:
        ranking.strongest_mechanism_id = (max(eligible or candidates, key=lambda c: _candidate_score(c, "core_mechanism_strength")).id if candidates else "")
    ranking.mutation_worthy_ids = [candidate_id for candidate_id in ranking.mutation_worthy_ids if candidate_id in ids]
    ranking.edge_value_ids = [candidate_id for candidate_id in ranking.edge_value_ids if candidate_id in ids]
    ranking.auxiliary_ids = [candidate_id for candidate_id in ranking.auxiliary_ids if candidate_id in ids]
    ranking.dormant_ids = [candidate_id for candidate_id in ranking.dormant_ids if candidate_id in ids]
    return ranking


__all__ = ["RelativeRater", "RelativeRankingResult", "relative_rater_schema"]
