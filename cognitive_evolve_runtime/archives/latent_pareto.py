"""Latent intent Pareto archive lane."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, coerce_str_list, utc_now

TERMINAL_FAILURE_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}

@dataclass
class LatentParetoIntentArchive:
    """Archive lane for latent intent-frontier candidates.

    This is a diversity/exploration index only.  Membership here does not imply
    answer-archive membership or final-answer eligibility.
    """

    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    intent_buckets: dict[str, list[str]] = field(default_factory=dict)
    candidate_intents: dict[str, list[str]] = field(default_factory=dict)
    candidate_observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_representatives_per_intent: int = 3
    stale_after_rounds: int = 6
    removed_total: int = 0
    removal_reasons: dict[str, int] = field(default_factory=dict)
    removal_log: list[dict[str, Any]] = field(default_factory=list)

    def add(self, candidate: CandidateGenome) -> None:
        current_round = _latent_archive_round(candidate)
        self.prune(current_round=current_round)
        intents = _latent_archive_candidate_intents(candidate)
        previous = coerce_dict(self.candidate_observations.get(candidate.id))
        added_round = _optional_int(previous.get("added_round"), default=current_round)
        data = candidate.to_dict()
        data["latent_archive_governance"] = {
            "intents": list(intents),
            "added_round": added_round,
            "last_seen_round": current_round,
            "stale_after_rounds": int(self.stale_after_rounds),
            "representative_scores": {intent: _latent_representative_score(data, intent) for intent in intents},
        }
        self.candidates[candidate.id] = data
        self.candidate_intents[candidate.id] = list(intents)
        self.candidate_observations[candidate.id] = {"added_round": added_round, "last_seen_round": current_round}
        self._rebuild_intent_buckets()
        self._enforce_intent_quotas()

    def discard(self, candidate_id: str, *, reason: str = "removed", record: bool = True) -> None:
        existed = candidate_id in self.candidates
        self.candidates.pop(candidate_id, None)
        self.candidate_intents.pop(candidate_id, None)
        self.candidate_observations.pop(candidate_id, None)
        for intent, ids in list(self.intent_buckets.items()):
            kept = [item for item in ids if item != candidate_id]
            if kept:
                self.intent_buckets[intent] = kept
            else:
                self.intent_buckets.pop(intent, None)
        if existed and record:
            self._record_removal(candidate_id, reason=reason)

    def prune(self, *, current_round: int | None = None) -> list[str]:
        self._rebuild_intent_buckets()
        removed: list[str] = []
        for candidate_id, data in list(self.candidates.items()):
            reason = ""
            fate = CandidateFate.normalize(data.get("current_fate"))
            if fate in TERMINAL_FAILURE_FATES:
                reason = "terminal_fate"
            elif coerce_dict(data.get("metadata")).get("latent_pareto_frontier") is not True:
                reason = "stale_not_frontier"
            else:
                observation = coerce_dict(self.candidate_observations.get(candidate_id))
                last_seen = _optional_int(observation.get("last_seen_round"), default=None)
                if current_round is not None and last_seen is not None and current_round - last_seen > int(self.stale_after_rounds):
                    reason = "stale_age"
            if reason:
                self.discard(candidate_id, reason=reason)
                removed.append(candidate_id)
        return removed

    def summary(self) -> dict[str, Any]:
        self._rebuild_intent_buckets()
        latest_round = self._latest_observed_round()
        stale_candidates = 0
        if latest_round is not None:
            for candidate_id in self.candidates:
                last_seen = _optional_int(coerce_dict(self.candidate_observations.get(candidate_id)).get("last_seen_round"), default=None)
                if last_seen is not None and latest_round - last_seen > int(self.stale_after_rounds):
                    stale_candidates += 1
        intent_count = len(self.intent_buckets)
        intent_selection_weights = {intent: (1.0 / intent_count if intent_count else 0.0) for intent in sorted(self.intent_buckets)}
        return {
            "candidates": len(self.candidates),
            "intent_count": intent_count,
            "intent_counts": {intent: len(ids) for intent, ids in sorted(self.intent_buckets.items())},
            "intent_buckets": {intent: list(ids) for intent, ids in sorted(self.intent_buckets.items())},
            "max_representatives_per_intent": int(self.max_representatives_per_intent),
            "stale_after_rounds": int(self.stale_after_rounds),
            "latest_observed_round": latest_round,
            "stale_candidates": stale_candidates,
            "removed_total": int(self.removed_total),
            "removal_reasons": dict(sorted(self.removal_reasons.items())),
            "intent_selection_weights": intent_selection_weights,
            "desirability_basis": "intent_bucket_uniform_not_archive_frequency",
        }

    def to_dict(self) -> dict[str, Any]:
        self._rebuild_intent_buckets()
        return {
            "candidates": self.candidates,
            "intent_buckets": {intent: list(ids) for intent, ids in self.intent_buckets.items()},
            "candidate_intents": {candidate_id: list(intents) for candidate_id, intents in self.candidate_intents.items()},
            "candidate_observations": {candidate_id: dict(observation) for candidate_id, observation in self.candidate_observations.items()},
            "max_representatives_per_intent": int(self.max_representatives_per_intent),
            "stale_after_rounds": int(self.stale_after_rounds),
            "removed_total": int(self.removed_total),
            "removal_reasons": dict(self.removal_reasons),
            "removal_log": [dict(item) for item in self.removal_log],
            "governance_summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentParetoIntentArchive":
        governance = coerce_dict(data.get("governance_summary"))
        archive = cls(
            candidates={str(k): dict(v) for k, v in coerce_dict(data.get("candidates")).items() if isinstance(v, dict)},
            intent_buckets={
                str(k): coerce_str_list(v)
                for k, v in coerce_dict(data.get("intent_buckets")).items()
            },
            candidate_intents={
                str(k): coerce_str_list(v)
                for k, v in coerce_dict(data.get("candidate_intents")).items()
            },
            candidate_observations={
                str(k): coerce_dict(v)
                for k, v in coerce_dict(data.get("candidate_observations")).items()
            },
            max_representatives_per_intent=max(
                1,
                _int_policy(
                    data.get("max_representatives_per_intent", governance.get("max_representatives_per_intent")),
                    default=3,
                )
                or 3,
            ),
            stale_after_rounds=max(
                0,
                _int_policy(data.get("stale_after_rounds", governance.get("stale_after_rounds")), default=6) or 0,
            ),
            removed_total=max(0, _int_policy(data.get("removed_total", governance.get("removed_total")), default=0) or 0),
            removal_reasons={
                str(k): max(0, _int_policy(v, default=0) or 0)
                for k, v in coerce_dict(data.get("removal_reasons", governance.get("removal_reasons"))).items()
            },
            removal_log=[dict(item) for item in data.get("removal_log", []) if isinstance(item, dict)][-50:],
        )
        archive._rebuild_intent_buckets()
        return archive

    def _rebuild_intent_buckets(self) -> None:
        buckets: dict[str, list[str]] = {}
        for candidate_id, data in list(self.candidates.items()):
            if not isinstance(data, dict):
                self.candidates.pop(candidate_id, None)
                continue
            intents = coerce_str_list(self.candidate_intents.get(candidate_id)) or _latent_archive_intents_from_data(data)
            self.candidate_intents[candidate_id] = intents
            observation = coerce_dict(self.candidate_observations.get(candidate_id))
            governance = coerce_dict(data.get("latent_archive_governance"))
            last_seen = _optional_int(observation.get("last_seen_round"), default=_optional_int(governance.get("last_seen_round"), default=None))
            added_round = _optional_int(observation.get("added_round"), default=_optional_int(governance.get("added_round"), default=last_seen))
            self.candidate_observations[candidate_id] = {"added_round": added_round, "last_seen_round": last_seen}
            for intent in intents:
                buckets.setdefault(intent, []).append(candidate_id)
        for intent, ids in buckets.items():
            ids.sort(key=lambda candidate_id: _latent_representative_sort_key(self.candidates[candidate_id], intent), reverse=True)
        self.intent_buckets = buckets

    def _enforce_intent_quotas(self) -> None:
        quota = max(1, int(self.max_representatives_per_intent))
        keep_ids: set[str] = set()
        for intent, ids in self.intent_buckets.items():
            keep_ids.update(ids[:quota])
        for candidate_id in list(self.candidates):
            if candidate_id not in keep_ids:
                self.discard(candidate_id, reason="intent_quota")
        self._rebuild_intent_buckets()

    def _latest_observed_round(self) -> int | None:
        rounds = [
            _optional_int(coerce_dict(observation).get("last_seen_round"), default=None)
            for observation in self.candidate_observations.values()
        ]
        present = [item for item in rounds if item is not None]
        return max(present) if present else None

    def _record_removal(self, candidate_id: str, *, reason: str) -> None:
        normalized = str(reason or "removed")
        self.removed_total += 1
        self.removal_reasons[normalized] = self.removal_reasons.get(normalized, 0) + 1
        self.removal_log.append({"candidate_id": str(candidate_id), "reason": normalized, "at": utc_now()})
        self.removal_log = self.removal_log[-50:]

def _candidate_is_latent_pareto_frontier(candidate: CandidateGenome) -> bool:
    return coerce_dict(getattr(candidate, "metadata", {})).get("latent_pareto_frontier") is True

def _latent_archive_removal_reason(candidate: CandidateGenome, fate: str) -> str:
    if CandidateFate.normalize(fate) in TERMINAL_FAILURE_FATES:
        return "terminal_fate"
    if not _candidate_is_latent_pareto_frontier(candidate):
        return "stale_not_frontier"
    return ""

def _latent_archive_candidate_intents(candidate: CandidateGenome) -> list[str]:
    return _latent_archive_intents_from_data(candidate.to_dict())

def _latent_archive_intents_from_data(data: dict[str, Any]) -> list[str]:
    metadata = coerce_dict(data.get("metadata"))
    explicit = coerce_str_list(
        metadata.get("latent_intent_ids")
        or metadata.get("target_intent_ids")
        or metadata.get("latent_target_intent_ids")
    )
    if explicit:
        return explicit
    single = str(metadata.get("latent_intent_id") or metadata.get("latent_intent") or metadata.get("intent_id") or "").strip()
    if single:
        return [single]
    intent_scores = _latent_intent_scores_from_data(data)
    if intent_scores:
        best_score = max(intent_scores.values())
        best = [intent for intent, value in sorted(intent_scores.items()) if value == best_score]
        return best or [sorted(intent_scores)[0]]
    governance = coerce_dict(data.get("latent_archive_governance"))
    stored = coerce_str_list(governance.get("intents"))
    if stored:
        return stored
    return ["unknown_intent"]

def _latent_archive_round(candidate: CandidateGenome) -> int | None:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in ("latent_archive_round", "latent_round", "created_in_round", "generation_plan_round"):
        parsed = _optional_int(metadata.get(key), default=None)
        if parsed is not None:
            return parsed
    return _optional_int(getattr(candidate, "generation", None), default=None)

def _latent_intent_scores_from_data(data: dict[str, Any]) -> dict[str, float]:
    metadata = coerce_dict(data.get("metadata"))
    raw = coerce_dict(metadata.get("latent_intent_scores") or data.get("latent_intent_scores"))
    scores: dict[str, float] = {}
    for intent, value in raw.items():
        parsed = _finite_float(value, default=None)
        if parsed is not None:
            scores[str(intent)] = parsed
    return scores

def _latent_representative_score(data: dict[str, Any], intent: str) -> float:
    return _latent_representative_sort_key(data, intent)[0]

def _latent_representative_sort_key(data: dict[str, Any], intent: str) -> tuple[float, float, float, int, str]:
    metadata = coerce_dict(data.get("metadata"))
    scores = coerce_dict(data.get("multihead_scores"))
    intent_scores = _latent_intent_scores_from_data(data)
    intent_score = intent_scores.get(intent)
    if intent_score is None:
        intent_score = _finite_float(metadata.get("latent_representative_score"), default=0.0) or 0.0
    reproductive_signal = max(
        _finite_float(scores.get("latent_reproductive_signal"), default=0.0) or 0.0,
        _finite_float(scores.get("novelty"), default=0.0) or 0.0,
        _finite_float(scores.get("rarity"), default=0.0) or 0.0,
    )
    quality = sum(
        (_finite_float(scores.get(axis), default=0.0) or 0.0)
        for axis in (
            "objective_alignment",
            "answer_likelihood",
            "verifiability",
            "tool_progress",
            "proof_progress",
            "evidence_progress",
        )
    ) / 6.0
    governance = coerce_dict(data.get("latent_archive_governance"))
    last_seen = _optional_int(governance.get("last_seen_round"), default=_optional_int(data.get("generation"), default=0)) or 0
    return (float(intent_score), float(reproductive_signal), float(quality), int(last_seen), str(data.get("id") or ""))

def _finite_float(value: Any, *, default: float | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed

def _optional_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _int_policy(value: Any, *, default: int | None) -> int | None:
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return default
    return _optional_int(value, default=default)

__all__ = [
    "LatentParetoIntentArchive",
    "_candidate_is_latent_pareto_frontier",
    "_latent_archive_removal_reason",
]
