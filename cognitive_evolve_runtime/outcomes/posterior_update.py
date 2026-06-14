"""Bounded deterministic posterior updates for M5.1 latent objectives."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.outcomes.latent import IntentHypothesis, LatentProblemState, PreferenceEvidence
from cognitive_evolve_runtime.outcomes.latent_ledger import LatentLedger


UPDATE_MODEL_VERSION = "latent-posterior-update/v1"


@dataclass(frozen=True)
class PosteriorUpdateConfig:
    update_model_version: str = UPDATE_MODEL_VERSION
    max_abs_log_update: float = 1.25
    default_source_log_update_cap: float = 0.35
    prior_floor: float = 0.03
    max_single_intent_posterior: float = 0.92
    stale_decay_per_round: float = 0.95
    min_uncertainty: float = 0.02
    source_log_update_caps: dict[str, float] = field(
        default_factory=lambda: {
            "verified_improvement_certificate": 1.25,
            "improvement_certificate": 1.25,
            "verifier": 0.70,
            "verification": 0.70,
            "archive": 0.25,
            "critique": 0.25,
            "trial_observation": 0.20,
            "model_narrative": 0.10,
            "unknown": 0.20,
        }
    )
    source_reliability: dict[str, float] = field(
        default_factory=lambda: {
            "verified_improvement_certificate": 1.00,
            "improvement_certificate": 0.75,
            "verifier": 0.85,
            "verification": 0.85,
            "archive": 0.45,
            "critique": 0.35,
            "trial_observation": 0.40,
            "model_narrative": 0.20,
            "unknown": 0.35,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class LatentPosteriorSnapshot:
    state: LatentProblemState
    ledger_cursor: int = 0
    active_evidence_ids: tuple[str, ...] = ()
    ledger_replay_hash: str = ""
    update_model_version: str = UPDATE_MODEL_VERSION
    update_config_hash: str = ""
    update_trace: dict[str, Any] = field(default_factory=dict)
    materialized_at_utc: str = field(default_factory=utc_now)
    version: str = "latent-posterior-snapshot/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "state": self.state.to_dict(),
            "state_hash": self.state.state_hash(),
            "ledger_cursor": int(self.ledger_cursor),
            "active_evidence_ids": list(self.active_evidence_ids),
            "ledger_replay_hash": self.ledger_replay_hash,
            "update_model_version": self.update_model_version,
            "update_config_hash": self.update_config_hash,
            "update_trace": dict(self.update_trace),
            "materialized_at_utc": self.materialized_at_utc,
        }

    def stable_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("materialized_at_utc", None)
        return payload

    def snapshot_hash(self) -> str:
        return stable_hash(self.stable_payload())

    def decision_trace(self, *, decision_type: str) -> dict[str, Any]:
        return {
            "latent_ledger_cursor": int(self.ledger_cursor),
            "latent_posterior_snapshot_hash": self.snapshot_hash(),
            "latent_update_model_version": self.update_model_version,
            "latent_decision_trace_ref": f"latent-decision:{decision_type}:{self.snapshot_hash()[:16]}:{self.ledger_cursor}",
        }


def materialize_posterior_snapshot(
    initial_state: LatentProblemState,
    ledger: LatentLedger | None = None,
    *,
    config: PosteriorUpdateConfig | None = None,
    cursor: int | None = None,
) -> LatentPosteriorSnapshot:
    config = config or PosteriorUpdateConfig()
    replay = (ledger or LatentLedger()).replay(cursor=cursor)
    state, trace = bounded_update_intent_posteriors(initial_state, list(replay.active_evidence), config=config)
    return LatentPosteriorSnapshot(
        state=state,
        ledger_cursor=replay.cursor,
        active_evidence_ids=replay.active_evidence_ids,
        ledger_replay_hash=replay.ledger_replay_hash,
        update_model_version=config.update_model_version,
        update_config_hash=config.config_hash(),
        update_trace=trace,
    )


def bounded_update_intent_posteriors(
    state: LatentProblemState,
    evidence: list[PreferenceEvidence],
    *,
    config: PosteriorUpdateConfig | None = None,
) -> tuple[LatentProblemState, dict[str, Any]]:
    """Update latent intent posteriors with caps and collapse guards.

    This is intentionally conservative: evidence can shift search pressure, but
    it cannot erase live hypotheses or by itself prove objective closure.
    """

    config = config or PosteriorUpdateConfig()
    known = {intent.id for intent in state.intents}
    raw_by_source_intent: dict[tuple[str, str], float] = {}
    support_by_intent: dict[str, float] = {}
    contradiction_by_intent: dict[str, float] = {}
    used_refs: list[str] = list(state.evidence_refs)
    ignored = 0
    for item in evidence:
        if item.intent_id not in known:
            ignored += 1
            continue
        source = str(item.source_type or "unknown")
        decay = _stale_decay(item, config=config)
        reliability = _source_reliability(item, config=config)
        raw_update = max(-config.max_abs_log_update, min(config.max_abs_log_update, item.log_update() * decay * reliability))
        raw_by_source_intent[(source, item.intent_id)] = raw_by_source_intent.get((source, item.intent_id), 0.0) + raw_update
        support_by_intent[item.intent_id] = support_by_intent.get(item.intent_id, 0.0) + item.weight * item.confidence * item.support * decay * reliability
        contradiction_by_intent[item.intent_id] = contradiction_by_intent.get(item.intent_id, 0.0) + item.weight * item.confidence * item.contradiction * decay * reliability
        if item.evidence_ref:
            used_refs.append(item.evidence_ref)

    capped_by_intent: dict[str, float] = {}
    capped_updates: dict[str, dict[str, float]] = {}
    for (source, intent_id), raw in sorted(raw_by_source_intent.items()):
        cap = float(config.source_log_update_caps.get(source, config.default_source_log_update_cap))
        capped = max(-cap, min(cap, raw))
        capped_by_intent[intent_id] = capped_by_intent.get(intent_id, 0.0) + capped
        capped_updates.setdefault(intent_id, {})[source] = capped

    unnormalized: list[tuple[IntentHypothesis, float]] = []
    for intent in state.intents:
        update = max(-config.max_abs_log_update, min(config.max_abs_log_update, capped_by_intent.get(intent.id, 0.0)))
        unnormalized.append((intent, intent.posterior * math.exp(update)))
    probabilities = _normalize_with_floor([value for _, value in unnormalized], floor=config.prior_floor)
    probabilities = _apply_collapse_guard(probabilities, cap=config.max_single_intent_posterior)

    updated: list[IntentHypothesis] = []
    for (intent, _), posterior in zip(unnormalized, probabilities, strict=False):
        support = support_by_intent.get(intent.id, 0.0)
        contradiction = contradiction_by_intent.get(intent.id, 0.0)
        agreement = abs(support - contradiction)
        conflict = min(support, contradiction)
        uncertainty = intent.uncertainty - 0.04 * agreement + 0.10 * conflict
        if conflict > 0:
            uncertainty = max(uncertainty, intent.uncertainty)
        updated.append(
            IntentHypothesis(
                id=intent.id,
                statement=intent.statement,
                posterior=posterior,
                utility_dimensions=intent.utility_dimensions,
                hard_constraints=intent.hard_constraints,
                representation_refs=intent.representation_refs,
                evaluator_refs=intent.evaluator_refs,
                uncertainty=max(config.min_uncertainty, min(1.0, uncertainty)),
            )
        )

    trace = {
        "update_model_version": config.update_model_version,
        "config_hash": config.config_hash(),
        "input_evidence_count": len(evidence),
        "used_evidence_count": len(evidence) - ignored,
        "ignored_evidence_count": ignored,
        "capped_source_updates": capped_updates,
        "support_by_intent": support_by_intent,
        "contradiction_by_intent": contradiction_by_intent,
        "source_reliability": {
            str(source): float(config.source_reliability.get(str(source), config.source_reliability.get("unknown", 0.35)))
            for source, _intent_id in sorted(raw_by_source_intent)
        },
        "prior_state_hash": state.state_hash(),
    }
    updated_state = LatentProblemState(
        intents=tuple(updated),
        frontier_candidates=state.frontier_candidates,
        actions=state.actions,
        evidence_refs=tuple(dict.fromkeys(used_refs)),
        version=state.version,
    )
    trace["posterior_state_hash"] = updated_state.state_hash()
    return updated_state, trace


def _stale_decay(evidence: PreferenceEvidence, *, config: PosteriorUpdateConfig) -> float:
    metadata = coerce_dict(evidence.metadata)
    if "stale_decay" in metadata:
        try:
            return max(0.0, min(1.0, float(metadata.get("stale_decay"))))
        except (TypeError, ValueError):
            return 1.0
    try:
        age_rounds = max(0.0, float(metadata.get("age_rounds", 0.0) or 0.0))
    except (TypeError, ValueError):
        age_rounds = 0.0
    return max(0.0, min(1.0, float(config.stale_decay_per_round) ** age_rounds))


def _source_reliability(evidence: PreferenceEvidence, *, config: PosteriorUpdateConfig) -> float:
    metadata = coerce_dict(evidence.metadata)
    if "source_reliability" in metadata:
        try:
            return max(0.0, min(1.0, float(metadata.get("source_reliability"))))
        except (TypeError, ValueError):
            pass
    source = str(evidence.source_type or "unknown")
    return max(0.0, min(1.0, float(config.source_reliability.get(source, config.source_reliability.get("unknown", 0.35)))))


def _normalize_with_floor(values: list[float], *, floor: float) -> list[float]:
    if not values:
        return []
    total = sum(max(0.0, value) for value in values)
    if total <= 0:
        probs = [1.0 / len(values) for _ in values]
    else:
        probs = [max(0.0, value) / total for value in values]
    bounded_floor = max(0.0, min(float(floor), 1.0 / max(1, len(values)) * 0.5))
    if bounded_floor <= 0:
        return probs
    floored = [max(bounded_floor, value) for value in probs]
    floored_total = sum(floored)
    return [value / floored_total for value in floored]


def _apply_collapse_guard(probabilities: list[float], *, cap: float) -> list[float]:
    if len(probabilities) <= 1:
        return probabilities
    cap = max(1.0 / len(probabilities), min(1.0, float(cap)))
    max_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    if probabilities[max_index] <= cap:
        return probabilities
    remainder = 1.0 - cap
    other_total = sum(value for index, value in enumerate(probabilities) if index != max_index)
    guarded = list(probabilities)
    guarded[max_index] = cap
    if other_total <= 0:
        share = remainder / (len(probabilities) - 1)
        for index in range(len(guarded)):
            if index != max_index:
                guarded[index] = share
    else:
        for index, value in enumerate(probabilities):
            if index != max_index:
                guarded[index] = remainder * value / other_total
    return guarded


__all__ = [
    "UPDATE_MODEL_VERSION",
    "LatentPosteriorSnapshot",
    "PosteriorUpdateConfig",
    "bounded_update_intent_posteriors",
    "materialize_posterior_snapshot",
]
