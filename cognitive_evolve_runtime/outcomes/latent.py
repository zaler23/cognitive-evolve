"""Latent problem-space primitives for ambiguous or open-ended evolution.

M5 proves that a challenger improved over a baseline under a frozen contract.
M5.1 learns which contract is worth freezing when the goal, representation, and
search space are still ambiguous.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.outcomes.improvement import OutcomeContract, OutcomeMetric


@dataclass(frozen=True)
class IntentHypothesis:
    """A possible interpretation of what "better" means."""

    id: str
    statement: str
    posterior: float = 1.0
    utility_dimensions: tuple[str, ...] = ("quality",)
    hard_constraints: tuple[str, ...] = ()
    representation_refs: tuple[str, ...] = ()
    evaluator_refs: tuple[str, ...] = ()
    uncertainty: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "posterior", max(0.0, float(self.posterior)))
        object.__setattr__(self, "uncertainty", min(1.0, max(0.0, float(self.uncertainty))))
        object.__setattr__(self, "utility_dimensions", tuple(str(item) for item in self.utility_dimensions if str(item)) or ("quality",))
        object.__setattr__(self, "hard_constraints", tuple(str(item) for item in self.hard_constraints if str(item)))
        object.__setattr__(self, "representation_refs", tuple(str(item) for item in self.representation_refs if str(item)))
        object.__setattr__(self, "evaluator_refs", tuple(str(item) for item in self.evaluator_refs if str(item)))

    def with_posterior(self, posterior: float) -> "IntentHypothesis":
        return IntentHypothesis(
            id=self.id,
            statement=self.statement,
            posterior=posterior,
            utility_dimensions=self.utility_dimensions,
            hard_constraints=self.hard_constraints,
            representation_refs=self.representation_refs,
            evaluator_refs=self.evaluator_refs,
            uncertainty=self.uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceEvidence:
    """Evidence that supports or weakens one intent interpretation."""

    intent_id: str
    support: float = 0.0
    contradiction: float = 0.0
    weight: float = 1.0
    evidence_ref: str = ""
    source_type: str = "unknown"
    provenance_ref: str = ""
    confidence: float = 1.0
    calibration: str = "uncalibrated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "support", max(0.0, float(self.support)))
        object.__setattr__(self, "contradiction", max(0.0, float(self.contradiction)))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "source_type", str(self.source_type or "unknown"))
        object.__setattr__(self, "provenance_ref", str(self.provenance_ref or ""))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "calibration", str(self.calibration or "uncalibrated"))
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    def log_update(self) -> float:
        return self.weight * self.confidence * (self.support - self.contradiction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierCandidate:
    """Candidate scored against multiple still-uncertain intents."""

    candidate_id: str
    utility_by_intent: dict[str, float]
    uncertainty_by_intent: dict[str, float] = field(default_factory=dict)
    novelty: float = 0.0
    risk: float = 0.0
    cost: float = 0.0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "utility_by_intent", _coerce_float_dict(self.utility_by_intent))
        object.__setattr__(self, "uncertainty_by_intent", _coerce_nonnegative_float_dict(self.uncertainty_by_intent))
        object.__setattr__(self, "novelty", max(0.0, float(self.novelty)))
        object.__setattr__(self, "risk", max(0.0, float(self.risk)))
        object.__setattr__(self, "cost", max(0.0, float(self.cost)))
        object.__setattr__(self, "evidence_refs", tuple(str(item) for item in self.evidence_refs if str(item)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float
    expected_utility: float
    uncertainty_penalty: float
    risk_penalty: float
    cost_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplorationAction:
    """A possible next step for learning the space or improving a candidate."""

    action_id: str
    kind: str
    target_intent_ids: tuple[str, ...] = ()
    expected_improvement: float = 0.0
    information_gain: float = 0.0
    diversity_gain: float = 0.0
    risk: float = 0.0
    cost: float = 0.0
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_intent_ids", tuple(str(item) for item in self.target_intent_ids if str(item)))
        for field_name in ["expected_improvement", "information_gain", "diversity_gain", "risk", "cost"]:
            object.__setattr__(self, field_name, max(0.0, float(getattr(self, field_name))))

    def acquisition_score(
        self,
        *,
        posterior_entropy: float = 0.0,
        beta: float = 1.0,
        gamma: float = 0.3,
        risk_weight: float = 1.0,
        cost_weight: float = 1.0,
    ) -> float:
        ambiguity_bonus = 1.0 + max(0.0, posterior_entropy)
        return (
            self.expected_improvement
            + beta * ambiguity_bonus * self.information_gain
            + gamma * self.diversity_gain
            - risk_weight * self.risk
            - cost_weight * self.cost
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatentProblemState:
    """Posterior state over possible objectives, candidates, and next probes."""

    intents: tuple[IntentHypothesis, ...]
    frontier_candidates: tuple[FrontierCandidate, ...] = ()
    actions: tuple[ExplorationAction, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    version: str = "latent-problem-state/v1"

    def __post_init__(self) -> None:
        if not self.intents:
            raise ValueError("latent problem state requires at least one intent hypothesis")
        normalized = _normalize_intents(self.intents)
        object.__setattr__(self, "intents", normalized)
        object.__setattr__(self, "frontier_candidates", tuple(self.frontier_candidates))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "evidence_refs", tuple(str(item) for item in self.evidence_refs if str(item)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intents": [intent.to_dict() for intent in self.intents],
            "frontier_candidates": [candidate.to_dict() for candidate in self.frontier_candidates],
            "actions": [action.to_dict() for action in self.actions],
            "evidence_refs": list(self.evidence_refs),
        }

    def state_hash(self) -> str:
        return stable_hash(self.to_dict())

    def top_intent(self) -> IntentHypothesis:
        return max(self.intents, key=lambda item: (item.posterior, -item.uncertainty, item.id))

    def posterior_entropy(self) -> float:
        if len(self.intents) <= 1:
            return 0.0
        entropy = -sum(intent.posterior * math.log(intent.posterior) for intent in self.intents if intent.posterior > 0)
        return entropy / math.log(len(self.intents))


@dataclass(frozen=True)
class ConvergenceAssessment:
    converged: bool
    selected_candidate_id: str
    reason_codes: tuple[str, ...]
    posterior_entropy: float
    best_action_score: float
    frontier_size: int
    improvement_certificate_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def update_intent_posteriors(state: LatentProblemState, evidence: list[PreferenceEvidence]) -> LatentProblemState:
    """Bayesian-style posterior update over latent intents."""

    updates: dict[str, float] = {}
    evidence_refs = list(state.evidence_refs)
    for item in evidence:
        updates[item.intent_id] = updates.get(item.intent_id, 0.0) + item.log_update()
        if item.evidence_ref:
            evidence_refs.append(item.evidence_ref)

    updated: list[IntentHypothesis] = []
    for intent in state.intents:
        multiplier = math.exp(max(-20.0, min(20.0, updates.get(intent.id, 0.0))))
        posterior = intent.posterior * multiplier
        uncertainty = max(0.0, intent.uncertainty - 0.05 * abs(updates.get(intent.id, 0.0)))
        updated.append(
            IntentHypothesis(
                id=intent.id,
                statement=intent.statement,
                posterior=posterior,
                utility_dimensions=intent.utility_dimensions,
                hard_constraints=intent.hard_constraints,
                representation_refs=intent.representation_refs,
                evaluator_refs=intent.evaluator_refs,
                uncertainty=uncertainty,
            )
        )
    return LatentProblemState(
        intents=tuple(updated),
        frontier_candidates=state.frontier_candidates,
        actions=state.actions,
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
    )


def rank_candidates(
    state: LatentProblemState,
    *,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.1,
    risk_weight: float = 0.5,
    cost_weight: float = 0.2,
) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for candidate in state.frontier_candidates:
        expected_utility = 0.0
        uncertainty_penalty = 0.0
        for intent in state.intents:
            expected_utility += intent.posterior * candidate.utility_by_intent.get(intent.id, 0.0)
            uncertainty_penalty += intent.posterior * candidate.uncertainty_by_intent.get(intent.id, intent.uncertainty)
        risk_penalty = risk_weight * candidate.risk
        cost_penalty = cost_weight * candidate.cost
        score = expected_utility - uncertainty_weight * uncertainty_penalty + novelty_weight * candidate.novelty - risk_penalty - cost_penalty
        scores.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                score=score,
                expected_utility=expected_utility,
                uncertainty_penalty=uncertainty_weight * uncertainty_penalty,
                risk_penalty=risk_penalty,
                cost_penalty=cost_penalty,
            )
        )
    return sorted(scores, key=lambda item: (item.score, item.expected_utility, item.candidate_id), reverse=True)


def pareto_frontier(state: LatentProblemState) -> tuple[FrontierCandidate, ...]:
    """Keep candidates that are not dominated across current latent intents, risk, and cost."""

    frontier: list[FrontierCandidate] = []
    for candidate in state.frontier_candidates:
        if not any(_dominates(other, candidate, state.intents) for other in state.frontier_candidates if other is not candidate):
            frontier.append(candidate)
    return tuple(frontier)


def select_exploration_action(
    state: LatentProblemState,
    *,
    beta: float = 1.0,
    gamma: float = 0.3,
    risk_weight: float = 1.0,
    cost_weight: float = 1.0,
) -> ExplorationAction | None:
    if not state.actions:
        return None
    entropy = state.posterior_entropy()
    known_intents = {intent.id for intent in state.intents}
    admissible = [
        action
        for action in state.actions
        if not action.target_intent_ids or all(intent_id in known_intents for intent_id in action.target_intent_ids)
    ]
    if not admissible:
        return None
    return max(
        admissible,
        key=lambda action: (
            action.acquisition_score(
                posterior_entropy=entropy,
                beta=beta,
                gamma=gamma,
                risk_weight=risk_weight,
                cost_weight=cost_weight,
            ),
            action.information_gain,
            action.action_id,
        ),
    )


def freeze_outcome_contract(state: LatentProblemState, *, intent_id: str | None = None, min_effect: float = 0.0) -> OutcomeContract:
    """Freeze the current best intent into an M5 contract for local proof."""

    intent = next((item for item in state.intents if item.id == intent_id), None) if intent_id else state.top_intent()
    if intent is None:
        raise ValueError(f"unknown intent hypothesis: {intent_id}")
    weight = 1.0 / max(1, len(intent.utility_dimensions))
    metrics = tuple(OutcomeMetric(id=dimension, weight=weight, direction="maximize") for dimension in intent.utility_dimensions)
    return OutcomeContract(
        objective=intent.statement,
        scope=f"latent-intent:{intent.id}",
        metrics=metrics,
        min_effect=min_effect,
        hard_constraints=intent.hard_constraints,
    )


def assess_convergence(
    state: LatentProblemState,
    *,
    improvement_certificate: Any | None = None,
    min_top_posterior: float = 0.75,
    max_entropy: float = 0.25,
    max_action_score: float = 0.05,
) -> ConvergenceAssessment:
    ranked = rank_candidates(state)
    best_candidate_id = ranked[0].candidate_id if ranked else ""
    best_action = select_exploration_action(state)
    entropy = state.posterior_entropy()
    best_action_score = best_action.acquisition_score(posterior_entropy=entropy) if best_action else 0.0
    certificate_verified = bool(getattr(improvement_certificate, "verified", False))
    top_intent = state.top_intent()

    reasons: list[str] = []
    if top_intent.posterior < min_top_posterior:
        reasons.append("intent_posterior_not_concentrated")
    if entropy > max_entropy:
        reasons.append("latent_intent_entropy_high")
    if best_action_score > max_action_score:
        reasons.append("valuable_exploration_remains")
    if not certificate_verified:
        reasons.append("missing_verified_improvement_certificate")
    if not best_candidate_id:
        reasons.append("empty_frontier")

    return ConvergenceAssessment(
        converged=not reasons,
        selected_candidate_id=best_candidate_id,
        reason_codes=tuple(reasons),
        posterior_entropy=entropy,
        best_action_score=best_action_score,
        frontier_size=len(pareto_frontier(state)),
        improvement_certificate_verified=certificate_verified,
    )


def _normalize_intents(intents: tuple[IntentHypothesis, ...]) -> tuple[IntentHypothesis, ...]:
    total = sum(intent.posterior for intent in intents)
    if total <= 0:
        weight = 1.0 / len(intents)
        return tuple(intent.with_posterior(weight) for intent in intents)
    return tuple(intent.with_posterior(intent.posterior / total) for intent in intents)


def _dominates(a: FrontierCandidate, b: FrontierCandidate, intents: tuple[IntentHypothesis, ...]) -> bool:
    better_or_equal = True
    strictly_better = False
    for intent in intents:
        a_value = a.utility_by_intent.get(intent.id, 0.0) - a.uncertainty_by_intent.get(intent.id, intent.uncertainty)
        b_value = b.utility_by_intent.get(intent.id, 0.0) - b.uncertainty_by_intent.get(intent.id, intent.uncertainty)
        if a_value < b_value:
            better_or_equal = False
            break
        if a_value > b_value:
            strictly_better = True
    if a.risk > b.risk or a.cost > b.cost:
        return False
    if a.risk < b.risk or a.cost < b.cost:
        strictly_better = True
    return better_or_equal and strictly_better


def _coerce_float_dict(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in coerce_dict(value).items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _coerce_nonnegative_float_dict(value: Any) -> dict[str, float]:
    return {key: max(0.0, val) for key, val in _coerce_float_dict(value).items()}


__all__ = [
    "CandidateScore",
    "ConvergenceAssessment",
    "ExplorationAction",
    "FrontierCandidate",
    "IntentHypothesis",
    "LatentProblemState",
    "PreferenceEvidence",
    "assess_convergence",
    "freeze_outcome_contract",
    "pareto_frontier",
    "rank_candidates",
    "select_exploration_action",
    "update_intent_posteriors",
]
