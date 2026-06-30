"""Cost-aware UCB scheduling for M6 niche lanes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict
from cognitive_evolve_runtime.outcomes.anytime_valid import AnytimeValidSolveCertificate, verify_anytime_valid_certificate

from .niches import normalize_niche_id


@dataclass
class NicheArmStats:
    niche_id: str
    pulls: int = 0
    verified_closures: int = 0
    reward_sum: float = 0.0
    total_cost: float = 0.0
    estimated_cost: float = 1.0
    rejected_rewards: int = 0
    last_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.niche_id = normalize_niche_id(self.niche_id)
        self.pulls = max(0, int(self.pulls or 0))
        self.verified_closures = max(0, int(self.verified_closures or 0))
        self.reward_sum = max(0.0, _finite_float(self.reward_sum))
        self.total_cost = max(0.0, _finite_float(self.total_cost))
        self.estimated_cost = max(1e-9, _finite_float(self.estimated_cost, default=1.0))
        self.rejected_rewards = max(0, int(self.rejected_rewards or 0))
        self.last_reward = max(0.0, _finite_float(self.last_reward))
        self.metadata = coerce_dict(self.metadata)

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0

    @property
    def mean_verified_reward(self) -> float:
        return self.reward_sum / self.verified_closures if self.verified_closures else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.pulls if self.pulls and self.total_cost > 0 else self.estimated_cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NicheArmStats":
        return cls(
            niche_id=str(data.get("niche_id") or ""),
            pulls=int(data.get("pulls") or 0),
            verified_closures=int(data.get("verified_closures") or 0),
            reward_sum=float(data.get("reward_sum") or 0.0),
            total_cost=float(data.get("total_cost") or 0.0),
            estimated_cost=float(data.get("estimated_cost") or 1.0),
            rejected_rewards=int(data.get("rejected_rewards") or 0),
            last_reward=float(data.get("last_reward") or 0.0),
            metadata=coerce_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class NicheScheduleDecision:
    niche_id: str
    score: float
    ucb_value: float
    mean_reward: float
    cost: float
    exploration_bonus: float
    pulls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CostAwareUCBScheduler:
    """Choose the next niche by UCB value per unit runtime cost.

    Only verified closure can contribute positive reward.  Failed, stale, or
    self-claimed closure still consumes pulls and cost, but leaves reward at
    zero so unverified lanes do not become scheduler attractors.
    """

    def __init__(self, arms: dict[str, NicheArmStats | dict[str, Any]] | None = None, *, exploration: float = 1.0, min_cost: float = 1e-9) -> None:
        self.exploration = max(0.0, _finite_float(exploration, default=1.0))
        self.min_cost = max(1e-12, _finite_float(min_cost, default=1e-9))
        self.arms: dict[str, NicheArmStats] = {}
        for niche_id, stats in (arms or {}).items():
            if isinstance(stats, NicheArmStats):
                arm = stats
            else:
                payload = coerce_dict(stats)
                payload.setdefault("niche_id", niche_id)
                arm = NicheArmStats.from_dict(payload)
            self.arms[normalize_niche_id(niche_id or arm.niche_id)] = arm

    def register_niche(self, niche_id: str, *, estimated_cost: float = 1.0, metadata: dict[str, Any] | None = None) -> NicheArmStats:
        niche = normalize_niche_id(niche_id)
        if niche not in self.arms:
            self.arms[niche] = NicheArmStats(niche_id=niche, estimated_cost=estimated_cost, metadata=coerce_dict(metadata))
        else:
            self.arms[niche].estimated_cost = max(self.min_cost, _finite_float(estimated_cost, default=self.arms[niche].estimated_cost))
            if metadata:
                self.arms[niche].metadata.update(coerce_dict(metadata))
        return self.arms[niche]

    register = register_niche

    def select(self) -> NicheScheduleDecision:
        if not self.arms:
            raise ValueError("cannot schedule without registered niches")
        decisions = [self.score(niche_id) for niche_id in sorted(self.arms)]
        return max(decisions, key=lambda decision: (decision.score, -decision.cost, decision.niche_id))

    def select_niche(self) -> str:
        return self.select().niche_id

    def score(self, niche_id: str) -> NicheScheduleDecision:
        niche = normalize_niche_id(niche_id)
        arm = self.arms[niche]
        total_pulls = max(1, sum(item.pulls for item in self.arms.values()) + len(self.arms))
        effective_pulls = max(1, arm.pulls)
        exploration_bonus = self.exploration * math.sqrt(2.0 * math.log(total_pulls + 1.0) / effective_pulls)
        mean_reward = arm.mean_reward
        ucb_value = mean_reward + exploration_bonus
        cost = max(self.min_cost, arm.mean_cost)
        return NicheScheduleDecision(
            niche_id=niche,
            score=ucb_value / cost,
            ucb_value=ucb_value,
            mean_reward=mean_reward,
            cost=cost,
            exploration_bonus=exploration_bonus,
            pulls=arm.pulls,
        )

    def record_trial(
        self,
        niche_id: str,
        *,
        closure_certificate: Any = None,
        candidate: CandidateGenome | None = None,
        reward: float = 1.0,
        cost: float = 1.0,
        verified: bool | None = None,
    ) -> bool:
        arm = self.register_niche(niche_id, estimated_cost=max(self.min_cost, _finite_float(cost, default=1.0)))
        verified = bool(verified) if verified is not None else closure_verified(closure_certificate if closure_certificate is not None else candidate)
        arm.pulls += 1
        arm.total_cost += max(self.min_cost, _finite_float(cost, default=1.0))
        if verified:
            bounded = _bounded_reward(reward)
            arm.verified_closures += 1
            arm.reward_sum += bounded
            arm.last_reward = bounded
        else:
            arm.rejected_rewards += 1
            arm.last_reward = 0.0
        return verified

    def update_reward(self, niche_id: str, *, closure_certificate: Any = None, reward: float = 1.0, cost: float = 1.0, verified: bool | None = None) -> bool:
        return self.record_trial(niche_id, closure_certificate=closure_certificate, reward=reward, cost=cost, verified=verified)

    record_result = update_reward
    record_closure = update_reward

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "m6-cost-aware-ucb-scheduler/v1",
            "exploration": self.exploration,
            "arms": {niche_id: arm.to_dict() for niche_id, arm in self.arms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostAwareUCBScheduler":
        return cls(arms=coerce_dict(data.get("arms")), exploration=float(data.get("exploration") or 1.0))


def closure_verified(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, AnytimeValidSolveCertificate):
        return verify_anytime_valid_certificate(raw)
    if isinstance(raw, CandidateGenome):
        return closure_verified(raw.verification_result) or closure_verified(raw.metadata)
    if hasattr(raw, "verified"):
        return bool(getattr(raw, "verified")) and not tuple(getattr(raw, "critical_failures", ()) or ())
    data = coerce_dict(raw)
    if not data:
        return False
    for nested_key in (
        "closure_certificate",
        "anytime_valid_certificate",
        "m6_solve_certificate",
        "solve_certificate",
        "improvement_certificate",
        "verification_result",
        "metadata",
    ):
        if nested_key in data and closure_verified(data.get(nested_key)):
            return True
    failures = tuple(str(item) for item in data.get("critical_failures", ()) if str(item))
    if failures:
        return False
    if bool(data.get("verified")):
        return True
    if str(data.get("status") or "").lower() == "verified":
        return True
    if bool(data.get("objective_solved")) and bool(data.get("verified_closure") or data.get("verified_solution") or data.get("improvement_verified")):
        return True
    return False


def verified_closure_reward(raw: Any, reward: float = 1.0) -> float:
    return _bounded_reward(reward) if closure_verified(raw) else 0.0


def _bounded_reward(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value)))


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


NicheScheduler = CostAwareUCBScheduler


__all__ = [
    "CostAwareUCBScheduler",
    "NicheArmStats",
    "NicheScheduleDecision",
    "NicheScheduler",
    "closure_verified",
    "verified_closure_reward",
]
