"""Multi-shot harvesting loops for seeds, plans, and offspring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from .descriptor_cells import descriptor_cell_key
from .fingerprints import candidate_fingerprint
from .math_model import batch_gain
from .relevance import relevance_score


@dataclass(frozen=True)
class HarvestPolicy:
    target_size: int
    max_batches: int
    min_batches: int = 1
    low_gain_patience: int = 2
    relevance_floor: float = 0.20
    stage: str = "candidate"


@dataclass
class HarvestResult:
    accepted: list[CandidateGenome] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    batches: int = 0
    stopped_reason: str = ""
    model_error: Exception | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_ids": [candidate.id for candidate in self.accepted],
            "rejected": list(self.rejected[-50:]),
            "batches": self.batches,
            "stopped_reason": self.stopped_reason,
            "model_error": f"{self.model_error.__class__.__name__}: {self.model_error}" if self.model_error else "",
        }


class CandidateHarvester:
    def __init__(self, *, deduper: CandidateDeduper | None = None, policy: HarvestPolicy) -> None:
        self.deduper = deduper or CandidateDeduper()
        self.policy = policy

    def harvest(
        self,
        *,
        request_batch: Callable[[int, list[CandidateGenome], list[dict[str, Any]]], Iterable[CandidateGenome]],
        on_error: Callable[[Exception], bool] | None = None,
        context: dict[str, Any] | None = None,
        recoverable_errors: tuple[type[Exception], ...] = (),
    ) -> HarvestResult:
        result = HarvestResult()
        low_gain_streak = 0
        context = dict(context or {})
        for batch_index in range(max(1, self.policy.max_batches)):
            try:
                batch = list(request_batch(batch_index, result.accepted, result.rejected))
            except Exception as exc:  # caller decides which boundary errors are recoverable
                if not recoverable_errors or not isinstance(exc, recoverable_errors):
                    raise
                result.model_error = exc
                if on_error is not None and on_error(exc):
                    raise
                result.stopped_reason = "model_error"
                break
            result.batches += 1
            before = len(result.accepted)
            relevant = 0
            novel = 0
            for candidate in batch:
                rel = relevance_score(candidate, **context)
                candidate.metadata.setdefault("search_kernel_stage", self.policy.stage)
                candidate.metadata["search_kernel_batch"] = batch_index
                candidate.metadata["search_kernel_relevance"] = rel
                candidate.metadata["search_kernel_fingerprint"] = candidate_fingerprint(candidate).to_dict()
                candidate.metadata["descriptor_cell"] = descriptor_cell_key(candidate)
                if rel < self.policy.relevance_floor:
                    result.rejected.append({"batch": batch_index, "candidate_id": candidate.id, "reason": "low_relevance", "relevance": rel})
                    continue
                relevant += 1
                if self.deduper.add(candidate):
                    novel += 1
                    result.accepted.append(candidate)
                else:
                    result.rejected.append({"batch": batch_index, "candidate_id": candidate.id, "reason": "duplicate_semantic_signature", "signature": candidate.metadata.get("dedupe_signature")})
            gain = batch_gain(accepted_count=len(result.accepted) - before, novel_count=novel, batch_size=max(1, len(batch)), relevant_count=relevant)
            if result.batches >= max(1, self.policy.min_batches) and len(result.accepted) >= self.policy.target_size:
                result.stopped_reason = "target_reached"
                break
            if gain <= 0.01:
                low_gain_streak += 1
            else:
                low_gain_streak = 0
            if low_gain_streak >= max(1, self.policy.low_gain_patience):
                result.stopped_reason = "low_gain_patience"
                break
        if not result.stopped_reason:
            result.stopped_reason = "batch_limit"
        return result


def plan_signature(plan: MutationPlan) -> str:
    return stable_hash(
        {
            "operator": plan.operator,
            "parent_ids": sorted(str(item) for item in plan.parent_ids),
            "instruction": str(plan.instruction or "").strip().lower()[:600],
            "rarity_seed": str(plan.rarity_seed or "").strip().lower()[:200],
            "effects": sorted(str(item) for item in plan.expected_gene_effects),
        }
    )


def dedupe_plans(plans: Iterable[MutationPlan]) -> tuple[list[MutationPlan], list[dict[str, Any]]]:
    accepted: list[MutationPlan] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in plans:
        sig = plan_signature(plan)
        plan.metadata.setdefault("search_kernel_plan_signature", sig)
        if sig in seen:
            rejected.append({"reason": "duplicate_plan_signature", "signature": sig, "operator": plan.operator, "parent_ids": list(plan.parent_ids)})
            continue
        seen.add(sig)
        accepted.append(plan)
    return accepted, rejected


__all__ = ["CandidateHarvester", "HarvestPolicy", "HarvestResult", "dedupe_plans", "plan_signature"]
