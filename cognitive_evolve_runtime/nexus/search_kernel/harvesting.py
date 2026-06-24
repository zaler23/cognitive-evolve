"""Multi-shot harvesting loops for seeds, plans, and offspring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.llm.fanout import model_fanout_workers, run_ordered_fanout
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.nextgen import ensure_nextgen_identity, record_candidate_budget_decision
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from .descriptor_cells import descriptor_cell_key
from .fingerprints import candidate_fingerprint
from .math_model import batch_gain
from .relevance import relevance_score


@dataclass(frozen=True)
class HarvestPolicy:
    target_size: int
    max_batches: int | None
    min_batches: int = 1
    low_gain_patience: int = 2
    relevance_floor: float = 0.20
    stage: str = "candidate"
    fanout_workers: int | None = None
    stop_at_target: bool = True
    exhaust_on_no_new: bool = False
    reservoir_mode: bool = False
    reservoir_limit: int = 256


@dataclass
class HarvestResult:
    accepted: list[CandidateGenome] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    batches: int = 0
    stopped_reason: str = ""
    fatal_model_error: Exception | None = None
    recoverable_batch_errors: list[dict[str, Any]] = field(default_factory=list)
    failed_batch_ids: list[int] = field(default_factory=list)
    reservoir: list[CandidateGenome] = field(default_factory=list)
    reservoir_truncated_count: int = 0
    reservoir_truncated_summaries: list[dict[str, Any]] = field(default_factory=list)
    stage: str = ""

    @property
    def model_error(self) -> Exception | None:
        """Backward-compatible fatal-error alias."""

        return self.fatal_model_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_ids": [candidate.id for candidate in self.accepted],
            "accepted_count": len(self.accepted),
            "rejected": list(self.rejected[-50:]),
            "rejected_count": len(self.rejected),
            "batches": self.batches,
            "stopped_reason": self.stopped_reason,
            "stage": self.stage,
            "model_error": f"{self.fatal_model_error.__class__.__name__}: {self.fatal_model_error}" if self.fatal_model_error else "",
            "fatal_model_error": f"{self.fatal_model_error.__class__.__name__}: {self.fatal_model_error}" if self.fatal_model_error else "",
            "recoverable_batch_errors": list(self.recoverable_batch_errors[-50:]),
            "failed_batch_ids": list(self.failed_batch_ids),
            "reservoir_ids": [candidate.id for candidate in self.reservoir[-100:]],
            "reservoir_count": len(self.reservoir),
            "reservoir_truncated_count": self.reservoir_truncated_count,
            "reservoir_truncated_summaries": list(self.reservoir_truncated_summaries[-50:]),
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
        result = HarvestResult(stage=self.policy.stage)
        low_gain_streak = 0
        context = dict(context or {})
        max_batches = None if self.policy.max_batches is None else max(1, int(self.policy.max_batches))
        workers = _harvest_workers(max_batches=max_batches, configured=self.policy.fanout_workers)
        batch_index = 0
        while (max_batches is None or batch_index < max_batches) and not result.stopped_reason:
            window_end = batch_index + workers if max_batches is None else min(max_batches, batch_index + workers)
            window = list(range(batch_index, window_end))
            accepted_snapshot = list(result.accepted)
            rejected_snapshot = list(result.rejected)

            def _request(index: int) -> tuple[int, list[CandidateGenome], Exception | None]:
                try:
                    return index, list(request_batch(index, accepted_snapshot, rejected_snapshot)), None
                except Exception as exc:  # caller decides which boundary errors are recoverable
                    if not recoverable_errors or not isinstance(exc, recoverable_errors):
                        raise
                    return index, [], exc

            successful_batches_in_window = 0
            first_recoverable_error: Exception | None = None
            for current_index, batch, error in run_ordered_fanout(window, _request, max_workers=workers):
                result.batches += 1
                if error is not None:
                    if on_error is not None and on_error(error):
                        result.fatal_model_error = error
                        raise error
                    first_recoverable_error = first_recoverable_error or error
                    result.failed_batch_ids.append(int(current_index))
                    error_record = {
                        "batch": int(current_index),
                        "reason": "recoverable_model_error",
                        "error_type": error.__class__.__name__,
                        "error": str(error),
                    }
                    result.recoverable_batch_errors.append(error_record)
                    result.rejected.append(error_record)
                    continue
                successful_batches_in_window += 1
                families_before = _harvest_family_set(result.accepted)
                accepted_before = len(result.accepted)
                gain = self._apply_batch(result, batch_index=current_index, batch=batch, context=context)
                accepted_delta = len(result.accepted) - accepted_before
                if self.policy.stop_at_target and result.batches >= max(1, self.policy.min_batches) and len(result.accepted) >= self.policy.target_size:
                    result.stopped_reason = "target_reached"
                    break
                if self.policy.exhaust_on_no_new and max_batches is None and self.policy.stage == "seed":
                    families_after = _harvest_family_set(result.accepted)
                    exhausted_batch = _unbounded_seed_handoff_exhausted(
                        accepted_before=accepted_before,
                        accepted_delta=accepted_delta,
                        family_count_before=len(families_before),
                        family_delta=len(families_after - families_before),
                        accepted_count=len(result.accepted),
                        family_count=len(families_after),
                        batches=result.batches,
                        target_size=self.policy.target_size,
                        min_batches=self.policy.min_batches,
                        low_gain_patience=self.policy.low_gain_patience,
                    )
                else:
                    exhausted_batch = accepted_delta <= 0 if self.policy.exhaust_on_no_new else gain <= 0.01
                if exhausted_batch:
                    low_gain_streak += 1
                else:
                    low_gain_streak = 0
                if low_gain_streak >= max(1, self.policy.low_gain_patience):
                    result.stopped_reason = "low_gain_patience"
                    break
            if successful_batches_in_window <= 0 and first_recoverable_error is not None and not result.stopped_reason:
                result.fatal_model_error = first_recoverable_error
                result.stopped_reason = "model_error"
            batch_index = window[-1] + 1
        if not result.stopped_reason:
            result.stopped_reason = "batch_limit"
        return result

    def _apply_batch(
        self,
        result: HarvestResult,
        *,
        batch_index: int,
        batch: list[CandidateGenome],
        context: dict[str, Any],
    ) -> float:
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
            ensure_nextgen_identity(candidate)
            if rel < self.policy.relevance_floor:
                record = {"batch": batch_index, "candidate_id": candidate.id, "reason": "low_relevance", "relevance": rel}
                result.rejected.append(record)
                record_candidate_budget_decision(candidate, source=f"{self.policy.stage}_harvester", reason="low_relevance", action="soft_reservoir" if self.policy.reservoir_mode else "soft_reject", details=record)
                if self.policy.reservoir_mode:
                    candidate.metadata.setdefault("seed_reservoir_reason", "low_relevance")
                    self._store_reservoir(result, candidate, reason="low_relevance")
                continue
            relevant += 1
            if self.deduper.add(candidate):
                novel += 1
                result.accepted.append(candidate)
            else:
                record = {"batch": batch_index, "candidate_id": candidate.id, "reason": "duplicate_semantic_signature", "signature": candidate.metadata.get("dedupe_signature")}
                result.rejected.append(record)
                record_candidate_budget_decision(candidate, source=f"{self.policy.stage}_harvester", reason="duplicate_semantic_signature", action="soft_reservoir" if self.policy.reservoir_mode else "soft_reject", details=record)
                if self.policy.reservoir_mode:
                    candidate.metadata.setdefault("seed_reservoir_reason", "duplicate_semantic_signature")
                    self._store_reservoir(result, candidate, reason="duplicate_semantic_signature")
        return batch_gain(accepted_count=len(result.accepted) - before, novel_count=novel, batch_size=max(1, len(batch)), relevant_count=relevant)

    def _store_reservoir(self, result: HarvestResult, candidate: CandidateGenome, *, reason: str) -> None:
        limit = max(0, int(self.policy.reservoir_limit or 0))
        if len(result.reservoir) < limit:
            result.reservoir.append(candidate)
            return
        result.reservoir_truncated_count += 1
        summary = {
            "candidate_id": candidate.id,
            "reason": reason,
            "concise_claim": str(candidate.concise_claim or candidate.core_mechanism or candidate.artifact or "")[:240],
        }
        result.reservoir_truncated_summaries.append(summary)
        del result.reservoir_truncated_summaries[:-50]
        candidate.metadata["seed_reservoir_truncated"] = summary


def _harvest_workers(*, max_batches: int | None, configured: int | None) -> int:
    if configured is None:
        return model_fanout_workers(max_batches)
    workers = max(1, int(configured or 1))
    return workers if max_batches is None else min(workers, max(1, int(max_batches or 1)))


def _unbounded_seed_handoff_exhausted(
    *,
    accepted_before: int,
    accepted_delta: int,
    family_count_before: int,
    family_delta: int,
    accepted_count: int,
    family_count: int,
    batches: int,
    target_size: int,
    min_batches: int,
    low_gain_patience: int,
) -> bool:
    if accepted_delta <= 0:
        return True
    if batches < max(1, int(min_batches or 1)):
        return False
    # low_gain_patience is applied by the caller's consecutive-streak counter;
    # multiplying it into this floor double-counts patience and delays handoff.
    handoff_floor = max(1, int(target_size or 1)) * max(1, int(min_batches or 1))
    if accepted_count < handoff_floor:
        return False
    if not _seed_pool_broad_enough(accepted_count=accepted_count, family_count=family_count, target_size=target_size):
        return False
    if accepted_before <= 0 or family_count_before <= 0:
        return False
    family_growth_rate = family_delta / max(1, family_count_before)
    candidate_growth_rate = accepted_delta / max(1, accepted_before)
    return family_growth_rate < candidate_growth_rate


def _seed_pool_broad_enough(*, accepted_count: int, family_count: int, target_size: int) -> bool:
    family_floor = max(3, min(12, max(1, int(target_size or 1)) // 8 or 1))
    return int(accepted_count or 0) > 0 and int(family_count or 0) >= family_floor


def _harvest_family_set(candidates: Iterable[CandidateGenome]) -> set[str]:
    return {key for key in (_harvest_family_key(candidate) for candidate in candidates or []) if key}


def _harvest_family_key(candidate: CandidateGenome) -> str:
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
            return text[:160]
    return ""


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
            plan.metadata["candidate_budget_decision"] = {
                "source": "dedupe_plans",
                "reason": "duplicate_plan_signature",
                "action": "soft_retain",
                "hard_gate": False,
            }
            rejected.append({"reason": "duplicate_plan_signature", "signature": sig, "operator": plan.operator, "parent_ids": list(plan.parent_ids), "action": "soft_retain"})
            accepted.append(plan)
            continue
        seen.add(sig)
        accepted.append(plan)
    return accepted, rejected


__all__ = ["CandidateHarvester", "HarvestPolicy", "HarvestResult", "dedupe_plans", "plan_signature"]
