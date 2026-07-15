"""Post-evaluation credit and productive branch allocation.

The allocator deliberately uses only runtime-observed outcomes for positive
credit.  Text novelty and model-authored evidence remain useful search hints,
but cannot make a lineage productive by themselves.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import evaluator_selection_key
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.receipts import transfer_credit_for_candidate
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import (
    base_mechanism_family,
    candidate_phenotype_signature,
)
from cognitive_evolve_runtime.theory.bandit import OperatorArmStats, suggest_budget_allocation


@dataclass(frozen=True)
class ProductiveOutcome:
    candidate_id: str
    arm_id: str
    outcome_cell: str
    reward: float
    risk: float
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reason_codes"] = list(self.reason_codes)
        return data


@dataclass(frozen=True)
class BranchSlot:
    slot_id: str
    arm_id: str
    parent_id: str
    intent: str
    variation_index: int
    ucb_score: float
    coverage_bonus: float = 0.0
    coverage_scale: float = 0.0
    allocation_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductiveBranchAllocation:
    slots: tuple[BranchSlot, ...]
    arms: tuple[OperatorArmStats, ...]
    credit_summary: dict[str, int]
    observed_family_counts: dict[str, int] = field(default_factory=dict)
    credited_transfer_artifact_hashes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "post_evaluation_productive_branch_allocation_v1",
            "arm_identity": "lineage_root",
            "slots": [slot.to_dict() for slot in self.slots],
            "arms": [
                {
                    **arm.to_dict(),
                    "mean_reward": round(arm.mean_reward, 6),
                    "mean_risk": round(arm.mean_risk, 6),
                }
                for arm in self.arms
            ],
            "credit_summary": dict(self.credit_summary),
            "observed_family_counts": dict(self.observed_family_counts),
            "credited_transfer_artifact_hashes": list(self.credited_transfer_artifact_hashes),
        }


def lineage_root(candidate: CandidateGenome) -> str:
    return str(candidate.lineage[0] if candidate.lineage else candidate.id)


def observed_outcome_cell(candidate: CandidateGenome) -> str:
    """Return a grounded runtime-state cell, excluding evaluator metrics."""

    metadata = coerce_dict(candidate.metadata)
    evaluator = coerce_dict(metadata.get("evaluator"))
    verification = _verification_state(candidate)
    patch = _patch_state(candidate)
    payload = {
        "evaluator_status": _evaluator_state(evaluator),
        "evaluator_passed": evaluator.get("passed") if isinstance(evaluator.get("passed"), bool) else None,
        "verification": verification,
        "patch": patch,
    }
    if not evaluator and verification == "not_run" and patch == "not_run":
        return ""
    return "outcome:" + stable_hash(payload)[:20]


def productive_outcomes(
    candidates: Iterable[CandidateGenome],
    *,
    metric_directions: dict[str, str] | None = None,
    credited_transfer_artifact_hashes: Iterable[str] = (),
) -> tuple[ProductiveOutcome, ...]:
    """Score candidates in causal order using post-evaluation observations."""

    selection_binding = _binding_from_metric_directions(metric_directions)
    ordered = sorted(_dedupe_candidates(candidates), key=_candidate_order)
    prior: list[CandidateGenome] = []
    seen_phenotypes: set[str] = set()
    seen_cells: set[str] = set()
    seen_resolved: set[str] = set()
    credited_transfer_hashes = {str(item) for item in credited_transfer_artifact_hashes if str(item)}
    outcomes: list[ProductiveOutcome] = []
    start = 0
    while start < len(ordered):
        cohort_key = _candidate_cohort(ordered[start])
        end = start + 1
        while end < len(ordered) and _candidate_cohort(ordered[end]) == cohort_key:
            end += 1
        cohort = ordered[start:end]
        cohort_phenotypes = [candidate_phenotype_signature(candidate) for candidate in cohort]
        cohort_unbound: list[bool] = []
        for candidate in cohort:
            metadata = coerce_dict(candidate.metadata)
            binding = str(metadata.get("branch_slot_binding_status") or "").strip()
            cohort_unbound.append(bool(binding and binding != "bound"))
        phenotype_counts: dict[str, int] = {}
        for value, unbound in zip(cohort_phenotypes, cohort_unbound):
            if value and not unbound:
                phenotype_counts[value] = phenotype_counts.get(value, 0) + 1
        facts: list[dict[str, Any]] = []
        for candidate, phenotype, unbound in zip(cohort, cohort_phenotypes, cohort_unbound):
            verification = _verification_state(candidate)
            terminal = _terminal(candidate)
            facts.append(
                {
                    "candidate": candidate,
                    "duplicate": bool(
                        not unbound
                        and phenotype
                        and (phenotype in seen_phenotypes or phenotype_counts.get(phenotype, 0) > 1)
                    ),
                    "cell": observed_outcome_cell(candidate),
                    "resolved": _resolved_challenges(candidate),
                    "verification": verification,
                    "terminal": terminal,
                    "unbound": unbound,
                    "observed_failure": _observed_failure(candidate),
                    "patch_verified": _patch_state(candidate) == "applied" and verification == "passed",
                    "transfer_credit": None,
                }
            )
        for fact in facts:
            if (
                not fact["duplicate"]
                and not fact["terminal"]
                and not fact["unbound"]
                and not fact["observed_failure"]
                and fact["verification"] != "failed"
            ):
                fact["transfer_credit"] = transfer_credit_for_candidate(
                    fact["candidate"],
                    credited_artifact_hashes=credited_transfer_hashes,
                )
        eligible = [
            index
            for index, fact in enumerate(facts)
            if not fact["duplicate"]
            and not fact["terminal"]
            and not fact["unbound"]
            and not fact["observed_failure"]
            and fact["verification"] != "failed"
            and (fact["transfer_credit"] is None or fact["transfer_credit"].productive_credit)
        ]
        eligible_set = set(eligible)
        new_cell_groups: dict[str, list[int]] = {}
        for index in eligible:
            cell = str(facts[index]["cell"] or "")
            if cell and cell not in seen_cells:
                new_cell_groups.setdefault(cell, []).append(index)
        new_cell_rewards = {
            index: 1.0 / len(indices)
            for indices in new_cell_groups.values()
            for index in indices
        }
        improvement_groups: dict[str, list[int]] = {}
        for index in eligible:
            if index in new_cell_rewards:
                continue
            cell = str(facts[index]["cell"] or "")
            candidate = facts[index]["candidate"]
            if cell and _same_cell_elite_improvement(candidate, prior, cell=cell, binding=selection_binding):
                improvement_groups.setdefault(cell, []).append(index)
        improvement_rewards = {
            index: 1.0 / len(indices)
            for indices in improvement_groups.values()
            for index in indices
        }
        challenge_groups: dict[str, list[int]] = {}
        for index in eligible:
            if index in new_cell_rewards or index in improvement_rewards:
                continue
            for challenge_id in facts[index]["resolved"] - seen_resolved:
                challenge_groups.setdefault(challenge_id, []).append(index)
        challenge_rewards: dict[int, float] = {}
        for indices in challenge_groups.values():
            share = 1.0 / len(indices)
            for index in indices:
                challenge_rewards[index] = min(1.0, challenge_rewards.get(index, 0.0) + share)
        pass_groups: dict[str, list[int]] = {}
        for index in eligible:
            if index in new_cell_rewards or index in improvement_rewards or index in challenge_rewards:
                continue
            candidate = facts[index]["candidate"]
            if _observed_pass(candidate):
                pass_groups.setdefault(str(facts[index]["cell"] or "observed_pass"), []).append(index)
        pass_rewards = {
            index: 0.5 / len(indices)
            for indices in pass_groups.values()
            for index in indices
        }
        probe_groups: dict[str, list[int]] = {}
        for index in eligible:
            if (
                index in new_cell_rewards
                or index in improvement_rewards
                or index in challenge_rewards
                or index in pass_rewards
            ):
                continue
            if _probe_survived(facts[index]["candidate"]):
                probe_groups.setdefault("parameterized_probe_survival", []).append(index)
        verified_pass_cap = min(pass_rewards.values()) if pass_rewards else None
        probe_rewards = {
            index: min(0.5 / len(indices), verified_pass_cap) if verified_pass_cap is not None else 0.5 / len(indices)
            for indices in probe_groups.values()
            for index in indices
        }
        cohort_cells: set[str] = set()
        cohort_resolved: set[str] = set()

        for index, fact in enumerate(facts):
            candidate = fact["candidate"]
            cell = fact["cell"]
            resolved = fact["resolved"]
            reasons: list[str] = []
            reward = 0.0
            risk = _text_risk(candidate)

            if fact["unbound"]:
                reasons.append("unbound_branch_slot")
                risk = 1.0
            elif fact["observed_failure"]:
                reasons.append("observed_evaluator_or_patch_failure")
                risk = 1.0
            elif fact["duplicate"]:
                reasons.append("exact_phenotype_duplicate")
                risk = 1.0
            elif fact["terminal"] or fact["verification"] == "failed":
                reasons.append("terminal_or_verification_failure")
                risk = 1.0
            elif fact["transfer_credit"] is not None and not fact["transfer_credit"].productive_credit:
                reasons.append("transfer_" + fact["transfer_credit"].reason)
            elif index in new_cell_rewards:
                reasons.append("new_grounded_outcome_cell")
                reward = new_cell_rewards[index]
            elif index in improvement_rewards:
                reasons.append("same_cell_evaluator_elite_improvement")
                reward = improvement_rewards[index]
            elif index in challenge_rewards:
                reasons.append("new_challenge_resolution")
                reward = challenge_rewards[index]
            elif fact["patch_verified"]:
                reasons.append("verified_applied_artifact")
                reward = 1.0
            elif index in pass_rewards:
                reasons.append("verified_or_evaluator_pass_survived")
                reward = pass_rewards[index]
            elif index in probe_rewards:
                reasons.append("parameterized_probe_survived")
                reward = probe_rewards[index]
            else:
                reasons.append("no_grounded_productive_event")

            if fact["transfer_credit"] is not None and fact["transfer_credit"].productive_credit:
                if reasons == ["no_grounded_productive_event"]:
                    reasons.clear()
                reasons.append("transfer_invariant_probe_survived")
                reward = max(0.5, reward)

            # Seeds establish baselines but are not reproduction pulls.
            if candidate.parent_ids:
                outcomes.append(
                    ProductiveOutcome(
                        candidate_id=candidate.id,
                        arm_id=lineage_root(candidate),
                        outcome_cell=cell,
                        reward=reward,
                        risk=risk,
                        reason_codes=tuple(reasons),
                    )
                )
            if index in eligible_set:
                if cell:
                    cohort_cells.add(cell)
                cohort_resolved.update(resolved)

        prior.extend(fact["candidate"] for index, fact in enumerate(facts) if index in eligible_set)
        seen_phenotypes.update(
            value
            for value, fact in zip(cohort_phenotypes, facts)
            if value and not fact["unbound"]
        )
        seen_cells.update(cohort_cells)
        seen_resolved.update(cohort_resolved)
        start = end
    return tuple(outcomes)


def allocate_productive_branches(
    *,
    parents: list[CandidateGenome],
    candidates: Iterable[CandidateGenome],
    budget_history: Iterable[dict[str, Any]] = (),
    metric_directions: dict[str, str] | None = None,
    total_slots: int,
    observed_family_counts: Mapping[str, int] | None = None,
    credited_transfer_artifact_hashes: Iterable[str] = (),
) -> ProductiveBranchAllocation:
    """Allocate real branch slots with lineage-root UCB and an exploration floor."""

    candidate_list = _dedupe_candidates(candidates)
    family_counts = (
        {family: observed_family_counts[family] for family in sorted(observed_family_counts)}
        if observed_family_counts is not None
        else count_observed_mechanism_families(candidate_list)
    )
    if not parents or total_slots <= 0:
        return ProductiveBranchAllocation(
            slots=(),
            arms=(),
            credit_summary={},
            observed_family_counts=family_counts,
        )
    history = [dict(item) for item in budget_history if isinstance(item, dict)]
    historical_transfer_hashes = _historical_transfer_credit_hashes(history)
    historical_transfer_hashes.update(str(item) for item in credited_transfer_artifact_hashes if str(item))
    outcomes = productive_outcomes(
        candidate_list,
        metric_directions=metric_directions,
        credited_transfer_artifact_hashes=historical_transfer_hashes,
    )
    planned_slots, rejected_slots = _historical_slot_events(history)
    allocation_epoch = max((_history_round(item) for item in history), default=-1) + 1
    planned_ids = {str(item.get("slot_id") or "") for item in planned_slots}
    stats: dict[str, OperatorArmStats] = {}

    def add(arm_id: str, *, pulls: int = 0, reward: float = 0.0, risk: float = 0.0) -> None:
        current = stats.get(arm_id, OperatorArmStats(arm_id=arm_id))
        stats[arm_id] = OperatorArmStats(
            arm_id=arm_id,
            pulls=current.pulls + pulls,
            reward_sum=current.reward_sum + reward,
            risk_sum=current.risk_sum + risk,
        )

    for slot in planned_slots:
        arm_id = str(slot.get("arm_id") or "")
        if arm_id:
            add(arm_id, pulls=1)
    rejected_ids: set[str] = set()
    for event in rejected_slots:
        slot_id = str(event.get("branch_slot_id") or event.get("slot_id") or "")
        if slot_id:
            rejected_ids.add(slot_id)
        arm_id = str(event.get("branch_arm_id") or event.get("arm_id") or "")
        if arm_id:
            add(arm_id, risk=1.0)
    by_id = {candidate.id: candidate for candidate in candidate_list}
    credited_slots: set[str] = set()
    reason_counts: dict[str, int] = {}
    for outcome in outcomes:
        candidate = by_id.get(outcome.candidate_id)
        metadata = coerce_dict(candidate.metadata) if candidate is not None else {}
        slot_id = str(metadata.get("branch_slot_id") or "")
        binding = str(metadata.get("branch_slot_binding_status") or "")
        if binding and binding != "bound":
            continue
        if slot_id and slot_id in rejected_ids:
            continue
        claimed_arm_id = str(metadata.get("branch_arm_id") or "")
        if claimed_arm_id and claimed_arm_id != outcome.arm_id:
            continue
        arm_id = outcome.arm_id
        if slot_id and slot_id in credited_slots:
            continue
        # A legacy child without a recorded slot is itself one inferred pull.
        inferred_pull = int(not slot_id or slot_id not in planned_ids)
        add(arm_id, pulls=inferred_pull, reward=outcome.reward, risk=outcome.risk)
        if slot_id:
            credited_slots.add(slot_id)
        for reason in outcome.reason_codes:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    parent_groups: dict[str, list[CandidateGenome]] = {}
    arm_order: list[str] = []
    for parent in parents:
        arm_id = lineage_root(parent)
        if arm_id not in parent_groups:
            arm_order.append(arm_id)
            parent_groups[arm_id] = []
        parent_groups[arm_id].append(parent)
        stats.setdefault(arm_id, OperatorArmStats(arm_id=arm_id))

    slots: list[BranchSlot] = []
    virtual = {arm_id: stats[arm_id] for arm_id in arm_order}
    arm_slot_counts = {arm_id: 0 for arm_id in arm_order}
    max_family_count = max(family_counts.values(), default=0)
    arm_coverage_bonus = {
        arm_id: sum(
            _family_coverage_bonus(base_mechanism_family(parent), family_counts, max_family_count)
            for parent in parent_groups[arm_id]
        )
        / len(parent_groups[arm_id])
        for arm_id in arm_order
    }

    # Explicitly try every unobserved selected lineage once before exploitation.
    for arm_id in (arm for arm in arm_order if virtual[arm].pulls == 0):
        if len(slots) >= total_slots:
            break
        _append_slot(
            slots,
            arm_id=arm_id,
            parent_groups=parent_groups,
            counts=arm_slot_counts,
            score=float("inf"),
            intent="explore_fresh",
            allocation_epoch=allocation_epoch,
        )
        virtual[arm_id] = _virtual_pull(virtual[arm_id])

    while len(slots) < total_slots:
        suggestions = suggest_budget_allocation(tuple(virtual[arm_id] for arm_id in arm_order))
        raw_scores = [suggestion.suggestion_score for suggestion in suggestions]
        ucb_span = max(raw_scores) - min(raw_scores)
        scored_suggestions = [
            (
                suggestion,
                suggestion.suggestion_score + arm_coverage_bonus[suggestion.arm_id] * ucb_span,
            )
            for suggestion in suggestions
        ]
        selected, selected_allocation_score = max(
            scored_suggestions,
            key=lambda item: (
                item[1],
                arm_coverage_bonus[item[0].arm_id],
                item[0].suggestion_score,
                item[0].arm_id,
            ),
        )
        arm_id = selected.arm_id
        best_mean = max((virtual[item].mean_reward for item in arm_order), default=0.0)
        intent = "exploit_deepen" if virtual[arm_id].mean_reward > 0.0 and virtual[arm_id].mean_reward >= best_mean else "standard_variation"
        _append_slot(
            slots,
            arm_id=arm_id,
            parent_groups=parent_groups,
            counts=arm_slot_counts,
            score=selected.suggestion_score,
            coverage_bonus=arm_coverage_bonus[arm_id],
            coverage_scale=ucb_span,
            allocation_score=selected_allocation_score,
            intent=intent,
            allocation_epoch=allocation_epoch,
        )
        virtual[arm_id] = _virtual_pull(virtual[arm_id])

    active_arms = tuple(stats[arm_id] for arm_id in arm_order)
    credited_transfer_artifact_hashes = tuple(sorted({
        str(coerce_dict(by_id[outcome.candidate_id].metadata).get("transfer_receipt", {}).get("artifact_hash") or "")
        for outcome in outcomes
        if outcome.candidate_id in by_id and "transfer_invariant_probe_survived" in outcome.reason_codes
    } - {""}))
    return ProductiveBranchAllocation(
        slots=tuple(slots),
        arms=active_arms,
        credit_summary=reason_counts,
        observed_family_counts=family_counts,
        credited_transfer_artifact_hashes=credited_transfer_artifact_hashes,
    )


def _append_slot(
    slots: list[BranchSlot],
    *,
    arm_id: str,
    parent_groups: dict[str, list[CandidateGenome]],
    counts: dict[str, int],
    score: float,
    coverage_bonus: float = 0.0,
    coverage_scale: float = 0.0,
    allocation_score: float = 0.0,
    intent: str,
    allocation_epoch: int,
) -> None:
    index = counts[arm_id]
    parent = parent_groups[arm_id][index % len(parent_groups[arm_id])]
    slot_id = "branch-" + stable_hash(
        {"epoch": allocation_epoch, "index": len(slots), "arm": arm_id, "parent": parent.id, "variation": index}
    )[:16]
    slots.append(
        BranchSlot(
            slot_id=slot_id,
            arm_id=arm_id,
            parent_id=parent.id,
            intent=intent,
            variation_index=index,
            ucb_score=0.0 if math.isinf(score) else round(float(score), 6),
            coverage_bonus=round(float(coverage_bonus), 6),
            coverage_scale=round(float(coverage_scale), 6),
            allocation_score=round(float(allocation_score), 6),
        )
    )
    counts[arm_id] += 1


def _virtual_pull(arm: OperatorArmStats) -> OperatorArmStats:
    return OperatorArmStats(
        arm_id=arm.arm_id,
        pulls=arm.pulls + 1,
        reward_sum=arm.reward_sum + arm.mean_reward,
        risk_sum=arm.risk_sum + arm.mean_risk,
    )


def count_observed_mechanism_families(candidates: Iterable[CandidateGenome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in _dedupe_candidates(candidates):
        family = base_mechanism_family(candidate)
        counts[family] = counts.get(family, 0) + 1
    return {family: counts[family] for family in sorted(counts)}


def _family_coverage_bonus(family: str, counts: Mapping[str, int], max_count: int) -> float:
    if max_count <= 0 or family == "general":
        return 0.0
    return 1.0 - counts.get(family, 0) / max_count


def _same_cell_elite_improvement(
    candidate: CandidateGenome,
    prior: list[CandidateGenome],
    *,
    cell: str,
    binding: dict[str, Any] | None,
) -> bool:
    candidate_key = evaluator_selection_key(candidate, binding)[:2]
    if candidate_key[0] < 0:
        return False
    peer_keys = [
        evaluator_selection_key(item, binding)[:2]
        for item in prior
        if observed_outcome_cell(item) == cell
    ]
    peer_keys = [item for item in peer_keys if item[0] >= 0]
    return bool(peer_keys) and candidate_key > max(peer_keys)


def _binding_from_metric_directions(metric_directions: dict[str, str] | None) -> dict[str, Any] | None:
    if not metric_directions:
        return None
    metric, direction = next(iter(metric_directions.items()))
    return {
        "metric": str(metric),
        "direction": str(direction),
        "value_type": "boolean" if str(direction) == "pass" else "number",
    }


def _verification_state(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    payloads = [
        coerce_dict(candidate.verification_result),
        coerce_dict(metadata.get("offspring_verification")),
        *(coerce_dict(item) for item in candidate.verification_trace[-3:]),
    ]
    if any(_payload_failed(item) for item in payloads if item):
        return "failed"
    if any(_payload_passed(item) for item in payloads if item):
        return "passed"
    return "not_run"


def _payload_passed(payload: dict[str, Any]) -> bool:
    status = str(payload.get("validation_status") or payload.get("status") or "").lower()
    if status in {"not_run", "inconclusive"}:
        return False
    return status in {"passed", "pass", "ok", "success", "preliminary_passed"} or payload.get("passed") is True


def _payload_failed(payload: dict[str, Any]) -> bool:
    status = str(payload.get("validation_status") or payload.get("status") or "").lower()
    if status in {"not_run", "inconclusive"}:
        return False
    return status in {"failed", "fail", "error", "preliminary_failed"} or payload.get("passed") is False


def _probe_survived(candidate: CandidateGenome) -> bool:
    payloads = [
        coerce_dict(candidate.verification_result),
        *(coerce_dict(item) for item in candidate.verification_trace[-3:]),
    ]
    for payload in reversed(payloads):
        metadata = coerce_dict(payload.get("metadata"))
        status = str(payload.get("validation_status") or metadata.get("validation_status") or "").lower()
        ratio = metadata.get("probe_survival_ratio", payload.get("probe_survival_ratio"))
        counterexamples = metadata.get("probe_counterexample_count", payload.get("probe_counterexample_count", 0))
        executed = metadata.get("probe_executed_count", payload.get("probe_executed_count", 0))
        try:
            if status == "inconclusive" and float(ratio or 0.0) > 0.0 and int(counterexamples or 0) == 0 and int(executed or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _patch_state(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    patch = coerce_dict(getattr(candidate, "patch_application_result", None) or metadata.get("patch_result"))
    status = str(patch.get("status") or "").strip().lower()
    if status in {"applied", "ok", "passed", "success"}:
        return "applied"
    if status in {"error", "failed", "rejected"}:
        return "failed"
    return "not_run" if not status else "unknown"


def _resolved_challenges(candidate: CandidateGenome) -> set[str]:
    metadata = coerce_dict(candidate.metadata)
    state = coerce_dict(metadata.get("evidence_state"))
    values = state.get("resolved_challenge_ids") or metadata.get("resolved_challenge_ids") or []
    return {str(item) for item in values if str(item or "").strip()} if isinstance(values, list) else set()


def _observed_pass(candidate: CandidateGenome) -> bool:
    evaluator = coerce_dict(coerce_dict(candidate.metadata).get("evaluator"))
    evaluator_status = str(evaluator.get("status") or "").strip().lower()
    return evaluator.get("passed") is True or evaluator_status in {"passed", "pass", "ok", "success"} or _verification_state(candidate) == "passed"


def _observed_failure(candidate: CandidateGenome) -> bool:
    evaluator = coerce_dict(coerce_dict(candidate.metadata).get("evaluator"))
    evaluator_status = str(evaluator.get("status") or "").strip().lower()
    return (
        evaluator.get("passed") is False
        or evaluator_status in {"failed", "fail", "error", "rejected"}
        or _patch_state(candidate) == "failed"
    )


def _terminal(candidate: CandidateGenome) -> bool:
    metadata = coerce_dict(candidate.metadata)
    return bool(
        CandidateFate.normalize(candidate.current_fate) in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}
        or metadata.get("terminal_failure")
        or metadata.get("terminal_reject")
        or metadata.get("terminal_reject_reason")
        or metadata.get("hard_reject_reason")
    )


def _text_risk(candidate: CandidateGenome) -> float:
    nextgen = coerce_dict(coerce_dict(candidate.metadata).get("nextgen"))
    observation = coerce_dict(nextgen.get("productive_child_observation"))
    return 1.0 if observation.get("near_verbatim_reskin") or observation.get("engineering_noise") else 0.0


def _evaluator_state(evaluator: dict[str, Any]) -> str:
    status = str(evaluator.get("status") or "").strip().lower()
    if evaluator.get("passed") is True or status in {"passed", "pass", "ok", "success"}:
        return "passed"
    if evaluator.get("passed") is False or status in {"failed", "fail", "error"}:
        return "failed"
    return "not_run" if not evaluator else "unknown"


def _candidate_cohort(candidate: CandidateGenome) -> tuple[int, int]:
    metadata = coerce_dict(candidate.metadata)
    return (int(metadata.get("created_in_round") or 0), int(candidate.generation or 0))


def _candidate_order(candidate: CandidateGenome) -> tuple[int, int, str]:
    return (*_candidate_cohort(candidate), candidate.id)


def _dedupe_candidates(candidates: Iterable[CandidateGenome]) -> list[CandidateGenome]:
    by_id: dict[str, CandidateGenome] = {}
    for candidate in candidates:
        current = by_id.get(candidate.id)
        if current is None or _grounded_field_count(candidate) > _grounded_field_count(current):
            by_id[candidate.id] = candidate
    return list(by_id.values())


def _grounded_field_count(candidate: CandidateGenome) -> int:
    metadata = coerce_dict(candidate.metadata)
    return sum(bool(item) for item in (metadata.get("evaluator"), metadata.get("evidence_state"), candidate.verification_result, getattr(candidate, "patch_application_result", None)))


def _historical_slot_events(history: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slots: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in history:
        plan = coerce_dict(item.get("generation_plan")) if isinstance(item, dict) else {}
        allocation = coerce_dict(plan.get("productive_branch_allocation"))
        slots.extend(dict(slot) for slot in allocation.get("slots", []) if isinstance(slot, dict))
        harvest = coerce_dict(plan.get("offspring_harvest"))
        rejected.extend(dict(event) for event in harvest.get("rejected", []) if isinstance(event, dict) and str(event.get("reason") or "").startswith("duplicate"))
        rejected.extend(dict(event) for event in plan.get("duplicate_offspring", []) if isinstance(event, dict))
    return slots, rejected


def _historical_transfer_credit_hashes(history: Iterable[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for item in history:
        plan = coerce_dict(item.get("generation_plan")) if isinstance(item, dict) else {}
        allocation = coerce_dict(plan.get("productive_branch_allocation"))
        hashes.update(str(value) for value in allocation.get("credited_transfer_artifact_hashes", []) if str(value))
    return hashes


def _history_round(item: dict[str, Any]) -> int:
    try:
        return int(item.get("round", -1))
    except (TypeError, ValueError):
        return -1


__all__ = [
    "BranchSlot",
    "ProductiveBranchAllocation",
    "ProductiveOutcome",
    "allocate_productive_branches",
    "count_observed_mechanism_families",
    "lineage_root",
    "observed_outcome_cell",
    "productive_outcomes",
]
