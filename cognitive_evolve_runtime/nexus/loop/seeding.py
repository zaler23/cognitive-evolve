"""Seed portfolio construction and generation for Nexus evolution."""
from __future__ import annotations

import os
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.exploration import amplify_population
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusSeedModelProtocol
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.nexus.minimal_core import apply_seed_active_frontier, run_core_ablation
from cognitive_evolve_runtime.nexus.seed_coverage import (
    SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY,
    assess_seed_coverage,
    seed_axis_contract_receipt,
    seed_reservoir_sidecar_payload,
)
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import (
    CandidateHarvester,
    HarvestPolicy,
    candidate_is_target_qualified,
    target_qualified_candidates,
)
from cognitive_evolve_runtime.nexus.search_kernel.skill_library import search_skill_payload
from cognitive_evolve_runtime.nexus._shared import (
    MODEL_BOUNDARY_ERRORS,
    call_with_optional_context,
    demote_model_candidate_runtime_payload,
    positive_int as _positive_int,
)
from cognitive_evolve_runtime.llm.env import LLMConfigurationError
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError

TEXT_SEED_TYPES = [
    "Direct Solver Seed",
    "Known Pattern Seed",
    "Edge Knowledge Seed",
    "Analogy Seed",
    "Inversion Seed",
    "Decomposition Seed",
    "Tool-Grounded Seed",
    "Wildcard Seed",
]

PROJECT_SEED_TYPES = [
    "Minimal Patch Seed",
    "Architecture Refactor Seed",
    "Test-First Seed",
    "Compatibility-Preserving Seed",
    "Internal Forgotten Pattern Seed",
]

# These are cognitive operations, not a hard-coded discipline ontology.  The
# model chooses the useful disciplines/lenses inside each family-axis slot.
SEED_COGNITIVE_AXES: tuple[dict[str, str], ...] = (
    {
        "id": "direct_mainstream",
        "instruction": "Develop the strongest direct or mainstream mechanism for this family, including the assumptions that make it work.",
    },
    {
        "id": "cross_domain_transfer",
        "instruction": "Transfer a mechanism from a model-chosen external discipline or problem class and state the structural correspondence.",
    },
    {
        "id": "edge_knowledge",
        "instruction": "Use relevant edge knowledge, uncommon results, boundary cases, or specialist heuristics that materially change the candidate.",
    },
    {
        "id": "counterexample_probe",
        "instruction": "Probe counterexamples, adversarial cases, failure modes, or assumption violations and turn the result into a candidate mechanism.",
    },
    {
        "id": "representation_shift",
        "instruction": "Change the representation, decomposition, scale, abstraction, or formalism so a different solution path becomes available.",
    },
    {
        "id": "tool_probe",
        "instruction": "Use a model-chosen tool, calculation, experiment, retrieval, simulation, or executable probe to expose a new mechanism.",
    },
)

SEED_AXIS_REQUIRED_CLAIMS = {
    "direct_mainstream": "metadata.search_space.seed_axis_claim",
    "cross_domain_transfer": "metadata.search_space.transfer_source_domain",
    "edge_knowledge": "edge_knowledge_seeds",
    "counterexample_probe": "metadata.search_space.probe_target_assumption",
    "representation_shift": "metadata.search_space.representation_shift.from/to",
    "tool_probe": "metadata.search_space.tool_probe_plan",
}

def seed_population(
    *,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    model: NexusModelLike | None = None,
    min_population_size: int | None = None,
    initial_candidates: list[CandidateGenome | dict[str, Any]] | None = None,
    provided_context: dict[str, Any] | None = None,
) -> CandidatePopulation:
    model_error: Exception | None = None
    model_candidates: list[CandidateGenome] = []
    rejected_model_seeds: list[dict[str, Any]] = []
    target_size = _seed_target_size(policy=policy, world=world, requested_minimum=min_population_size)
    incumbents = _prepare_initial_candidates(initial_candidates, contract=contract)
    full_seed_portfolio = _allocate_seed_slots(policy, requested_count=target_size)
    uncovered_slots = _uncovered_seed_slots(full_seed_portfolio, incumbents)
    remaining_target = max(0, target_size - len(incumbents), len(uncovered_slots))
    if remaining_target and isinstance(model, NexusSeedModelProtocol):
        model_candidates, rejected_model_seeds, model_error = _generate_model_seed_batches(
            model=model,
            contract=contract,
            world=world,
            policy=policy,
            target_size=remaining_target,
            initial_candidates=incumbents,
            provided_context=provided_context,
            full_seed_portfolio=full_seed_portfolio,
        )
    elif remaining_target and model is not None:
        model_error = LLMConfigurationError("configured seed model does not implement NexusSeedModelProtocol")
        if isinstance(policy.metadata, dict):
            policy.metadata["seed_harvest"] = {
                "accepted_count": 0,
                "batches": 0,
                "stopped_reason": "model_error",
                "fatal_model_error": f"{model_error.__class__.__name__}: {model_error}",
            }
    candidates: list[CandidateGenome] = [*incumbents, *model_candidates]
    if model is not None:
        population = CandidatePopulation(candidates)
    else:
        population = amplify_population(
            population=CandidatePopulation(candidates),
            contract=contract,
            world=world,
            policy=policy,
            minimum_size=target_size,
        )
    if rejected_model_seeds:
        for candidate in model_candidates:
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata.setdefault("model_seed_rejections", list(rejected_model_seeds[:10]))
    for candidate in population.candidates:
        candidate.metadata.setdefault("created_in_round", 0)
    return population


def _prepare_initial_candidates(initial_candidates: list[CandidateGenome | dict[str, Any]] | None, *, contract: NexusObjectiveContract) -> list[CandidateGenome]:
    candidates = [item if isinstance(item, CandidateGenome) else candidate_from_dict(item) for item in initial_candidates or []]
    contract_hash = contract.contract_hash()
    for candidate in candidates:
        candidate.parent_ids = []
        candidate.generation = 0
        candidate.lineage = [candidate.id]
        candidate.contract_hash = contract_hash
        candidate.metadata.setdefault("operator_provided_incumbent", True)
        candidate.metadata.setdefault("exploration_source", "operator_provided_incumbent")
        candidate.metadata.setdefault("created_in_round", 0)
    return candidates


def _seed_target_size(*, policy: EvolutionPolicy, world: Any, requested_minimum: int | None) -> int:
    requested = int(requested_minimum) if requested_minimum and requested_minimum > 0 else 0
    configured = _positive_int((policy.metadata or {}).get("initial_candidate_count")) or 0
    families = _seed_families(policy)
    portfolio_minimum = sum(max(len(SEED_COGNITIVE_AXES), _family_quota(family)) for family in families)
    if families:
        return max(1, requested, configured, portfolio_minimum)
    if not requested and not configured:
        niche_count = len({str(item).strip().lower() for item in policy.candidate_niches if str(item).strip()})
        template_count = len(PROJECT_SEED_TYPES if getattr(world, "kind", "text") == "project" else TEXT_SEED_TYPES)
        configured = max(1, niche_count or template_count)
    return max(1, requested, configured)


def _generate_model_seed_batches(
    *,
    model: NexusSeedModelProtocol,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    target_size: int,
    initial_candidates: list[CandidateGenome] | None = None,
    provided_context: dict[str, Any] | None = None,
    full_seed_portfolio: list[dict[str, Any]] | None = None,
) -> tuple[list[CandidateGenome], list[dict[str, Any]], Exception | None]:
    deduper = CandidateDeduper()
    incumbent_candidates = list(initial_candidates or [])
    for candidate in incumbent_candidates:
        deduper.add(candidate)
    portfolio = list(full_seed_portfolio or _allocate_seed_slots(policy, requested_count=target_size + len(incumbent_candidates)))
    if portfolio:
        target_size = max(target_size, len(_uncovered_seed_slots(portfolio, incumbent_candidates)))
        if isinstance(policy.metadata, dict):
            policy.metadata["seed_portfolio"] = [dict(slot) for slot in portfolio]
            policy.metadata["seed_portfolio_contract"] = _seed_portfolio_contract()
    harvester = CandidateHarvester(
        deduper=deduper,
        policy=HarvestPolicy(
            target_size=target_size,
            max_batches=1 if portfolio else _seed_safety_batch_limit(policy=policy),
            min_batches=1 if portfolio else _seed_min_batches(policy=policy),
            low_gain_patience=_seed_low_novelty_patience(policy=policy),
            relevance_floor=0.20,
            stage="seed",
            fanout_workers=1 if portfolio else _seed_fanout_workers(policy=policy, target_size=target_size),
            stop_at_target=True,
            exhaust_on_no_new=False,
            reservoir_mode=True,
        ),
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        accepted_with_incumbent = [*incumbent_candidates, *accepted]
        batch_slots = _uncovered_seed_slots(portfolio, accepted_with_incumbent)
        batch_policy = _policy_for_seed_batch(
            policy,
            batch_index=batch_index,
            accepted=accepted_with_incumbent,
            rejected=rejected,
            target_size=len(portfolio) or target_size + len(incumbent_candidates),
            seed_portfolio=batch_slots,
        )
        raw = call_with_optional_context(
            model.seed_population,
            contract=contract,
            world=world,
            policy=batch_policy,
            provided_context=provided_context,
        )
        batch = _order_seed_batch_for_portfolio(_coerce_seed_batch(raw), batch_slots)
        priority = _seed_family_priority(policy, accepted_with_incumbent)
        origin = _seed_origin_metadata(model)
        for candidate in batch:
            # Every accepted seed starts an independent lineage arm.  Model-
            # supplied synthetic parents would collapse the entire portfolio
            # back into one bandit arm before any outcome is observed.
            candidate.parent_ids = []
            candidate.generation = 0
            candidate.lineage = [candidate.id]
            candidate.metadata.setdefault("exploration_source", "nexus_model_seed_batch")
            candidate.metadata.setdefault("created_in_round", 0)
            candidate.metadata["model_seed_batch"] = batch_index
            for key, value in origin.items():
                candidate.metadata.setdefault(key, value)
            candidate.metadata.setdefault("seed_family_priority_trace", {"batch_index": batch_index, "source": priority.get("source"), "requested_families": [item.get("id") or item.get("name") for item in list(priority.get("families") or [])[:4]]})
        return batch

    result = harvester.harvest(
        request_batch=_request,
        on_error=is_quota_error,
        context={"contract": contract, "policy": policy, "world": world},
        recoverable_errors=MODEL_BOUNDARY_ERRORS,
    )
    if not result.accepted and not incumbent_candidates and result.fatal_model_error is None:
        result.fatal_model_error = ModelResponseSchemaError("nexus_seed_population returned no valid candidates")
        result.stopped_reason = "model_error"
    coverage = assess_seed_coverage(
        target_qualified_candidates([*incumbent_candidates, *result.accepted]),
        reservoir=result.reservoir,
        rejected=result.rejected,
        harvest_summary=result.to_dict(),
        contract=contract,
        policy=policy,
    )
    if isinstance(policy.metadata, dict):
        all_seed_candidates = [*incumbent_candidates, *result.accepted]
        frontier = apply_seed_active_frontier(all_seed_candidates, limit=_seed_active_frontier_limit(policy=policy))
        ablation = run_core_ablation(all_seed_candidates, policy=policy)
        policy.metadata["seed_harvest"] = result.to_dict()
        policy.metadata["seed_coverage"] = coverage
        policy.metadata["seed_active_frontier"] = frontier
        policy.metadata["minimal_core_ablation"] = ablation
        if result.reservoir:
            policy.metadata[SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY] = seed_reservoir_sidecar_payload(result.reservoir)
        policy.metadata["algorithm_efficiency"] = {
            "seed_batches": result.batches,
            "accepted_per_batch": round(len(result.accepted) / max(1, result.batches), 4),
            "target_qualified_per_batch": round(result.target_qualified_count / max(1, result.batches), 4),
            "reservoir_count": len(result.reservoir),
            "partial_failure_count": len(result.failed_batch_ids),
            "advisory_frontier_size": len(frontier.get("selected_ids") or []),
            "advisory_unselected_seed_count": int(frontier.get("advisory_unselected_count") or 0),
            "policy": "measure_only_no_capability_tradeoff",
        }
        policy.metadata["model_parallel_efficiency"] = {
            "seed_fanout_workers": 1 if portfolio else _seed_fanout_workers(policy=policy, target_size=target_size),
            "max_batches": 1 if portfolio else _seed_safety_batch_limit(policy=policy),
            "policy": "parallelism_observed_not_seed_breadth_reduced",
        }
    for candidate in result.accepted:
        candidate.metadata.setdefault("seed_harvest", _candidate_seed_harvest_trace(result, candidate))
        candidate.metadata.setdefault("seed_coverage", _candidate_seed_coverage_trace(coverage))
        candidate.metadata.setdefault("minimal_core_ablation_profile", ablation.get("recommendation", "advisory"))
        if result.reservoir:
            candidate.metadata.setdefault(
                "seed_reservoir",
                {
                    "mode": "soft_reject_retention",
                    "candidate_ids": [item.id for item in result.reservoir[-100:]],
                    "count": len(result.reservoir),
                    "checkpoint_policy": "coverage_summary_plus_candidate_ids",
                },
            )
    return result.accepted, result.rejected, result.fatal_model_error


def _candidate_seed_harvest_trace(result: Any, candidate: CandidateGenome) -> dict[str, Any]:
    return {
        "schema": "seed_harvest_candidate_trace.v1",
        "stage": str(getattr(result, "stage", "") or "seed"),
        "candidate_id": candidate.id,
        "batch": int((candidate.metadata or {}).get("model_seed_batch") or (candidate.metadata or {}).get("search_kernel_batch") or 0),
        "batches": int(getattr(result, "batches", 0) or 0),
        "accepted_count": len(getattr(result, "accepted", []) or []),
        "target_qualified_count": int(getattr(result, "target_qualified_count", 0) or 0),
        "carried_low_relevance_count": int(getattr(result, "carried_low_relevance_count", 0) or 0),
        "rejected_count": len(getattr(result, "rejected", []) or []),
        "reservoir_count": len(getattr(result, "reservoir", []) or []),
        "stopped_reason": str(getattr(result, "stopped_reason", "") or ""),
        "failed_batch_ids": list(getattr(result, "failed_batch_ids", []) or []),
        "partial_failure_count": len(getattr(result, "failed_batch_ids", []) or []),
        "fatal_model_error": f"{result.fatal_model_error.__class__.__name__}: {result.fatal_model_error}" if getattr(result, "fatal_model_error", None) else "",
        "policy": "per_candidate_compact_trace_full_harvest_in_policy_metadata",
    }


def _candidate_seed_coverage_trace(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "seed_coverage_candidate_trace.v1",
        "status": coverage.get("status") or coverage.get("coverage_status") or "",
        "coverage_status": coverage.get("coverage_status") or coverage.get("status") or "",
        "contract_coverage_status": coverage.get("contract_coverage_status") or "not_planned",
        "candidate_count": coverage.get("candidate_count"),
        "family_count": coverage.get("family_count"),
        "singleton_family_count": coverage.get("singleton_family_count"),
        "top1_family_share": coverage.get("top1_family_share"),
        "top3_family_share": coverage.get("top3_family_share"),
        "fingerprint": coverage.get("fingerprint"),
        "needs_more_seed": coverage.get("needs_more_seed"),
        "needs_target_perturb": coverage.get("needs_target_perturb"),
        "required_slot_count": coverage.get("required_slot_count"),
        "covered_slot_count": coverage.get("covered_slot_count"),
        "missing_slot_count": len(coverage.get("missing_slot_ids") or []),
        "missing_edge_slot_count": len(coverage.get("missing_edge_slot_ids") or []),
        "redundant_edge_slot_count": len(coverage.get("redundant_edge_slot_ids") or []),
        "missing_lens_slot_count": len(coverage.get("missing_lens_slot_ids") or []),
        "redundant_lens_slot_count": len(coverage.get("redundant_lens_slot_ids") or []),
        "missing_outcome_slot_count": len(coverage.get("missing_outcome_slot_ids") or []),
        "capability_status": coverage.get("capability_status") or "pending_pba_outcome",
        "policy": "compact_candidate_trace_full_coverage_in_policy_metadata",
    }


def _seed_origin_metadata(model: Any) -> dict[str, Any]:
    metadata = getattr(model, "metadata", {}) if isinstance(getattr(model, "metadata", None), dict) else {}
    spec = metadata.get("model_spec") if isinstance(metadata.get("model_spec"), dict) else {}
    provider = str(spec.get("provider") or "").strip()
    model_name = str(spec.get("model") or spec.get("fixture") or "").strip()
    profile_id = str(spec.get("profile_id") or spec.get("model_profile_id") or "").strip()
    origin = "/".join(part for part in (provider, model_name) if part)
    if not origin:
        origin = str(metadata.get("transport") or type(model).__name__).strip()
    out = {"origin_model": origin or type(model).__name__}
    if profile_id:
        out["model_profile_id"] = profile_id
    spec_hash = str(metadata.get("model_spec_hash") or "").strip()
    if spec_hash:
        out["origin_model_spec_hash"] = spec_hash
    return out


def _seed_active_frontier_limit(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(metadata.get("seed_active_frontier_size") or metadata.get("active_frontier_size") or metadata.get("seed_active_evaluation_budget"))
    if configured:
        return configured
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_ACTIVE_FRONTIER_SIZE"))
    if configured:
        return configured
    return 64


def _coerce_seed_batch(raw: Any) -> list[CandidateGenome]:
    if isinstance(raw, CandidatePopulation):
        return [_candidate_from_model_seed(item) for item in raw.candidates]
    if isinstance(raw, list):
        return [_candidate_from_model_seed(item) for item in raw if isinstance(item, (CandidateGenome, dict))]
    return []


def _candidate_from_model_seed(item: CandidateGenome | dict[str, Any]) -> CandidateGenome:
    data = demote_model_candidate_runtime_payload(item.to_dict() if isinstance(item, CandidateGenome) else dict(item))
    metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
    claimed_id = str(data.pop("id", "") or "").strip()
    metadata.pop("model_claimed_candidate_id", None)
    if claimed_id:
        metadata["model_claimed_candidate_id"] = claimed_id
    data["metadata"] = metadata
    return candidate_from_dict(_normalize_seed_item(data))


def _normalize_seed_item(item: dict[str, Any]) -> dict[str, Any]:
    """Preserve seed-specific structured fields for direct and adapter models."""

    data = dict(item)
    metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
    if isinstance(data.get("search_space"), dict):
        metadata.setdefault("search_space", dict(data.get("search_space") or {}))
    evaluation_dimensions = [str(value).strip() for value in data.get("evaluation_dimensions", []) or [] if str(value).strip()]
    if evaluation_dimensions:
        structured = dict(metadata.get("structured_output_fields") or {}) if isinstance(metadata.get("structured_output_fields"), dict) else {}
        structured["evaluation_dimensions"] = evaluation_dimensions
        metadata["structured_output_fields"] = structured
    if metadata:
        data["metadata"] = metadata
    return data


def _seed_family_plan(policy: EvolutionPolicy) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    typed_plan = policy.search_space if isinstance(policy.search_space, dict) else {}
    legacy_plan = metadata.get("search_space_plan") if isinstance(metadata.get("search_space_plan"), dict) else {}
    for plan in (typed_plan, legacy_plan):
        raw = plan.get("candidate_families") or plan.get("exploration_planes") or plan.get("families") or plan.get("planes") or []
        families = [dict(item) for item in raw if isinstance(item, dict) and str(item.get("id") or item.get("name") or "").strip()]
        if families:
            return plan, families
    return {}, []


def _seed_families(policy: EvolutionPolicy) -> list[dict[str, Any]]:
    """Return active model, contract, or domain-neutral fallback families."""

    return _seed_family_plan(policy)[1]


def _family_quota(family: dict[str, Any]) -> int:
    return _positive_int(family.get("quota_min")) or 1


def _seed_slot(family: dict[str, Any], *, axis_index: int, occurrence: int) -> dict[str, Any]:
    family_id = str(family.get("id") or family.get("name") or "").strip()
    axis = SEED_COGNITIVE_AXES[axis_index % len(SEED_COGNITIVE_AXES)]
    variant = occurrence // len(SEED_COGNITIVE_AXES) + 1
    return {
        "slot_id": f"{family_id}::{axis['id']}::{variant}",
        "family_id": family_id,
        "family_variant_index": variant,
        "seed_axis": axis["id"],
        "axis_instruction": axis["instruction"],
        "required_claim": SEED_AXIS_REQUIRED_CLAIMS[axis["id"]],
    }


def _allocate_seed_slots(policy: EvolutionPolicy, *, requested_count: int) -> list[dict[str, Any]]:
    """Allocate an outcome-ready family × cognitive-axis seed portfolio.

    Every model-authored family receives all six cognitive operations.  A
    larger ``quota_min`` or requested population adds round-robin variants;
    neither disciplines nor task domains are named by the runtime.
    """

    families = _seed_families(policy)
    if not families:
        return []
    family_targets = [max(len(SEED_COGNITIVE_AXES), _family_quota(family)) for family in families]
    target = max(int(requested_count or 0), sum(family_targets))
    counts = [0 for _ in families]
    slots: list[dict[str, Any]] = []

    # Axis-major order makes a short provider response cover families before it
    # deepens one family, while the complete call still carries the Cartesian
    # family × cognitive-operation portfolio.
    for axis_index in range(len(SEED_COGNITIVE_AXES)):
        for family_index, family in enumerate(families):
            slots.append(_seed_slot(family, axis_index=axis_index, occurrence=counts[family_index]))
            counts[family_index] += 1

    while len(slots) < target:
        eligible = [index for index, count in enumerate(counts) if count < family_targets[index]]
        if not eligible:
            eligible = list(range(len(families)))
        family_index = min(
            eligible,
            key=lambda index: (counts[index] / max(1, family_targets[index]), index),
        )
        occurrence = counts[family_index]
        slots.append(_seed_slot(families[family_index], axis_index=occurrence, occurrence=occurrence))
        counts[family_index] += 1
    return slots


def _seed_declaration(candidate: CandidateGenome) -> dict[str, str]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    search_space = metadata.get("search_space") if isinstance(metadata.get("search_space"), dict) else {}
    return {
        "slot_id": str(metadata.get("seed_type") or "").strip(),
        "family_id": str(search_space.get("family_id") or search_space.get("plane_id") or "").strip(),
        "seed_axis": str(search_space.get("seed_axis") or "").strip(),
        "seed_axis_claim": str(search_space.get("seed_axis_claim") or "").strip(),
    }


def _candidate_matches_seed_slot(candidate: CandidateGenome, slot: dict[str, Any], *, require_receipt: bool) -> bool:
    return bool(
        seed_axis_contract_receipt(
            candidate,
            slot,
            require_slot_id=require_receipt,
        )["complete"]
    )


def _uncovered_seed_slots(portfolio: list[dict[str, Any]], candidates: list[CandidateGenome]) -> list[dict[str, Any]]:
    remaining_candidates = target_qualified_candidates(candidates)
    missing: list[dict[str, Any]] = []
    for slot in portfolio:
        match_index = next(
            (index for index, candidate in enumerate(remaining_candidates) if _candidate_matches_seed_slot(candidate, slot, require_receipt=True)),
            None,
        )
        if match_index is None:
            missing.append(dict(slot))
        else:
            remaining_candidates.pop(match_index)
    return missing


def _order_seed_batch_for_portfolio(batch: list[CandidateGenome], portfolio: list[dict[str, Any]]) -> list[CandidateGenome]:
    """Prefer contract-complete slot/family coverage without rejecting fallbacks."""

    if not portfolio or not batch:
        return list(batch)
    remaining = list(batch)
    ordered: list[CandidateGenome] = []
    for slot in portfolio:
        match_index: int | None = None
        for predicate in (
            lambda candidate: _candidate_matches_seed_slot(candidate, slot, require_receipt=True),
            lambda candidate: _candidate_matches_seed_slot(candidate, slot, require_receipt=False),
            lambda candidate: _seed_declaration(candidate)["family_id"] == str(slot.get("family_id") or "")
            and _seed_declaration(candidate)["seed_axis"] == str(slot.get("seed_axis") or ""),
            lambda candidate: _seed_declaration(candidate)["family_id"] == str(slot.get("family_id") or ""),
        ):
            match_index = next((index for index, candidate in enumerate(remaining) if predicate(candidate)), None)
            if match_index is not None:
                break
        if match_index is not None:
            ordered.append(remaining.pop(match_index))
    return [*ordered, *remaining]


def _seed_portfolio_contract() -> dict[str, Any]:
    return {
        "mode": "family_x_cognitive_axis",
        "required_axes": [axis["id"] for axis in SEED_COGNITIVE_AXES],
        "discipline_policy": "model_selects_relevant_disciplines_no_runtime_discipline_ontology",
        "slot_receipt": "metadata.seed_type must equal slot_id",
        "axis_declaration": "metadata.search_space must include matching family_id and seed_axis; direct mainstream requires seed_axis_claim, cross-domain transfer_source_domain plus metadata.transfer_receipt structure mapping, counterexample probe_target_assumption, representation shift from/to, tool probe tool_probe_plan, and edge knowledge non-empty edge_knowledge_seeds.",
        "axis_required_claims": dict(SEED_AXIS_REQUIRED_CLAIMS),
        "outcome_readiness": "Each slot needs a materially distinct artifact, pairwise non-redundant model-chosen niche_memberships, and non-empty evaluation_dimensions; edge_knowledge_seeds are required only for edge_knowledge slots.",
        "acceptance_policy": "Missing declarations create coverage shortfall but do not hard-reject a candidate.",
        "capability_policy": "Slot labels are contract receipts only; productive capability is credited later from PBA grounded outcomes.",
    }


def _seed_family_priority(policy: EvolutionPolicy, accepted: list[CandidateGenome]) -> dict[str, Any]:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    plan, families = _seed_family_plan(policy)
    source = str(plan.get("source") or metadata.get("seed.family_priority_source") or metadata.get("seed_family_priority_source") or "model_authored_search_space")
    counts: dict[str, int] = {}
    for candidate in target_qualified_candidates(accepted):
        candidate_metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
        search_space = candidate_metadata.get("search_space") if isinstance(candidate_metadata, dict) else {}
        if isinstance(search_space, dict):
            family_id = str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()
            if family_id:
                counts[family_id] = counts.get(family_id, 0) + 1
    prioritized = []
    for index, family in enumerate(families):
        family_id = str(family.get("id") or family.get("name") or "").strip()
        if not family_id:
            continue
        item = dict(family)
        accepted_count = counts.get(family_id, 0)
        portfolio_quota = max(len(SEED_COGNITIVE_AXES), _family_quota(family))
        item["accepted_count"] = accepted_count
        item["portfolio_quota"] = portfolio_quota
        item["quota_deficit"] = max(0, portfolio_quota - accepted_count)
        item["priority_reason"] = "undercovered_search_family" if accepted_count == 0 else "covered_family_soft_followup"
        item["_stable_index"] = index
        prioritized.append(item)
    prioritized.sort(
        key=lambda item: (
            0 if int(item.get("accepted_count") or 0) == 0 else 1,
            int(item.get("accepted_count") or 0) / max(1, int(item.get("portfolio_quota") or 1)),
            int(item.get("_stable_index") or 0),
        )
    )
    for item in prioritized:
        item.pop("_stable_index", None)
    if not prioritized:
        source = "objective_placeholder"
    return {"source": source, "families": prioritized, "coverage": counts}


def _policy_for_seed_batch(
    policy: EvolutionPolicy,
    *,
    batch_index: int,
    accepted: list[CandidateGenome],
    rejected: list[dict[str, Any]],
    target_size: int = 0,
    seed_portfolio: list[dict[str, Any]] | None = None,
) -> EvolutionPolicy:
    data = policy.to_dict()
    metadata = dict(data.get("metadata") or {})
    family_priority = _seed_family_priority(policy, accepted)
    portfolio = list(seed_portfolio) if seed_portfolio is not None else _uncovered_seed_slots(
        _allocate_seed_slots(policy, requested_count=max(1, int(target_size or 1))),
        accepted,
    )
    target_qualified_count = len(target_qualified_candidates(accepted))
    carried_low_relevance_count = sum(not candidate_is_target_qualified(candidate) for candidate in accepted)
    requested_count = len(portfolio) if portfolio else max(1, int(target_size or 1) - target_qualified_count)
    seed_instruction = (
        "In this single seed call return exactly one materially distinct candidate per seed_portfolio slot. "
        "For every candidate set metadata.seed_type to slot_id and metadata.search_space to the slot family_id, "
        "and seed_axis. For direct_mainstream add a non-empty seed_axis_claim explaining the mainstream mechanism. "
        "For cross_domain_transfer add transfer_source_domain plus metadata.transfer_receipt with source/target relations, element-level mapping, preserved invariant, predicted break condition, probe_ref, and artifact_hash; for counterexample_probe add probe_target_assumption; "
        "for representation_shift add representation_shift.from and .to; for tool_probe add a concrete tool_probe_plan; "
        "only edge_knowledge slots require non-empty, pairwise non-redundant edge_knowledge_seeds. Use pairwise non-redundant niche_memberships for model-chosen angles, "
        "perspectives, and disciplines; and provide non-empty evaluation_dimensions tied to observable outcomes. "
        "Do not merely relabel or rephrase one mechanism. The runtime does not prescribe disciplines."
        if portfolio
        else (
            "Generate the requested number of semantically distinct candidates from model-authored objective-level families. "
            "Choose relevant angles and disciplines yourself; populate concrete edge_knowledge_seeds and observable evaluation_dimensions; "
            "do not rephrase the same mechanism."
        )
    )
    metadata.update(
        {
            "seed_batch_index": batch_index,
            "accepted_seed_signatures": [candidate.metadata.get("dedupe_signature") for candidate in accepted[-12:] if candidate.metadata.get("dedupe_signature")],
            "rejected_seed_count": len(rejected),
            "seed_instruction": seed_instruction,
            "seed_family_priority": list(family_priority.get("families") or []),
            "seed_family_priority_source": str(family_priority.get("source") or "objective_placeholder"),
            "seed_family_coverage_snapshot": dict(family_priority.get("coverage") or {}),
            "seed_portfolio": [dict(slot) for slot in portfolio],
            "seed_portfolio_contract": _seed_portfolio_contract(),
            "requested_candidate_count": requested_count,
            "seed_target_size": max(1, int(target_size or 1)),
            "accepted_seed_count": target_qualified_count,
            "carried_low_relevance_count": carried_low_relevance_count,
            "search_kernel_skills": search_skill_payload(limit=4),
        }
    )
    data["metadata"] = metadata
    return EvolutionPolicy.from_dict(data)


SEED_BATCH_DEFAULT_MAX = 1
_UNBOUNDED_SEED_LIMIT_VALUES = {"0", "none", "no_limit", "unbounded", "until_exhausted"}


def _seed_safety_batch_limit(*, policy: EvolutionPolicy) -> int | None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    raw_configured = (
        metadata.get("seed_safety_max_batches")
        or metadata.get("seed_harvest_safety_max_batches")
        or metadata.get("seed_max_batches")
    )
    if _seed_limit_is_unbounded(raw_configured):
        return None
    configured = _positive_int(
        raw_configured
    )
    if configured:
        return configured
    raw_env = os.environ.get("COGEV_NEXUS_SEED_BATCH_LIMIT")
    if _seed_limit_is_unbounded(raw_env):
        return None
    configured = _positive_int(raw_env)
    if configured:
        return configured
    return SEED_BATCH_DEFAULT_MAX


def _seed_limit_is_unbounded(value: Any) -> bool:
    return str(value or "").strip().lower() in _UNBOUNDED_SEED_LIMIT_VALUES


def _seed_min_batches(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(metadata.get("seed_min_batches") or metadata.get("seed_min_batches_before_exhaustion"))
    if configured:
        max_batches = _seed_safety_batch_limit(policy=policy)
        return max(1, configured if max_batches is None else min(max_batches, configured))
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_MIN_BATCHES"))
    if configured:
        max_batches = _seed_safety_batch_limit(policy=policy)
        return max(1, configured if max_batches is None else min(max_batches, configured))
    return 1

def _seed_low_novelty_patience(*, policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(
        metadata.get("seed_no_new_patience")
        or metadata.get("seed_low_novelty_patience")
        or metadata.get("seed_exhaustion_patience")
    )
    if configured:
        return configured
    configured = _positive_int(os.environ.get("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE"))
    if configured:
        return configured
    return 1


def _seed_fanout_workers(*, policy: EvolutionPolicy, target_size: int) -> int | None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    configured = _positive_int(
        metadata.get("seed_fanout_concurrency")
        or metadata.get("seed_batch_concurrency")
        or metadata.get("seed_harvest_fanout_workers")
    )
    if configured:
        return configured
    # No seed-specific override: follow the shared model fanout governor.
    # Concurrent seed prompts intentionally share a previous-window snapshot of
    # accepted signatures; the post-fanout harvester remains the serial
    # dedupe/merge authority for deterministic acceptance.
    return None


__all__ = ["TEXT_SEED_TYPES", "PROJECT_SEED_TYPES", "seed_population"]
