"""Offspring planning, allocation, and model generation for Nexus rounds."""
from __future__ import annotations

import os
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus._serde import coerce_str_list, stable_hash
from cognitive_evolve_runtime.llm.fanout import run_ordered_fanout
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.llm.session import logical_llm_call
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol
from cognitive_evolve_runtime.nexus._shared import (
    MODEL_BOUNDARY_ERRORS,
    call_with_optional_context,
    demote_model_candidate_runtime_payload,
    demote_model_runtime_metadata,
    positive_int,
)
from cognitive_evolve_runtime.llm.env import LLMConfigurationError
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import (
    CandidateHarvester,
    HarvestPolicy,
    plan_signature,
    target_qualified_candidates,
)
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_materialized_artifact
from cognitive_evolve_runtime.nexus.search_kernel.skill_library import search_skill_payload

from .policy_directives import _attach_policy_directives_to_plans

_OFFSPRING_PROMPT_TEMPLATE_VERSION = "nexus-offspring/v1"

_TRUSTED_BRANCH_DIRECTIVE_KEYS = (
    "search_pressure",
    "search_pressure_id",
    "target_challenge_ids",
    "artifact_policy",
    "latent_exploration_action",
    "latent_decision_trace",
    "problem_model_discrimination_action",
    "problem_model_decision_trace",
    "problem_model_snapshot_hash",
    "problem_model_ledger_cursor",
)

_LINEAGE_ENVELOPE_METADATA_KEYS = (
    "action_palette",
    "archive_search_memory",
    "branch_slots",
    "completion_mode",
    "latent_exploration",
    "policy_directives",
    "requested_candidate_count",
    "search_diagnosis",
    "semantic_mutation_owner",
)


def _plan_mutations(
    *,
    model: NexusModelLike | None,
    mutation_planner: MutationPlanner,
    parents: list[CandidateGenome],
    actions: list[str],
    archives: ArchiveManager,
    diagnosis: SearchDiagnosis,
    policy: EvolutionPolicy,
    provided_context: dict[str, Any] | None = None,
    target_count: int | None = None,
) -> list[MutationPlan]:
    if model is None:
        fallback = mutation_planner.plan_from_actions(parents, actions, rarity_seeds=archives.rarity_archive.rare_seeds(limit=max(2, len(parents))))
        return _attach_policy_directives_to_plans(fallback, policy, parents=parents)
    if not isinstance(model, NexusMutationPlannerModelProtocol):
        raise LLMConfigurationError("configured model does not implement NexusMutationPlannerModelProtocol")
    target = max(1, int(target_count or len(parents)))
    accepted: list[MutationPlan] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    low_gain_streak = 0
    valid_parent_ids = {parent.id for parent in parents}
    for batch_index in range(_mutation_plan_batch_limit(target)):
        raw = call_with_optional_context(
            model.plan_mutations,
            parents=parents,
            actions=actions,
            archives=archives,
            diagnosis=diagnosis,
            policy=_policy_for_generation_batch(policy, batch_index=batch_index, accepted_signatures=list(seen), rejected=rejected, kind="mutation_plan"),
            provided_context=provided_context,
        )
        model_plans = [item if isinstance(item, MutationPlan) else MutationPlan.from_dict(item) for item in (raw or []) if isinstance(item, (MutationPlan, dict))]
        model_plans = [
            MutationPlan.from_dict(
                {
                    **plan.to_dict(),
                    "metadata": demote_model_runtime_metadata(plan.metadata),
                }
            )
            for plan in model_plans
        ]
        model_plans = _attach_policy_directives_to_plans(list(model_plans), policy, parents=parents)
        batch_new = 0
        for plan in model_plans:
            claimed_parent_ids = list(dict.fromkeys(str(item) for item in plan.parent_ids if item is not None and str(item).strip()))
            bound_parent_ids = [parent_id for parent_id in claimed_parent_ids if parent_id in valid_parent_ids]
            dropped_parent_ids = [parent_id for parent_id in claimed_parent_ids if parent_id not in valid_parent_ids]
            if not bound_parent_ids:
                rejected.append({"batch": batch_index, "reason": "invalid_plan_parent_ids", "parent_ids": claimed_parent_ids})
                continue
            if dropped_parent_ids:
                metadata = dict(plan.metadata or {})
                metadata["dropped_unavailable_parent_ids"] = dropped_parent_ids
                metadata["model_claimed_parent_ids"] = claimed_parent_ids
                plan = MutationPlan.from_dict({**plan.to_dict(), "parent_ids": bound_parent_ids, "metadata": metadata})
            metadata = dict(plan.metadata or {})
            claimed_plan_id = str(metadata.pop("plan_id", "") or metadata.get("id") or "").strip()
            metadata.pop("id", None)
            claimed_plan_source = str(metadata.pop("plan_source", "") or "").strip()
            metadata.pop("model_claimed_plan_id", None)
            metadata.pop("model_claimed_plan_source", None)
            if claimed_plan_id:
                metadata["model_claimed_plan_id"] = claimed_plan_id
            if claimed_plan_source:
                metadata["model_claimed_plan_source"] = claimed_plan_source
            plan = MutationPlan.from_dict({**plan.to_dict(), "metadata": metadata})
            sig = plan_signature(plan)
            plan.metadata["search_kernel_plan_signature"] = sig
            plan.metadata["plan_id"] = sig
            plan.metadata["search_kernel_batch"] = batch_index
            if sig in seen:
                rejected.append({"batch": batch_index, "reason": "duplicate_plan_signature", "signature": sig, "operator": plan.operator})
                continue
            seen.add(sig)
            accepted.append(plan)
            batch_new += 1
        low_gain_streak = low_gain_streak + 1 if batch_new <= 0 else 0
        if len(accepted) >= target and batch_index + 1 >= _mutation_plan_min_batches(target):
            break
        if low_gain_streak >= _mutation_plan_low_gain_patience(target):
            break
    if not accepted:
        raise ModelResponseSchemaError("nexus_plan_mutations returned no valid mutation plans")
    for plan in accepted:
        plan.metadata.setdefault("search_kernel_plan_harvest", {"accepted": len(accepted), "rejected": rejected[-20:]})
    return accepted[:target]


def _parent_for_plan(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    metadata = plan.metadata if isinstance(plan.metadata, dict) else {}
    if metadata.get("branch_slot_binding_status") == "planned":
        allocated_parent = by_id.get(str(metadata.get("branch_slot_parent_id") or ""))
        if allocated_parent is not None:
            return allocated_parent
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _generate_offspring(
    *,
    model: NexusModelLike | None,
    mutation_engine: MutationEngine,
    parents: list[CandidateGenome],
    plans: list[MutationPlan],
    world: Any,
    contract: NexusObjectiveContract,
    policy: EvolutionPolicy,
    candidate_pool: list[CandidateGenome] | None = None,
    ca_config: CACrossoverConfig | None = None,
    provided_context: dict[str, Any] | None = None,
    target_size: int | None = None,
    harvest_outcome: dict[str, Any] | None = None,
) -> list[CandidateGenome]:
    if harvest_outcome is not None:
        harvest_outcome.clear()
    if model is None:
        return _deterministic_fallback_offspring(
            mutation_engine=mutation_engine,
            parents=parents,
            plans=plans,
            candidate_pool=candidate_pool or parents,
            ca_config=ca_config,
        )
    if not isinstance(model, NexusOffspringModelProtocol):
        raise LLMConfigurationError("configured model does not implement NexusOffspringModelProtocol")
    branch_slots = _branch_slots_from_plans(plans)
    policy_metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    mode = str(
        policy_metadata.get("offspring_parallel_mode")
        or os.environ.get("COGEV_OFFSPRING_PARALLEL_MODE")
        or "slot"
    ).strip().lower()
    if mode not in {"slot", "single_batch"}:
        raise ValueError("COGEV_OFFSPRING_PARALLEL_MODE must be slot or single_batch")
    if branch_slots and mode == "slot":
        return _generate_slot_offspring(
            model=model,
            parents=parents,
            plans=plans,
            branch_slots=branch_slots,
            world=world,
            contract=contract,
            policy=policy,
            candidate_pool=candidate_pool or [],
            provided_context=provided_context,
            harvest_outcome=harvest_outcome,
        )
    harvest_error: Exception | None = None
    saw_model_items = False
    target = max(1, int(target_size or len(plans) or 1))
    direct_single_batch = target_size is not None and any(_branch_slots_from_plans(plans))
    existing_candidates = list(candidate_pool or [])
    existing_ids = {candidate.id for candidate in existing_candidates}
    existing_candidates.extend(parent for parent in parents if parent.id not in existing_ids)
    harvester = CandidateHarvester(
        deduper=CandidateDeduper(existing_candidates),
        policy=HarvestPolicy(
            target_size=target,
            max_batches=1 if direct_single_batch else _offspring_batch_limit(target),
            min_batches=1 if direct_single_batch else _offspring_min_batches(target),
            low_gain_patience=_offspring_low_gain_patience(target),
            relevance_floor=0.15,
            stage="offspring",
            fanout_workers=1 if direct_single_batch else None,
        ),
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        nonlocal saw_model_items
        batch_policy = _policy_for_generation_batch(
            policy,
            batch_index=batch_index,
            accepted_signatures=[c.metadata.get("dedupe_signature", "") for c in accepted],
            accepted_candidates=accepted,
            rejected=rejected,
            kind="offspring",
        )
        batch_policy.metadata["requested_candidate_count"] = max(1, target - len(target_qualified_candidates(accepted)))
        raw = call_with_optional_context(
            model.generate_offspring,
            plans=plans,
            parents=parents,
            world=world,
            contract=contract,
            policy=batch_policy,
            provided_context=provided_context,
        )
        raw_items = list(raw or [])
        saw_model_items = saw_model_items or bool(raw_items)
        model_offspring = [_candidate_from_model_offspring(item) for item in raw_items if isinstance(item, (CandidateGenome, dict))]
        if model_offspring:
            _merge_plan_metadata_into_model_offspring(model_offspring, plans, parents)
        return model_offspring

    result = harvester.harvest(
        request_batch=_request,
        context={"contract": contract, "policy": policy, "world": world},
        recoverable_errors=MODEL_BOUNDARY_ERRORS,
    )
    harvest_error = result.fatal_model_error
    if result.accepted:
        if harvest_outcome is not None:
            harvest_outcome.update(result.to_dict())
            harvest_outcome["status"] = "accepted"
        offspring = list(result.accepted)
        for candidate in offspring:
            candidate.metadata.setdefault("offspring_harvest", result.to_dict())
            if harvest_error is not None:
                candidate.metadata.setdefault("partial_model_offspring_error", f"{harvest_error.__class__.__name__}: {harvest_error}")
            elif result.recoverable_batch_errors:
                candidate.metadata.setdefault(
                    "partial_model_offspring_error",
                    "; ".join(f"{item.get('error_type')}: {item.get('error')}" for item in result.recoverable_batch_errors[:3]),
                )
        return offspring
    if harvest_error is not None:
        raise harvest_error
    if not saw_model_items:
        if harvest_outcome is not None:
            harvest_outcome.update(result.to_dict())
            harvest_outcome["status"] = "model_abstained"
        return []
    rejection_reasons = {str(item.get("reason") or "") for item in result.rejected}
    if rejection_reasons and rejection_reasons <= {"duplicate_materialized_artifact"}:
        if harvest_outcome is not None:
            harvest_outcome.update(result.to_dict())
            harvest_outcome["status"] = "duplicate_exhausted"
        return []
    raise ModelResponseSchemaError("nexus_generate_offspring returned no valid offspring")


def _generate_slot_offspring(
    *,
    model: NexusOffspringModelProtocol,
    parents: list[CandidateGenome],
    plans: list[MutationPlan],
    branch_slots: list[dict[str, Any]],
    world: Any,
    contract: NexusObjectiveContract,
    policy: EvolutionPolicy,
    candidate_pool: list[CandidateGenome],
    provided_context: dict[str, Any] | None,
    harvest_outcome: dict[str, Any] | None,
) -> list[CandidateGenome]:
    parent_by_id = {parent.id: parent for parent in parents}
    source_plan = plans[0]
    parent_plan_id = str((source_plan.metadata or {}).get("plan_id") or "runtime-lineage-envelope")

    def _request_slot(slot: dict[str, Any]) -> tuple[CandidateGenome | None, Exception | None]:
        slot_id = str(slot["slot_id"])
        parent_id = str(slot["parent_id"])
        parent = parent_by_id[parent_id]
        slot_plan_id = stable_hash({"parent_plan_id": parent_plan_id, "slot_id": slot_id})[:20]
        slot_metadata = {
            **dict(source_plan.metadata or {}),
            "parent_plan_id": parent_plan_id,
            "plan_id": slot_plan_id,
            "branch_slots": [dict(slot)],
        }
        slot_plan = MutationPlan.from_dict(
            {
                **source_plan.to_dict(),
                "parent_ids": [parent_id],
                "metadata": slot_metadata,
            }
        )
        slot_policy = EvolutionPolicy.from_dict(policy.to_dict())
        slot_policy.metadata["requested_candidate_count"] = 1
        slot_policy.metadata["offspring_slot_id"] = slot_id
        sampling_policy = _slot_sampling_policy(policy, slot)
        try:
            with logical_llm_call(
                f"{parent_plan_id}/{slot_id}",
                template_version=_OFFSPRING_PROMPT_TEMPLATE_VERSION,
                request_policy=sampling_policy,
            ):
                raw = call_with_optional_context(
                    model.generate_offspring,
                    plans=[slot_plan],
                    parents=[parent],
                    world=world,
                    contract=contract,
                    policy=slot_policy,
                    provided_context=provided_context,
                )
            raw_items = list(raw or [])
            if len(raw_items) != 1 or not isinstance(raw_items[0], (CandidateGenome, dict)):
                raise ModelResponseSchemaError(f"offspring slot {slot_id} must return exactly one candidate")
            candidate = _candidate_from_model_offspring(raw_items[0])
            _merge_plan_metadata_into_model_offspring([candidate], [slot_plan], [parent])
            return candidate, None
        except MODEL_BOUNDARY_ERRORS as exc:
            return None, exc

    slot_results = run_ordered_fanout(branch_slots, _request_slot, thread_name_prefix="cogev-offspring-slot")
    successful = [candidate for candidate, error in slot_results if candidate is not None and error is None]
    errors = [error for candidate, error in slot_results if candidate is None and error is not None]
    if not successful:
        raise errors[0]

    existing_candidates = list(candidate_pool)
    existing_ids = {candidate.id for candidate in existing_candidates}
    existing_candidates.extend(parent for parent in parents if parent.id not in existing_ids)
    harvester = CandidateHarvester(
        deduper=CandidateDeduper(existing_candidates),
        policy=HarvestPolicy(
            target_size=len(successful),
            max_batches=1,
            min_batches=1,
            relevance_floor=0.15,
            stage="offspring",
            fanout_workers=1,
        ),
    )
    result = harvester.harvest(
        request_batch=lambda _batch, _accepted, _rejected: list(successful),
        context={"contract": contract, "policy": policy, "world": world},
    )
    if harvest_outcome is not None:
        harvest_outcome.update(result.to_dict())
        harvest_outcome["status"] = "accepted" if result.accepted else "duplicate_exhausted"
        harvest_outcome["slot_errors"] = [f"{error.__class__.__name__}: {error}" for error in errors]
    if errors:
        summary = "; ".join(f"{error.__class__.__name__}: {error}" for error in errors)
        for candidate in result.accepted:
            candidate.metadata["partial_model_offspring_error"] = summary
    return list(result.accepted)


def _slot_sampling_policy(policy: EvolutionPolicy, slot: dict[str, Any]) -> LLMRequestPolicy | None:
    configured = (policy.metadata or {}).get("slot_sampling_profiles")
    if not isinstance(configured, dict):
        return None
    profiles = configured.get(str(slot.get("intent") or ""))
    if profiles in (None, []):
        return None
    if not isinstance(profiles, list):
        raise ValueError("slot_sampling_profiles intent value must be a list")
    profile = profiles[int(slot.get("variation_index") or 0) % len(profiles)]
    if not isinstance(profile, dict):
        raise ValueError("slot_sampling_profiles entries must be objects")
    return LLMRequestPolicy(
        temperature=float(profile["temperature"]) if profile.get("temperature") is not None else None,
        top_p=float(profile["top_p"]) if profile.get("top_p") is not None else None,
        seed=int(profile["seed"]) if profile.get("seed") is not None else None,
    )


def _deterministic_fallback_offspring(
    *,
    mutation_engine: MutationEngine,
    parents: list[CandidateGenome],
    plans: list[MutationPlan],
    candidate_pool: list[CandidateGenome],
    ca_config: CACrossoverConfig | None,
) -> list[CandidateGenome]:
    offspring: list[CandidateGenome] = []
    for index, plan in enumerate(plans):
        parent = _parent_for_plan(plan, parents, index)
        if parent is None:
            continue
        if str(plan.operator) == MutationOperator.CROSSOVER:
            child = _deterministic_crossover_child(parent=parent, plan=plan, parents=parents, candidate_pool=candidate_pool, ca_config=ca_config)
            if child is not None:
                _bind_deterministic_branch_slot(child, plan)
                offspring.append(child)
                continue
        child = mutation_engine.mutate(parent, plan)
        _bind_deterministic_branch_slot(child, plan)
        offspring.append(child)
    return offspring


def _bind_deterministic_branch_slot(candidate: CandidateGenome, plan: MutationPlan) -> None:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    plan_metadata = plan.metadata if isinstance(plan.metadata, dict) else {}
    if plan_metadata.get("branch_slot_binding_status") != "planned":
        return
    arm_id = str(plan_metadata.get("branch_arm_id") or "")
    parent_id = str(plan_metadata.get("branch_slot_parent_id") or "")
    root_id = str(candidate.lineage[0] if candidate.lineage else candidate.id)
    if candidate.parent_ids and candidate.parent_ids[0] == parent_id and arm_id == root_id:
        metadata["branch_slot_binding_status"] = "bound"
    else:
        metadata["branch_slot_binding_status"] = "unbound_parent_mismatch"
        metadata.pop("branch_slot_id", None)
        metadata.pop("branch_arm_id", None)
        metadata.pop("branch_slot_parent_id", None)
        metadata.pop("branch_intent", None)
    candidate.metadata = metadata


def _deterministic_crossover_child(
    *,
    parent: CandidateGenome,
    plan: MutationPlan,
    parents: list[CandidateGenome],
    candidate_pool: list[CandidateGenome],
    ca_config: CACrossoverConfig | None,
) -> CandidateGenome | None:
    by_id = {candidate.id: candidate for candidate in [*candidate_pool, *parents]}
    partner: CandidateGenome | None = None
    for parent_id in plan.parent_ids:
        candidate = by_id.get(parent_id)
        if candidate is not None and candidate.id != parent.id:
            partner = candidate
            break
    if partner is None:
        partner = neighborhood_crossover_partner(parent, list(by_id.values()), ca_config)
    if partner is None or partner.id == parent.id:
        return None
    child = crossover(parent, partner, instruction=plan.instruction or "descriptor-neighborhood crossover")
    if not isinstance(child.metadata, dict):
        child.metadata = {}
    child.metadata.update(dict(plan.metadata or {}))
    child.metadata["ca_crossover"] = {
        "parent_ids": [parent.id, partner.id],
        "selection": "descriptor_neighborhood_or_configured_global_donor",
        "operator": MutationOperator.CROSSOVER,
    }
    return child



def _policy_for_generation_batch(
    policy: EvolutionPolicy,
    *,
    batch_index: int,
    accepted_signatures: list[str],
    rejected: list[dict[str, Any]],
    kind: str,
    accepted_candidates: list[CandidateGenome] | None = None,
) -> EvolutionPolicy:
    data = policy.to_dict()
    metadata = dict(data.get("metadata") or {})
    metadata.update(
        {
            f"{kind}_batch_index": batch_index,
            f"accepted_{kind}_signatures": [sig for sig in accepted_signatures[-16:] if sig],
            f"rejected_{kind}_count": len(rejected),
            f"rejected_{kind}_feedback": [
                {
                    key: item.get(key)
                    for key in ("batch", "candidate_id", "reason", "signature", "operator", "candidate")
                    if item.get(key) is not None
                }
                for item in rejected[-16:]
            ],
            f"{kind}_instruction": "Produce alternatives that land in new descriptor cells and avoid accepted signatures; do not merely paraphrase.",
            "search_kernel_skills": search_skill_payload(limit=4),
        }
    )
    if kind == "offspring" and accepted_candidates:
        metadata["accepted_offspring_candidates"] = [
            {
                "id": candidate.id,
                "artifact": candidate_materialized_artifact(candidate),
                "artifact_type": candidate.artifact_type,
                "concise_claim": candidate.concise_claim,
                "core_mechanism": candidate.core_mechanism,
            }
            for candidate in accepted_candidates
        ]
    data["metadata"] = metadata
    return EvolutionPolicy.from_dict(data)


def _mutation_plan_batch_limit(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_MUTATION_PLAN_BATCH_LIMIT", maximum=16)
    if configured:
        return configured
    return 1


def _mutation_plan_min_batches(target: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_MUTATION_PLAN_MIN_BATCHES"))
    if configured:
        return max(1, min(_mutation_plan_batch_limit(target), configured))
    return 1


def _mutation_plan_low_gain_patience(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_MUTATION_PLAN_LOW_GAIN_PATIENCE", maximum=8)
    if configured:
        return configured
    return 2 if target <= 4 else 3


def _offspring_batch_limit(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", maximum=16)
    if configured:
        return configured
    return 1


def _offspring_min_batches(target: int) -> int:
    configured = _positive_int(os.environ.get("COGEV_NEXUS_OFFSPRING_MIN_BATCHES"))
    if configured:
        return max(1, min(_offspring_batch_limit(target), configured))
    return 1


def _offspring_low_gain_patience(target: int) -> int:
    configured = _bounded_env_int("COGEV_NEXUS_OFFSPRING_LOW_GAIN_PATIENCE", maximum=8)
    if configured:
        return configured
    return 2 if target <= 4 else 3


def _bounded_env_int(name: str, *, maximum: int) -> int | None:
    configured = _positive_int(os.environ.get(name))
    if configured:
        return min(maximum, configured)
    return None


def _candidate_from_model_offspring(item: CandidateGenome | dict[str, Any]) -> CandidateGenome:
    data = demote_model_candidate_runtime_payload(item.to_dict() if isinstance(item, CandidateGenome) else dict(item))
    metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
    claimed_id = str(data.pop("id", "") or "").strip()
    metadata.pop("model_claimed_candidate_id", None)
    if claimed_id:
        metadata["model_claimed_candidate_id"] = claimed_id
    data["metadata"] = metadata
    return candidate_from_dict(data)


def _demote_model_offspring_runtime_controls(candidate: CandidateGenome) -> None:
    metadata = dict(candidate.metadata or {})
    controls = dict(metadata.pop("model_claimed_runtime_controls", {}))
    if CandidateFate.normalize(candidate.current_fate) != CandidateFate.ACTIVE.value:
        controls["current_fate"] = candidate.current_fate
    candidate.current_fate = CandidateFate.ACTIVE.value
    candidate.verification_result = {}
    candidate.verification_trace = []
    patch_result = getattr(candidate, "patch_application_result", None)
    if patch_result:
        controls["patch_application_result"] = patch_result
        candidate.patch_application_result = {}
    if controls:
        metadata["model_claimed_runtime_controls"] = controls
    candidate.metadata = metadata

def _merge_plan_metadata_into_model_offspring(offspring: list[CandidateGenome], plans: list[MutationPlan], parents: list[CandidateGenome] | None = None) -> None:
    if not plans:
        if offspring:
            raise ModelResponseSchemaError("model offspring cannot be bound without mutation plans")
        return
    parents = parents or []
    parent_by_id = {parent.id: parent for parent in parents}
    plan_by_id = {str((plan.metadata or {}).get("plan_id") or (plan.metadata or {}).get("id") or ""): plan for plan in plans if isinstance(plan.metadata, dict)}
    branch_slots = _branch_slots_from_plans(plans)
    used_branch_slot_ids: set[str] = set()
    for candidate in offspring:
        if not _artifact_has_content(candidate.artifact):
            raise ModelResponseSchemaError(f"model offspring {candidate.id} is missing a concrete artifact")
        if not candidate.parent_ids:
            raise ModelResponseSchemaError(f"model offspring {candidate.id} is missing parent_ids")
        plan = _plan_for_model_offspring(candidate, plans=plans, plan_by_id=plan_by_id)
        if not isinstance(candidate.metadata, dict):
            candidate.metadata = {}
        if plan is None:
            claimed_parent_ids = list(dict.fromkeys(coerce_str_list(candidate.parent_ids)))
            if (
                not claimed_parent_ids
                or any(parent_id not in parent_by_id for parent_id in claimed_parent_ids)
                or not _artifact_differs_from_parents(candidate.artifact, claimed_parent_ids, parent_by_id)
            ):
                raise ModelResponseSchemaError(f"model offspring {candidate.id} cannot be resolved to one mutation plan")
            plan = MutationPlan(
                operator="ModelDirected",
                parent_ids=claimed_parent_ids,
                instruction="",
                metadata={"unplanned_model_variation": True, "plan_binding_status": "unplanned_model_variation"},
            )
        claimed_plan_id = str(candidate.metadata.get("plan_id") or candidate.metadata.get("mutation_plan_id") or "").strip()
        authoritative_plan_id = str((plan.metadata or {}).get("plan_id") or (plan.metadata or {}).get("id") or "").strip()
        if claimed_plan_id and claimed_plan_id != authoritative_plan_id:
            candidate.metadata["model_claimed_plan_id"] = claimed_plan_id
        candidate.metadata.pop("mutation_plan_id", None)
        if authoritative_plan_id:
            candidate.metadata["plan_id"] = authoritative_plan_id
        else:
            candidate.metadata.pop("plan_id", None)
        claimed_plan_source = str(candidate.metadata.get("plan_source") or "").strip()
        authoritative_plan_source = str((plan.metadata or {}).get("plan_source") or "").strip()
        if claimed_plan_source and claimed_plan_source != authoritative_plan_source:
            candidate.metadata["model_claimed_plan_source"] = claimed_plan_source
        if authoritative_plan_source:
            candidate.metadata["plan_source"] = authoritative_plan_source
        else:
            candidate.metadata.pop("plan_source", None)
        lineage_envelope = authoritative_plan_source == "runtime_lineage_envelope"
        if lineage_envelope:
            for key in _LINEAGE_ENVELOPE_METADATA_KEYS:
                candidate.metadata.pop(key, None)
        else:
            for key, value in (plan.metadata or {}).items():
                if key not in {"id", "model_claimed_runtime_controls", "plan_id", "plan_source", "branch_slots"}:
                    candidate.metadata.setdefault(key, value)
        _merge_edge_lineage_fields(candidate, plan.metadata or {})
        claimed_parent_ids = list(dict.fromkeys(_offspring_parent_ids(candidate, plan)))
        plan_parent_ids = list(dict.fromkeys(str(item) for item in plan.parent_ids if item is not None and str(item).strip()))
        if lineage_envelope:
            alias_targets: dict[str, list[str]] = {}
            for parent_id in plan_parent_ids:
                parent = parent_by_id.get(parent_id)
                alias = str((parent.metadata or {}).get("model_claimed_candidate_id") or "").strip() if parent else ""
                if alias:
                    alias_targets.setdefault(alias, []).append(parent_id)
            alias_to_parent = {alias: ids[0] for alias, ids in alias_targets.items() if len(ids) == 1}
            canonical_parent_ids = list(
                dict.fromkeys(
                    parent_id if parent_id in plan_parent_ids else alias_to_parent.get(parent_id, parent_id)
                    for parent_id in claimed_parent_ids
                )
            )
            if canonical_parent_ids != claimed_parent_ids:
                candidate.metadata["model_claimed_parent_ids"] = claimed_parent_ids
                claimed_parent_ids = canonical_parent_ids
        if not claimed_parent_ids or not set(claimed_parent_ids).issubset(set(plan_parent_ids)):
            raise ModelResponseSchemaError(f"model offspring {candidate.id} has parent_ids outside its mutation plan")
        bound_parents = [parent_by_id[item] for item in claimed_parent_ids if item in parent_by_id]
        if len(bound_parents) != len(claimed_parent_ids):
            raise ModelResponseSchemaError(f"model offspring {candidate.id} references an unavailable parent")
        candidate.parent_ids = claimed_parent_ids
        candidate.generation = max(parent.generation for parent in bound_parents) + 1
        _demote_model_offspring_runtime_controls(candidate)
        _bind_branch_slot(
            candidate,
            branch_slots=branch_slots,
            used_slot_ids=used_branch_slot_ids,
            parent_by_id=parent_by_id,
        )
        candidate.lineage = list(dict.fromkeys([item for parent in bound_parents for item in parent.lineage] + [candidate.id]))
        _merge_parent_edge_lineage(candidate, bound_parents)


def _branch_slots_from_plans(plans: list[MutationPlan]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in plans:
        raw = (plan.metadata or {}).get("branch_slots")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            slot_id = str(item.get("slot_id") or "")
            if slot_id and slot_id not in seen:
                slots.append(dict(item))
                seen.add(slot_id)
    return slots


def _bind_branch_slot(
    candidate: CandidateGenome,
    *,
    branch_slots: list[dict[str, Any]],
    used_slot_ids: set[str],
    parent_by_id: dict[str, CandidateGenome],
) -> None:
    if not branch_slots:
        return
    metadata = candidate.metadata
    claimed_id = str(metadata.get("branch_slot_id") or "")
    primary_parent_id = str(candidate.parent_ids[0] if candidate.parent_ids else "")
    primary_parent = parent_by_id.get(primary_parent_id)
    primary_arm_id = str(primary_parent.lineage[0] if primary_parent and primary_parent.lineage else primary_parent_id)
    by_id = {str(item.get("slot_id") or ""): item for item in branch_slots}
    selected = by_id.get(claimed_id) if claimed_id else next(
        (
            item
            for item in branch_slots
            if str(item.get("slot_id") or "") not in used_slot_ids
            and str(item.get("parent_id") or "") == primary_parent_id
            and str(item.get("arm_id") or "") == primary_arm_id
        ),
        None,
    )
    selected_id = str((selected or {}).get("slot_id") or "")
    parent_matches = bool(selected and str(selected.get("parent_id") or "") == primary_parent_id)
    arm_matches = bool(selected and str(selected.get("arm_id") or "") == primary_arm_id)
    claimed_controls = dict(metadata.get("model_claimed_runtime_controls") or {})
    for key in (*_TRUSTED_BRANCH_DIRECTIVE_KEYS, "branch_slot_directive"):
        if key in metadata:
            claimed_controls.setdefault(key, metadata.pop(key))
    if claimed_controls:
        metadata["model_claimed_runtime_controls"] = claimed_controls
    if selected is None or selected_id in used_slot_ids or not parent_matches or not arm_matches:
        metadata["branch_slot_binding_status"] = "unbound_parent_mismatch"
        if claimed_id:
            metadata["model_claimed_branch_slot_id"] = claimed_id
        metadata.pop("branch_slot_id", None)
        metadata.pop("branch_arm_id", None)
        metadata.pop("branch_slot_parent_id", None)
        metadata.pop("branch_intent", None)
        return
    used_slot_ids.add(selected_id)
    metadata["branch_slot_id"] = selected_id
    metadata["branch_arm_id"] = str(selected.get("arm_id") or "")
    metadata["branch_slot_parent_id"] = str(selected.get("parent_id") or "")
    metadata["branch_intent"] = str(selected.get("intent") or "")
    metadata["branch_slot_binding_status"] = "bound"
    if selected.get("island_id") is not None:
        if "island_id" in metadata and metadata.get("island_id") != selected.get("island_id"):
            metadata["model_claimed_island_id"] = metadata.get("island_id")
        metadata["island_id"] = int(selected["island_id"])
    directive = selected.get("directive") if isinstance(selected.get("directive"), dict) else {}
    if directive:
        metadata["branch_slot_directive"] = dict(directive)
    for key in _TRUSTED_BRANCH_DIRECTIVE_KEYS:
        if key in directive:
            metadata[key] = directive[key]


def _artifact_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_artifact_has_content(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_artifact_has_content(item) for item in value)
    return True


def _plan_for_model_offspring(candidate: CandidateGenome, *, plans: list[MutationPlan], plan_by_id: dict[str, MutationPlan]) -> MutationPlan | None:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    plan_id = str(metadata.get("plan_id") or metadata.get("mutation_plan_id") or "").strip()
    if plan_id and plan_id in plan_by_id:
        return plan_by_id[plan_id]
    for parent_ids in (
        set(coerce_str_list(getattr(candidate, "parent_ids", []))),
        set(coerce_str_list(metadata.get("parent_ids")) + coerce_str_list(metadata.get("parent_id"))),
    ):
        if not parent_ids:
            continue
        matches = [plan for plan in plans if parent_ids.intersection(set(str(item) for item in plan.parent_ids))]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            eligible_parent_ids = list(dict.fromkeys(str(item) for item in candidate.parent_ids if item is not None and str(item).strip()))
            if eligible_parent_ids and all(
                parent_id in {str(item) for plan in matches for item in plan.parent_ids}
                for parent_id in eligible_parent_ids
            ):
                plan_ids = [str((plan.metadata or {}).get("plan_id") or (plan.metadata or {}).get("id") or "") for plan in matches]
                return MutationPlan(
                    operator="ModelDirected",
                    parent_ids=eligible_parent_ids,
                    instruction="",
                    metadata={
                        "ambiguous_plan_binding": True,
                        "plan_binding_status": "ambiguous_parent_lineage_only",
                        "candidate_plan_ids": [item for item in plan_ids if item],
                    },
                )
    if len(plans) == 1:
        return plans[0]
    return None


def _artifact_differs_from_parents(artifact: Any, parent_ids: list[str], parent_by_id: dict[str, CandidateGenome]) -> bool:
    return all(artifact != parent_by_id[parent_id].artifact for parent_id in parent_ids if parent_id in parent_by_id)


def _offspring_parent_ids(candidate: CandidateGenome, plan: MutationPlan) -> list[str]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    ids = coerce_str_list(getattr(candidate, "parent_ids", [])) or coerce_str_list(metadata.get("parent_ids")) or coerce_str_list(metadata.get("parent_id")) or coerce_str_list(plan.parent_ids)
    return list(dict.fromkeys(str(item) for item in ids if item is not None and str(item).strip()))


def _merge_parent_edge_lineage(candidate: CandidateGenome, parents: list[CandidateGenome]) -> None:
    if not parents:
        return
    for parent in parents:
        _merge_edge_lineage_fields(candidate, parent)


def _merge_edge_lineage_fields(candidate: CandidateGenome, source: Any) -> None:
    for attr in ("edge_knowledge_seeds", "inherited_genes", "novelty_descriptors", "niche_memberships"):
        current = list(getattr(candidate, attr, []) or [])
        merged = list(current)
        values = source.get(attr) if isinstance(source, dict) else getattr(source, attr, [])
        for item in coerce_str_list(values):
            if item not in merged:
                merged.append(item)
        setattr(candidate, attr, merged)


def _positive_int(value: Any) -> int | None:
    return positive_int(value)


def _best_auxiliary_id(candidates: list[CandidateGenome]) -> str:
    auxiliary = [c for c in candidates if c.current_fate == CandidateFate.AUXILIARY or c.multihead_scores.get("auxiliary_value", 0.0) > 0]
    if not auxiliary:
        return ""
    return max(auxiliary, key=lambda c: c.multihead_scores.get("auxiliary_value", 0.0)).id


__all__ = ["_best_auxiliary_id", "_generate_offspring", "_merge_plan_metadata_into_model_offspring", "_plan_mutations"]
