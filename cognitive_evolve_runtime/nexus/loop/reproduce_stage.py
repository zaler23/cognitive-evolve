"""Reproduction half of one Nexus evolution round."""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

from cognitive_evolve_runtime.archives.quality_diversity import candidate_bin_key
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, evidence_advisory_features
from cognitive_evolve_runtime.evaluators.evidence import evaluator_selection_key, select_preliminary_incumbent
from cognitive_evolve_runtime.nexus.critique import CandidateCritique
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, assert_stage_ready, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.nextgen import ensure_nextgen_identity, structurally_blocked
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.prompt_view import archive_prompt_view
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.receipts import DONOR_ROLES, record_reproduction_receipts
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_outcome_signature, candidate_phenotype_signature
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import ProductiveBranchAllocation
from cognitive_evolve_runtime.nexus.search_kernel.islands import allocate_logical_islands, assign_candidate_islands, derive_island_count
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    apply_latent_exploration_to_mutation_plans,
    latent_exploration_plan_for_contract,
)
from cognitive_evolve_runtime.nexus.reproduction import (
    dedupe_offspring_against_population,
    elite_gap_merge_offspring,
    parents_for_crossover,
    ranked_repair_fallback_parents,
    sync_repair_parent_attempts_to_dormant_archive,
    verify_offspring,
)
from cognitive_evolve_runtime.theory import build_population_representation
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig, V23TheoryRuntimeConfig
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult

from .offspring import _generate_offspring, _plan_mutations, _slot_sampling_policy
from .policy_directives import _attach_policy_directives_to_plans, _critique_actions
from .stage_helpers import _eligibility_policy, _theory_config_from_policy


_REQUIRED_CONTRIBUTION_MAP = {
    "generic_space_mapping": "element-level shared structural role mapping",
    "retained_from_primary": "primary elements retained by the child",
    "borrowed_from_donor": "donor elements materially present in the child",
    "structural_correspondence": "cross-parent structural relation",
    "emergent_delta": "child-only structure",
    "incompatibilities": "mapping conflicts",
    "unresolved_obligations": "open merge obligations",
}


class ReproduceStage:
    def reproduce(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        critiques: list[CandidateCritique],
        offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None,
        repair_parent_candidates: list[CandidateGenome] | None = None,
        provided_context: dict[str, Any] | None = None,
        context_provider: Callable[[list[CandidateGenome], str], dict[str, Any] | None] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        plan = GenerationPlan.from_dict(self.last_generation_plan) if self.last_generation_plan else None
        completed_stage_ops = list(self.last_completed_stage_ops or self.last_generation_plan.get("completed_stage_ops") or [])
        if plan is not None:
            assert_stage_ready(plan, "select_parents", completed_stage_ops)
        island_config = dict(policy.metadata.get("islands") or {}) if isinstance(policy.metadata, dict) and isinstance(policy.metadata.get("islands"), dict) else {}
        coverage_floor = diagnosis.metadata.get("axis_family_coverage_floor") if isinstance(diagnosis.metadata, dict) else None
        if isinstance(coverage_floor, dict):
            island_config["coverage_floor_targets"] = list(coverage_floor.get("targets") or [])
            island_config["coverage_floor_slots"] = int(coverage_floor.get("floor_slots") or 0)
        archive_admission, archive_admission_audit = _archive_elite_admission_view(
            population=population.candidates,
            archives=archives,
            limit=_archive_elite_reentry_limit(policy),
            current_round=current_round,
        )
        active_lineages = {
            candidate.lineage[0] if candidate.lineage else candidate.id
            for candidate in population.candidates
            if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}
        }
        island_count = derive_island_count(
            total_slots=self._branch_limit(),
            lineage_count=len(active_lineages),
            configured=island_config.get("count", "auto"),
        )
        parents = self._select_reproduction_parents_by_island(
            island_count=island_count,
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            rankings=rankings,
            diagnosis=diagnosis,
            repair_parent_candidates=repair_parent_candidates,
            archive_admission_candidates=archive_admission,
        )
        if not parents:
            return "no_parents_available", [], {}
        evaluator_spec = EvaluatorSpec.from_mapping(dict(self.adaptive.config.evaluator or {}))
        metric_directions = {item.name: item.direction for item in evaluator_spec.metrics}
        if island_count > 1:
            island_config["count"] = island_count
        island_allocation = allocate_logical_islands(
            parents=parents,
            candidates=_branch_credit_candidates(
                population=population,
                archives=archives,
                repair_parent_candidates=repair_parent_candidates,
            ),
            budget_history=self.budget.history,
            metric_directions=metric_directions,
            total_slots=self._branch_limit(),
            config=island_config,
            current_round=current_round,
        )
        branch_allocation = island_allocation.branches
        for candidate in population.candidates:
            if candidate.id in island_allocation.candidate_islands:
                candidate.metadata["island_id"] = island_allocation.candidate_islands[candidate.id]
        parent_by_id = {parent.id: parent for parent in parents}
        allocated_parent_ids = list(dict.fromkeys(slot.parent_id for slot in branch_allocation.slots))
        parents = [parent_by_id[parent_id] for parent_id in allocated_parent_ids]
        archive_admission_audit["selected_candidate_ids"] = [
            parent.id
            for parent in parents
            if bool(parent.metadata.get("archive_elite_reentry"))
        ]
        if plan is not None:
            completed_stage_ops.append("select_parents")
            self.last_generation_plan["parent_ids"] = [parent.id for parent in parents]
            self.last_generation_plan["productive_branch_allocation"] = branch_allocation.to_dict()
            self.last_generation_plan["logical_islands"] = island_allocation.to_dict()
            self.last_generation_plan["archive_elite_reentry"] = archive_admission_audit
            if isinstance(coverage_floor, dict):
                self.last_generation_plan["axis_family_coverage_floor"] = {
                    **coverage_floor,
                    "reserved_slot_ids": list(branch_allocation.coverage_floor.get("reserved_slot_ids") or []),
                }
            self._refresh_generation_plan_id()
            self._record_generation_stage_progress(completed_stage_ops)
        generation_policy = EvolutionPolicy.from_dict(policy.to_dict())
        generation_policy.metadata["productive_branch_allocation"] = branch_allocation.to_dict()
        generation_policy.metadata["slot_islands"] = dict(island_allocation.slot_islands)
        generation_policy.metadata["requested_candidate_count"] = len(branch_allocation.slots)
        slot_sampling_profiles = []
        for slot in branch_allocation.slots:
            sampling = _slot_sampling_policy(generation_policy, slot.to_dict())
            if sampling is not None:
                slot_sampling_profiles.append(
                    {
                        "slot_id": slot.slot_id,
                        "intent": slot.intent,
                        "search_phase": sampling.search_phase,
                        "sampling_profile_id": sampling.sampling_profile_id,
                        "temperature": sampling.temperature,
                        "top_p": sampling.top_p,
                        "seed": sampling.seed,
                    }
                )
        if plan is not None and slot_sampling_profiles:
            self.last_generation_plan["slot_sampling_profiles"] = slot_sampling_profiles
        model_backed_path = self.model is not None
        evaluator_led = self._evaluator_led()
        actions = action_palette_for_round(
            current_round,
            diagnosis.recommended_actions if diagnosis.stagnation_detected else _critique_actions(critiques),
        )
        latent_exploration_plan = latent_exploration_plan_for_contract(contract, limit=self._branch_limit())
        latent_actions = [str(item) for item in latent_exploration_plan.get("mutation_actions", []) if item]
        if latent_actions:
            actions = list(dict.fromkeys(latent_actions + actions))
        round_context = provided_context
        if context_provider is not None and not model_backed_path:
            round_context = context_provider(parents, "; ".join(actions)) or provided_context
        if plan is not None:
            assert_stage_ready(plan, "plan_mutations", completed_stage_ops)
        self._sync_model_context_controls()
        if model_backed_path:
            direct_plan, latent_exploration_plan = self._direct_model_plan(
                parents=parents,
                current_round=current_round,
                branch_allocation=branch_allocation,
                actions=actions,
                diagnosis=diagnosis,
                contract=contract,
                archives=archives,
                population=population.candidates,
                latent_exploration_plan=latent_exploration_plan,
                policy=generation_policy,
                evaluator_led=evaluator_led,
            )
            plans = [direct_plan]
            if context_provider is not None:
                slot_instructions = [
                    str((slot.get("directive") or {}).get("instruction") or "")
                    for slot in direct_plan.metadata.get("branch_slots", [])
                    if isinstance(slot, dict) and isinstance(slot.get("directive"), dict)
                ]
                context_instruction = "\n\n".join(
                    item for item in [direct_plan.instruction, *slot_instructions] if item
                )
                round_context = context_provider(parents, context_instruction) or provided_context
        else:
            plans = _plan_mutations(
                model=None,
                mutation_planner=self.mutation_planner,
                parents=parents,
                actions=actions,
                archives=archives,
                diagnosis=diagnosis,
                policy=generation_policy,
                provided_context=round_context,
                target_count=len(branch_allocation.slots),
            )
            plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(plans, contract, exploration=latent_exploration_plan)
            plans = self._apply_search_pressure_to_plans(plans, parents=parents)
        unallocated_plans: list[dict[str, Any]] = []
        if not model_backed_path:
            plans = _attach_branch_allocation_to_plans(
                plans,
                branch_allocation,
                include_manifest=False,
                rejected_out=unallocated_plans,
            )
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        if not model_backed_path:
            plans = self._apply_ca_crossover_to_plans(plans, parents=parents, population=population.candidates, config=v23_config.ca_crossover)
        if plan is not None:
            completed_stage_ops.append("plan_mutations")
            self.last_generation_plan["mutation_objectives"] = list(actions)
            self.last_generation_plan["mutation_plan_count"] = len(plans)
            if unallocated_plans:
                self.last_generation_plan["unallocated_mutation_plans"] = unallocated_plans
            if model_backed_path:
                self.last_generation_plan["mutation_plan_source"] = "runtime_lineage_envelope"
            if latent_exploration_plan:
                self.last_generation_plan["latent_exploration_planning"] = latent_exploration_plan
            self._refresh_generation_plan_id()
            self._record_generation_stage_progress(completed_stage_ops)
            assert_stage_ready(plan, "generate_offspring", completed_stage_ops)
        offspring = self._build_reproduction_offspring(
            current_round=current_round,
            parents=parents,
            plans=plans,
            population=population,
            archives=archives,
            policy=generation_policy,
            contract=contract,
            world=world,
            rankings=rankings,
            diagnosis=diagnosis,
            provided_context=round_context,
            requested_offspring_count=len(branch_allocation.slots),
        )
        duplicate_exhausted = self.last_offspring_harvest_outcome.get("status") == "duplicate_exhausted"
        model_abstained = self.model is not None and not offspring and not duplicate_exhausted
        duplicate_offspring: list[dict[str, Any]] = []
        offspring = dedupe_offspring_against_population(offspring, population, rejected_out=duplicate_offspring)
        activation_map = _cell_activation_map(parents=parents, plans=plans, offspring=offspring)
        if activation_map:
            self.adaptive.record_cell_activation_map(activation_map, round_index=current_round, history_limit=v23_config.ca_crossover.activation_history_limit)
        canonical_metrics = _canonical_family_metrics([*population.candidates, *offspring])
        if canonical_metrics:
            self.adaptive.record_canonical_family_metrics(canonical_metrics, round_index=current_round)
        if plan is not None:
            completed_stage_ops.append("generate_offspring")
            self.last_generation_plan["offspring_ids"] = [candidate.id for candidate in offspring]
            self.last_generation_plan["offspring_harvest"] = dict(self.last_offspring_harvest_outcome)
            self.last_generation_plan["offspring_transport"] = dict(self.last_offspring_harvest_outcome.get("transport") or {})
            self.last_generation_plan["duplicate_offspring"] = duplicate_offspring
            self.last_generation_plan["cell_activation_map"] = activation_map
            self._record_generation_stage_progress(completed_stage_ops)
        if not offspring:
            if duplicate_exhausted:
                return "no_new_unique_offspring", [], {}
            if model_abstained:
                incumbent = _accepted_preliminary_incumbent(population.candidates)
                if evaluator_led and incumbent is not None:
                    if plan is not None:
                        self.last_generation_plan["offspring_outcome"] = "model_abstained_empty_batch"
                    return "", [], {}
                raise ModelResponseSchemaError(
                    "nexus_generate_offspring returned an empty batch without an accepted preliminary incumbent"
                )
            return "no_new_unique_offspring", [], {}
        for candidate in offspring:
            candidate.metadata["created_in_round"] = current_round
        if plan is not None:
            assert_stage_ready(plan, "verify_offspring", completed_stage_ops)
        result = self._verify_and_integrate_offspring(
            offspring=offspring,
            offspring_verifier=offspring_verifier,
            population=population,
            archives=archives,
            policy=policy,
            current_round=current_round,
            generation_plan=plan,
            completed_stage_ops=completed_stage_ops,
        )
        if plan is not None:
            record_reproduction_receipts(
                self.last_generation_plan,
                diagnosis=diagnosis,
                mutation_plans=plans,
                offspring=offspring,
                outcomes=result[1],
            )
            self._refresh_generation_plan_id()
        return result

    def _record_generation_stage_progress(self, completed_stage_ops: list[str]) -> None:
        self.last_completed_stage_ops = list(completed_stage_ops)
        if self.last_generation_plan:
            self.last_generation_plan["completed_stage_ops"] = list(completed_stage_ops)

    def _refresh_generation_plan_id(self) -> None:
        if not self.last_generation_plan:
            return
        plan = GenerationPlan.from_dict(self.last_generation_plan)
        self.last_generation_plan["plan_id"] = expected_generation_plan_id(plan)

    def _branch_limit(self) -> int:
        return max(2, int(self.budget.branch_factor or 2))

    def _evaluator_led(self, evaluator_spec: EvaluatorSpec | None = None) -> bool:
        spec = evaluator_spec or EvaluatorSpec.from_mapping(dict(self.adaptive.config.evaluator or {}))
        return bool(self.adaptive.evaluator_enabled and spec.enabled)

    def _direct_model_plan(
        self,
        *,
        parents: list[CandidateGenome],
        current_round: int,
        branch_allocation: ProductiveBranchAllocation,
        actions: list[str],
        diagnosis: SearchDiagnosis,
        contract: NexusObjectiveContract,
        archives: ArchiveManager,
        population: list[CandidateGenome],
        latent_exploration_plan: dict[str, Any],
        policy: EvolutionPolicy,
        evaluator_led: bool,
    ) -> tuple[MutationPlan, dict[str, Any]]:
        parent_ids = list(dict.fromkeys(parent.id for parent in parents if parent.id))
        plan_id = "runtime-direct-" + stable_hash(
            {
                "round": current_round,
                "parent_ids": parent_ids,
                "slot_ids": [slot.slot_id for slot in branch_allocation.slots],
            }
        )[:16]
        action_palette = list(dict.fromkeys(str(action) for action in actions if str(action).strip()))
        policy_metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
        configured_quota = policy_metadata.get("crossover_slot_quota")
        crossover_quota = (
            max(0, int(configured_quota))
            if configured_quota is not None
            else (1 if diagnosis.stagnation_detected else 0)
        )
        crossover_slots = _role_crossover_slots(
            branch_allocation,
            parent_ids=parent_ids,
            quota=crossover_quota,
            roles=policy_metadata.get("crossover_donor_roles"),
        )
        slot_plans: list[MutationPlan] = []
        for index, slot in enumerate(branch_allocation.slots):
            crossover_slot = crossover_slots.get(slot.slot_id, {})
            slot_parent_ids = [slot.parent_id]
            if crossover_slot:
                slot_parent_ids.append(str(crossover_slot["donor_parent_id"]))
            instruction = (
                f"Branch intent: {slot.intent}. "
                + (
                    "Fill the zero-occupancy coverage target "
                    f"axis={slot.coverage_target.get('axis') or 'unchanged'}, "
                    f"family={slot.coverage_target.get('family') or 'unchanged'}. "
                    if slot.coverage_target
                    else ""
                )
                + (
                    f"Optional semantic direction: {action_palette[index % len(action_palette)]}. "
                    if action_palette
                    else ""
                )
                + "Choose the concrete mutation strategy that best advances the objective; the hint is not a fixed operator."
            )
            if crossover_slot:
                instruction += (
                    " This is a role-constrained crossover slot: materially combine the primary and donor, and return "
                    "metadata.blend_receipt with every field declared in required_contribution_map."
                )
            slot_plans.append(
                MutationPlan(
                    operator="ModelDirected",
                    parent_ids=slot_parent_ids,
                    instruction=instruction,
                ),
            )
        slot_plans = _attach_policy_directives_to_plans(slot_plans, policy, parents=parents)
        slot_plans, latent_exploration_plan = apply_latent_exploration_to_mutation_plans(
            slot_plans,
            contract,
            exploration=latent_exploration_plan,
        )
        slot_plans = self._apply_search_pressure_to_plans(slot_plans, parents=parents)
        branch_slots: list[dict[str, Any]] = []
        for index, (slot, slot_plan) in enumerate(zip(branch_allocation.slots, slot_plans)):
            plan_metadata = dict(slot_plan.metadata or {})
            runtime_keys = (
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
            directive = {
                "instruction": slot_plan.instruction,
                "action_hint": action_palette[index % len(action_palette)] if action_palette else "",
            }
            for key in runtime_keys:
                if key in plan_metadata:
                    directive[key] = plan_metadata[key]
            policy_directives = {
                key: value
                for key, value in plan_metadata.items()
                if key not in runtime_keys
            }
            if policy_directives:
                directive["policy_directives"] = policy_directives
            branch_slots.append(
                {
                    **slot.to_dict(),
                    **crossover_slots.get(slot.slot_id, {}),
                    "island_id": policy.metadata.get("slot_islands", {}).get(slot.slot_id),
                    "directive": directive,
                }
            )
        diagnosis_context = {
            "stagnation_detected": diagnosis.stagnation_detected,
            "stagnation_type": diagnosis.stagnation_type,
            "over_explored_families": list(diagnosis.over_explored_families),
            "under_explored_families": list(diagnosis.under_explored_families),
            "recommended_actions": list(diagnosis.recommended_actions),
            "notes": diagnosis.notes,
            "grounded_information_gain": dict(diagnosis.grounded_information_gain),
        }
        return MutationPlan(
            operator="ModelDirected",
            parent_ids=parent_ids,
            instruction=(
                "Directly evolve actual task artifacts from the allocated primary parents. "
                "For every branch slot return exactly one materially changed artifact, copy slot_id into metadata.branch_slot_id, "
                "and put slot.parent_id first in truthful parent_ids. A slot with donor_parent_id must receive both parents and "
                "must return a blend receipt; otherwise use only its primary parent. "
                "Actions and slot directives are semantic search guidance, not fixed operators; choose the mutation, transfer, "
                "representation, probe, or crossover strategy yourself. Return artifacts, not plans, commentary, or unchanged copies."
            ),
            metadata={
                "plan_id": plan_id,
                "plan_source": "runtime_lineage_envelope",
                "completion_mode": "complete_task_artifact_only" if evaluator_led else "concrete_progress_allowed",
                "semantic_mutation_owner": "model",
                "requested_candidate_count": len(branch_allocation.slots),
                "branch_slots": branch_slots,
                "action_palette": action_palette,
                "search_diagnosis": diagnosis_context,
                "latent_exploration": dict(latent_exploration_plan),
                "archive_search_memory": archive_prompt_view(archives, population=population),
            },
        ), latent_exploration_plan

    def _sync_model_context_controls(self) -> None:
        metadata = getattr(self.model, "metadata", None)
        if not isinstance(metadata, dict):
            return
        plan = self.adaptive.verification_plan_dict()
        if not plan:
            return
        metadata["prompt_context_controls"] = {"verification_plan": plan}

    def _select_reproduction_parents(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        repair_parent_candidates: list[CandidateGenome] | None,
        archive_admission_candidates: list[CandidateGenome] | None = None,
        limit_override: int | None = None,
    ) -> list[CandidateGenome]:
        limit = max(1, int(limit_override or self._branch_limit()))
        preliminary_incumbent = select_preliminary_incumbent(population.candidates)

        def _with_incumbent(selected: list[CandidateGenome]) -> list[CandidateGenome]:
            if preliminary_incumbent is None or structurally_blocked(preliminary_incumbent):
                return selected[:limit]
            ordered = (
                [*selected, preliminary_incumbent]
                if self.budget.search_phase == "explore"
                else [preliminary_incumbent, *selected]
            )
            return list({candidate.id: candidate for candidate in ordered}.values())[:limit]

        advisory_features = self._combined_advisory_features(policy=policy, candidates=population.candidates, current_round=current_round)
        selection_candidates = [*population.candidates, *(archive_admission_candidates or [])]
        if self.budget.search_phase == "explore" and repair_parent_candidates:
            selection_candidates = [*selection_candidates, *repair_parent_candidates]
        selection_candidates = list({candidate.id: candidate for candidate in selection_candidates}.values())
        parents = self.selector.select(
            selection_candidates,
            archives,
            limit=limit,
            eligibility_policy=_eligibility_policy(policy),
            advisory_features=advisory_features,
            search_phase=self.budget.search_phase,
        )
        if parents:
            return _with_incumbent(parents)
        parents = ranked_repair_fallback_parents(population.candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
        if parents:
            return _with_incumbent(parents)
        if repair_parent_candidates:
            parents = ranked_repair_fallback_parents(repair_parent_candidates, rankings=rankings, diagnosis=diagnosis, limit=limit, current_round=current_round)
            if parents:
                return _with_incumbent(parents)
        parents = recover_repairable_dormant_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents:
            return _with_incumbent(parents)
        parents = recover_failure_archive_repair_seeds(archives=archives, diagnosis=diagnosis, policy=policy, limit=limit, current_round=current_round)
        if parents or population.candidates:
            return _with_incumbent(parents)
        return _with_incumbent(emergency_activation_reseed(
            contract=contract,
            world=world,
            policy=policy,
            limit=max(1, min(2, limit)),
            current_round=current_round,
        ))

    def _select_reproduction_parents_by_island(
        self,
        *,
        island_count: int,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        repair_parent_candidates: list[CandidateGenome] | None,
        archive_admission_candidates: list[CandidateGenome] | None = None,
    ) -> list[CandidateGenome]:
        if island_count <= 1:
            return self._select_reproduction_parents(
                current_round=current_round,
                population=population,
                archives=archives,
                policy=policy,
                contract=contract,
                world=world,
                rankings=rankings,
                diagnosis=diagnosis,
                repair_parent_candidates=repair_parent_candidates,
                archive_admission_candidates=archive_admission_candidates,
            )
        all_candidates = [*population.candidates, *(repair_parent_candidates or []), *(archive_admission_candidates or [])]
        assignments = assign_candidate_islands(all_candidates, island_count=island_count)
        for candidate in all_candidates:
            candidate.metadata["island_id"] = assignments[candidate.id]
        base_slots, extra_slots = divmod(self._branch_limit(), island_count)
        selected: list[CandidateGenome] = []
        for island_id in range(island_count):
            island_population = CandidatePopulation(
                [candidate for candidate in population.candidates if assignments[candidate.id] == island_id]
            )
            island_repair = [
                candidate
                for candidate in repair_parent_candidates or []
                if assignments[candidate.id] == island_id
            ]
            island_admission = [
                candidate
                for candidate in archive_admission_candidates or []
                if assignments[candidate.id] == island_id
            ]
            parents = self._select_reproduction_parents(
                current_round=current_round,
                population=island_population,
                archives=archives,
                policy=policy,
                contract=contract,
                world=world,
                rankings=rankings,
                diagnosis=diagnosis,
                repair_parent_candidates=island_repair,
                archive_admission_candidates=island_admission,
                limit_override=base_slots + int(island_id < extra_slots),
            )
            if not parents:
                return self._select_reproduction_parents(
                    current_round=current_round,
                    population=population,
                    archives=archives,
                    policy=policy,
                    contract=contract,
                    world=world,
                    rankings=rankings,
                    diagnosis=diagnosis,
                    repair_parent_candidates=repair_parent_candidates,
                    archive_admission_candidates=archive_admission_candidates,
                )
            selected.extend(parents)
        return list({candidate.id: candidate for candidate in selected}.values())

    def _combined_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for source in (
            self._theory_advisory_features(policy=policy, candidates=candidates, current_round=current_round),
            evidence_advisory_features(candidates),
        ):
            for candidate_id, feature in (source or {}).items():
                current = dict(combined.get(candidate_id) or {})
                data = dict(feature) if isinstance(feature, dict) else {
                    "rank_prior": getattr(feature, "rank_prior", 0.0),
                    "plan_value": getattr(feature, "plan_value", 0.0),
                    "diversity": getattr(feature, "diversity", 0.0),
                    "risk": getattr(feature, "risk", 0.0),
                }
                for key in ("rank_prior", "plan_value", "diversity"):
                    current[key] = max(float(current.get(key, 0.0) or 0.0), float(data.get(key, 0.0) or 0.0))
                current["risk"] = max(float(current.get("risk", 0.0) or 0.0), float(data.get("risk", 0.0) or 0.0))
                combined[str(candidate_id)] = current
        return combined

    def _theory_advisory_features(self, *, policy: EvolutionPolicy, candidates: list[CandidateGenome], current_round: int) -> dict[str, Any]:
        config = _theory_config_from_policy(policy)
        if not config.enabled:
            return {}
        representation = build_population_representation(candidates, cycle_id=f"round:{current_round}")
        return self.theory_layer.advisory_features_for_population(representation, config=config)

    def _apply_search_pressure_to_plans(self, plans: list[MutationPlan], *, parents: list[CandidateGenome]) -> list[MutationPlan]:
        if not plans or not parents or not self.adaptive.enabled:
            return plans
        out: list[MutationPlan] = []
        by_id = {parent.id: parent for parent in parents}
        for index, plan in enumerate(plans):
            parent = None
            for parent_id in plan.parent_ids:
                parent = by_id.get(parent_id)
                if parent is not None:
                    break
            if parent is None:
                parent = parents[index % len(parents)]
            pressure = self.adaptive.compile_search_pressure(parent_id=parent.id, scope="candidate", parent=parent, candidates=parents)
            if pressure is None or not _search_pressure_has_effect(pressure):
                out.append(plan)
                continue
            metadata = dict(plan.metadata or {})
            metadata["search_pressure"] = pressure.to_dict()
            metadata["search_pressure_id"] = pressure.id
            if pressure.target_challenge_ids:
                metadata["target_challenge_ids"] = list(pressure.target_challenge_ids)
            metadata["artifact_policy"] = dict(pressure.artifact_requirements or {})
            instruction = plan.instruction
            if pressure.mutation_instruction and pressure.mutation_instruction not in instruction:
                instruction = (instruction.rstrip() + "\n\n" + pressure.mutation_instruction).strip()
            out.append(MutationPlan.from_dict({**plan.to_dict(), "instruction": instruction, "metadata": metadata}))
        return out

    def _apply_ca_crossover_to_plans(
        self,
        plans: list[MutationPlan],
        *,
        parents: list[CandidateGenome],
        population: list[CandidateGenome],
        config: CACrossoverConfig,
    ) -> list[MutationPlan]:
        if not plans or not parents:
            return plans
        by_id = {candidate.id: candidate for candidate in [*population, *parents]}
        out: list[MutationPlan] = []
        for index, plan in enumerate(plans):
            if str(plan.operator) != MutationOperator.CROSSOVER:
                out.append(plan)
                continue
            pivot = _parent_for_plan_id(plan, parents, index)
            if pivot is None:
                out.append(plan)
                continue
            partner = None
            for parent_id in plan.parent_ids:
                candidate = by_id.get(parent_id)
                if candidate is not None and candidate.id != pivot.id:
                    partner = candidate
                    break
            if partner is None:
                partner = neighborhood_crossover_partner(pivot, list(by_id.values()), config)
            if partner is None:
                out.append(plan)
                continue
            metadata = dict(plan.metadata or {})
            metadata["ca_crossover"] = {
                "pivot_id": pivot.id,
                "partner_id": partner.id,
                "selection": "descriptor_neighborhood_or_configured_global_donor",
            }
            parent_ids = list(dict.fromkeys([pivot.id, partner.id]))
            out.append(MutationPlan.from_dict({**plan.to_dict(), "parent_ids": parent_ids, "metadata": metadata}))
        return out

    def _build_reproduction_offspring(
        self,
        *,
        current_round: int,
        parents: list[CandidateGenome],
        plans: list[MutationPlan],
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        world: Any,
        rankings: RelativeRankingResult,
        diagnosis: SearchDiagnosis,
        provided_context: dict[str, Any] | None = None,
        requested_offspring_count: int | None = None,
    ) -> list[CandidateGenome]:
        sync_repair_parent_attempts_to_dormant_archive(archives, parents)
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        harvest_outcome: dict[str, Any] = {}
        offspring = _generate_offspring(
            model=self.model,
            mutation_engine=self.mutation_engine,
            parents=parents,
            plans=plans,
            world=world,
            contract=contract,
            policy=policy,
            candidate_pool=population.candidates,
            ca_config=v23_config.ca_crossover,
            provided_context=provided_context,
            target_size=requested_offspring_count,
            harvest_outcome=harvest_outcome,
            budget_history=self.budget.history,
        )
        self.last_offspring_harvest_outcome = harvest_outcome
        if self.model is None:
            if len(parents) >= 2 and rankings.crossover_pairs:
                first, second = parents_for_crossover(parents, rankings.crossover_pairs[0])
                ranked_child = crossover(first, second)
                ranked_child.metadata["ca_crossover"] = {
                    "parent_ids": [first.id, second.id],
                    "selection": "ranking_pair",
                    "operator": MutationOperator.CROSSOVER,
                }
                offspring.append(ranked_child)
            offspring.extend(elite_gap_merge_offspring(population.candidates, archives=archives, policy=policy, branch_factor=self.budget.branch_factor))
        reactivated = archives.reactivate_dormant() if "reactivate_dormant" in diagnosis.recommended_actions else None
        if reactivated:
            reactivated.metadata["reactivated_in_round"] = current_round
            reactivated.metadata.setdefault("created_in_round", current_round)
            offspring.append(reactivated)
        return offspring

    def _verify_and_integrate_offspring(
        self,
        *,
        offspring: list[CandidateGenome],
        offspring_verifier: Callable[[list[CandidateGenome]], list[Any]] | None,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        current_round: int,
        generation_plan: GenerationPlan | None = None,
        completed_stage_ops: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        offspring_verification = verify_offspring(offspring, offspring_verifier)
        reproduction_archive_updates: list[dict[str, Any]] = []
        if offspring_verification:
            failed_ids = {item.get("candidate_id") for item in offspring_verification if item.get("passed") is False}
            failed = [
                candidate
                for candidate in offspring
                if candidate.id in failed_ids and CandidateFate.normalize(candidate.current_fate) == CandidateFate.FAILED.value
            ]
            if failed:
                failed_assignments = archives.update({candidate.id: CandidateFate.FAILED for candidate in failed}, candidates=failed)
                reproduction_archive_updates.extend(
                    {
                        "candidate_id": assignment.candidate_id,
                        "fate": assignment.fate,
                        "source": "verify_offspring",
                    }
                    for assignment in failed_assignments
                )
        population.integrate(offspring)
        reproduction_compaction = compact_live_population(
            population,
            archives,
            policy,
            branch_factor=self.budget.branch_factor,
            round_index=current_round,
        )
        if generation_plan is not None:
            completed = list(completed_stage_ops or [])
            completed.append("verify_offspring")
            self.last_generation_plan["offspring_verification_count"] = len(offspring_verification)
            self.last_generation_plan["reproduction_archive_updates"] = reproduction_archive_updates
            self._record_generation_stage_progress(completed)
        return "", offspring_verification, reproduction_compaction.to_dict()




def _search_pressure_has_effect(pressure: Any) -> bool:
    return bool(
        getattr(pressure, "target_challenge_ids", None)
        or getattr(pressure, "avoid_challenge_ids", None)
        or getattr(pressure, "artifact_requirements", None)
        or getattr(pressure, "success_criteria", None)
        or str(getattr(pressure, "mutation_instruction", "") or "").strip()
    )


def _accepted_preliminary_incumbent(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    incumbent = select_preliminary_incumbent(candidates)
    if incumbent is None:
        return None
    metadata = incumbent.metadata if isinstance(incumbent.metadata, dict) else {}
    evaluator = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
    status = str(evaluator.get("status") or "").strip().lower()
    if evaluator.get("passed") is True or status in {"passed", "pass", "ok", "success"}:
        return incumbent
    return None


def _parent_for_plan_id(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _attach_branch_allocation_to_plans(
    plans: list[MutationPlan],
    allocation: ProductiveBranchAllocation,
    *,
    include_manifest: bool,
    rejected_out: list[dict[str, Any]] | None = None,
) -> list[MutationPlan]:
    if not plans or not allocation.slots:
        return plans
    unused = list(allocation.slots)
    out: list[MutationPlan] = []
    for plan in plans:
        slot = next((item for item in unused if item.parent_id in plan.parent_ids), None)
        if slot is None:
            if rejected_out is not None:
                rejected_out.append(
                    {
                        "reason": "parent_slot_quota_exceeded",
                        "plan_id": str((plan.metadata or {}).get("plan_id") or ""),
                        "parent_ids": list(plan.parent_ids),
                    }
                )
            continue
        metadata = dict(plan.metadata or {})
        if not include_manifest or len(plans) > 1 or len(allocation.slots) == 1:
            unused.remove(slot)
            metadata.update(
                {
                    "branch_slot_id": slot.slot_id,
                    "branch_arm_id": slot.arm_id,
                    "branch_slot_parent_id": slot.parent_id,
                    "branch_intent": slot.intent,
                    "branch_slot_binding_status": "planned",
                    "coverage_target": dict(slot.coverage_target),
                }
            )
        if include_manifest and not out:
            metadata["branch_slots"] = [item.to_dict() for item in allocation.slots]
        out.append(MutationPlan.from_dict({**plan.to_dict(), "metadata": metadata}))
    return out


def _archive_elite_reentry_limit(policy: EvolutionPolicy) -> int:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    return max(0, int(metadata.get("archive_elite_reentry_limit", 1)))


def _archive_elite_admission_view(
    *,
    population: list[CandidateGenome],
    archives: ArchiveManager,
    limit: int,
    current_round: int,
) -> tuple[list[CandidateGenome], dict[str, Any]]:
    existing_ids = {candidate.id for candidate in population}
    by_id: dict[str, tuple[float, CandidateGenome]] = {}
    stores = (
        archives.quality_diversity.elites_by_niche,
        archives.quality_diversity.cell_elites,
    )
    excluded: list[dict[str, str]] = []
    for store in stores:
        for raw in store.values():
            if not isinstance(raw, dict) or not isinstance(raw.get("candidate"), dict):
                continue
            candidate = candidate_from_dict(copy.deepcopy(raw["candidate"]))
            if candidate.id in existing_ids:
                continue
            if structurally_blocked(candidate):
                excluded.append({"candidate_id": candidate.id, "reason": "structural_or_safety_blocked"})
                continue
            score = max(float(raw.get("search_quality", -1.0)), float(raw.get("final_quality", -1.0)))
            current = by_id.get(candidate.id)
            if current is None or score > current[0]:
                by_id[candidate.id] = (score, candidate)
    ranked = sorted(
        by_id.values(),
        key=lambda item: (*evaluator_selection_key(item[1])[:2], item[0], item[1].id),
        reverse=True,
    )
    admitted: list[CandidateGenome] = []
    for score, candidate in ranked[: max(0, int(limit))]:
        candidate.current_fate = CandidateFate.ELITE.value
        candidate.metadata["archive_elite_reentry"] = {
            "source": "quality_diversity_archive",
            "round": int(current_round),
            "archive_quality": score,
            "effect": "parent_selection_admission_only",
        }
        admitted.append(candidate)
    audit = {
        "limit": max(0, int(limit)),
        "admitted_candidate_ids": [candidate.id for candidate in admitted],
        "selected_candidate_ids": [],
        "excluded": excluded,
        "effect": "parent_selector_admission_only_archive_unchanged",
    }
    return admitted, audit


def _branch_credit_candidates(
    *,
    population: CandidatePopulation,
    archives: ArchiveManager,
    repair_parent_candidates: list[CandidateGenome] | None,
) -> list[CandidateGenome]:
    candidates = [*population.candidates, *(repair_parent_candidates or [])]
    direct_stores = (
        archives.answer_archive,
        archives.mechanism_archive,
        archives.novelty_archive,
        archives.project_patch_archive,
        archives.rarity_archive.candidates,
        archives.auxiliary_archive.candidates,
        archives.dormant_archive.candidates,
        archives.latent_pareto_archive.candidates,
    )
    for store in direct_stores:
        candidates.extend(candidate_from_dict(data) for data in store.values() if isinstance(data, dict) and data.get("id"))
    for store in (archives.quality_diversity.elites_by_niche, archives.quality_diversity.cell_elites):
        candidates.extend(
            candidate_from_dict(item["candidate"])
            for item in store.values()
            if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
        )
    return candidates


def _canonical_family_metrics(candidates: list[CandidateGenome]) -> dict[str, Any]:
    if not candidates:
        return {}
    family_counts: dict[str, int] = {}
    bin_keys: set[str] = set()
    descriptive_families: set[str] = set()
    descriptive_bins: set[str] = set()
    seen_phenotypes: set[str] = set()
    outcome_signatures: set[str] = set()
    novelty_term_counts: list[tuple[str, int]] = []
    migration_samples = 0
    migration_changed = 0
    for candidate in candidates:
        meta = ensure_nextgen_identity(candidate)
        family = str(meta.get("canonical_mechanism_family_id") or candidate.id)
        bin_key = candidate_bin_key(candidate)
        descriptive_families.add(family)
        descriptive_bins.add(bin_key)
        phenotype = candidate_phenotype_signature(candidate)
        duplicate_phenotype = bool(phenotype and phenotype in seen_phenotypes)
        if phenotype:
            seen_phenotypes.add(phenotype)
        migration = meta.get("canonical_mechanism_migration") if isinstance(meta.get("canonical_mechanism_migration"), dict) else {}
        if migration:
            migration_samples += 1
            if str(migration.get("from_canonical_mechanism_family_id") or "") != str(migration.get("to_canonical_mechanism_family_id") or ""):
                migration_changed += 1
        if duplicate_phenotype:
            continue
        outcome = candidate_outcome_signature(candidate)
        if outcome:
            outcome_signatures.add(outcome)
        behavior_key = outcome or phenotype or family
        family_counts[behavior_key] = family_counts.get(behavior_key, 0) + 1
        bin_keys.add(outcome or phenotype or bin_key)
        novelty_terms = {str(item) for item in [*candidate.novelty_descriptors, *candidate.niche_memberships] if str(item or "").strip()}
        novelty_term_counts.append((behavior_key, len(novelty_terms)))
    population_count = len(candidates)
    phenotype_count = sum(family_counts.values())
    probabilities = [count / max(1, phenotype_count) for count in family_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probabilities if p > 0.0)
    top_count = max(family_counts.values()) if family_counts else 0
    # Divide by the final family size so the ratio does not depend on iteration order.
    ratios = sorted(count / max(1, family_counts[family]) for family, count in novelty_term_counts)
    return {
        "population_count": population_count,
        "phenotype_count": phenotype_count,
        "exact_clone_excess": max(0, population_count - phenotype_count),
        "evaluator_outcome_signature_count": len(outcome_signatures),
        "canonical_family_entropy": round(entropy, 6),
        "max_canonical_family_share": round(top_count / max(1, phenotype_count), 6),
        "top_canonical_family_count": top_count,
        "distinct_canonical_family_count": len(family_counts),
        "canonical_family_count_to_population_count": round(len(family_counts) / max(1, population_count), 6),
        "candidate_bin_count": len(bin_keys),
        "candidate_bin_count_to_canonical_family_count": round(len(bin_keys) / max(1, len(family_counts)), 6),
        "descriptive_canonical_family_count": len(descriptive_families),
        "descriptive_candidate_bin_count": len(descriptive_bins),
        "same_declared_changed_canonical_share": round(migration_changed / max(1, migration_samples), 6),
        "same_declared_changed_canonical_sample_count": migration_samples,
        "novelty_to_canonical_family_ratio_p50": round(_percentile(ratios, 0.50), 6),
        "novelty_to_canonical_family_ratio_p95": round(_percentile(ratios, 0.95), 6),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(float(quantile or 0.0) * len(values)) - 1))
    return float(values[index])


def _role_crossover_slots(
    allocation: ProductiveBranchAllocation,
    *,
    parent_ids: list[str],
    quota: int,
    roles: Any = None,
) -> dict[str, dict[str, Any]]:
    if quota <= 0 or len(parent_ids) < 2:
        return {}
    role_items = (
        [str(item) for item in roles if str(item)]
        if isinstance(roles, list)
        else [
            "representation_donor",
            "repair_pattern_donor",
            "mechanism_fragment_donor",
        ]
    )
    invalid = [role for role in role_items if role not in DONOR_ROLES]
    if invalid:
        raise ValueError(f"unsupported crossover donor role: {invalid[0]}")
    if not role_items:
        raise ValueError("crossover donor roles must not be empty")
    assigned: dict[str, dict[str, Any]] = {}
    for slot in allocation.slots:
        if len(assigned) >= quota:
            break
        if slot.parent_id not in parent_ids:
            continue
        primary_index = parent_ids.index(slot.parent_id)
        donor_id = next(
            (
                parent_ids[(primary_index + offset) % len(parent_ids)]
                for offset in range(1, len(parent_ids))
                if parent_ids[(primary_index + offset) % len(parent_ids)] != slot.parent_id
            ),
            "",
        )
        if not donor_id:
            continue
        role = role_items[len(assigned) % len(role_items)]
        assigned[slot.slot_id] = {
            "primary_parent_id": slot.parent_id,
            "donor_parent_id": donor_id,
            "donor_role": role,
            "required_contribution_map": dict(_REQUIRED_CONTRIBUTION_MAP),
        }
    return assigned


def _cell_activation_map(*, parents: list[CandidateGenome], plans: list[MutationPlan], offspring: list[CandidateGenome]) -> dict[str, Any]:
    activation: dict[str, dict[str, Any]] = {}

    def _entry(cell: str) -> dict[str, Any]:
        return activation.setdefault(cell, {"parent_ids": [], "offspring_ids": [], "operators": []})

    for parent in parents:
        cell = candidate_bin_key(parent)
        entry = _entry(cell)
        if parent.id not in entry["parent_ids"]:
            entry["parent_ids"].append(parent.id)
    for plan in plans:
        operator = str(plan.operator or "")
        parent_ids = [str(item) for item in plan.parent_ids if item]
        for parent in parents:
            if parent_ids and parent.id not in parent_ids:
                continue
            entry = _entry(candidate_bin_key(parent))
            if operator and operator not in entry["operators"]:
                entry["operators"].append(operator)
    for child in offspring:
        cell = candidate_bin_key(child)
        entry = _entry(cell)
        if child.id not in entry["offspring_ids"]:
            entry["offspring_ids"].append(child.id)
        for operator in getattr(child, "mutation_history", []) or []:
            op = str(operator or "")
            if op and op not in entry["operators"]:
                entry["operators"].append(op)
    return {cell: entry for cell, entry in activation.items() if entry.get("parent_ids") or entry.get("offspring_ids")}


def _prompt_verification_regime_item(obligation: dict[str, Any]) -> dict[str, Any]:
    """Return a model-facing obligation view without certification shortcuts."""

    allowed = {
        "id",
        "origin",
        "must_pass",
        "exogeneity_probe",
        "variety_probe",
        "falsification_budget",
        "replay_record",
    }
    out = {key: obligation.get(key) for key in allowed if key in obligation}
    for forbidden in ("strength_contribution", "replayable", "strength", "strength_value", "measured_strength"):
        out.pop(forbidden, None)
    return out



__all__ = [
    "ReproduceStage",
    "_attach_branch_allocation_to_plans",
    "_canonical_family_metrics",
    "_cell_activation_map",
]
