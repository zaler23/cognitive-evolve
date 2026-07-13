"""Last-resort activation reseeding for empty Nexus populations.

This module is intentionally small: it only handles the degenerate case where
normal selection, dormant repair, and failure archive recovery all return no
parents.  It creates non-final Incubating material from the model-defined
artifact contract without choosing a domain-specific answer taxonomy.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


def emergency_activation_reseed(
    *,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    limit: int,
    current_round: int,
) -> list[CandidateGenome]:
    """Create non-final, contract-derived repair seeds when every pool is empty."""

    del world  # The dynamic artifact contract carries the domain semantics.
    target = max(0, int(limit or 0))
    if target <= 0:
        return []
    dac = {}
    if isinstance(getattr(contract, "dynamic_artifact_contract", None), dict):
        dac = dict(getattr(contract, "dynamic_artifact_contract") or {})
    if not dac and isinstance(getattr(contract, "outcome_policy", None), dict):
        nested = contract.outcome_policy.get("dynamic_artifact_contract")
        if isinstance(nested, dict):
            dac = dict(nested)
    objective = str(getattr(contract, "normalized_goal", "") or getattr(contract, "original_user_goal", "") or "user objective")
    required = dac.get("required_work_product") if isinstance(dac, dict) else {}
    delta = dac.get("minimum_concrete_delta") if isinstance(dac, dict) else {}
    seeds: list[CandidateGenome] = []
    policy_niches = [str(item).strip() for item in getattr(policy, "candidate_niches", []) or [] if str(item).strip()]
    descriptors = policy_niches[:target] or [str(dac.get("artifact_domain_label") or "model_defined_artifact")]
    for index, descriptor in enumerate(descriptors[:target]):
        seed = CandidateGenome(
            id=f"EA{current_round:04d}{index:02d}",
            generation=max(0, int(current_round or 0)),
            artifact={
                "source": "emergency_activation_reseed",
                "objective": objective,
                "model_defined_required_work_product": required,
                "model_defined_minimum_delta": delta,
                "instruction": "Materialize the smallest actual artifact or repair step required by the dynamic artifact contract; do not return commentary only.",
            },
            artifact_type="activation_repair_seed",
            concise_claim=f"Recover empty population by materializing the contract-defined artifact for: {objective}",
            core_mechanism=descriptor,
            missing_parts=["actual contract-defined artifact", "measurable delta", "verification or comparison evidence"],
            uncertainty_notes=["emergency activation reseed is search material, not final answer material"],
            mutation_history=["EmergencyActivationReseed"],
            novelty_descriptors=[descriptor, "empty_population_recovery"],
            niche_memberships=[descriptor],
            current_fate=CandidateFate.INCUBATING.value,
            multihead_scores={
                "objective_alignment": 0.25,
                "answer_likelihood": 0.0,
                "verifiability": 0.1,
                "novelty": 0.35,
                "rarity": 0.2,
                "deferral_risk": 0.9,
            },
            contract_hash=contract.contract_hash(),
            metadata={
                "created_in_round": int(current_round or 0),
                "search_seed_not_final": True,
                "final_answer_blocked_until_repaired": True,
                "exploration_source": "emergency_activation_reseed",
                "repair_required": {
                    "blockers": ["empty_population_no_parent_pool", "artifact_object_absent", "concrete_delta_absent"],
                    "evidence_needed": ["actual_artifact", "artifact_delta", "contract_relevant_evidence"],
                    "next_actions": ["produce the smallest actual artifact body or structured object that satisfies the dynamic artifact contract"],
                    "source": "emergency_activation_reseed",
                },
            },
        )
        seeds.append(seed)
    return seeds


__all__ = ["emergency_activation_reseed"]
