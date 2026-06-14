"""Evolution policy schemas for Nexus runtime."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.mutation import MutationOperator
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, stable_hash, utc_now
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike

DEFAULT_FITNESS_AXES = [
    "objective_alignment",
    "answer_likelihood",
    "core_mechanism_strength",
    "novelty",
    "rarity",
    "verifiability",
    "proof_progress",
    "evidence_progress",
    "internal_coherence",
    "tool_progress",
    "robustness",
    "simplicity",
    "transfer_potential",
    "auxiliary_value",
    "deferral_risk",
]

DEFAULT_ARCHIVES = [
    "AnswerArchive",
    "MechanismArchive",
    "NoveltyArchive",
    "RarityArchive",
    "FailureArchive",
    "AuxiliaryArchive",
    "DormantArchive",
    "ProjectPatchArchive",
]

DEFAULT_ELIGIBILITY_POLICY = {
    # This is a deterministic fallback policy for offline tests and model
    # outages.  Model-backed runs can replace or tune it through
    # ``build_evolution_policy(...).metadata["eligibility_policy"]``.
    "source": "offline_fallback_model_overridable",
    "model_driven": True,
    "stage_mode": "model_or_signal_adaptive",
    # Numeric stage/candidate-age boundaries are intentionally absent here.
    # A model-authored EvolutionPolicy may provide stage_fractions or explicit
    # candidate_age_windows; otherwise the runtime uses signal-adaptive phases
    # and only treats the safety checkpoint itself as final pressure.
    "stage_fractions": {},
    "candidate_age_fractions": {},
    "claim_maturity_can_raise_stage_before_late": False,
    "max_incubation_attempts": "auto",
    "max_incubation_age_fraction": "auto",
    "min_incubation_age_rounds": 4,
    "max_repeated_repair_blockers": 3,
    "active_floor": {"enabled": True, "branch_multiplier": "auto", "minimum": "auto"},
    "repair_selection": {"enabled": True, "sqrt_incubating_slots": True, "max_parent_fraction": "auto"},
    "dormant_repair_reactivation": {
        "enabled": True,
        "max_seeds": "auto",
        "max_repair_attempts": "auto",
        "max_per_group": "auto",
    },
    "failure_archive_reseed": {
        "enabled": True,
        "max_seeds": "auto",
        "max_per_group": "auto",
        "require_repair_signal": True,
    },
    "selection_pressure": {
        "enabled": True,
        "over_explored_penalty": "auto",
        "under_explored_bonus": "auto",
        "prematurely_culled_bonus": "auto",
    },
    "elite_gap_merge": {"enabled": True, "max_fraction_of_branch_factor": 0.5},
}


@dataclass
class EvolutionPolicy:
    candidate_niches: list[str] = field(default_factory=lambda: ["direct", "known_pattern", "edge", "analogy", "inversion", "decomposition", "tool_grounded", "wildcard"])
    fitness_axes: list[str] = field(default_factory=lambda: list(DEFAULT_FITNESS_AXES))
    mutation_operators: list[str] = field(default_factory=lambda: list(MutationOperator.ALL))
    archive_schema: dict[str, Any] = field(default_factory=lambda: {name: {"enabled": True} for name in DEFAULT_ARCHIVES})
    parent_selection_preferences: dict[str, Any] = field(default_factory=lambda: {"prefer_reproductive_value_over_winner_only": True})
    culling_principles: list[str] = field(default_factory=lambda: ["extract_inheritable_genes_before_cull", "preserve_rare_or_incomplete_non_dominated_candidates"])
    rarity_budget: float = 0.2
    tool_preferences: list[str] = field(default_factory=lambda: ["local_verification", "schema_validation"])
    stagnation_actions: list[str] = field(default_factory=lambda: ["increase_rarity_budget", "reactivate_dormant", "cross_archives", "compress_lessons"])
    synthesis_policy: dict[str, Any] = field(default_factory=lambda: {"auxiliary_candidates_are_not_main_winners_by_default": True})
    policy_id: str = "nexus-evolution-policy"
    version: str = "nexus/evolution-policy/v1"
    created_at: str = field(default_factory=utc_now)
    updated_from_diagnoses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidate_niches = coerce_str_list(self.candidate_niches)
        self.fitness_axes = coerce_str_list(self.fitness_axes) or list(DEFAULT_FITNESS_AXES)
        self.mutation_operators = [op for op in coerce_str_list(self.mutation_operators) if op] or list(MutationOperator.ALL)
        self.archive_schema = coerce_dict(self.archive_schema) or {name: {"enabled": True} for name in DEFAULT_ARCHIVES}
        self.parent_selection_preferences = coerce_dict(self.parent_selection_preferences)
        self.culling_principles = coerce_str_list(self.culling_principles)
        self.tool_preferences = coerce_str_list(self.tool_preferences)
        self.stagnation_actions = coerce_str_list(self.stagnation_actions)
        self.synthesis_policy = coerce_dict(self.synthesis_policy)
        self.updated_from_diagnoses = coerce_str_list(self.updated_from_diagnoses)
        self.metadata = coerce_dict(self.metadata)
        self.metadata.setdefault("eligibility_policy", dict(DEFAULT_ELIGIBILITY_POLICY))
        try:
            self.rarity_budget = max(0.0, float(self.rarity_budget))
        except (TypeError, ValueError):
            self.rarity_budget = 0.2

    @property
    def policy_hash(self) -> str:
        data = self.to_dict()
        data.pop("created_at", None)
        return stable_hash(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionPolicy":
        return cls(
            candidate_niches=coerce_str_list(data.get("candidate_niches")),
            fitness_axes=coerce_str_list(data.get("fitness_axes")),
            mutation_operators=coerce_str_list(data.get("mutation_operators")),
            archive_schema=coerce_dict(data.get("archive_schema")),
            parent_selection_preferences=coerce_dict(data.get("parent_selection_preferences")),
            culling_principles=coerce_str_list(data.get("culling_principles")),
            rarity_budget=float(data.get("rarity_budget", 0.2) or 0.2),
            tool_preferences=coerce_str_list(data.get("tool_preferences")),
            stagnation_actions=coerce_str_list(data.get("stagnation_actions")),
            synthesis_policy=coerce_dict(data.get("synthesis_policy")),
            policy_id=str(data.get("policy_id") or "nexus-evolution-policy"),
            version=str(data.get("version") or "nexus/evolution-policy/v1"),
            created_at=str(data.get("created_at") or utc_now()),
            updated_from_diagnoses=coerce_str_list(data.get("updated_from_diagnoses")),
            metadata=coerce_dict(data.get("metadata")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "EvolutionPolicy":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("evolution policy JSON must decode to an object")
        return cls.from_dict(data)


class EvolutionPolicyBuilder:
    def build(self, *, contract: Any, world: Any, model: NexusModelLike | None = None) -> EvolutionPolicy:
        if model is not None and hasattr(model, "build_evolution_policy"):
            try:
                raw = model.build_evolution_policy(contract=contract, world=world)
                if isinstance(raw, EvolutionPolicy):
                    return raw
                if isinstance(raw, dict):
                    return EvolutionPolicy.from_dict(raw)
            except (LLMResponseError, TypeError, ValueError) as exc:
                return _fallback_policy_from_contract(
                    contract,
                    world,
                    fallback_reason="model_policy_unavailable",
                    error=exc,
                )
        return _fallback_policy_from_contract(contract, world)


def _fallback_policy_from_contract(
    contract: Any,
    world: Any,
    *,
    fallback_reason: str = "deterministic_policy",
    error: Exception | None = None,
) -> EvolutionPolicy:
        niches = list(EvolutionPolicy().candidate_niches)
        search_planes = _search_plane_ids_from_contract(contract)
        if search_planes:
            niches.extend(search_planes)
        policy = EvolutionPolicy(candidate_niches=list(dict.fromkeys(niches)))
        if error is not None:
            policy.metadata["model_policy_fallback"] = {
                "source": fallback_reason,
                "error_type": error.__class__.__name__,
                "error_category": provider_error_category(error),
                "message": str(error)[:500],
                "final_answer_blocked": True,
            }
        if search_planes:
            policy.metadata["search_space_plan"] = {"source": "objective_contract", "candidate_families": [{"id": item} for item in search_planes]}
        elif getattr(world, "kind", "text") == "project":
            policy.metadata["search_space_plan_required"] = (
                "project snapshots expose local files for grounding, but the model must author objective-level search planes; "
                "the runtime must not default to minimal_patch or executor-loop families."
            )
        return policy


def _search_plane_ids_from_contract(contract: Any) -> list[str]:
    data = contract.to_dict() if hasattr(contract, "to_dict") else coerce_dict(contract)
    outcome = coerce_dict(data.get("outcome_policy"))
    dac = coerce_dict(data.get("dynamic_artifact_contract") or outcome.get("dynamic_artifact_contract"))
    sources = [
        data.get("search_space_plan"),
        data.get("search_space"),
        outcome.get("search_space_plan"),
        outcome.get("search_space"),
        dac.get("search_space_plan"),
        dac.get("search_space"),
    ]
    ids: list[str] = []
    for source in sources:
        plan = coerce_dict(source)
        families = plan.get("candidate_families") or plan.get("exploration_planes") or plan.get("planes") or []
        if not isinstance(families, list):
            continue
        for item in families:
            if isinstance(item, dict):
                value = str(item.get("id") or item.get("name") or "").strip()
            else:
                value = str(item or "").strip()
            if value:
                ids.append(value)
    return list(dict.fromkeys(ids))


__all__ = ["DEFAULT_ARCHIVES", "DEFAULT_ELIGIBILITY_POLICY", "DEFAULT_FITNESS_AXES", "EvolutionPolicy", "EvolutionPolicyBuilder"]
