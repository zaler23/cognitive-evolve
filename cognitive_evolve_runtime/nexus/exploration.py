"""Exploration helpers that port the useful breadth of the old adaptive path.

The old adaptive runtime was good at leaving a wide search trail: direct routes,
counter-routes, tool-grounded fragments, and weird edge ideas all existed before
ranking narrowed the pool.  Nexus keeps that advantage without restoring the old
parallel architecture by treating these as task-neutral *search moves* encoded in
CandidateGenome metadata.  They are seeds for evolution, not claims of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


@dataclass(frozen=True)
class ExplorationSeedTemplate:
    seed_type: str
    niche: str
    prompt: str
    novelty: float = 0.4
    rarity: float = 0.0
    verifiability: float = 0.35
    answer_likelihood: float = 0.45
    edge_tokens: tuple[str, ...] = field(default_factory=tuple)


EXPLORATION_SEED_TEMPLATES: tuple[ExplorationSeedTemplate, ...] = (
    ExplorationSeedTemplate("Direct Solver Seed", "direct", "Attempt the objective head-on and expose the actual solution mechanism.", answer_likelihood=0.62),
    ExplorationSeedTemplate("Known Pattern Seed", "known_pattern", "Try the most standard route, but record why it might be insufficient.", answer_likelihood=0.55),
    ExplorationSeedTemplate("Counterexample / Negative Construction Seed", "counterexample_or_negative_construction", "Look for a falsifying object, obstruction, or minimal bad case before assuming a positive route.", novelty=0.62, rarity=0.35, edge_tokens=("negative_construction",)),
    ExplorationSeedTemplate("Analogy Transfer Seed", "analogy_transfer", "Map the problem to a neighboring structure and import only the transferable mechanism.", novelty=0.65, rarity=0.25, edge_tokens=("analogy_transfer",)),
    ExplorationSeedTemplate("Inversion / Dual Seed", "inversion_dual", "Invert the target: swap construction with obstruction, upper with lower, direct with dual.", novelty=0.7, rarity=0.3, edge_tokens=("duality", "inversion")),
    ExplorationSeedTemplate("Decomposition Seed", "decomposition", "Split the task into independently testable subclaims and identify the narrowest hard core.", verifiability=0.55),
    ExplorationSeedTemplate("Tool-Grounded Seed", "tool_grounded", "Turn part of the idea into a local check, enumeration, static validation, or executable micro-test.", verifiability=0.82),
    ExplorationSeedTemplate("Adversarial Critic Seed", "adversarial_critic", "Attack the current best route and preserve any reusable core that survives.", novelty=0.55, rarity=0.2),
    ExplorationSeedTemplate("Rare Recall Seed", "rare_recall", "Use obscure or low-frequency knowledge only as a seed, then require cross-checking.", novelty=0.82, rarity=0.88, edge_tokens=("rare_recall",)),
    ExplorationSeedTemplate("Wildcard Seed", "wildcard", "Try a high-variance route that would normally be pruned too early.", novelty=0.9, rarity=0.62, answer_likelihood=0.28, edge_tokens=("wildcard",)),
)

PROJECT_EXPLORATION_TEMPLATES: tuple[ExplorationSeedTemplate, ...] = (
    ExplorationSeedTemplate("Minimal Patch Seed", "minimal_patch", "Make the smallest contract-preserving patch and verify it locally.", verifiability=0.78, answer_likelihood=0.58),
    ExplorationSeedTemplate("Test-First Seed", "test_first", "Capture the desired behavior as a test or executable check before broad changes.", verifiability=0.86),
    ExplorationSeedTemplate("Compatibility-Preserving Seed", "compatibility_preserving", "Preserve current public behavior while changing the internal mechanism.", verifiability=0.65),
    ExplorationSeedTemplate("Internal Forgotten Pattern Seed", "internal_forgotten_pattern", "Search for an existing local pattern that can be revived instead of adding a new parallel path.", novelty=0.68, rarity=0.7, edge_tokens=("internal_forgotten_pattern",)),
)


def required_population_size(policy: EvolutionPolicy, *, requested_minimum: int | None = None, world: Any | None = None) -> int:
    """Return a minimum population width without making it a hard domain rule."""

    if requested_minimum and requested_minimum > 0:
        return int(requested_minimum)
    configured = _positive_int((policy.metadata or {}).get("initial_candidate_count"))
    if configured:
        return configured
    unique_niches = {str(item).strip().lower() for item in policy.candidate_niches or [] if str(item).strip()}
    search_planes = _search_plane_templates(contract=None, policy=policy, used_keys=set())
    if search_planes:
        return max(len(search_planes), len(unique_niches) or 0, 1)
    if unique_niches:
        return len(unique_niches)
    return max(1, len(EXPLORATION_SEED_TEMPLATES))


def amplify_population(
    *,
    population: CandidatePopulation,
    contract: NexusObjectiveContract,
    world: Any,
    policy: EvolutionPolicy,
    minimum_size: int | None = None,
) -> CandidatePopulation:
    """Ensure Nexus starts with the old runtime's exploration breadth.

    Model-generated candidates are never replaced.  We only add structured search
    seeds when the model produced a narrow pool or deterministic fallback is in
    use.  Supplemental candidates are marked ``search_seed_not_final`` so final
    synthesis does not mistake a prompt-like seed for a solved answer.
    """

    candidates = list(population.candidates)
    used_keys = {_candidate_key(candidate) for candidate in candidates}
    edge_pool = coerce_str_list(getattr(world, "edge_seed_pool", [])) or ["rare_recall_seed"]
    plane_templates = _search_plane_templates(contract=contract, policy=policy, used_keys=used_keys)
    templates = list(plane_templates)
    templates.extend(EXPLORATION_SEED_TEMPLATES)
    templates.extend(_policy_niche_templates(policy, used_keys))
    minimum = max(required_population_size(policy, requested_minimum=minimum_size, world=world), len(plane_templates) if plane_templates else 0)
    if len(candidates) >= minimum:
        return CandidatePopulation(candidates)
    for template in templates:
        if len(candidates) >= minimum:
            break
        if template.niche in used_keys or template.seed_type.lower() in used_keys:
            continue
        candidates.append(_candidate_from_template(template, contract=contract, world=world, edge_pool=edge_pool))
        used_keys.add(template.niche)
    index = 0
    while len(candidates) < minimum:
        round_robin = tuple(templates or EXPLORATION_SEED_TEMPLATES)
        template = round_robin[index % len(round_robin)]
        index += 1
        candidates.append(_candidate_from_template(template, contract=contract, world=world, edge_pool=edge_pool, suffix=f"-{index}"))
    return CandidatePopulation(candidates)


def action_palette_for_round(round_index: int, diagnosis_actions: Iterable[str] | None = None) -> list[str]:
    """Blend diagnosis actions with the old generate-reflect-evolve breadth."""

    actions = [str(action) for action in diagnosis_actions or [] if action and str(action) != "continue"]
    hard_proof_actions = [
        action
        for action in actions
        if action in {"instantiate_formal_artifact", "discharge_obligation", "case_split", "construct_witness", "route_kill"}
    ]
    if hard_proof_actions:
        actions = hard_proof_actions + [action for action in actions if action not in hard_proof_actions]
    if round_index <= 1:
        actions.extend(["deepen", "tool_ground", "rare_inject"])
    elif round_index == 2:
        actions.extend(["invert", "transfer", "adversarial_patch"])
    else:
        actions.extend(["cross_archives", "rare_inject", "repair", "scaffold_removal"])
    deduped: list[str] = []
    for action in actions or ["deepen"]:
        if action not in deduped:
            deduped.append(action)
    return deduped


def _candidate_from_template(
    template: ExplorationSeedTemplate,
    *,
    contract: NexusObjectiveContract,
    world: Any,
    edge_pool: list[str],
    suffix: str = "",
) -> CandidateGenome:
    artifact = (
        f"{template.seed_type}: {template.prompt}\n"
        f"Objective: {contract.normalized_goal}\n"
        "This is a search seed, not a verified final answer."
    )
    edge_seeds = list(template.edge_tokens)
    if template.rarity > 0.5:
        edge_seeds.extend(edge_pool[:2])
    scores = {
        "objective_alignment": 0.48,
        "answer_likelihood": template.answer_likelihood,
        "core_mechanism_strength": 0.45,
        "novelty": template.novelty,
        "rarity": template.rarity,
        "verifiability": template.verifiability,
        "internal_coherence": 0.52,
        "tool_progress": 0.25 if template.verifiability >= 0.75 else 0.0,
        "robustness": 0.35,
        "simplicity": 0.48,
        "transfer_potential": 0.55 if "transfer" in template.niche else 0.25,
        "auxiliary_value": 0.15 if "critic" in template.niche else 0.0,
        "deferral_risk": 0.25 if template.answer_likelihood < 0.35 else 0.12,
    }
    metadata = {
        "seed_type": template.seed_type + suffix,
        "search_seed_not_final": True,
        "exploration_source": "nexus_exploration_amplifier",
        "legacy_advantage_absorbed": "broad_generate_reflect_rank_evolve_seed_palette",
        "created_in_round": 0,
        "search_space": {
            "family_id": template.niche,
            "seed_type": template.seed_type,
            "distinct_from_local_surface": "seed is a search-plane obligation, not a commitment to the nearest implementation surface",
        },
    }
    if getattr(world, "kind", "text") == "project" and template.niche in {item.niche for item in PROJECT_EXPLORATION_TEMPLATES}:
        return ProjectCandidateGenome(
            generation=0,
            artifact_type="activation_seed",
            artifact={
                "seed_type": template.seed_type,
                "instruction": template.prompt,
                "required_work": "generate a real project patch or repair obligation in a later offspring; this seed itself is not a patch",
            },
            patch_set=[],
            concise_claim=f"{template.seed_type} for {contract.normalized_goal}",
            core_mechanism=template.niche,
            edge_knowledge_seeds=edge_seeds,
            novelty_descriptors=[template.niche, template.seed_type.lower().replace(" ", "_")],
            niche_memberships=[template.niche],
            multihead_scores=scores,
            contract_hash=contract.contract_hash(),
            expected_effects=[template.prompt, "materialize into runtime/test/schema patch before final eligibility"],
            metadata=metadata,
        )
    return CandidateGenome(
        generation=0,
        artifact=artifact,
        artifact_type="answer",
        concise_claim=f"{template.seed_type} exploration route",
        core_mechanism=template.niche,
        edge_knowledge_seeds=edge_seeds,
        novelty_descriptors=[template.niche, template.seed_type.lower().replace(" ", "_")],
        niche_memberships=[template.niche],
        missing_parts=["requires evolution beyond initial seed", "requires verification before final use"],
        multihead_scores=scores,
        contract_hash=contract.contract_hash(),
        metadata=metadata,
    )


def _search_plane_templates(
    *,
    contract: NexusObjectiveContract | None,
    policy: EvolutionPolicy,
    used_keys: set[str],
) -> list[ExplorationSeedTemplate]:
    planes = _search_planes_from_contract(contract)
    policy_plan = coerce_dict((policy.metadata or {}).get("search_space_plan"))
    planes.extend(_planes_from_mapping(policy_plan))
    out: list[ExplorationSeedTemplate] = []
    for index, plane in enumerate(planes, start=1):
        raw_id = str(plane.get("id") or plane.get("name") or f"model_defined_plane_{index}").strip()
        normalized = raw_id.lower().replace(" ", "_")
        if not normalized or normalized in used_keys:
            continue
        prompt = str(plane.get("description") or plane.get("intent") or plane.get("goal") or raw_id)
        out.append(
            ExplorationSeedTemplate(
                seed_type=f"Model-Defined Search Plane Seed: {raw_id}",
                niche=normalized,
                prompt=(
                    "Explore this model-authored search plane from the objective, not merely the nearest local implementation surface: "
                    + prompt
                ),
                novelty=0.62,
                rarity=0.22,
                verifiability=0.42,
                answer_likelihood=0.48,
                edge_tokens=(normalized,),
            )
        )
    return out


def _search_planes_from_contract(contract: NexusObjectiveContract | None) -> list[dict[str, Any]]:
    if contract is None:
        return []
    data = contract.to_dict() if hasattr(contract, "to_dict") else coerce_dict(contract)
    outcome = coerce_dict(data.get("outcome_policy"))
    dac = coerce_dict(data.get("dynamic_artifact_contract") or outcome.get("dynamic_artifact_contract"))
    planes: list[dict[str, Any]] = []
    for source in (
        data.get("search_space_plan"),
        data.get("search_space"),
        outcome.get("search_space_plan"),
        outcome.get("search_space"),
        dac.get("search_space_plan"),
        dac.get("search_space"),
    ):
        planes.extend(_planes_from_mapping(coerce_dict(source)))
    return planes


def _planes_from_mapping(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("candidate_families", "exploration_planes", "planes", "families"):
        raw = plan.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                out.append(dict(item))
            elif str(item or "").strip():
                out.append({"id": str(item).strip(), "description": str(item).strip()})
    return out


def _policy_niche_templates(policy: EvolutionPolicy, used_keys: set[str]) -> list[ExplorationSeedTemplate]:
    out: list[ExplorationSeedTemplate] = []
    for niche in policy.candidate_niches:
        normalized = str(niche or "").strip().lower().replace(" ", "_")
        if not normalized or normalized in used_keys:
            continue
        out.append(
            ExplorationSeedTemplate(
                seed_type=f"Policy Niche Seed: {normalized}",
                niche=normalized,
                prompt="Explore this model-selected niche and preserve any inheritable mechanism.",
                novelty=0.5,
                rarity=0.15,
            )
        )
    return out


def _candidate_key(candidate: CandidateGenome) -> str:
    if candidate.niche_memberships:
        return str(candidate.niche_memberships[0]).lower()
    return (candidate.core_mechanism or candidate.concise_claim or candidate.id).strip().lower().replace(" ", "_")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "EXPLORATION_SEED_TEMPLATES",
    "PROJECT_EXPLORATION_TEMPLATES",
    "ExplorationSeedTemplate",
    "action_palette_for_round",
    "amplify_population",
    "required_population_size",
]
