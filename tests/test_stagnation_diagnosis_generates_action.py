from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _plan_mutations
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def test_stagnation_diagnosis_generates_action_for_auxiliary_collapse() -> None:
    population = [
        CandidateGenome(id="aux1", current_fate=CandidateFate.AUXILIARY, multihead_scores={"auxiliary_value": 0.9}),
        CandidateGenome(id="aux2", current_fate=CandidateFate.AUXILIARY, multihead_scores={"auxiliary_value": 0.8}),
        CandidateGenome(id="weak", multihead_scores={"answer_likelihood": 0.1}),
    ]

    diagnosis = SearchStateDiagnoser().diagnose(population=population, archives=ArchiveManager(), policy=EvolutionPolicy())
    updated = PolicyUpdater().update(EvolutionPolicy(mutation_operators=[]), diagnosis)

    assert diagnosis.stagnation_detected is True
    assert diagnosis.stagnation_type == "AuxiliaryCollapse"
    assert {"core_extraction", "rare_inject"} & set(diagnosis.recommended_actions)
    assert updated.mutation_operators


def test_failed_elite_frontier_triggers_verification_bottleneck() -> None:
    candidate = CandidateGenome(
        id="elite-failed",
        generation=2,
        current_fate=CandidateFate.ELITE,
        core_mechanism="candidate route",
        multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.8},
        verification_result={"passed": False, "rank_eligible": False, "final_eligible": False, "diagnostics": ["source_binding_absent"]},
        verification_trace=[{"status": "failed", "diagnostics": ["source_binding_absent"]}],
    )

    diagnosis = SearchStateDiagnoser().diagnose(population=[candidate], archives=ArchiveManager(), policy=EvolutionPolicy(rarity_budget=0))

    assert diagnosis.stagnation_detected is True
    assert diagnosis.stagnation_type == "DiversityCollapse"
    assert "increase_rarity_budget" in diagnosis.recommended_actions or "continue" in diagnosis.recommended_actions


def test_failed_dormant_frontier_triggers_verification_bottleneck_but_terminal_failures_do_not() -> None:
    dormant = CandidateGenome(
        id="dormant-failed",
        generation=3,
        current_fate=CandidateFate.DORMANT,
        core_mechanism="rare route",
        edge_knowledge_seeds=["rare"],
        multihead_scores={"objective_alignment": 0.8, "rarity": 0.9},
        verification_result={"passed": False, "rank_eligible": False, "final_eligible": False, "diagnostics": ["evidence_ref_absent"]},
        verification_trace=[{"status": "failed"}],
    )
    terminal = CandidateGenome(
        id="terminal-failed",
        generation=3,
        current_fate=CandidateFate.FAILED,
        core_mechanism="dead route",
        verification_result={"passed": False, "diagnostics": ["evidence_ref_absent"]},
        verification_trace=[{"status": "failed"}],
    )

    diagnosis = SearchStateDiagnoser().diagnose(population=[dormant], archives=ArchiveManager(), policy=EvolutionPolicy(rarity_budget=0))
    terminal_only = SearchStateDiagnoser().diagnose(population=[terminal], archives=ArchiveManager(), policy=EvolutionPolicy(rarity_budget=0))

    assert diagnosis.stagnation_type == "DiversityCollapse"
    assert terminal_only.stagnation_type != "VerificationBottleneck"


def test_policy_updater_scales_rarity_budget_from_observed_archive_pressure() -> None:
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="DiversityCollapse",
        recommended_actions=["increase_rarity_budget"],
    )
    shallow = ArchiveManager()
    shallow.fates = {f"candidate-{index}": CandidateFate.DORMANT.value for index in range(4)}
    deep = ArchiveManager()
    deep.fates = {f"candidate-{index}": CandidateFate.DORMANT.value for index in range(20)}
    for index in range(10):
        rare = CandidateGenome(
            id=f"rare-{index}",
            current_fate=CandidateFate.DORMANT,
            edge_knowledge_seeds=[f"rare-seed-{index}"],
            multihead_scores={"rarity": 0.9},
        )
        deep.rarity_archive.add(rare)

    base = EvolutionPolicy(rarity_budget=0.2, mutation_operators=[])
    shallow_update = PolicyUpdater().update(base, diagnosis, archives=shallow)
    deep_update = PolicyUpdater().update(base, diagnosis, archives=deep)

    shallow_increment = shallow_update.metadata["rarity_budget_update"]["increment"]
    deep_increment = deep_update.metadata["rarity_budget_update"]["increment"]
    assert shallow_increment > deep_increment
    assert shallow_increment != 0.2
    assert shallow_update.rarity_budget == base.rarity_budget + shallow_increment
    assert deep_update.rarity_budget == base.rarity_budget + deep_increment
    assert shallow_update.metadata["rarity_budget_update"]["population_size"] == 4
    assert deep_update.metadata["rarity_budget_update"]["rare_archive_depth"] == 10


def test_diversity_collapse_pressure_uses_open_family_distribution_not_fixed_placeholders() -> None:
    dominant = [
        CandidateGenome(
            id=f"dom-{index}",
            lineage=["dominant-lineage", f"dom-{index}"],
            core_mechanism="dominant basin",
            concise_claim=f"dominant variant {index}",
            niche_memberships=["dominant-basin"],
            multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.8},
        )
        for index in range(4)
    ]
    rare = CandidateGenome(
        id="rare",
        lineage=["rare-lineage", "rare"],
        core_mechanism="cold path",
        concise_claim="low sample cold path",
        niche_memberships=["cold-path-family"],
        edge_knowledge_seeds=["cold edge"],
        multihead_scores={"objective_alignment": 0.65, "answer_likelihood": 0.55, "novelty": 0.9, "rarity": 0.9},
    )

    diagnosis = SearchStateDiagnoser().diagnose(population=[*dominant, rare], archives=ArchiveManager(), policy=EvolutionPolicy())
    updated = PolicyUpdater().update(EvolutionPolicy(), diagnosis)
    pressure = updated.metadata["selection_pressure"]

    assert diagnosis.stagnation_detected is True
    assert diagnosis.stagnation_type == "SemanticLooping"
    assert "cold_path_family" in pressure["under_explored_families"]
    assert "dominant-lineage" in pressure["over_explored_families"]
    assert not {"rarity", "dormant", "crossover"}.intersection(set(pressure["under_explored_families"]))

    selected = ParentSelector().select(
        [*dominant, rare],
        ArchiveManager(),
        limit=2,
        eligibility_policy=updated.metadata["eligibility_policy"],
    )

    assert rare.id in [candidate.id for candidate in selected]


def test_semantic_looping_replays_under_explored_family_into_reproduction_plan_and_offspring() -> None:
    dominant = [
        CandidateGenome(
            id=f"dom-{index}",
            lineage=["dominant-lineage", f"dom-{index}"],
            core_mechanism="dominant basin",
            concise_claim=f"dominant variant {index}",
            niche_memberships=["dominant-basin"],
            multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.8},
        )
        for index in range(4)
    ]
    rare = CandidateGenome(
        id="rare",
        lineage=["rare-lineage", "rare"],
        core_mechanism="cold path",
        concise_claim="low sample cold path",
        niche_memberships=["cold-path-family"],
        edge_knowledge_seeds=["cold edge"],
        multihead_scores={"objective_alignment": 0.65, "answer_likelihood": 0.55, "novelty": 0.9, "rarity": 0.9},
    )
    archives = ArchiveManager()
    population = [*dominant, rare]

    diagnosis = SearchStateDiagnoser().diagnose(population=population, archives=archives, policy=EvolutionPolicy())
    updated = PolicyUpdater().update(EvolutionPolicy(), diagnosis)
    selected = ParentSelector().select(
        population,
        archives,
        limit=2,
        eligibility_policy=updated.metadata["eligibility_policy"],
    )
    plans = _plan_mutations(
        model=None,
        mutation_planner=MutationPlanner(),
        parents=selected,
        actions=action_palette_for_round(3, diagnosis.recommended_actions),
        archives=archives,
        diagnosis=diagnosis,
        policy=updated,
    )
    offspring = _generate_offspring(
        model=None,
        mutation_engine=MutationEngine(),
        parents=selected,
        plans=plans,
        world={},
        contract=NexusObjectiveContract(original_user_goal="x", normalized_goal="x"),
        policy=updated,
    )

    assert diagnosis.stagnation_type == "SemanticLooping"
    assert "cold_path_family" in updated.metadata["selection_pressure"]["under_explored_families"]
    assert rare.id in [candidate.id for candidate in selected]
    assert any(rare.id in plan.parent_ids for plan in plans)
    assert any(child.parent_ids == [rare.id] for child in offspring)
