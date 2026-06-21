from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.loop import TEXT_SEED_TYPES, seed_population
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


def test_text_seed_population_has_required_entry_points() -> None:
    packet = TextInputPacket.from_text("Solve a hard problem with rare edge knowledge and inversion.")
    world = TextWorldModel.from_packet(packet)
    contract = NexusObjectiveContract(original_user_goal=packet.raw_text, normalized_goal="solve hard problem")
    population = seed_population(contract=contract, world=world, policy=EvolutionPolicy())

    assert isinstance(population, CandidatePopulation)
    assert len(population.candidates) >= len(TEXT_SEED_TYPES)
    assert any(candidate.edge_knowledge_seeds for candidate in population.candidates)
    assert all(candidate.contract_hash == contract.contract_hash() for candidate in population.candidates)


def test_nexus_runtime_text_one_round_persists(tmp_path) -> None:
    result = NexusRuntime(output_dir=tmp_path).run_text("Answer with a rare analogy route.", max_rounds=1)

    assert result.mode == "text"
    assert result.final_answer
    assert (tmp_path / "population.json").exists()
    assert (tmp_path / "archives.json").exists()
    assert (tmp_path / "checkpoint.json").exists()
    assert result.evolution["progress_events"][0]["type"] == "evolution_progress"
    certificate = result.evolution["synthesis"]["closure_certificate"]
    assert certificate["version"] == "closure_certificate_v1"
    assert certificate["terminal_status"] == result.evolution["completion_status"]
    assert certificate["objective_solved"] is result.evolution["synthesis"]["objective_solved"]


def test_project_fallback_seeds_are_activation_material_not_docs_only_patches() -> None:
    world = type("ProjectWorld", (), {"kind": "project", "edge_seed_pool": []})()
    contract = NexusObjectiveContract(original_user_goal="Improve project runtime.", normalized_goal="improve project runtime")
    population = seed_population(contract=contract, world=world, policy=EvolutionPolicy(candidate_niches=[]), model=None, min_population_size=3)

    assert population.candidates
    assert all(candidate.metadata.get("search_seed_not_final") for candidate in population.candidates)
    assert not any(getattr(candidate, "patch_set", []) for candidate in population.candidates)
    assert not any("NEXUS_SEED_NOTE" in str(candidate.to_dict()) for candidate in population.candidates)


def test_model_defined_search_planes_drive_project_seed_breadth_without_low_level_default() -> None:
    world = type("ProjectWorld", (), {"kind": "project", "edge_seed_pool": []})()
    contract = NexusObjectiveContract(
        original_user_goal="Find a more elegant self-evolution core architecture.",
        normalized_goal="find elegant self-evolution core architecture",
        outcome_policy={
            "search_space_plan": {
                "candidate_families": [
                    {"id": "candidate_lifecycle", "description": "candidate survival, repair, and parenthood"},
                    {"id": "search_distribution", "description": "how exploration breadth is allocated"},
                    {"id": "final_answer_boundary", "description": "when advisory final telemetry should be recorded"},
                ]
            }
        },
    )

    population = seed_population(contract=contract, world=world, policy=EvolutionPolicy(candidate_niches=[]), model=None, min_population_size=3)
    niches = {candidate.metadata["search_space"]["family_id"] for candidate in population.candidates}

    assert {"candidate_lifecycle", "search_distribution", "final_answer_boundary"}.issubset(niches)
    assert "minimal_patch" not in niches
