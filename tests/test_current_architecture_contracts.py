from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from cognitive_evolve_runtime.engine.result import NexusEngineResult
from cognitive_evolve_runtime.nexus.state import nexus_runtime_state, nexus_search_state

ROOT = Path(__file__).resolve().parents[1]


def _minimal_result(prompt: str) -> NexusEngineResult:
    evolution = {"synthesis": {"final_answer": "ok"}, "population": {"candidates": []}, "archives": {}, "progress_events": []}
    return NexusEngineResult(
        prompt=prompt,
        mode="text",
        contract={"normalized_goal": prompt},
        policy={},
        world={},
        evolution=evolution,
        verification_results={"passed": True},
    )


def test_absent_import_paths_do_not_resolve() -> None:
    removed = [
        "cognitive_evolve_runtime.objective_contract",
        "cognitive_evolve_runtime.evidence_planner",
        "cognitive_evolve_runtime.evidence_ledger",
        "cognitive_evolve_runtime.candidates.elo",
        "cognitive_evolve_runtime.archive",
        "cognitive_evolve_runtime.optimizer",
        "cognitive_evolve_runtime.adaptive_engine",
        "cognitive_evolve_runtime.candidate_search",
        "cognitive_evolve_runtime.multi_agent_optimizer",
        "cognitive_evolve_runtime.selection_contracts",
        "cognitive_evolve_runtime.routing",
        "cognitive_evolve_runtime.semantic_controller",
        "cognitive_evolve_runtime.intake",
        "cognitive_evolve_runtime.intake_contracts",
        "cognitive_evolve_runtime.intake_questions",
        "cognitive_evolve_runtime.optimization",
        "cognitive_evolve_runtime.native_eval",
        "cognitive_evolve_runtime.runtime_validation",
        "cognitive_evolve_runtime.llm_client",
        "cognitive_evolve_runtime.request_context",
        "cognitive_evolve_runtime.state_contract",
        "cognitive_evolve_runtime.budget_policy",
        "cognitive_evolve_runtime.capability_runtime",
        "cognitive_evolve_runtime.search_space",
        "cognitive_evolve_runtime.search_descriptors",
        "cognitive_evolve_runtime.semantic_adapter",
        "cognitive_evolve_runtime.textual_gradient",
        "cognitive_evolve_runtime.verifier_stack",
        "cognitive_evolve_runtime.task_types",
    ]
    for module_name in removed:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_canonical_imports_are_direct() -> None:
    import cognitive_evolve_runtime.contracts.objective_contract as contracts
    import cognitive_evolve_runtime.evidence.planner as planner
    import cognitive_evolve_runtime.evidence.ledger as ledger
    import cognitive_evolve_runtime.ranking.multihead_elo as elo

    assert hasattr(contracts, "NexusObjectiveContract")
    assert hasattr(planner, "EvidencePlanner")
    assert hasattr(ledger, "EvidenceLedger")
    assert hasattr(elo, "MultiHeadElo")


def test_nexus_engine_result_to_dict_is_canonical() -> None:
    result = _minimal_result("x")
    data = result.to_dict()
    assert data["runtime_architecture"] == "nexus"
    assert data["runtime_path"] == "nexus"
    assert data["final_answer"] == "ok"
    assert "adaptive" not in str(data).lower()


def test_runtime_state_uses_current_single_runtime_key(tmp_path: Path) -> None:
    run_data = {
        "contract": {"normalized_goal": "x"},
        "policy": {},
        "evolution": {"synthesis": {"final_answer": "ok"}, "population": {"candidates": []}, "archives": {}, "progress_events": []},
        "artifacts": {},
    }
    state = nexus_runtime_state(task_dir=tmp_path, prompt="x", run_data=run_data, selected_capabilities=[])
    assert state["single_runtime"] == {"enforced": True, "source_of_truth": "NexusRuntime"}
    assert "search_state" not in state and "internal_evolution" not in state


def test_nexus_search_uses_selected_candidate_not_frontier_schema() -> None:
    state = nexus_search_state({"evolution": {"synthesis": {"best_candidate_id": "C1"}, "population": {"candidates": []}, "archives": {}}})
    assert state["selected_candidate"]["id"] == "C1"
    assert "selection" in state and "pareto_frontier" not in state


def test_pyproject_declares_nexus_as_only_runtime_authority() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'runtime_authority = "nexus"' in text
    assert "alternate_runtime_backends = []" in text
    assert "local_runtime_backends" not in text
    assert "optional_external_backends" not in text
    assert "[project.optional-dependencies]\nexternal-backends" not in text


def test_core_algorithm_documentation_states_model_driven_evolution_boundary() -> None:
    text = (ROOT / "docs" / "CORE_EVOLVE_ALGORITHM.md").read_text(encoding="utf-8")

    assert "model-driven iterative search" in text
    assert "not a claim that every mutation/crossover" in text
    assert "biological genetic operator" in text
