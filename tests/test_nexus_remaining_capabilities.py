from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.candidates.patch_merge import merge_patch_sets
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError, StructuredModelAdapter
from cognitive_evolve_runtime.nexus.protocols import NexusModelProtocol
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


def test_archive_update_applies_assignments_once() -> None:
    candidate = ProjectCandidateGenome(id="P1", patch_set=[PatchOperation(path="a.py", content="x=1\n")])
    archives = ArchiveManager()

    applied = archives.update([FateAssignment("P1", CandidateFate.ELITE)], candidates=[candidate])

    assert len(applied) == 1
    assert candidate.current_fate == CandidateFate.ELITE
    assert list(archives.answer_archive) == ["P1"]
    assert len(archives.history) == 1
    assert archives.project_patch_archive["P1"]["patch_set"][0]["path"] == "a.py"


def test_project_population_roundtrip_preserves_patch_candidates() -> None:
    population = CandidatePopulation([
        ProjectCandidateGenome(id="P2", patch_set=[PatchOperation(path="pkg/mod.py", operation="append", content="\n")])
    ])

    loaded = CandidatePopulation.from_dict(population.to_dict())

    assert isinstance(loaded.candidates[0], ProjectCandidateGenome)
    assert loaded.candidates[0].patch_set[0].path == "pkg/mod.py"


def test_project_mutation_preserves_patch_set_and_adds_rare_seed() -> None:
    parent = ProjectCandidateGenome(
        id="P3",
        edge_knowledge_seeds=["old_seed"],
        patch_set=[PatchOperation(path="mod.py", operation="write", content="x=1\n")],
    )

    child = MutationEngine().mutate(parent, MutationPlan(operator=MutationOperator.RARE_INJECT, rarity_seed="edge_seed"))

    assert isinstance(child, ProjectCandidateGenome)
    assert child.parent_ids == ["P3"]
    assert any(op.path == "NEXUS_RARE_SEED.md" for op in child.patch_set)
    assert "edge_seed" in child.edge_knowledge_seeds


def test_project_patch_crossover_detects_conflicts() -> None:
    left = [PatchOperation(path="mod.py", operation="write", content="x=1\n")]
    right = [PatchOperation(path="mod.py", operation="write", content="x=2\n")]

    merged = merge_patch_sets(left, right)

    assert not merged.clean
    assert merged.conflicts[0].path == "mod.py"
    assert merged.conflicts[0].reason == "same_path_incompatible_patch_operations"


def test_structured_model_adapter_validates_response_schema() -> None:
    def caller(request_type: str, payload: dict, schema: dict) -> dict:
        assert request_type == "nexus_request_context"
        return {"need_files": ["a.py"], "need_symbols": [], "need_tests": [], "reason": "unit test"}

    adapter = StructuredModelAdapter(caller=caller)

    result = adapter.request_context(contract={}, world={}, parents=[], archives={})

    assert isinstance(adapter, NexusModelProtocol)
    assert result["need_files"] == ["a.py"]


def test_structured_model_adapter_rejects_invalid_response() -> None:
    adapter = StructuredModelAdapter(caller=lambda *_: {"need_files": "not-a-list", "need_symbols": [], "need_tests": [], "reason": "bad"})

    with pytest.raises(ModelResponseSchemaError, match="schema validation"):
        adapter.request_context(contract={}, world={}, parents=[], archives={})


def test_nexus_project_runtime_verifies_context_and_resume(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    out = tmp_path / "out"

    result = NexusRuntime(output_dir=out).run_project(repo, user_goal="Improve mod.py safely", max_rounds=1)

    assert result.mode == "project"
    assert result.context_protocol["packets"]
    assert all(summary["patch_result"]["status"] == "applied" for summary in result.verification_summaries)
    assert "NEXUS_SEED_NOTE" not in json.dumps(result.evolution["population"], ensure_ascii=False)
    assert (out / "run-result.json").exists()
    assert any(event.get("type") == "pipeline_progress" for event in result.pipeline_events)

    checkpoint = CheckpointStore(out / "checkpoint.json").load()
    assert checkpoint is not None
    assert checkpoint.contract["normalized_goal"] == "Improve mod.py safely"
    assert checkpoint.world["project_world_model"]["kind"] == "project"

    resumed = NexusRuntime(output_dir=out).resume_from_checkpoint(max_rounds=2)
    assert resumed.evolution["progress_events"][-1]["round"] == 2


def test_model_project_seed_population_preserves_project_candidate_type() -> None:
    from cognitive_evolve_runtime.contracts.objective_contract import NexusProjectObjectiveContract
    from cognitive_evolve_runtime.nexus.loop import seed_population
    from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy

    class World:
        kind = "project"
        edge_seed_pool: list[str] = []

    patch_candidate = ProjectCandidateGenome(
        id="MP1",
        artifact_type="project_patch",
        patch_set=[PatchOperation(path="pkg/x.py", operation="write", content="x=1\n")],
    )

    class Model:
        def seed_population(self, **_: object) -> list[dict]:
            return [patch_candidate.to_dict()]

    population = seed_population(
        contract=NexusProjectObjectiveContract(original_user_goal="g", normalized_goal="g"),
        world=World(),
        policy=EvolutionPolicy(),
        model=Model(),
    )

    assert isinstance(population.candidates[0], ProjectCandidateGenome)
    assert population.candidates[0].patch_set[0].path == "pkg/x.py"
