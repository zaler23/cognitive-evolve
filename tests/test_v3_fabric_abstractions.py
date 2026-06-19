from __future__ import annotations

import json

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.fabric import (
    CandidateDossier,
    DossierIndexEntry,
    FabricCheckpointState,
    ExplorationTask,
    TaskGraph,
    TaskKind,
    TaskStatus,
    assert_advisory_payload,
)
from cognitive_evolve_runtime.persistence.checkpoint import NexusCheckpoint, build_checkpoint_state


def test_dossier_roundtrip_and_advisory_coercion() -> None:
    dossier = CandidateDossier.from_dict(
        {
            "candidate_id": "C1",
            "summary": "generic expansion",
            "expanded_content": {"shape": "domain neutral"},
            "applicability_bounds": ["bounded"],
            "assumptions": ["assumption"],
            "risks": ["risk"],
            "validation_paths": ["suggested validation"],
            "variants": ["variant"],
            "counterexamples": ["counterexample"],
            "differentiators": ["different"],
            "effect_hypotheses": ["effect"],
            "maturity_level": 2,
            "advisory": False,
        }
    )
    restored = CandidateDossier.from_dict(dossier.to_dict())
    assert restored.advisory is True
    assert restored.field_completeness == 10
    assert "advisory_false_coerced_to_true" in restored.diagnostics


def test_dossier_rejects_authority_payload() -> None:
    with pytest.raises(ValueError, match="verification-authority"):
        CandidateDossier.from_dict({"candidate_id": "C1", "expanded_content": {"objective_solved": True}})


def test_dossier_index_entry_is_bounded_and_advisory() -> None:
    entry = DossierIndexEntry(candidate_id="C1", summary="x" * 2000, field_completeness=7, content_ref="dossiers/C1.json", content_sha256="abc")
    data = entry.to_dict()
    assert data["advisory"] is True
    assert len(data["summary"]) == 1000
    assert DossierIndexEntry.from_dict(data).content_ref == "dossiers/C1.json"


def test_task_roundtrip() -> None:
    task = ExplorationTask(task_id="T1", kind=TaskKind.EVALUATE, target_ids=["C1"], priority=0.5, payload={"advisory_note": "ok"})
    restored = ExplorationTask.from_dict(task.to_dict())
    assert restored.task_id == "T1"
    assert restored.kind == TaskKind.EVALUATE
    assert restored.status == TaskStatus.PENDING
    assert restored.advisory is True


def test_task_rejects_non_verify_authority_payload() -> None:
    with pytest.raises(ValueError, match="verification-authority"):
        ExplorationTask(task_id="T1", kind=TaskKind.EXPAND, payload={"passed": True}).to_dict()
    verify_task = ExplorationTask(task_id="T2", kind=TaskKind.VERIFY, payload={"passed": True})
    assert verify_task.to_dict()["payload"]["passed"] is True


def test_task_graph_ready_topology_and_drained() -> None:
    graph = TaskGraph()
    graph.add(ExplorationTask(task_id="A", kind=TaskKind.EVALUATE))
    graph.add(ExplorationTask(task_id="B", kind=TaskKind.REPRODUCE, depends_on=["A"]))
    assert [task.task_id for task in graph.ready_tasks()] == ["A"]
    graph.mark("A", TaskStatus.DONE, {"ok": True})
    assert [task.task_id for task in graph.ready_tasks()] == ["B"]
    assert graph.topological_order() == ["A", "B"]
    graph.mark("B", TaskStatus.DONE)
    assert graph.is_drained() is True
    assert TaskGraph.from_dict(graph.to_dict()).is_drained() is True


def test_task_graph_cycle_detection() -> None:
    graph = TaskGraph()
    graph.add(ExplorationTask(task_id="A", kind=TaskKind.EVALUATE))
    graph.add(ExplorationTask(task_id="B", kind=TaskKind.REPRODUCE, depends_on=["A"]))
    graph.tasks["A"].depends_on = ["B"]
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_order()


def test_task_graph_recover_inflight() -> None:
    graph = TaskGraph({"A": ExplorationTask(task_id="A", kind=TaskKind.EVALUATE, status=TaskStatus.RUNNING)})
    graph.recover_inflight()
    assert graph.tasks["A"].status == TaskStatus.READY
    assert graph.tasks["A"].error["recovered_from_inflight"] is True


def test_old_checkpoint_resumes_without_fabric_fields() -> None:
    checkpoint = build_checkpoint_state(round=0, max_rounds=1, population=CandidatePopulation([CandidateGenome(id="C1")]), archives=ArchiveManager())
    data = checkpoint.to_dict()
    data.pop("fabric", None)
    restored = NexusCheckpoint.from_dict(data)
    assert restored.fabric == {}


def test_checkpoint_persists_fabric_state_without_embedding_large_sidecars() -> None:
    graph = TaskGraph({"A": ExplorationTask(task_id="A", kind=TaskKind.EVALUATE)})
    state = FabricCheckpointState(
        graph=graph,
        dossier_index={"C1": DossierIndexEntry(candidate_id="C1", summary="short", field_completeness=3, content_ref="dossiers/C1.json", content_sha256="sha")},
    )
    checkpoint = build_checkpoint_state(round=1, max_rounds=2, population=CandidatePopulation([CandidateGenome(id="C1")]), archives=ArchiveManager(), fabric=state.to_dict())
    payload = checkpoint.to_dict()
    assert payload["fabric"]["dossier_index"]["C1"]["content_ref"] == "dossiers/C1.json"
    assert "expanded_content" not in json.dumps(payload["fabric"], sort_keys=True)


def test_advisory_guard_rejects_nested_authority_keys() -> None:
    assert_advisory_payload({"nested": [{"priority": 0.1}]})
    with pytest.raises(ValueError, match="verification_result"):
        assert_advisory_payload({"nested": [{"verification_result": {"passed": True}}]})
