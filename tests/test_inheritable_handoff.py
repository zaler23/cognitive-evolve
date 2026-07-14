from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from cognitive_evolve_runtime.archives.failure import FailureArchive
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.handoff import (
    build_inheritable_handoff,
    load_inherited_gene_entries,
)
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopResult
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.nexus.runtime import NexusRunResult, NexusRuntime
from cognitive_evolve_runtime.nexus.runtime_services import NexusPersistenceService
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult


def _population_and_failures() -> tuple[CandidatePopulation, FailureArchive]:
    live_a = CandidateGenome(
        id="live-a",
        artifact={"answer": "A"},
        core_mechanism="live mechanism A",
        edge_knowledge_seeds=["edge A"],
    )
    live_b = CandidateGenome(id="live-b", artifact={"answer": "B"}, core_mechanism="live mechanism B")
    failure_only = CandidateGenome(
        id="failed-z",
        artifact={"discarded": True},
        core_mechanism="failure mechanism",
        failure_lessons=["counterexample lesson"],
    )
    failures = FailureArchive()
    failures.add(failure_only, signature="counterexample-z")
    failures.records["live-a"] = {
        "candidate_id": "live-a",
        "failure_signature": "old failure A",
        "inherited_gene_summary": "stale compacted summary",
        "covered_by": "",
        "future_reactivation_condition": "new evidence A",
    }
    return CandidatePopulation([live_a, live_b]), failures


def test_handoff_is_stable_union_of_final_population_and_failure_archive(tmp_path: Path) -> None:
    population, failures = _population_and_failures()

    handoff = build_inheritable_handoff(
        population=population,
        failure_archive=failures,
        source_run_id="run-1",
        project_signature="project-1",
    )
    entries = handoff["entries"]

    assert [entry["candidate_id"] for entry in entries] == ["live-a", "live-b", "failed-z"]
    assert entries[0]["sources"] == ["final_population", "failure_archive"]
    assert entries[0]["gene_summary"] == population.candidates[0].extract_inheritable_gene_summary()
    assert entries[0]["failure_signature"] == "old failure A"
    assert entries[0]["future_reactivation_condition"] == "new evidence A"
    assert entries[2]["artifact_digest"] is None
    assert entries[2]["gene_summary"] == failures.records["failed-z"]["inherited_gene_summary"]
    assert all(entry["entry_digest"] for entry in entries)

    path = tmp_path / "inheritable-handoff.v1.json"
    path.write_text(json.dumps(handoff, ensure_ascii=False), encoding="utf-8")
    selected = load_inherited_gene_entries(path, ["failed-z", "live-a"])

    assert [entry["candidate_id"] for entry in selected] == ["failed-z", "live-a"]
    assert selected[0]["gene_summary"] == entries[2]["gene_summary"]
    assert selected[1]["gene_summary"] == entries[0]["gene_summary"]
    assert set(selected[0]) == {"candidate_id", "gene_summary"}


def test_handoff_selection_requires_path_and_ids_together_and_validates_digest(tmp_path: Path) -> None:
    population, failures = _population_and_failures()
    handoff = build_inheritable_handoff(
        population=population,
        failure_archive=failures,
        source_run_id="run-1",
        project_signature="project-1",
    )
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")

    with pytest.raises(ValueError, match="must be provided together"):
        load_inherited_gene_entries(path, None)
    with pytest.raises(ValueError, match="must be provided together"):
        load_inherited_gene_entries(None, ["live-a"])

    handoff["entries"][0]["gene_summary"] = "tampered"
    path.write_text(json.dumps(handoff), encoding="utf-8")
    with pytest.raises(ValueError, match="entry_digest mismatch: live-a"):
        load_inherited_gene_entries(path, ["live-a"])


def test_persistence_writes_handoff_inside_runtime_artifact_root(tmp_path: Path) -> None:
    population, failures = _population_and_failures()
    archives = ArchiveManager()
    archives.failure_archive = failures
    result = EvolutionLoopResult(
        population=population,
        archives=archives,
        policy=EvolutionPolicy(),
        diagnosis=SearchDiagnosis(),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate output"),
        current_round=0,
        max_rounds=1,
        stop_reason="max_rounds",
        completion_status="completed",
    )
    run = NexusRunResult(
        mode="project",
        contract={"contract_hash": "contract-1"},
        policy={},
        world={"snapshot": {"root_hash": "project-root-hash"}},
        evolution={},
    )

    artifacts = NexusPersistenceService(output_dir=tmp_path).persist(
        run,
        result,
        contract={},
        world=run.world,
        budget_history=[],
        budget=EvolutionBudget(max_rounds=1),
    )

    path = tmp_path / "inheritable-handoff.v1.json"
    assert path.exists()
    assert artifacts["inheritable_handoff"] == "inheritable-handoff.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["project_signature"] == "project-root-hash"
    assert [entry["candidate_id"] for entry in payload["entries"]] == ["live-a", "live-b", "failed-z"]


def test_inherited_gene_entries_are_protected_and_never_truncated() -> None:
    entries = [
        {"candidate_id": "B", "gene_summary": "BEGIN-B\n" + "B" * 4000 + "\nEND-B"},
        {"candidate_id": "A", "gene_summary": "BEGIN-A\n" + "A" * 4000 + "\nEND-A"},
    ]
    view = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": {},
            "world": {},
            "policy": {},
            "source_context": {"inherited_gene_entries": entries},
        },
        max_chars=500,
        schema_hint={"type": "object"},
    )

    assert view.payload["source_context"]["inherited_gene_entries"] == entries
    assert view.metadata["protected_over_budget"] is True


def test_inherited_context_over_cap_fails_before_provider_with_selection_details(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit-test-model")
    entries = [{"candidate_id": "A", "gene_summary": "A" * 2000}]

    class NeverProvider:
        provider_id = "never"

        def complete_json(self, **kwargs):  # noqa: ANN003
            pytest.fail("provider must not be called for an over-cap inherited handoff")

    with pytest.raises(LLMResponseError, match=r"selected_ids=\['A'\].*inherited_chars=.*cap=400"):
        llm_json(
            "nexus_seed_population",
            {"source_context": {"inherited_gene_entries": entries}},
            system="Return JSON",
            schema_hint={},
            provider=NeverProvider(),
            request_policy=LLMRequestPolicy(structured_prompt=True, max_prompt_chars=400),
        )


def test_public_runtime_and_orchestrator_forward_explicit_handoff_selection(monkeypatch, tmp_path: Path) -> None:
    assert "inherited_handoff_path" in inspect.signature(NexusRuntime.run_text).parameters
    assert "inherited_candidate_ids" in inspect.signature(NexusRuntime.run_text).parameters
    assert "inherited_handoff_path" in inspect.signature(NexusRuntime.run_project).parameters
    assert "inherited_candidate_ids" in inspect.signature(NexusRuntime.run_project).parameters
    captured: dict[str, object] = {}

    def fake_run_text(self, text: str, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return NexusRunResult(mode="text", contract={}, policy={}, world={}, evolution={})

    monkeypatch.setattr(NexusRuntime, "run_text", fake_run_text)
    EngineOrchestrator(model=object()).run(
        "goal",
        context={
            "inherited_handoff_path": str(tmp_path / "handoff.json"),
            "inherited_candidate_ids": ["B", "A"],
        },
    )

    assert captured["inherited_handoff_path"] == str(tmp_path / "handoff.json")
    assert captured["inherited_candidate_ids"] == ["B", "A"]
