from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.runtime import runtime_run, runtime_status
from cognitive_evolve_runtime.nexus.evaluation import write_runtime_validation_report
from cognitive_evolve_runtime.persistence.transactional_snapshot import NexusSnapshotTransaction, SnapshotWrite


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_backed_runtime_run_writes_valid_artifacts(tmp_path, monkeypatch, capsys) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    fixture = ROOT / "tests" / "fixtures" / "llm_fixture.json"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(fixture))
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime-root"))

    rc = runtime_run(str(task_dir), "agent system evolution tuning architecture conflict", activate_all=True, rounds=2)

    assert rc == 0
    state = json.loads((task_dir / "runtime-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["single_runtime"]["source_of_truth"] == "NexusRuntime"
    assert state["interaction_mode"] == "one_shot"
    assert state["external_questions_allowed"] is False
    assert state["nexus_evolution"]["actual_rounds"] >= 1
    assert (task_dir / "nexus-runtime" / "run-result.json").exists()
    assert (task_dir / "nexus-runtime" / "population.json").exists()

    validation = write_runtime_validation_report(task_dir)
    assert validation["suite"] == "runtime-validation"
    assert validation["status"] == "pass"

    assert runtime_status(str(task_dir)) == 0
    out = capsys.readouterr().out
    assert "status: completed" in out


def test_runtime_validation_treats_not_run_as_non_blocking_when_runtime_integrity_passed(tmp_path) -> None:
    task_dir = tmp_path / "task"
    nexus_dir = task_dir / "nexus-runtime"
    evaluations = task_dir / "evaluations"
    evaluations.mkdir(parents=True)
    (evaluations / "llm-runtime-report.json").write_text("{}\n", encoding="utf-8")
    (nexus_dir / "events.jsonl").parent.mkdir(parents=True)
    (nexus_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (task_dir / "runtime-state.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "runtime_path": "nexus",
                "single_runtime": {"enforced": True},
                "interaction_mode": "one_shot",
                "external_questions_allowed": False,
                "nodes": [{"id": "nexus_runtime"}],
                "nexus_evolution": {"actual_rounds": 1},
                "nexus_search": {"archive_summary": {}},
                "verification_results": {
                    "passed": None,
                    "validation_status": "not_run",
                    "runtime_integrity_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    NexusSnapshotTransaction(nexus_dir).commit(
        [
            SnapshotWrite("run-result.json", "json", {"status": "best_current"}),
            SnapshotWrite("population.json", "json", {"candidates": [{"id": "C1"}]}),
            SnapshotWrite("archives.json", "json", {"archive_schema": {}}),
            SnapshotWrite("checkpoint.json", "json", {"round": 1}),
            SnapshotWrite("final-answer.md", "text", "best current\n", sort_keys=False),
        ]
    )

    report = write_runtime_validation_report(task_dir)

    assert report["status"] == "pass"
    assert report["summary"]["validation_status"] == "not_run"
