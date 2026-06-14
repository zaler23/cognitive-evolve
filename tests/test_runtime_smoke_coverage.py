from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.runtime import runtime_run, runtime_status
from cognitive_evolve_runtime.nexus.evaluation import write_runtime_validation_report


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_backed_runtime_run_writes_valid_artifacts(tmp_path, monkeypatch, capsys) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    fixture = ROOT / "tests" / "fixtures" / "llm_fixture.json"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(fixture))
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime-root"))

    rc = runtime_run(str(task_dir), "agent系统 演化 调优 架构 冲突", activate_all=True, rounds=2)

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
