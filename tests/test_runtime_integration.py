from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.artifacts.task_files import new_task
from cognitive_evolve_runtime.doctor import doctor
from cognitive_evolve_runtime.runtime import runtime_run, runtime_status
from cognitive_evolve_runtime.nexus.evaluation import runtime_validation_run

import cognitive_evolve_runtime.core as core
import cognitive_evolve_runtime.core.paths as paths
import cognitive_evolve_runtime.artifacts.store as store
import cognitive_evolve_runtime.artifacts.task_files as task_files
import cognitive_evolve_runtime.validation.project_health as project_health
import cognitive_evolve_runtime.validation.standalone_runtime as standalone_runtime


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_backed_runtime_smoke_passes_core_and_runtime_doctor(tmp_path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    tasks_root = runtime_root / ".cogev" / "tasks"
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("COGEV_TASKS_ROOT", str(tasks_root))
    for module in (paths, core, store, task_files, project_health, standalone_runtime):
        monkeypatch.setattr(module, "LOCAL_RUNTIME_ROOT", runtime_root, raising=False)
        monkeypatch.setattr(module, "TASKS", tasks_root, raising=False)
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(ROOT / "tests" / "fixtures" / "llm_fixture.json"))
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    task_dir = new_task("test", "fixture-runtime-smoke")

    assert runtime_run(str(task_dir), "Refactor architecture boundaries and verify release readiness", activate_all=True, rounds=1) == 0
    assert runtime_status(str(task_dir)) == 0
    assert runtime_validation_run(str(task_dir)) == 0
    assert doctor("core") == 0
    assert doctor("runtime") == 0

    state = json.loads((task_dir / "runtime-state.json").read_text(encoding="utf-8"))
    assert state["version"] == "2.0"
    assert state["interaction_mode"] == "one_shot"
    assert state["external_questions_allowed"] is False
    assert state["runtime_path"] == "nexus"
    assert state["status"] == "completed"
    assert state["verification_results"]["passed"] is True
    assert state["single_runtime"]["source_of_truth"] == "NexusRuntime"

    native_eval = json.loads((task_dir / "evaluations" / "native-eval-report.json").read_text(encoding="utf-8"))
    assert native_eval["suite"] == "runtime-validation"
    assert native_eval["status"] == "pass"
    assert native_eval["passed"] == native_eval["total"]
