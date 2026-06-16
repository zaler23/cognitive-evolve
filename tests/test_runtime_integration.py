from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.artifacts.task_files import new_task
from cognitive_evolve_runtime.doctor import doctor
from cognitive_evolve_runtime.runtime import _seed_prompt_with_artifact_policy, runtime_run, runtime_status
from cognitive_evolve_runtime.nexus.semantics import ensure_enhanced_task_contract
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


def test_runtime_artifact_policy_hint_is_appended_from_adaptive_config() -> None:
    prompt = "Design a general policy artifact."

    augmented = _seed_prompt_with_artifact_policy(
        prompt,
        {
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
                "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
                "field_aliases": {"eviction_scoring": "eviction"},
                "metadata": {"domain_vocabulary": ["cache", "hit", "miss"], "forbidden_semantic_terms": ["checkpoint"]},
            }
        },
    )

    assert augmented.startswith(prompt)
    assert "CognitiveEvolve machine artifact contract" in augmented
    assert "artifact_type exactly: `cache_policy`" in augmented
    assert "`admission`, `eviction`, `parameters`, `update_or_state_update`" in augmented
    assert "Do not use field aliases: `eviction_scoring`" in augmented
    assert "Avoid forbidden semantic terms" in augmented


def test_semantic_intake_preserves_user_research_brief_and_writes_route_summary(tmp_path: Path) -> None:
    user_brief = "# Research Brief\n\nThis is the full user-authored benchmark specification.\n"
    (tmp_path / "research-brief.md").write_text(user_brief, encoding="utf-8")
    long_prompt = "Full task prompt. " * 80
    short_prompt = "short route summary"

    ensure_enhanced_task_contract(tmp_path, long_prompt, force=True)
    ensure_enhanced_task_contract(tmp_path, short_prompt, force=True)

    assert (tmp_path / "research-brief.md").read_text(encoding="utf-8") == user_brief
    assert "Nexus Route Summary" in (tmp_path / "intake" / "route-summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "intake" / "user-input.md").read_text(encoding="utf-8") == long_prompt
