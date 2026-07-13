from __future__ import annotations

from cognitive_evolve_runtime.doctor import doctor
from cognitive_evolve_runtime.validation import project_health, source_runtime_sync


def test_doctor_core_is_hermetic_without_llm_fixture(monkeypatch) -> None:
    monkeypatch.delenv("COGEV_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("COGEV_LLM_FIXTURE", raising=False)
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    assert doctor("core") == 0


def test_installed_runtime_core_health_does_not_require_source_checkout_files(tmp_path, monkeypatch) -> None:
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir()
    runtime_root = tmp_path / "runtime"
    tasks_root = runtime_root / ".cogev" / "tasks"

    monkeypatch.setattr(project_health, "ROOT", installed_root)
    monkeypatch.setattr(project_health, "COGEV", installed_root / ".cogev")
    monkeypatch.setattr(project_health, "LOCAL_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(project_health, "TASKS", tasks_root)
    monkeypatch.setattr(source_runtime_sync, "ROOT", installed_root)

    assert project_health.source_checkout_detected() is False
    assert all(condition for condition, _message in project_health.core_project_validation())
    assert source_runtime_sync.source_runtime_coverage_validation() == [
        (True, "source/runtime mirror coverage is not applicable to an installed runtime")
    ]


def test_source_checkout_core_health_still_requires_source_files(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "pyproject.toml").write_text("", encoding="utf-8")

    monkeypatch.setattr(project_health, "ROOT", source_root)
    monkeypatch.setattr(project_health, "COGEV", source_root / ".cogev")
    monkeypatch.setattr(project_health, "LOCAL_RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(project_health, "TASKS", tmp_path / "runtime" / ".cogev" / "tasks")

    checks = project_health.core_project_validation()
    assert project_health.source_checkout_detected() is True
    assert (False, "source directory exists: scripts") in checks
    assert (False, "source file exists: README.md") in checks
