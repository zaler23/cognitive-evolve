from __future__ import annotations

from pathlib import Path
import sys

import pytest

from cognitive_evolve_runtime import commands
from cognitive_evolve_runtime.nexus.semantics import NexusRoute as Route


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["cogev", *argv])
    return commands.main()


def test_build_routed_prompt_and_standalone_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "intake").mkdir(parents=True)
    (task_dir / "intake" / "enhanced-task-contract.md").write_text("contract", encoding="utf-8")
    route = Route("L4_evolutionary", "deep", True, True, True, "architecture task")
    routed = commands.build_routed_prompt("refactor architecture", route, task_dir=task_dir)
    assert "independent review" in routed.lower()
    assert "One-shot policies" in routed
    assert str(task_dir) in routed

    monkeypatch.setattr(commands, "classify", lambda prompt: route)
    monkeypatch.setattr(commands, "new_task", lambda task_type, slug: task_dir)
    monkeypatch.setattr(commands, "ensure_enhanced_task_contract", lambda *args, **kwargs: {})
    assert commands.run_standalone("refactor architecture", dry_run=True) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "L4_evolutionary" in out


def test_command_router_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[str, object]] = []
    task_dir = tmp_path / "new-task"
    task_dir.mkdir()

    def mark(name: str, rc: int = 0):
        def inner(*args, **kwargs):
            calls.append((name, (args, kwargs)))
            return rc
        return inner

    monkeypatch.setattr(commands, "new_task", lambda task_type, slug: task_dir)
    monkeypatch.setattr(commands, "ensure_enhanced_task_contract", mark("intake", rc={}))
    monkeypatch.setattr(commands, "list_tasks", mark("list"))
    monkeypatch.setattr(commands, "config_init", mark("config", rc=0))
    monkeypatch.setattr(commands, "doctor", mark("doctor"))
    monkeypatch.setattr(commands, "check_task", mark("check"))
    monkeypatch.setattr(commands, "enhance_request", mark("enhance"))
    monkeypatch.setattr(commands, "list_capabilities", mark("cap-list"))
    monkeypatch.setattr(commands, "list_ports", mark("cap-ports"))
    monkeypatch.setattr(commands, "show_capability", mark("cap-show"))
    monkeypatch.setattr(commands, "select_capabilities", mark("cap-select"))
    monkeypatch.setattr(commands, "runtime_run", mark("runtime-run"))
    monkeypatch.setattr(commands, "runtime_status", mark("runtime-status"))
    monkeypatch.setattr(commands, "native_eval_run", mark("eval"))
    monkeypatch.setattr(commands, "native_optimize_run", mark("optimize"))
    monkeypatch.setattr(commands, "route_prompt", mark("route"))
    monkeypatch.setattr(commands, "load_service_env", mark("env", rc=Path(".env")))
    monkeypatch.setattr(commands, "llm_status_cli", mark("llm"))
    monkeypatch.setattr(commands, "api_status_cli", mark("api-status"))
    monkeypatch.setattr(commands, "api_serve", mark("api-serve"))
    monkeypatch.setattr(commands, "run_standalone", mark("run"))
    monkeypatch.setattr(commands, "quickstart", mark("quickstart"))

    cases = [
        ["new", "--type", "general", "--slug", "hello"],
        ["new", "--type", "general", "--slug", "hello", "--no-enhance"],
        ["list"],
        ["config", "init", "--profile", "fixture", "--output", str(tmp_path / "generated.env")],
        ["doctor", "--scope", "core"],
        ["check", str(tmp_path)],
        ["enhance", "--path", str(tmp_path), "--json", "build", "plan"],
        ["capability", "list"],
        ["capability", "ports"],
        ["capability", "show", "task_scoping"],
        ["capability", "select", "architecture", "adapter"],
        ["runtime", "run", str(tmp_path), "--prompt", "go", "--all", "--rounds", "2"],
        ["runtime", "status", str(tmp_path)],
        ["eval", "run", str(tmp_path)],
        ["optimize", "run", str(tmp_path), "--source", "final.md"],
        ["route", "analyze", "architecture"],
        ["llm", "status"],
        ["api", "status"],
        ["api", "serve"],
        ["run", "--dry-run", "do", "work"],
        ["quickstart", "do", "work"],
    ]
    for argv in cases:
        assert _run_cli(monkeypatch, argv) == 0

    assert _run_cli(monkeypatch, ["capability", "select"]) == 2
    assert _run_cli(monkeypatch, ["route"]) == 2
    assert _run_cli(monkeypatch, ["run"]) == 2
    assert _run_cli(monkeypatch, ["quickstart"]) == 2
    stderr = capsys.readouterr().err
    assert "Missing prompt" in stderr
    called_names = [name for name, _ in calls]
    assert "config" in called_names
    assert "runtime-run" in called_names and "api-serve" in called_names and "llm" in called_names
    assert "quickstart" in called_names


def test_quickstart_writes_minimal_env_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    answers = iter(["direct_http", "openai/test", "2", "3"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(commands.getpass, "getpass", lambda _prompt="": "secret-key")
    monkeypatch.setattr(commands, "load_service_env", lambda: tmp_path / ".env")
    monkeypatch.setattr(commands, "run_standalone", lambda prompt: 0 if prompt == "do work" else 1)

    assert commands.quickstart("do work") == 0
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "COGEV_LLM_PROVIDER=direct_http" in env_text
    assert "COGEV_LLM_MODEL=openai/test" in env_text
    assert "COGEV_LLM_API_KEY=secret-key" in env_text
    assert "COGEV_LLM_MAX_CONCURRENT=2" in env_text
    assert "COGEV_NEXUS_MAX_ROUNDS=3" in env_text
