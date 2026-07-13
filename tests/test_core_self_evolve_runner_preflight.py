from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-core-self-evolve-openai.py"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["COGEV_HERMETIC_TEST"] = "1"
    env["COGEV_LLM_PROVIDER"] = "litellm"
    env["COGEV_LLM_MODEL"] = "openai/example-model"
    env["COGEV_LLM_API_BASE"] = "https://provider.example/v1"
    env["COGEV_LLM_API_KEY"] = "example-upstream-key"
    return env


def test_runner_dry_run_reports_quota_and_upstream_preflight_without_model_call() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--project-dir",
            str(ROOT),
            "--dry-run",
            "--label",
            "pytest-preflight",
            "--not-before",
            "2099-01-01T00:00:00+00:00",
            "--require-upstream-health",
            "--upstream-health-url",
            "http://127.0.0.1:9/health",
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run"
    assert payload["preflight"]["ok"] is False
    assert any(item.startswith("waiting_until:") for item in payload["preflight"]["blockers"])
    assert "upstream_health_unavailable" in payload["preflight"]["blockers"]
    assert payload["llm_temperature"] == "0.7"


def test_runner_non_dry_run_stops_before_model_call_when_not_before_is_future(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--project-dir",
            str(ROOT),
            "--run-dir",
            str(run_dir),
            "--not-before",
            "2099-01-01T00:00:00+00:00",
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )

    assert completed.returncode == 75, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "waiting_preflight"
    status = json.loads((run_dir / "self-evolve-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "waiting_preflight"
    assert status["preflight"]["ok"] is False
    assert not (run_dir / "self-evolve-result.json").exists()
    assert not (run_dir / "self-evolve-error.txt").exists()
