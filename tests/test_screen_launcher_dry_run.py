from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch-core-self-evolve-openai-screen.sh"


def test_screen_launcher_dry_run_prints_safe_plan_without_starting_services() -> None:
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "--dry-run",
            "--label",
            "pytest-launch",
            "--not-before",
            "2099-01-01T00:00:00+00:00",
            "--require-upstream-health",
            "--upstream-health-url",
            "http://127.0.0.1:9/health",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DRY_RUN" in completed.stdout
    assert "cogev_pytest-launch_services" in completed.stdout
    assert "cogev_pytest-launch_runner_" in completed.stdout
    assert "--require-upstream-health" in completed.stdout
    assert "--not-before 2099-01-01T00:00:00+00:00" in completed.stdout
