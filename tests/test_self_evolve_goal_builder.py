from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_build_self_evolve_goal_composes_preamble_and_markdown_inputs(tmp_path: Path) -> None:
    preamble = tmp_path / "goal-preamble.md"
    preamble.write_text("# 探索版目标\n允许结构化设计候选。", encoding="utf-8")
    input_zip = tmp_path / "input.zip"
    with zipfile.ZipFile(input_zip, "w") as archive:
        archive.writestr("00-context.md", "# Context\nA")
        archive.writestr("06-plan.md", "# Plan\nB")
        archive.writestr("notes/raw.txt", "not included by default")
    output = tmp_path / "bootstrap-goal.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build-self-evolve-goal.py",
            "--input-zip",
            str(input_zip),
            "--goal-preamble",
            str(preamble),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert str(output) in completed.stdout
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# 探索版目标")
    assert "## 00-context.md" in text
    assert "## 06-plan.md" in text
    assert "not included by default" not in text


def test_build_self_evolve_goal_rejects_zip_traversal(tmp_path: Path) -> None:
    preamble = tmp_path / "goal-preamble.md"
    preamble.write_text("safe", encoding="utf-8")
    input_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(input_zip, "w") as archive:
        archive.writestr("../escape.md", "bad")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build-self-evolve-goal.py",
            "--input-zip",
            str(input_zip),
            "--goal-preamble",
            str(preamble),
            "--output",
            str(tmp_path / "out.md"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "unsafe zip member" in completed.stderr
