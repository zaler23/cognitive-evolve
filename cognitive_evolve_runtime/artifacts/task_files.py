#!/usr/bin/env python3
"""Task folder creation, skeleton, and artifact helpers."""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

from ..core import (
    TEMPLATES,
    TASKS,
    TEMPLATE_PLACEHOLDER_MARKERS,
)
from .store import (
    dt,
    Path,
    TASKS,
    ensure_dirs,
    slugify,
    latest_task_dir,
    _first_nonempty_line,
    _read,
)
TASK_TEMPLATE_FILES = {
    "research-brief.md": "research-brief.md",
    "problem-contract.md": "problem-contract.md",
    "decision-record.md": "decision-record.md",
    "validation-plan.md": "validation-plan.md",
    "feedback.md": "feedback.md",
    "working-memory.md": "working-memory.md",
    "candidates/candidate-001.md": "candidate.md",
    "evaluations/review-report.md": "review-report.md",
    "evaluations/checkmodel-report.md": "checkmodel-report.md",
}

def new_task(task_type: str, slug: str) -> Path:
    ensure_dirs()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{slugify(slug)}"
    task_dir = TASKS / name
    task_dir.mkdir(parents=True)
    for d in ["candidates", "evaluations", "evidence", "logs"]:
        (task_dir / d).mkdir()
    (task_dir / "trace.jsonl").write_text("", encoding="utf-8")
    for dest, tmpl in TASK_TEMPLATE_FILES.items():
        shutil.copyfile(TEMPLATES / tmpl, task_dir / dest)
    meta = task_dir / "task.yaml"
    meta.write_text(
        (
            f"task_type: {task_type}\n"
            f"slug: {slug}\n"
            f"created_at: {stamp}\n"
            "status: proposed\n"
            "level: pending_classification\n"
        ),
        encoding="utf-8",
    )
    latest = TASKS / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(task_dir.name)
    except OSError:
        (TASKS / "LATEST.txt").write_text(task_dir.name, encoding="utf-8")
    return task_dir

def _task_yaml_field(task_dir: Path, field: str) -> str:
    prefix = f"{field}:"
    for line in _read(task_dir / "task.yaml").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    return ""

def _task_seed_prompt(task_dir: Path, prompt: str | None = None) -> str:
    if prompt and prompt.strip():
        return prompt.strip()
    user_input = _read(task_dir / "intake" / "user-input.md")
    if user_input.strip() and not _is_template_text(user_input):
        legacy_fenced = _legacy_fenced_user_input(user_input)
        return legacy_fenced or user_input.strip()
    slug = _task_yaml_field(task_dir, "slug")
    if slug:
        return slug
    problem = _read(task_dir / "problem-contract.md")
    if problem and not _is_template_text(problem):
        return _first_nonempty_line(problem, task_dir.name)
    return task_dir.name

def _is_template_text(text: str) -> bool:
    """Return True only for untouched template artifacts.

    Earlier versions treated any occurrence of the word ``placeholder`` as an
    untouched template.  That caused real task artifacts to be repeatedly
    re-seeded when they merely *discussed* placeholder checks.  The LLM-first
    runtime now uses stricter structural markers plus explicit TBD/replace-this
    wording.
    """
    normalized = text.strip()
    low = normalized.lower()
    if low in {"tbd placeholder", "replace this placeholder before execution."}:
        return True
    if any(token in low for token in ["{{placeholder}}", "[placeholder]", "placeholder before execution"]):
        return True
    for marker in TEMPLATE_PLACEHOLDER_MARKERS:
        marker_low = marker.lower()
        if marker_low == "placeholder":
            continue
        if marker_low in low:
            return True
    return False


def _legacy_fenced_user_input(text: str) -> str:
    """Return payload from old all-fenced user-input files only.

    New runs write the user's prompt directly.  A real prompt may itself contain
    fenced code or text blocks, so a fence appearing anywhere in the file is not
    enough evidence to discard the surrounding objective.
    """

    stripped = text.strip()
    if not stripped.startswith("```text"):
        return ""
    body = stripped.split("```text", 1)[1]
    if "```" not in body:
        return ""
    payload, tail = body.split("```", 1)
    if tail.strip():
        return ""
    return payload.strip()

def _is_template_file(path: Path) -> bool:
    return _is_template_text(_read(path))

def _core_artifacts_have_templates(task_dir: Path) -> bool:
    rels = [
        "problem-contract.md",
        "candidates/candidate-001.md",
        "evaluations/review-report.md",
        "evaluations/checkmodel-report.md",
        "decision-record.md",
        "validation-plan.md",
        "working-memory.md",
    ]
    return any(_is_template_file(task_dir / rel) for rel in rels)

def _has_completed_intake(task_dir: Path) -> bool:
    task_meta = _read(task_dir / "task.yaml")
    return (
        "intake_status: completed" in task_meta
        and (task_dir / "intake" / "enhanced-task-contract.md").exists()
        and (task_dir / "intake" / "enhanced-task-contract.json").exists()
        and (task_dir / "intake" / "external-questioning-disabled.json").exists()
        and (task_dir / "intake" / "internal-resolution-ledger.json").exists()
    )

def ensure_task_skeleton(task_dir: Path, task_type: str = "general", slug: str = "task") -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for d in ["candidates", "evaluations", "evidence", "logs"]:
        (task_dir / d).mkdir(exist_ok=True)
    if not (task_dir / "trace.jsonl").exists():
        (task_dir / "trace.jsonl").write_text("", encoding="utf-8")
    for dest, tmpl in TASK_TEMPLATE_FILES.items():
        target = task_dir / dest
        if not target.exists():
            shutil.copyfile(TEMPLATES / tmpl, target)
    meta = task_dir / "task.yaml"
    if not meta.exists():
        meta.write_text(
            (
                f"task_type: {task_type}\n"
                f"slug: {slug}\n"
                f"created_at: {dt.datetime.now().strftime('%Y%m%d-%H%M%S')}\n"
                "status: proposed\n"
                "level: pending_classification\n"
            ),
            encoding="utf-8",
        )

def list_tasks() -> None:
    ensure_dirs()
    for p in sorted(TASKS.iterdir()):
        if p.is_dir() and p.name != "latest":
            print(p.name)

def check_task(path: str | None) -> int:
    task_dir = Path(path) if path else (latest_task_dir() or TASKS / "latest")
    required = [
        "task.yaml",
        "research-brief.md",
        "problem-contract.md",
        "decision-record.md",
        "validation-plan.md",
        "feedback.md",
        "working-memory.md",
        "trace.jsonl",
        "evaluations/checkmodel-report.md",
    ]
    missing = [f for f in required if not (task_dir / f).exists()]
    if missing:
        print("Missing:")
        for f in missing:
            print(f"  - {f}")
        return 1
    print(f"OK: {task_dir}")
    return 0

def _slug_from_prompt(prompt: str) -> str:
    return slugify(_first_nonempty_line(prompt, "cognitive-intake")[:80])
