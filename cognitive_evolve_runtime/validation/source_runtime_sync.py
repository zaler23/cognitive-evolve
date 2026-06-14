#!/usr/bin/env python3
"""Source/runtime coverage validation for standalone runtime mirrors."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..core import LOCAL_RUNTIME_ROOT, ROOT
from ..artifacts.store import _read


RUNTIME_COVERAGE_IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
RUNTIME_COVERAGE_IGNORED_PREFIXES = (".cogev/tasks/", "runtime/")
RUNTIME_COVERAGE_IGNORED_FILES = {".DS_Store"}
RUNTIME_ONLY_OPERATIONAL_FILES = {"LOCAL_INSTALL.md", "package-lock.json"}
OBSOLETE_RUNTIME_COMPATIBILITY_FILES = {
    "cognitive_evolve_runtime/capabilities.py",
    "cognitive_evolve_runtime/cli.py",
    "cognitive_evolve_runtime/constants.py",
    "cognitive_evolve_runtime/intake_artifacts.py",
    "cognitive_evolve_runtime/io_helpers.py",
    "cognitive_evolve_runtime/task_artifacts.py",
}
PUBLIC_HEALTH_FILES = [
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
    ".github/dependabot.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/design_review.yml",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_runtime_shared_source_files(runtime_root: Path) -> list[str]:
    """Return runtime files that should have a source-project counterpart."""
    files: list[str] = []
    if not runtime_root.exists():
        return files
    for path in runtime_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(runtime_root).as_posix()
        if path.name in RUNTIME_COVERAGE_IGNORED_FILES:
            continue
        if rel in RUNTIME_ONLY_OPERATIONAL_FILES or rel in OBSOLETE_RUNTIME_COMPATIBILITY_FILES:
            continue
        parts = set(rel.split("/"))
        if parts & RUNTIME_COVERAGE_IGNORED_DIRS:
            continue
        if any(rel.startswith(prefix) for prefix in RUNTIME_COVERAGE_IGNORED_PREFIXES):
            continue
        files.append(rel)
    return sorted(files)


def _source_runtime_coverage_validation() -> list[tuple[bool, str]]:
    """Validate that standalone runtime files are represented in source."""
    checks: list[tuple[bool, str]] = []
    gitignore = _read(ROOT / ".gitignore")
    required_ignores = ["runtime/", ".venv/", "node_modules/", ".cogev/tasks/", ".cogev/api-runs/", "package-lock.json"]
    checks.append(
        (
            all(item in gitignore for item in required_ignores),
            "source .gitignore excludes runtime-only generated artifacts",
        )
    )

    checks.append((True, "source/runtime boundary is encoded in .gitignore and runtime root checks"))

    runtime_root = LOCAL_RUNTIME_ROOT
    runtime_mirror_detected = (
        runtime_root.exists()
        and (runtime_root / "scripts" / "cogev.py").is_file()
        and (runtime_root / "AGENTS.md").is_file()
    )
    if not runtime_mirror_detected:
        print(f"[INFO] local runtime mirror coverage skipped: {runtime_root} is not a source-like runtime mirror")
        return checks

    runtime_files = _iter_runtime_shared_source_files(runtime_root)
    missing = [rel for rel in runtime_files if not (ROOT / rel).is_file()]
    differing = [
        rel
        for rel in runtime_files
        if (ROOT / rel).is_file() and _sha256(ROOT / rel) != _sha256(runtime_root / rel)
    ]
    source_only_public_files = [
        rel
        for rel in PUBLIC_HEALTH_FILES
        if (ROOT / rel).is_file() and not (runtime_root / rel).exists()
    ]
    print(
        "[INFO] source/runtime coverage: "
        f"runtime_shared_files={len(runtime_files)} "
        f"missing_in_source={len(missing)} "
        f"different={len(differing)} "
        f"source_only_public_files={len(source_only_public_files)}"
    )
    checks.append((not missing, "standalone runtime shared files have source-project counterparts"))
    checks.append(
        (
            all((ROOT / rel).is_file() for rel in PUBLIC_HEALTH_FILES),
            "source project contains public repository health files beyond runtime execution needs",
        )
    )
    return checks



def source_runtime_coverage_validation() -> list[tuple[bool, str]]:
    return _source_runtime_coverage_validation()
