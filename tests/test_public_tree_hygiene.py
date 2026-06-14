from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIR_NAMES = {
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "htmlcov",
    "build",
    "dist",
    "test-runs",
}
FORBIDDEN_FILE_NAMES = {
    ".coverage",
    "coverage.xml",
    ".DS_Store",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}
FORBIDDEN_LEGACY_MODULE_SUFFIXES = (
    "_old.py",
    "_backup.py",
    "_legacy.py",
)
FORBIDDEN_LOCAL_BRIDGE_NAME_TOKENS = (
    "anti" + "gravity",
    "gemi" + "ni_bridge",
    "gemi" + "ni-bridge",
    "local_bridge",
    "local-bridge",
    "openai" + "_proxy",
    "openai" + "-proxy",
    "codex" + "_openai",
    "host" + "_app_relay",
    "host" + "-app-relay",
)
SKIP_DIR_NAMES = {".git"}
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
LARGE_FILE_ALLOWLIST = {"uv.lock"}


def _public_paths() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(ROOT).parts):
            continue
        paths.append(path)
    return paths


def test_public_tree_has_no_cache_runtime_or_backup_artifacts() -> None:
    offenders: list[str] = []
    for path in _public_paths():
        rel = path.relative_to(ROOT).as_posix()
        if path.is_dir() and (path.name in FORBIDDEN_DIR_NAMES or path.name.endswith(".egg-info")):
            offenders.append(rel + "/")
        elif path.is_file():
            if path.name in FORBIDDEN_FILE_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
                offenders.append(rel)
            if path.name.endswith(FORBIDDEN_LEGACY_MODULE_SUFFIXES):
                offenders.append(rel)
            lower_name = path.name.lower()
            if any(token in lower_name for token in FORBIDDEN_LOCAL_BRIDGE_NAME_TOKENS):
                offenders.append(rel)
            if path.stat().st_size > MAX_PUBLIC_FILE_BYTES and path.name not in LARGE_FILE_ALLOWLIST:
                offenders.append(rel)

    assert offenders == []


def test_public_tree_has_no_broken_symlinks() -> None:
    broken = [
        path.relative_to(ROOT).as_posix()
        for path in _public_paths()
        if path.is_symlink() and not path.exists()
    ]

    assert broken == []


def test_public_tree_has_no_absolute_local_developer_paths_or_secret_files() -> None:
    offenders: list[str] = []
    for path in _public_paths():
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "tests/test_public_tree_hygiene.py":
            continue
        if path.name == ".env":
            offenders.append(rel)
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == "uv.lock":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ("/Users/" + "zzz/") in text or ("file://" + "/Users/") in text:
            offenders.append(rel)
        lower = text.lower()
        forbidden_text = (
            ("anti" + "gravity"),
            ("gemi" + "ni-3.5"),
            ("gemi" + "ni 3.5"),
            "codex" + "-local-" + "bridge",
            "local model " + "bridge",
            "host-app " + "bridge",
        )
        if any(token in lower for token in forbidden_text):
            offenders.append(rel)

    assert offenders == []


def test_public_release_automation_matches_source_tree_shape() -> None:
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert 'package-ecosystem: "github-actions"' in dependabot
    assert 'package-ecosystem: "pip"' in dependabot
    assert 'package-ecosystem: "npm"' not in dependabot
    assert not (ROOT / "package.json").exists()


def test_public_api_launcher_avoids_project_local_venv_and_implicit_port_kill() -> None:
    launcher = (ROOT / "scripts" / "start-cognitive-evolve-api.sh").read_text(encoding="utf-8")

    assert 'VENV_DIR="${COGEV_VENV_DIR:-$PROJECT_DIR/.venv}"' not in launcher
    assert 'pip install -e "$PROJECT_DIR"' not in launcher
    assert 'pip install "$PROJECT_DIR"' in launcher
    assert '$CACHE_ROOT/cognitive-evolve/venv' in launcher
    assert "COGEV_STOP_EXISTING_PORT=1" in launcher
    assert "Port $SERVICE_PORT is already in use." in launcher
    assert 'STOP_EXISTING_PORT="${COGEV_STOP_EXISTING_PORT:-0}"' in launcher
