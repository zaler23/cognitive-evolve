"""Project snapshot builder for offline Nexus evolution."""
from __future__ import annotations

import hashlib
import json
import os
import fnmatch
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_str_list, stable_hash, utc_now

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "api-runs",
    "runs",
}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".dylib", ".class", ".o"}
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".sqlite", ".db"}
SENSITIVE_FILE_NAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}
SENSITIVE_PATTERNS = {
    ".env.*",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*.kubeconfig",
    "*secret*",
    "*credential*",
    "*token*",
}


@dataclass
class ProjectSnapshot:
    snapshot_id: str
    root_hash: str
    file_manifest: list[dict[str, Any]] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    language_profile: dict[str, int] = field(default_factory=dict)
    detected_commands: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    binary_files: list[str] = field(default_factory=list)
    ignored_generated_files: list[str] = field(default_factory=list)
    root_path: str = ""
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_path(cls, root: str | Path) -> "ProjectSnapshot":
        root_path = Path(root).resolve()
        manifest: list[dict[str, Any]] = []
        hashes: dict[str, str] = {}
        language_profile: dict[str, int] = {}
        binary_files: list[str] = []
        ignored_generated: list[str] = []
        for path in sorted(root_path.rglob("*")):
            if path.is_symlink():
                try:
                    rel = path.relative_to(root_path).as_posix()
                except ValueError:
                    rel = path.name
                ignored_generated.append(rel)
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(root_path).as_posix()
            if _ignored(path, root_path):
                ignored_generated.append(rel)
                continue
            stat = path.stat()
            suffix = path.suffix.lower()
            language = _language_for_suffix(suffix)
            language_profile[language] = language_profile.get(language, 0) + 1
            is_binary = suffix in BINARY_EXTENSIONS or _is_probably_binary(path)
            if is_binary:
                binary_files.append(rel)
            digest = _sha256_file(path)
            hashes[rel] = digest
            manifest.append({"path": rel, "size": stat.st_size, "sha256": digest, "language": language, "binary": is_binary})
        root_hash = stable_hash({"files": hashes})
        return cls(
            snapshot_id="snapshot-" + root_hash[:16],
            root_hash=root_hash,
            file_manifest=manifest,
            file_hashes=hashes,
            language_profile=language_profile,
            detected_commands=_detected_commands(root_path, manifest),
            package_managers=_package_managers(root_path),
            binary_files=binary_files,
            ignored_generated_files=ignored_generated,
            root_path=str(root_path),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectSnapshot":
        return cls(
            snapshot_id=str(data.get("snapshot_id") or ""),
            root_hash=str(data.get("root_hash") or ""),
            file_manifest=[dict(item) for item in data.get("file_manifest", []) if isinstance(item, dict)],
            file_hashes={str(k): str(v) for k, v in dict(data.get("file_hashes") or {}).items()},
            language_profile={str(k): int(v) for k, v in dict(data.get("language_profile") or {}).items()},
            detected_commands=coerce_str_list(data.get("detected_commands")),
            package_managers=coerce_str_list(data.get("package_managers")),
            binary_files=coerce_str_list(data.get("binary_files")),
            ignored_generated_files=coerce_str_list(data.get("ignored_generated_files")),
            root_path=str(data.get("root_path") or ""),
            created_at=str(data.get("created_at") or utc_now()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "ProjectSnapshot":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("project snapshot JSON must decode to an object")
        return cls.from_dict(data)


def tree_hash(root: str | Path) -> str:
    return ProjectSnapshot.from_path(root).root_hash


def _ignored(path: Path, root: Path) -> bool:
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return True
    if path.is_symlink():
        return True
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return True
    parts = set(rel_path.parts)
    if parts & IGNORE_DIRS:
        return True
    rel = rel_path.as_posix()
    name = path.name
    if name in SENSITIVE_FILE_NAMES:
        return True
    if any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern) for pattern in SENSITIVE_PATTERNS):
        return True
    if path.suffix.lower() in GENERATED_SUFFIXES:
        return True
    return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_probably_binary(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return True
    return b"\0" in sample


def _language_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c_header",
    }.get(suffix, suffix.lstrip(".") or "unknown")


def _package_managers(root: Path) -> list[str]:
    managers: list[str] = []
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        managers.append("python")
    if (root / "package.json").exists():
        managers.append("npm")
    if (root / "Cargo.toml").exists():
        managers.append("cargo")
    if (root / "go.mod").exists():
        managers.append("go")
    return managers


def _detected_commands(root: Path, manifest: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    paths = {str(item.get("path")) for item in manifest}
    if any(path.endswith(".py") for path in paths):
        commands.append("python -m compileall -q .")
    if "pytest.ini" in paths or "pyproject.toml" in paths or any(path.startswith("tests/") for path in paths):
        commands.append("python -m pytest -q")
    if "package.json" in paths:
        commands.append("npm test")
    return commands


__all__ = ["ProjectSnapshot", "tree_hash"]
