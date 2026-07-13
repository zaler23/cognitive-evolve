"""Project world model extraction from a ProjectSnapshot."""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .project_snapshot import ProjectSnapshot


@dataclass
class ProjectWorldModel:
    kind: str = "project"
    snapshot_id: str = ""
    file_roles: dict[str, str] = field(default_factory=dict)
    symbol_graph: dict[str, list[str]] = field(default_factory=dict)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    test_map: dict[str, list[str]] = field(default_factory=dict)
    config_map: dict[str, str] = field(default_factory=dict)
    docs_map: dict[str, str] = field(default_factory=dict)
    command_graph: dict[str, list[str]] = field(default_factory=dict)
    hotspot_map: dict[str, float] = field(default_factory=dict)
    objective_relevance_map: dict[str, float] = field(default_factory=dict)
    project_summary: str = ""

    @classmethod
    def from_snapshot(cls, snapshot: ProjectSnapshot, *, objective: str = "") -> "ProjectWorldModel":
        root = Path(snapshot.root_path) if snapshot.root_path else None
        file_roles: dict[str, str] = {}
        symbol_graph: dict[str, list[str]] = {}
        dependency_graph: dict[str, list[str]] = {}
        test_map: dict[str, list[str]] = {}
        config_map: dict[str, str] = {}
        docs_map: dict[str, str] = {}
        hotspot_map: dict[str, float] = {}
        relevance: dict[str, float] = {}
        for item in snapshot.file_manifest:
            rel = str(item.get("path") or "")
            role = _role_for_path(rel)
            file_roles[rel] = role
            hotspot_map[rel] = _hotspot(item)
            symbols: list[str] = []
            if root and rel.endswith(".py"):
                path = root / rel
                if path.exists():
                    symbols, imports = _python_symbols_and_imports(path)
                    symbol_graph[rel] = symbols
                    dependency_graph[rel] = imports
            relevance[rel] = _relevance(rel, objective, role=role, symbols=symbols)
            if role == "test":
                test_map.setdefault(_probable_source_for_test(rel), []).append(rel)
            elif role == "config":
                config_map[rel] = Path(rel).name
            elif role == "docs":
                docs_map[rel] = Path(rel).name
        return cls(
            snapshot_id=snapshot.snapshot_id,
            file_roles=file_roles,
            symbol_graph=symbol_graph,
            dependency_graph=dependency_graph,
            test_map=test_map,
            config_map=config_map,
            docs_map=docs_map,
            command_graph={"detected": list(snapshot.detected_commands)},
            hotspot_map=hotspot_map,
            objective_relevance_map=_top_relevance(relevance),
            project_summary=f"{len(snapshot.file_manifest)} files; languages={snapshot.language_profile}; commands={snapshot.detected_commands}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectWorldModel":
        return cls(
            kind=str(data.get("kind") or "project"),
            snapshot_id=str(data.get("snapshot_id") or ""),
            file_roles=dict(data.get("file_roles") or {}),
            symbol_graph={str(k): [str(x) for x in v] for k, v in dict(data.get("symbol_graph") or {}).items()},
            dependency_graph={str(k): [str(x) for x in v] for k, v in dict(data.get("dependency_graph") or {}).items()},
            test_map={str(k): [str(x) for x in v] for k, v in dict(data.get("test_map") or {}).items()},
            config_map=dict(data.get("config_map") or {}),
            docs_map=dict(data.get("docs_map") or {}),
            command_graph={str(k): [str(x) for x in v] for k, v in dict(data.get("command_graph") or {}).items()},
            hotspot_map={str(k): float(v) for k, v in dict(data.get("hotspot_map") or {}).items()},
            objective_relevance_map={str(k): float(v) for k, v in dict(data.get("objective_relevance_map") or {}).items()},
            project_summary=str(data.get("project_summary") or ""),
        )


def _role_for_path(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if lowered.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    if name in {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "package.json", "tsconfig.json"} or lowered.startswith(".github/"):
        return "config"
    if lowered.endswith(".md") or lowered.startswith("docs/"):
        return "docs"
    if lowered.endswith(('.py', '.js', '.ts', '.rs', '.go', '.java', '.c', '.cpp')):
        return "implementation"
    return "asset"


def _hotspot(item: dict[str, Any]) -> float:
    size = float(item.get("size") or 0)
    return min(1.0, size / 20000.0)


def _relevance(path: str, objective: str, *, role: str = "", symbols: list[str] | None = None) -> float:
    tokens = _tokens(objective)
    if not tokens:
        return 0.0
    path_tokens = _tokens(path)
    symbol_tokens = set().union(*(_tokens(symbol) for symbol in (symbols or []))) if symbols else set()
    score = 0.0
    score += 0.55 * (len(tokens & path_tokens) / max(1, len(tokens)))
    score += 0.35 * (len(tokens & symbol_tokens) / max(1, len(tokens)))
    if role == "implementation" and tokens & (path_tokens | symbol_tokens):
        score += 0.10
    elif role == "test" and ({"test", "pytest", "regression", "fix"} & tokens):
        score += 0.08
    elif role in {"config", "docs"} and tokens & path_tokens:
        score += 0.04
    return min(1.0, round(score, 6))


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", str(text).lower()) if len(token) > 2}


def _top_relevance(relevance: dict[str, float], *, limit: int = 80) -> dict[str, float]:
    ranked = sorted(relevance.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return {path: score for path, score in ranked}


def _probable_source_for_test(path: str) -> str:
    name = Path(path).name
    if name.startswith("test_"):
        return name.removeprefix("test_").removesuffix(".py") + ".py"
    return "unknown"


def _python_symbols_and_imports(path: Path) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    symbols: list[str] = []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return symbols, [item for item in imports if item]


__all__ = ["ProjectWorldModel"]
