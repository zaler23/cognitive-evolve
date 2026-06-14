from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THEORY = ROOT / "cognitive_evolve_runtime" / "theory"

FORBIDDEN_THEORY_PREFIXES = (
    "cognitive_evolve_runtime.nexus",
    "cognitive_evolve_runtime.ranking",
    "cognitive_evolve_runtime.archives",
)
M5_M6_FILES = [
    ROOT / "cognitive_evolve_runtime" / "outcomes" / "improvement.py",
    ROOT / "cognitive_evolve_runtime" / "outcomes" / "latent.py",
    ROOT / "cognitive_evolve_runtime" / "outcomes" / "runtime_bridge.py",
    ROOT / "cognitive_evolve_runtime" / "nexus" / "final_gate.py",
]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module)
    return imports


def test_theory_package_does_not_import_runtime_internals() -> None:
    offenders: list[str] = []
    for path in THEORY.glob("*.py"):
        for module in _imports(path):
            if module.startswith(FORBIDDEN_THEORY_PREFIXES):
                offenders.append(f"{path.name}:{module}")
    assert offenders == []


def test_m5_m6_gate_modules_do_not_import_theory() -> None:
    offenders: list[str] = []
    for path in M5_M6_FILES:
        if not path.exists():
            continue
        for module in _imports(path):
            if module.startswith("cognitive_evolve_runtime.theory"):
                offenders.append(f"{path.relative_to(ROOT)}:{module}")
    assert offenders == []
