from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKED = [
    ROOT / "cognitive_evolve_runtime/verification/synthesizer.py",
    *sorted((ROOT / "cognitive_evolve_runtime/verification/modalities").glob("*.py")),
    ROOT / "cognitive_evolve_runtime/verification/obligation_runner.py",
    ROOT / "cognitive_evolve_runtime/nexus/loop/round.py",
]


def test_no_runtime_strength_assignment() -> None:
    violations: list[str] = []
    for path in CHECKED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if _assigns_verification_strength_member(node.value) and not _diagnostic_assignment(node):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Call):
                if _constructs_result_with_strength_member(node) and not _call_has_diagnostic_only(node):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def _assigns_verification_strength_member(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "VerificationStrength":
        return node.attr != "from_value"
    if isinstance(node, ast.IfExp):
        return _assigns_verification_strength_member(node.body) or _assigns_verification_strength_member(node.orelse)
    return False


def _constructs_result_with_strength_member(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "strength" and _assigns_verification_strength_member(keyword.value):
            return True
    return False


def _call_has_diagnostic_only(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg in {"diagnostics_only", "legacy"}:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        if keyword.arg == "metadata" and isinstance(keyword.value, ast.Dict):
            keys = [key.value for key in keyword.value.keys if isinstance(key, ast.Constant)]
            values = keyword.value.values
            for key, value in zip(keys, values):
                if key in {"diagnostics_only", "legacy"} and isinstance(value, ast.Constant) and value.value is True:
                    return True
    return False


def _diagnostic_assignment(node: ast.Assign) -> bool:
    return any(isinstance(target, ast.Name) and target.id.startswith("diagnostic") for target in node.targets)
