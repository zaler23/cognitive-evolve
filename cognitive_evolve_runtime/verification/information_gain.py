"""Engine-computed grounded information-gain signals."""
from __future__ import annotations

import ast
import json
from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash

RUNTIME_KEYS = {"runtime", "jitter", "name", "round", "round_index", "id", "candidate_id", "created_at", "updated_at"}
UNDEFINED_SIGNATURE = "UNDEFINED"


def grounded_signature(candidate: Any) -> str:
    artifact = getattr(candidate, "artifact", candidate)
    if isinstance(artifact, str):
        code_sig = _code_signature(artifact)
        if code_sig:
            return code_sig
        return _obligation_set_signature(candidate)
    if isinstance(artifact, dict):
        return "spec-" + stable_hash(_strip_runtime_keys(artifact))[:32]
    if isinstance(artifact, (list, tuple)):
        return "spec-" + stable_hash(_strip_runtime_keys(list(artifact)))[:32]
    return _obligation_set_signature(candidate)


def compute_marginal_gain(
    current_signatures: set[str],
    history_signatures: set[str],
    measured_strength_delta: float,
    evidence_delta_count: int,
) -> float:
    if not current_signatures or current_signatures == {UNDEFINED_SIGNATURE}:
        return 0.0
    defined_current = {sig for sig in current_signatures if sig != UNDEFINED_SIGNATURE}
    if not defined_current:
        return 0.0
    new_sig_ratio = len(defined_current - history_signatures) / max(1, len(defined_current))
    return (
        0.6 * new_sig_ratio
        + 0.3 * min(1.0, max(0.0, float(measured_strength_delta or 0.0)))
        + 0.1 * min(1.0, max(0, int(evidence_delta_count or 0)) / 5)
    )


def population_information_gain_report(candidates: list[Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    current = {grounded_signature(candidate) for candidate in candidates or []}
    history_signatures: set[str] = set()
    for item in history or []:
        if not isinstance(item, dict):
            continue
        gain = item.get("grounded_information_gain")
        if isinstance(gain, dict):
            history_signatures.update(str(sig) for sig in gain.get("current_signatures", []) if sig)
    measured_delta = _measured_strength_delta(candidates, history or [])
    max_strength = _max_measured_strength(candidates)
    evidence_delta_count = _evidence_delta_count(candidates)
    gain_value = compute_marginal_gain(current, history_signatures, measured_delta, evidence_delta_count)
    return {
        "current_signatures": sorted(current),
        "history_signature_count": len(history_signatures),
        "marginal_information_gain": gain_value,
        "measured_strength_delta": measured_delta,
        "max_measured_strength_value": max_strength,
        "evidence_delta_count": evidence_delta_count,
        "undefined_count": len([sig for sig in current if sig == UNDEFINED_SIGNATURE]),
    }


def _code_signature(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If, ast.Return)) for node in ast.walk(tree)):
        return ""
    normalizer = _AstNormalizer()
    normalizer.visit(tree)
    ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return "ast-" + stable_hash(dumped)[:32]


class _AstNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_", annotation=self.visit(node.annotation) if node.annotation else None, type_comment=None), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "_fn"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "_fn"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "_cls"
        self.generic_visit(node)
        return node


def _obligation_set_signature(candidate: Any) -> str:
    passed: set[str] = set()
    failed: set[str] = set()
    for item in getattr(candidate, "verification_trace", []) or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        oid = str(metadata.get("obligation_id") or item.get("obligation_id") or "")
        if not oid:
            continue
        if item.get("passed") is True:
            passed.add(oid)
        elif item.get("passed") is False:
            failed.add(oid)
    if not passed and not failed:
        return UNDEFINED_SIGNATURE
    return "obl-" + stable_hash({"passed": sorted(passed), "failed": sorted(failed)})[:32]


def _strip_runtime_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strip_runtime_keys(v) for k, v in sorted(value.items(), key=lambda item: str(item[0])) if str(k) not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [_strip_runtime_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_runtime_keys(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _measured_strength_delta(candidates: list[Any], history: list[dict[str, Any]]) -> float:
    current = _max_measured_strength(candidates)
    previous = 0
    for item in history:
        gain = item.get("grounded_information_gain") if isinstance(item, dict) else None
        if isinstance(gain, dict):
            previous = max(previous, int(gain.get("max_measured_strength_value") or 0))
    return max(0.0, (current - previous) / 5.0)


def _max_measured_strength(candidates: list[Any]) -> int:
    current = 0
    for candidate in candidates or []:
        result = getattr(candidate, "verification_result", {}) if isinstance(getattr(candidate, "verification_result", {}), dict) else {}
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        current = max(current, int(metadata.get("measured_strength_value") or 0))
    return current


def _evidence_delta_count(candidates: list[Any]) -> int:
    count = 0
    for candidate in candidates or []:
        if getattr(candidate, "evidence_delta", None):
            count += 1
        metadata = getattr(candidate, "metadata", {}) if isinstance(getattr(candidate, "metadata", None), dict) else {}
        if metadata.get("evidence_state") or metadata.get("evidence_records"):
            count += 1
    return count


__all__ = ["UNDEFINED_SIGNATURE", "compute_marginal_gain", "grounded_signature", "population_information_gain_report"]
