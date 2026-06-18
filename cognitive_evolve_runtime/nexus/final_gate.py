"""Strict final-answer gate for project/source-grounded candidates.

This gate is deliberately about *final eligibility*, not search survival.  A
hybrid/design candidate may remain useful as repair material, but a project
update is not final unless its sources, symbols, evidence, and completion flags
are concrete enough to re-verify locally.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.obligations import (
    candidate_evidence_refs,
    candidate_source_bindings,
    requires_source_grounding,
)
from cognitive_evolve_runtime.nexus.source_lineage import analyze_source_lineage
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings, final_candidate_source_bindings_allowed
from cognitive_evolve_runtime.nexus.artifact_contract import (
    contract_requires_adapter,
    dynamic_artifact_contract_from,
    materialization_scope_from_contract,
)


@dataclass(frozen=True)
class FinalGateSummary:
    """Final-only project eligibility summary.

    ``rank_eligible`` remains true so exploratory candidates are not killed just
    because they are not yet publishable.
    """

    candidate_id: str
    required: bool
    rank_eligible: bool = True
    final_eligible: bool = True
    diagnostics: list[str] = field(default_factory=list)
    missing_symbols: list[dict[str, str]] = field(default_factory=list)
    evidence_refs_checked: int = 0
    source_bindings_checked: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FINAL_BLOCKING_METADATA_FLAGS = (
    "final_answer_blocked_until_repaired",
    "final_answer_blocked_until_reverified",
    "final_answer_blocked_until_verified",
    "search_seed_not_final",
)

NON_FINAL_ARTIFACT_TYPES = {"hybrid", "narrative", "design", "pseudo_code", "proposal", "analysis"}
PATCH_ARTIFACT_TYPES = {"project_patch", "patch", "code_patch"}


def final_gate_summary(
    candidate: CandidateGenome,
    *,
    contract: NexusObjectiveContract | None = None,
    project_root: str | Path | None = None,
) -> FinalGateSummary:
    source_bindings = candidate_source_bindings(candidate)
    dac = dynamic_artifact_contract_from(contract=contract, candidate=candidate)
    source_adapter_required = contract_requires_adapter(contract, "source", candidate=candidate)
    patch_adapter_required = contract_requires_adapter(contract, "patch", candidate=candidate)
    if dac is not None:
        source_required = bool(source_adapter_required or patch_adapter_required or _has_project_source_binding(source_bindings))
    else:
        source_required = requires_source_grounding(contract, candidate=candidate) or _has_project_source_binding(source_bindings)
    update_required = _requires_concrete_project_update(candidate, contract=contract)
    required = bool(source_required or update_required or _has_final_blocking_metadata(candidate))
    if not required:
        return FinalGateSummary(candidate_id=candidate.id, required=False)

    diagnostics: list[str] = []
    missing_symbols: list[dict[str, str]] = []
    root = _resolved_project_root(project_root)
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    try:
        manifest = annotate_candidate_source_bindings(candidate, project_root=root)
        if not final_candidate_source_bindings_allowed(candidate):
            diagnostics.append("source_binding_unresolved_or_invented")
            diagnostics.extend(str(item) for item in manifest.diagnostics[:4])
    except Exception:
        diagnostics.append("source_binding_manifest_unavailable")

    if _has_final_blocking_metadata(candidate):
        for key in FINAL_BLOCKING_METADATA_FLAGS:
            if metadata.get(key):
                diagnostics.append(key)

    if candidate.missing_parts and (source_required or update_required):
        diagnostics.append("final_missing_parts_unresolved")

    artifact_type = str(getattr(candidate, "artifact_type", "") or "").strip().lower()
    if dac is None and update_required and artifact_type in NON_FINAL_ARTIFACT_TYPES:
        diagnostics.append("final_artifact_type_not_publishable")

    if update_required and not _has_concrete_project_update(candidate):
        diagnostics.append("final_update_artifact_absent")

    if source_required and not source_bindings:
        diagnostics.append("source_binding_absent")

    if root is not None:
        lineage_summary = analyze_source_lineage(candidate, project_root=root, materialization_scope=materialization_scope_from_contract(contract, candidate=candidate))
        if lineage_summary.required:
            for fact in lineage_summary.facts:
                if "source_binding_missing_symbol" in fact.diagnostics and fact.symbol:
                    missing_symbols.append({"path": fact.path, "symbol": fact.symbol})
            diagnostics.extend(lineage_summary.diagnostics)
        else:
            missing_paths = []
            for binding in source_bindings:
                path = _project_relative_path(binding.get("path") or binding.get("file") or binding.get("source_path"), root=root)
                if not path:
                    continue
                if not _path_exists(root, path):
                    missing_paths.append(path)
                    continue
                symbol = str(binding.get("symbol") or binding.get("name") or "").strip()
                if symbol and _is_python_file(path) and not _python_symbol_exists(root / path, symbol):
                    missing_symbols.append({"path": path, "symbol": symbol})
            if missing_paths:
                diagnostics.append("source_binding_missing_path")
            if missing_symbols:
                diagnostics.append("source_binding_missing_symbol")

    evidence_refs = candidate_evidence_refs(candidate)
    if source_required and source_bindings and not _evidence_covers_source_bindings(evidence_refs, source_bindings, root=root):
        diagnostics.append("evidence_ref_not_source_relevant")

    final_eligible = not diagnostics
    return FinalGateSummary(
        candidate_id=candidate.id,
        required=required,
        final_eligible=final_eligible,
        diagnostics=list(dict.fromkeys(diagnostics)),
        missing_symbols=missing_symbols,
        evidence_refs_checked=len(evidence_refs),
        source_bindings_checked=len(source_bindings),
    )


def _has_final_blocking_metadata(candidate: CandidateGenome) -> bool:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    return any(bool(metadata.get(key)) for key in FINAL_BLOCKING_METADATA_FLAGS)


def _has_project_source_binding(source_bindings: list[dict[str, Any]]) -> bool:
    for binding in source_bindings:
        kind = str(binding.get("kind") or binding.get("type") or binding.get("binding_type") or "").strip().lower()
        path = str(binding.get("path") or binding.get("file") or binding.get("source_path") or "").strip().lower()
        if kind in {"source_file", "schema_field", "test", "patch", "implementation", "runtime_file"}:
            return True
        if path.endswith((".py", ".pyi", ".toml", ".yaml", ".yml", ".json")):
            return True
    return False


def _requires_concrete_project_update(candidate: CandidateGenome, *, contract: NexusObjectiveContract | None) -> bool:
    dac = dynamic_artifact_contract_from(contract=contract, candidate=candidate)
    if dac is not None:
        return bool(contract_requires_adapter(contract, "patch", candidate=candidate) or contract_requires_adapter(contract, "source", candidate=candidate) or isinstance(candidate, ProjectCandidateGenome) or getattr(candidate, "patch_set", None))
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").strip().lower()
    if artifact_type in PATCH_ARTIFACT_TYPES or isinstance(candidate, ProjectCandidateGenome) or getattr(candidate, "patch_set", None):
        return True
    text_parts = [artifact_type, str(getattr(candidate, "concise_claim", "") or ""), str(getattr(candidate, "core_mechanism", "") or "")]
    if contract is not None:
        text_parts.extend(
            [
                str(getattr(contract, "original_user_goal", "") or ""),
                str(getattr(contract, "normalized_goal", "") or ""),
                " ".join(str(item) for item in getattr(contract, "expected_output_forms", []) or []),
                " ".join(str(item) for item in getattr(contract, "verification_preferences", []) or []),
            ]
        )
    text = " ".join(text_parts).lower()
    return any(token in text for token in ("patch", "code", "source", "runtime", "implementation", "test", "pytest", "schema", "project", "代码", "源码", "运行时", "实现", "测试", "项目"))


def _has_concrete_project_update(candidate: CandidateGenome) -> bool:
    if isinstance(candidate, ProjectCandidateGenome) and getattr(candidate, "patch_set", None):
        return True
    if getattr(candidate, "patch_set", None):
        return True
    patch_result = getattr(candidate, "patch_application_result", None)
    if isinstance(patch_result, dict) and patch_result.get("status") == "applied" and patch_result.get("applied_files"):
        return True
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").strip().lower()
    artifact = getattr(candidate, "artifact", None)
    if artifact_type in PATCH_ARTIFACT_TYPES and isinstance(artifact, dict):
        return any(isinstance(artifact.get(key), str) and artifact.get(key).strip() for key in ("patch", "patch_content", "diff", "unified_diff"))
    return False


def _resolved_project_root(explicit_root: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root))
    candidates.append(Path.cwd())
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent)
    seen: set[Path] = set()
    for raw in candidates:
        root = raw.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "pyproject.toml").exists() and (root / "cognitive_evolve_runtime").is_dir():
            return root
    return None


def _project_relative_path(raw_path: Any, *, root: Path) -> str:
    text = str(raw_path or "").strip().replace("\\", "/")
    if not text or text == "/dev/null" or "://" in text:
        return ""
    if text.startswith(("a/", "b/")):
        text = text[2:]
    path = Path(text)
    if path.is_absolute():
        try:
            text = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
        except ValueError:
            return ""
    parts = Path(text).parts
    if any(part in {"", ".", ".."} for part in parts):
        return ""
    return Path(text).as_posix()


def _path_exists(root: Path, relative_path: str) -> bool:
    target = root / relative_path
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return False
    return target.exists()


def _is_python_file(path: str) -> bool:
    return path.endswith(".py")


def _python_symbol_exists(path: Path, symbol: str) -> bool:
    if not symbol:
        return True
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    wanted = symbol.split(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == wanted:
            return True
        if isinstance(node, ast.Assign):
            if any(_assignment_target_name(target) == wanted for target in node.targets):
                return True
        if isinstance(node, ast.AnnAssign) and _assignment_target_name(node.target) == wanted:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", 1)[0]
                if name == wanted:
                    return True
    return False


def _assignment_target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _evidence_covers_source_bindings(evidence_refs: list[dict[str, Any]], source_bindings: list[dict[str, Any]], *, root: Path | None) -> bool:
    binding_terms = _binding_terms(source_bindings)
    if not binding_terms:
        return False
    for ref in evidence_refs:
        if _evidence_ref_covers_terms(ref, binding_terms, root=root):
            return True
    return False


def _binding_terms(source_bindings: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for binding in source_bindings:
        path = str(binding.get("path") or binding.get("file") or binding.get("source_path") or "").strip()
        symbol = str(binding.get("symbol") or binding.get("name") or "").strip()
        if path:
            normalized = path.replace("\\", "/").lower()
            terms.add(normalized)
            terms.add(Path(normalized).stem)
            parts = normalized.split("/")
            for index in range(1, len(parts) + 1):
                suffix = "/".join(parts[-index:])
                if suffix:
                    terms.add(suffix)
        if symbol:
            terms.add(symbol.lower())
    return {term for term in terms if term}


def _evidence_ref_covers_terms(ref: dict[str, Any], terms: set[str], *, root: Path | None) -> bool:
    text = " ".join(str(value or "") for value in ref.values()).replace("\\", "/").lower()
    if any(term and term in text for term in terms):
        return True
    test_name = str(ref.get("test_name") or ref.get("id") or "").strip()
    if not test_name or root is None:
        return False
    test_path = _find_test_path(root, test_name)
    if test_path is None:
        return False
    try:
        source = test_path.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    module_terms = {term for term in terms if "/" not in term and "." not in term and len(term) >= 3}
    return any(re.search(rf"\b{re.escape(term)}\b", source) for term in module_terms)


def _find_test_path(root: Path, name: str) -> Path | None:
    candidate = Path(name)
    if candidate.suffix == ".py":
        path = root / candidate
        return path if path.exists() else None
    normalized = name if name.endswith(".py") else f"{name}.py"
    for base in (root / "tests", root):
        path = base / normalized
        if path.exists():
            return path
    matches = list((root / "tests").glob(f"**/{normalized}")) if (root / "tests").exists() else []
    return matches[0] if matches else None


__all__ = ["FINAL_BLOCKING_METADATA_FLAGS", "FinalGateSummary", "final_gate_summary"]
