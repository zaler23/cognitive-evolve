"""Source-binding lineage analysis; Chinese intent tokens support multilingual patch candidates.

The gate answers one narrow question: does the proposed patch/test/source
reality support the candidate's declared path+symbol binding?  It is deliberately
mode-neutral: existing-file refinement, existing-file extension, and new-file
materialization are all valid evolution shapes when their evidence is concrete.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict

PATCH_ARTIFACT_TYPES = {"project_patch", "patch", "code_patch"}
MATERIALIZATION_REPAIR_DIAGNOSTICS = {
    "declared_new_file_not_created",
    "new_file_patch_absent",
    "declared_new_symbol_not_created",
    "new_file_integration_absent",
}
MATERIALIZATION_HARD_DIAGNOSTICS = {
    "new_file_path_out_of_scope",
}
SOURCE_LINEAGE_DIAGNOSTICS = MATERIALIZATION_REPAIR_DIAGNOSTICS | MATERIALIZATION_HARD_DIAGNOSTICS | {
    "source_binding_missing_path",
    "patch_target_missing",
    "source_binding_missing_symbol",
}
ALLOWED_MATERIALIZATION_PREFIXES = ("cognitive_evolve_runtime/", "tests/")


@dataclass(frozen=True)
class SourceLineageFact:
    path: str
    symbol: str = ""
    declared_mode: str = ""
    lineage_mode: str = ""
    path_exists: bool = False
    symbol_exists_pre: bool = False
    patch_touches_path: bool = False
    patch_creates_file: bool = False
    patch_defines_symbol: bool = False
    integration_present: bool = False
    allowed_new_path: bool = False
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceLineageAnalysis:
    candidate_id: str
    required: bool
    facts: list[SourceLineageFact] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    @property
    def final_eligible(self) -> bool:
        return not self.diagnostics

    @property
    def diagnostic_fragments(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for fact in self.facts:
            fragment = f"{fact.path}::{fact.symbol}" if fact.symbol else fact.path
            for diagnostic in fact.diagnostics:
                out.setdefault(diagnostic, [])
                if fragment not in out[diagnostic]:
                    out[diagnostic].append(fragment)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "required": self.required,
            "passed": self.passed,
            "final_eligible": self.final_eligible,
            "diagnostics": list(self.diagnostics),
            "facts": [fact.to_dict() for fact in self.facts],
        }


def analyze_source_lineage(candidate: CandidateGenome, *, project_root: str | Path | None = None, materialization_scope: list[str] | tuple[str, ...] | None = None) -> SourceLineageAnalysis:
    """Classify declared source bindings without privileging old or new files.

    A missing path/symbol is only a hallucination when the candidate's claim and
    patch reality disagree.  If the candidate is plausibly trying to create the
    path/symbol, the analysis emits repairable materialization diagnostics so the
    lineage can incubate and produce the missing patch/test evidence.
    """

    if not is_project_patch_candidate(candidate):
        return SourceLineageAnalysis(candidate_id=candidate.id, required=False)
    root = resolved_project_root(project_root)
    if root is None:
        return SourceLineageAnalysis(candidate_id=candidate.id, required=False)

    patch_paths = patch_declared_paths(candidate, root=root)
    created_paths = patch_created_paths(candidate, root=root)
    bindings = declared_source_bindings(candidate, root=root)
    path_only_targets = [path for path in declared_existing_patch_target_paths(candidate, root=root) if path not in {item["path"] for item in bindings}]

    facts: list[SourceLineageFact] = []
    for binding in bindings:
        facts.append(_analyze_binding(candidate, binding=binding, root=root, patch_paths=patch_paths, created_paths=created_paths, materialization_scope=materialization_scope))
    for path in path_only_targets:
        facts.append(_analyze_binding(candidate, binding={"path": path, "symbol": "", "declared_mode": "", "target_only": "1"}, root=root, patch_paths=patch_paths, created_paths=created_paths, materialization_scope=materialization_scope))

    diagnostics: list[str] = []
    for fact in facts:
        diagnostics.extend(fact.diagnostics)
    return SourceLineageAnalysis(
        candidate_id=candidate.id,
        required=bool(facts),
        facts=facts,
        diagnostics=list(dict.fromkeys(diagnostics)),
    )


def is_project_patch_candidate(candidate: CandidateGenome) -> bool:
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").lower()
    if isinstance(candidate, ProjectCandidateGenome) or artifact_type in PATCH_ARTIFACT_TYPES:
        return True
    if bool(getattr(candidate, "patch_set", None)):
        return True
    artifact = getattr(candidate, "artifact", None)
    return isinstance(artifact, dict) and any(key in artifact for key in ("path", "patch", "patch_content", "diff", "unified_diff"))


def resolved_project_root(explicit_root: str | Path | None = None) -> Path | None:
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


def declared_source_bindings(candidate: CandidateGenome, *, root: Path) -> list[dict[str, str]]:
    raw_items: list[Any] = list(getattr(candidate, "source_bindings", []) or [])
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    raw_items.extend(metadata.get("source_bindings", []) or [])
    repair = metadata.get("repair_required")
    if isinstance(repair, dict):
        raw_items.extend(repair.get("source_bindings", []) or [])
    out: dict[tuple[str, str], dict[str, str]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path = project_relative_path(item.get("path") or item.get("file") or item.get("source_path"), root=root)
        if not path:
            continue
        symbol = str(item.get("symbol") or item.get("name") or "").strip()
        mode = str(item.get("mode") or item.get("binding_mode") or item.get("lineage_mode") or item.get("intent") or "").strip().lower()
        out[(path, symbol)] = {"path": path, "symbol": symbol, "declared_mode": mode}
    return list(out.values())


def declared_existing_patch_target_paths(candidate: CandidateGenome, *, root: Path) -> list[str]:
    paths: list[str] = []
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            raw_path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            operation = str(getattr(op, "operation", "") if not isinstance(op, dict) else op.get("operation", "") or "").lower()
            path = project_relative_path(raw_path, root=root)
            if path and operation in {"replace", "delete"}:
                paths.append(path)
            elif path and operation in {"write", "append"} and not path_exists(root, path):
                # A write/append to a missing file is a materialization attempt,
                # not an existing target.  It is analyzed through changed paths.
                continue
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        explicit_path = project_relative_path(artifact.get("path") or artifact.get("file") or artifact.get("target_path"), root=root)
        patch_text = artifact_patch_text(artifact)
        created = created_paths_from_patch_text(patch_text, root=root)
        if explicit_path and explicit_path not in created:
            paths.append(explicit_path)
        for path in existing_paths_from_patch_text(patch_text, root=root):
            if path and path not in created:
                paths.append(path)
    return list(dict.fromkeys(paths))


def patch_declared_paths(candidate: CandidateGenome, *, root: Path) -> set[str]:
    paths: set[str] = set()
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            raw_path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            path = project_relative_path(raw_path, root=root)
            if path:
                paths.add(path)
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        for key in ("path", "file", "target_path"):
            path = project_relative_path(artifact.get(key), root=root)
            if path:
                paths.add(path)
        paths.update(project_relative_path(raw, root=root) for raw in raw_paths_from_patch_text(artifact_patch_text(artifact)))
    return {path for path in paths if path}


def patch_created_paths(candidate: CandidateGenome, *, root: Path) -> set[str]:
    created: set[str] = set()
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            raw_path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            operation = str(getattr(op, "operation", "") if not isinstance(op, dict) else op.get("operation", "") or "").lower()
            path = project_relative_path(raw_path, root=root)
            if path and operation in {"write", "append"} and not path_exists(root, path):
                created.add(path)
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        created.update(created_paths_from_patch_text(artifact_patch_text(artifact), root=root))
    return created


def patch_defines_symbol(candidate: CandidateGenome, *, path: str, symbol: str, root: Path) -> bool:
    wanted = symbol.split(".", 1)[0].strip()
    if not wanted:
        return True
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            op_path = project_relative_path(getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", ""), root=root)
            if op_path and op_path != path:
                continue
            content_parts = [
                getattr(op, "content", "") if not isinstance(op, dict) else op.get("content", ""),
                getattr(op, "new_text", "") if not isinstance(op, dict) else op.get("new_text", ""),
            ]
            if any(text_defines_symbol(str(part or ""), wanted) for part in content_parts):
                return True
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        patch_text = artifact_patch_text(artifact)
        if patch_text and patch_text_adds_symbol(patch_text, wanted, path=path, root=root):
            return True
        for key in ("content", "replacement", "new_text"):
            value = artifact.get(key)
            if isinstance(value, str) and text_defines_symbol(value, wanted):
                return True
    return False


def has_integration_reference(candidate: CandidateGenome, *, path: str, symbol: str, root: Path) -> bool:
    if path.startswith("tests/"):
        return True
    module_stem = Path(path).stem.replace("-", "_")
    terms = {path.lower(), module_stem.lower()}
    if symbol:
        terms.add(symbol.split(".", 1)[0].lower())
    for ref in list(getattr(candidate, "evidence_refs", []) or []):
        if not isinstance(ref, dict):
            continue
        kind = str(ref.get("kind") or ref.get("type") or "").strip().lower()
        text = " ".join(str(value or "") for value in ref.values()).replace("\\", "/").lower()
        is_integration_evidence = kind in {"test", "verification", "command"} or "tests/" in text or "pytest" in text
        if is_integration_evidence and any(term and term in text for term in terms):
            return True
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            op_path = project_relative_path(getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", ""), root=root)
            if not op_path or op_path == path:
                continue
            text = "\n".join(
                str(part or "")
                for part in (
                    getattr(op, "content", "") if not isinstance(op, dict) else op.get("content", ""),
                    getattr(op, "new_text", "") if not isinstance(op, dict) else op.get("new_text", ""),
                )
            ).lower()
            if op_path.startswith("tests/") or any(term and term in text for term in terms):
                return True
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        patch_text = artifact_patch_text(artifact)
        for changed_path, added_text in added_text_by_patch_path(patch_text, root=root).items():
            if not changed_path or changed_path == path:
                continue
            lowered = added_text.lower()
            if changed_path.startswith("tests/") or any(term and term in lowered for term in terms):
                return True
    return False


def _analyze_binding(
    candidate: CandidateGenome,
    *,
    binding: dict[str, str],
    root: Path,
    patch_paths: set[str],
    created_paths: set[str],
    materialization_scope: list[str] | tuple[str, ...] | None = None,
) -> SourceLineageFact:
    path = binding["path"]
    symbol = binding.get("symbol", "")
    declared_mode = binding.get("declared_mode", "")
    exists = path_exists(root, path)
    symbol_exists = bool(exists and path.endswith(".py") and (not symbol or python_symbol_exists(root / path, symbol)))
    touches = path in patch_paths
    creates = path in created_paths
    defines = patch_defines_symbol(candidate, path=path, symbol=symbol, root=root) if symbol else True
    integration = has_integration_reference(candidate, path=path, symbol=symbol, root=root)
    allowed_new = allowed_materialization_path(path, materialization_scope=materialization_scope)
    diagnostics: list[str] = []
    mode = "existing_file_refinement"

    if exists:
        if symbol and not symbol_exists:
            if defines:
                mode = "existing_file_extension"
            else:
                mode = "existing_file_extension_unmaterialized" if _extension_intent(candidate, declared_mode=declared_mode) else "existing_file_symbol_mismatch"
                diagnostics.append("declared_new_symbol_not_created" if mode == "existing_file_extension_unmaterialized" else "source_binding_missing_symbol")
        else:
            mode = "existing_file_refinement"
    else:
        materialization_intent = creates or _materialization_intent(candidate, declared_mode=declared_mode)
        target_only = str(binding.get("target_only") or "") == "1"
        mode = "new_file_materialization" if creates else ("new_file_unmaterialized" if materialization_intent else "missing_existing_binding")
        if materialization_intent and not allowed_new:
            diagnostics.append("new_file_path_out_of_scope")
        elif not materialization_intent:
            diagnostics.append("patch_target_missing" if target_only else "source_binding_missing_path")
            if not target_only and touches:
                diagnostics.append("patch_target_missing")
        elif not creates:
            diagnostics.extend(["declared_new_file_not_created", "new_file_patch_absent"])
        else:
            if symbol and not defines:
                diagnostics.append("declared_new_symbol_not_created")
            if not integration:
                diagnostics.append("new_file_integration_absent")

    return SourceLineageFact(
        path=path,
        symbol=symbol,
        declared_mode=declared_mode,
        lineage_mode=mode,
        path_exists=exists,
        symbol_exists_pre=symbol_exists,
        patch_touches_path=touches,
        patch_creates_file=creates,
        patch_defines_symbol=defines,
        integration_present=integration,
        allowed_new_path=allowed_new,
        diagnostics=list(dict.fromkeys(diagnostics)),
    )


def _extension_intent(candidate: CandidateGenome, *, declared_mode: str) -> bool:
    if declared_mode in {"extend", "extension", "new_symbol", "materialize", "new_file", "create"}:
        return True
    text = " ".join(str(part or "") for part in (candidate.concise_claim, candidate.core_mechanism)).lower()
    return any(token in text for token in ("add ", "create ", "introduce ", "define ", "new symbol", "new function", "new class", "新增", "创建", "引入"))


def _materialization_intent(candidate: CandidateGenome, *, declared_mode: str) -> bool:
    if declared_mode in {"materialize", "new_file", "new file", "create_file", "create"}:
        return True
    text = " ".join(str(part or "") for part in (candidate.concise_claim, candidate.core_mechanism)).lower()
    return any(token in text for token in ("new file", "new module", "materialize", "create a new", "create new", "新增文件", "新文件", "新模块"))


def allowed_materialization_path(path: str, *, materialization_scope: list[str] | tuple[str, ...] | None = None) -> bool:
    normalized = str(path or "").strip().replace("\\", "/").lstrip("./")
    if not normalized or normalized.startswith(("/", "../")) or "/../" in normalized:
        return False
    scopes = tuple(str(item).strip().replace("\\", "/").lstrip("./") for item in (materialization_scope or ALLOWED_MATERIALIZATION_PREFIXES) if str(item).strip())
    if not scopes:
        return False
    return normalized.startswith(scopes) and normalized.endswith((".py", ".pyi", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"))


def artifact_patch_text(artifact: dict[str, Any]) -> str:
    for key in ("patch", "patch_content", "diff", "unified_diff"):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = artifact.get("content")
    if isinstance(content, str) and looks_like_patch_text(content):
        return content
    return ""


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(.+?)\s*$")


def looks_like_patch_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lines = text.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_git = any(_DIFF_GIT_RE.match(line.strip()) for line in lines)
    return bool(has_hunk and (has_git or (has_old and has_new)))


def raw_paths_from_patch_text(text: str) -> list[str]:
    paths: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            paths.extend([git_match.group(1), git_match.group(2)])
            continue
        path_match = _DIFF_PATH_RE.match(stripped)
        if path_match:
            raw = path_match.group(1).strip()
            if "\t" in raw:
                raw = raw.split("\t", 1)[0]
            paths.append(raw)
    return [path for path in paths if path and path != "/dev/null"]


def existing_paths_from_patch_text(text: str, *, root: Path) -> list[str]:
    paths: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            path = project_relative_path(git_match.group(1), root=root)
            if path:
                paths.append(path)
            continue
        if stripped.startswith("--- "):
            raw = stripped[4:].strip().split("\t", 1)[0]
            if raw != "/dev/null":
                path = project_relative_path(raw, root=root)
                if path:
                    paths.append(path)
    return list(dict.fromkeys(paths))


def created_paths_from_patch_text(text: str, *, root: Path) -> set[str]:
    created: set[str] = set()
    previous_was_dev_null = False
    last_git_new_path = ""
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            last_git_new_path = project_relative_path(git_match.group(2), root=root)
            previous_was_dev_null = False
            continue
        if stripped.startswith("new file mode") and last_git_new_path:
            created.add(last_git_new_path)
            continue
        if stripped.startswith("--- "):
            previous_was_dev_null = stripped[4:].strip().split("\t", 1)[0] == "/dev/null"
            continue
        if previous_was_dev_null and stripped.startswith("+++ "):
            path = project_relative_path(stripped[4:].strip().split("\t", 1)[0], root=root)
            if path:
                created.add(path)
            previous_was_dev_null = False
    return created


def added_text_by_patch_path(text: str, *, root: Path) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current_path = ""
    previous_was_dev_null = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            current_path = project_relative_path(git_match.group(2), root=root)
            previous_was_dev_null = False
            continue
        if stripped.startswith("--- "):
            previous_was_dev_null = stripped[4:].strip().split("\t", 1)[0] == "/dev/null"
            continue
        if stripped.startswith("+++ "):
            path = project_relative_path(stripped[4:].strip().split("\t", 1)[0], root=root)
            if path:
                current_path = path
            previous_was_dev_null = False
            continue
        if current_path and line.startswith("+") and not line.startswith("+++"):
            out.setdefault(current_path, []).append(line[1:])
    return {path: "\n".join(lines) for path, lines in out.items()}


def patch_text_adds_symbol(patch_text: str, symbol: str, *, path: str, root: Path) -> bool:
    for changed_path, added_text in added_text_by_patch_path(patch_text, root=root).items():
        if changed_path == path and text_defines_symbol(added_text, symbol):
            return True
    return False


def text_defines_symbol(text: str, symbol: str) -> bool:
    if not symbol:
        return True
    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b", re.MULTILINE)
    return bool(pattern.search(str(text or "")))


def python_symbol_exists(path: Path, symbol: str) -> bool:
    if not symbol:
        return True
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    parts = [part for part in symbol.split(".") if part]
    if not parts:
        return True
    if len(parts) == 1:
        return any(_node_declares_symbol(node, parts[0]) for node in ast.walk(tree))
    owner, member = parts[0], parts[1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == owner:
            return any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == member for child in node.body)
    return False


def _node_declares_symbol(node: ast.AST, wanted: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == wanted:
        return True
    if isinstance(node, ast.Assign):
        return any(_assignment_target_name(target) == wanted for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return _assignment_target_name(node.target) == wanted
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


def project_relative_path(raw_path: Any, *, root: Path) -> str:
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


def path_exists(root: Path, relative_path: str) -> bool:
    path = project_relative_path(relative_path, root=root)
    if not path:
        return False
    target = root / path
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return False
    return target.exists()


__all__ = [
    "ALLOWED_MATERIALIZATION_PREFIXES",
    "MATERIALIZATION_HARD_DIAGNOSTICS",
    "MATERIALIZATION_REPAIR_DIAGNOSTICS",
    "SOURCE_LINEAGE_DIAGNOSTICS",
    "SourceLineageAnalysis",
    "SourceLineageFact",
    "allowed_materialization_path",
    "analyze_source_lineage",
    "is_project_patch_candidate",
    "resolved_project_root",
]
