"""Context packets for project evolution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_str_list, stable_hash
from .project_snapshot import ProjectSnapshot
from .project_map import ProjectWorldModel


@dataclass
class ContextRequest:
    need_files: list[str] = field(default_factory=list)
    need_symbols: list[str] = field(default_factory=list)
    need_tests: list[str] = field(default_factory=list)
    target_obligation_ids: list[str] = field(default_factory=list)
    evidence_need: str = ""
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextRequest":
        return cls(
            need_files=coerce_str_list(data.get("need_files")),
            need_symbols=coerce_str_list(data.get("need_symbols")),
            need_tests=coerce_str_list(data.get("need_tests")),
            target_obligation_ids=coerce_str_list(data.get("target_obligation_ids")),
            evidence_need=str(data.get("evidence_need") or ""),
            reason=str(data.get("reason") or ""),
        )


@dataclass
class ContextPacket:
    objective_contract: dict[str, Any]
    project_summary: str
    selected_file_summaries: dict[str, str] = field(default_factory=dict)
    raw_file_slices: dict[str, str] = field(default_factory=dict)
    symbol_dependencies: dict[str, list[str]] = field(default_factory=dict)
    test_feedback: list[dict[str, Any]] = field(default_factory=list)
    parent_candidate_summaries: list[dict[str, Any]] = field(default_factory=list)
    archive_hints: dict[str, Any] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    evidence_need: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    mutation_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextSelector:
    def __init__(self, *, max_file_chars: int = 0) -> None:
        self.max_file_chars = max_file_chars

    def build_context_packet(
        self,
        *,
        contract: Any,
        snapshot: ProjectSnapshot,
        world: ProjectWorldModel,
        request: ContextRequest | dict[str, Any] | None = None,
        parent_candidates: list[Any] | None = None,
        archive_hints: dict[str, Any] | None = None,
        mutation_instruction: str = "",
    ) -> ContextPacket:
        req = request if isinstance(request, ContextRequest) else ContextRequest.from_dict(request or {})
        safe_manifest = _snapshot_safe_manifest(snapshot)
        requested_paths = _resolve_selected_paths(req.need_files + req.need_tests, world, allowed_paths=safe_manifest)
        symbol_paths, unresolved_symbols = _resolve_symbol_paths(req.need_symbols, world, allowed_paths=safe_manifest)
        selected = list(dict.fromkeys([*requested_paths, *symbol_paths]))
        selected = _include_imported_sources(selected, world, allowed_paths=safe_manifest)
        if not selected:
            selected = [path for path in _default_files(world) if path in safe_manifest]
        root = Path(snapshot.root_path) if snapshot.root_path else None
        raw_slices: dict[str, str] = {}
        summaries: dict[str, str] = {}
        source_hashes: dict[str, str] = {}
        truncated_files: dict[str, dict[str, int]] = {}
        for rel in selected:
            summaries[rel] = world.file_roles.get(rel, "unknown")
            if root:
                path = _safe_project_file(root, rel, safe_manifest)
                if path is not None:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    source_text = _bounded_source_text(text, self.max_file_chars)
                    raw_slices[rel] = source_text
                    if source_text != text:
                        truncated_files[rel] = {"original_chars": len(text), "selected_chars": len(source_text)}
                    source_hashes[rel] = stable_hash({"path": rel, "content": text})
        contract_dict = contract.to_dict() if hasattr(contract, "to_dict") else dict(contract or {})
        coverage = {
            "requested_files": list(req.need_files),
            "requested_symbols": list(req.need_symbols),
            "requested_tests": list(req.need_tests),
            "selected_files": list(selected),
            "symbol_selected_files": list(symbol_paths),
            "unresolved_symbols": unresolved_symbols,
            "raw_slice_files": sorted(raw_slices),
            "source_text_policy": "full_files" if self.max_file_chars <= 0 else f"explicit_per_file_cap_{self.max_file_chars}_chars",
            "truncated_files": truncated_files,
            "target_obligation_ids": list(req.target_obligation_ids),
            "evidence_need": req.evidence_need,
        }
        return ContextPacket(
            objective_contract=contract_dict,
            project_summary=world.project_summary,
            selected_file_summaries=summaries,
            raw_file_slices=raw_slices,
            symbol_dependencies={rel: world.dependency_graph.get(rel, []) for rel in selected},
            test_feedback=[],
            parent_candidate_summaries=[candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate) for candidate in (parent_candidates or [])],
            archive_hints=archive_hints or {},
            source_hashes=source_hashes,
            evidence_need=req.evidence_need,
            coverage=coverage,
            mutation_instruction=mutation_instruction,
        )


def _default_files(world: ProjectWorldModel) -> list[str]:
    ranked = sorted(world.objective_relevance_map, key=lambda path: world.objective_relevance_map[path], reverse=True)
    if ranked:
        return ranked[:5]
    impl = [path for path, role in world.file_roles.items() if role in {"implementation", "test", "config"}]
    return impl[:5]


def _resolve_selected_paths(paths: list[str], world: ProjectWorldModel, *, allowed_paths: set[str]) -> list[str]:
    manifest = [path for path in world.file_roles if path in allowed_paths]
    resolved: list[str] = []
    for raw in paths:
        item = str(raw or "").strip().lstrip("./")
        if not item or any(char in item for char in "*?[") or _unsafe_request_path(item):
            continue
        match = _resolve_manifest_path(item, manifest)
        if match and match in allowed_paths:
            resolved.append(match)
    return list(dict.fromkeys(resolved))


def _resolve_symbol_paths(symbols: list[str], world: ProjectWorldModel, *, allowed_paths: set[str]) -> tuple[list[str], list[str]]:
    """Resolve requested symbol names to every manifest file declaring them."""

    resolved: list[str] = []
    unresolved: list[str] = []
    for raw in symbols:
        requested = str(raw or "").strip()
        if not requested:
            continue
        matched: list[str] = []
        for path, declared_symbols in world.symbol_graph.items():
            if path not in allowed_paths:
                continue
            if any(_symbol_request_matches(requested, declared) for declared in declared_symbols):
                matched.append(path)
        if matched:
            resolved.extend(matched)
        else:
            unresolved.append(requested)
    return list(dict.fromkeys(resolved)), list(dict.fromkeys(unresolved))


def _symbol_request_matches(requested: str, declared: str) -> bool:
    query = str(requested or "").strip().lower()
    symbol = str(declared or "").strip().lower()
    return bool(symbol) and (
        query == symbol
        or query.endswith("." + symbol)
        or query.endswith(":" + symbol)
        or query.endswith("#" + symbol)
    )


def _include_imported_sources(selected: list[str], world: ProjectWorldModel, *, allowed_paths: set[str], per_test_limit: int = 4, total_limit: int = 16) -> list[str]:
    manifest = set(world.file_roles).intersection(allowed_paths)
    expanded = list(selected)
    query_tokens = _tokens(" ".join(selected))
    imported: list[str] = []
    for rel in selected:
        if world.file_roles.get(rel) != "test":
            continue
        per_test: list[tuple[float, str]] = []
        for module in world.dependency_graph.get(rel, []):
            path = _module_to_project_path(module)
            if not path or path not in manifest:
                continue
            score = _path_score(path, query_tokens)
            per_test.append((score, path))
        for _, path in sorted(per_test, key=lambda item: (-item[0], item[1]))[:per_test_limit]:
            imported.append(path)
    for path in list(dict.fromkeys(imported))[:total_limit]:
        expanded.append(path)
    return list(dict.fromkeys(expanded))


def _snapshot_safe_manifest(snapshot: ProjectSnapshot) -> set[str]:
    safe: set[str] = set()
    for item in snapshot.file_manifest:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().lstrip("./")
        if not path or bool(item.get("binary")) or _unsafe_request_path(path):
            continue
        safe.add(path)
    if not safe:
        safe.update(path for path in snapshot.file_hashes if not _unsafe_request_path(path))
    return safe


def _unsafe_request_path(path: str) -> bool:
    item = str(path or "").strip()
    if not item or item.startswith("/") or item.startswith("~"):
        return True
    parts = Path(item).parts
    return any(part in {"", ".", ".."} for part in parts)


def _safe_project_file(root: Path, rel: str, allowed_paths: set[str]) -> Path | None:
    if rel not in allowed_paths or _unsafe_request_path(rel):
        return None
    root_resolved = root.resolve()
    path = root_resolved / rel
    if path.is_symlink():
        return None
    try:
        resolved = path.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None
    return resolved


def _resolve_manifest_path(item: str, manifest: list[str]) -> str:
    if item in manifest:
        return item
    name = Path(item).name
    basename_matches = [path for path in manifest if Path(path).name == name]
    if len(basename_matches) == 1:
        return basename_matches[0]
    suffix_matches = [path for path in manifest if "/" in item and path.endswith(item)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    query = _tokens(item)
    if not query:
        return ""
    scored = [(_path_score(path, query), path) for path in manifest if path.endswith(".py")]
    scored = [(score, path) for score, path in scored if score >= 2.0]
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    if len(scored) == 1 or scored[0][0] > scored[1][0]:
        return scored[0][1]
    return ""


def _module_to_project_path(module: str) -> str:
    text = str(module or "").strip()
    if not text.startswith("cognitive_evolve_runtime."):
        return ""
    return text.replace(".", "/") + ".py"


def _path_score(path: str, query_tokens: set[str]) -> float:
    path_tokens = _tokens(path)
    score = float(len(path_tokens & query_tokens))
    if Path(path).name.removesuffix(".py") in query_tokens:
        score += 1.0
    if path.startswith("tests/"):
        score -= 0.5
    return score


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower().replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " ")
    return {token for token in lowered.split() if len(token) > 2 and token not in {"the", "and", "test", "tests", "runtime", "cognitive", "evolve"}}


def _bounded_source_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n...[source region omitted from {len(text)} chars]...\n"
    available = max(3, max_chars - 2 * len(marker))
    head = max(1, available // 3)
    middle = max(1, available // 3)
    tail = max(1, available - head - middle)
    middle_start = max(head, (len(text) - middle) // 2)
    return text[:head] + marker + text[middle_start : middle_start + middle] + marker + text[-tail:]


__all__ = ["ContextRequest", "ContextPacket", "ContextSelector"]
