"""Resolve candidate source-binding claims into admission routes.

The resolver is intentionally generic: models may claim paths, symbols, or
commands, but only engine resolution determines whether a candidate may enter
normal crossover/archive/final lanes.  Unresolved or invented bindings remain
usable as repair targets, not as implementation evidence.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import stable_hash


@dataclass(frozen=True)
class BindingClaim:
    path: str = ""
    symbol: str = ""
    command: str = ""
    kind: str = ""
    source: str = "source_bindings"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BindingResolution:
    claim: BindingClaim
    status: str
    diagnostics: list[str] = field(default_factory=list)
    resolved_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["claim"] = self.claim.to_dict()
        return data


@dataclass(frozen=True)
class SourceBindingManifest:
    candidate_id: str
    binding_class: str
    admission_route: str
    claims: list[BindingClaim] = field(default_factory=list)
    resolutions: list[BindingResolution] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["claims"] = [claim.to_dict() for claim in self.claims]
        data["resolutions"] = [resolution.to_dict() for resolution in self.resolutions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SourceBindingManifest":
        raw = data if isinstance(data, dict) else {}
        return cls(
            candidate_id=str(raw.get("candidate_id") or ""),
            binding_class=str(raw.get("binding_class") or "no_binding"),
            admission_route=str(raw.get("admission_route") or _route_for_class(str(raw.get("binding_class") or "no_binding"))),
            claims=[BindingClaim(**{k: str(item.get(k) or "") for k in ("path", "symbol", "command", "kind", "source")}) for item in raw.get("claims", []) if isinstance(item, dict)],
            resolutions=[_resolution_from_dict(item) for item in raw.get("resolutions", []) if isinstance(item, dict)],
            diagnostics=[str(item) for item in raw.get("diagnostics", []) if item],
            manifest_hash=str(raw.get("manifest_hash") or ""),
        )


def resolve_candidate_source_bindings(candidate: CandidateGenome, *, project_root: str | Path | None = None) -> SourceBindingManifest:
    root = _resolved_project_root(project_root)
    claims = _claims_from_candidate(candidate)
    if not claims:
        manifest = SourceBindingManifest(
            candidate_id=candidate.id,
            binding_class="no_binding",
            admission_route="repair_only",
            claims=[],
            resolutions=[],
            diagnostics=["no_source_binding_claims"],
        )
        return _with_hash(manifest)
    resolutions = [_resolve_claim(claim, root=root, candidate=candidate) for claim in claims]
    statuses = {resolution.status for resolution in resolutions}
    diagnostics = list(dict.fromkeys(diag for resolution in resolutions for diag in resolution.diagnostics))
    if statuses and statuses <= {"resolved"}:
        binding_class = "resolved"
    elif "invented" in statuses:
        binding_class = "invented"
    elif "negative_fixture_only" in statuses:
        binding_class = "negative_fixture_only"
    elif "unresolved" in statuses:
        binding_class = "unresolved"
    elif "unbindable" in statuses:
        binding_class = "unbindable"
    else:
        binding_class = "no_binding"
    manifest = SourceBindingManifest(
        candidate_id=candidate.id,
        binding_class=binding_class,
        admission_route=_route_for_class(binding_class),
        claims=claims,
        resolutions=resolutions,
        diagnostics=diagnostics,
    )
    return _with_hash(manifest)


def annotate_candidate_source_bindings(candidate: CandidateGenome, *, project_root: str | Path | None = None) -> SourceBindingManifest:
    manifest = resolve_candidate_source_bindings(candidate, project_root=project_root)
    if not isinstance(candidate.metadata, dict):
        candidate.metadata = {}
    candidate.metadata["source_binding_manifest"] = manifest.to_dict()
    candidate.metadata["source_binding_class"] = manifest.binding_class
    candidate.metadata["source_binding_admission_route"] = manifest.admission_route
    if manifest.binding_class in {"invented", "unresolved", "no_binding"}:
        candidate.metadata["selection_deprioritized_until_new_delta"] = True
    return manifest


def candidate_source_binding_class(candidate: CandidateGenome) -> str:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    manifest = metadata.get("source_binding_manifest") if isinstance(metadata.get("source_binding_manifest"), dict) else {}
    return str(manifest.get("binding_class") or metadata.get("source_binding_class") or "no_binding")


def candidate_admission_route(candidate: CandidateGenome) -> str:
    cls = candidate_source_binding_class(candidate)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    manifest = metadata.get("source_binding_manifest") if isinstance(metadata.get("source_binding_manifest"), dict) else {}
    return str(manifest.get("admission_route") or metadata.get("source_binding_admission_route") or _route_for_class(cls))


def final_candidate_source_bindings_allowed(candidate: CandidateGenome) -> bool:
    cls = candidate_source_binding_class(candidate)
    return cls not in {"invented", "unresolved"}


def _claims_from_candidate(candidate: CandidateGenome) -> list[BindingClaim]:
    claims: list[BindingClaim] = []
    for item in getattr(candidate, "source_bindings", []) or []:
        if not isinstance(item, dict):
            continue
        claims.append(
            BindingClaim(
                path=str(item.get("path") or item.get("file") or item.get("source_path") or ""),
                symbol=str(item.get("symbol") or item.get("name") or ""),
                command=str(item.get("command") or ""),
                kind=str(item.get("binding_mode") or item.get("kind") or item.get("type") or item.get("binding_type") or "source_file"),
                source="source_bindings",
            )
        )
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for command in metadata.get("targeted_commands", []) if isinstance(metadata.get("targeted_commands"), list) else []:
        claims.append(BindingClaim(command=str(command), kind="command", source="metadata.targeted_commands"))
    return claims


def _resolve_claim(claim: BindingClaim, *, root: Path | None, candidate: CandidateGenome | None = None) -> BindingResolution:
    diagnostics: list[str] = []
    kind = claim.kind.lower()
    if kind in {"negative_fixture", "negative_fixture_only", "counterexample"}:
        return BindingResolution(claim, "negative_fixture_only", ["negative_fixture_binding"], "")
    if claim.command and not claim.path:
        allowed = _command_allowed(claim.command)
        return BindingResolution(claim, "resolved" if allowed else "unresolved", [] if allowed else ["command_not_allowlisted"], "")
    if not claim.path:
        return BindingResolution(claim, "no_binding", ["binding_path_absent"], "")
    if root is None:
        return BindingResolution(claim, "unresolved", ["project_root_unavailable"], "")
    rel = _project_relative_path(claim.path, root=root)
    if not rel:
        return BindingResolution(claim, "invented", ["binding_path_outside_project"], "")
    target = root / rel
    if not target.exists():
        materialize = str(kind or "").lower() in {"materialize", "artifact_file", "extend"} or "materialize" in str(claim.kind).lower()
        if materialize and _patch_materializes(candidate, rel, claim.symbol):
            return BindingResolution(claim, "resolved", [], rel)
        return BindingResolution(claim, "unresolved" if materialize else "invented", ["binding_path_missing"], rel)
    if claim.symbol and target.suffix == ".py" and not _python_symbol_exists(target, claim.symbol):
        if kind in {"extend", "materialize"} and _patch_materializes(candidate, rel, claim.symbol):
            return BindingResolution(claim, "resolved", [], rel)
        return BindingResolution(claim, "unresolved", ["binding_symbol_missing"], rel)
    return BindingResolution(claim, "resolved", [], rel)


def _route_for_class(binding_class: str) -> str:
    return {
        "resolved": "normal",
        "negative_fixture_only": "negative_archive_only",
        "unresolved": "repair_only",
        "invented": "repair_only",
        "unbindable": "repair_only",
        "no_binding": "repair_only",
    }.get(str(binding_class or ""), "repair_only")


def _with_hash(manifest: SourceBindingManifest) -> SourceBindingManifest:
    data = manifest.to_dict()
    data.pop("manifest_hash", None)
    return SourceBindingManifest(
        candidate_id=manifest.candidate_id,
        binding_class=manifest.binding_class,
        admission_route=manifest.admission_route,
        claims=manifest.claims,
        resolutions=manifest.resolutions,
        diagnostics=manifest.diagnostics,
        manifest_hash="source-binding-" + stable_hash(data)[:20],
    )


def _resolution_from_dict(item: dict[str, Any]) -> BindingResolution:
    claim_data = item.get("claim") if isinstance(item.get("claim"), dict) else {}
    return BindingResolution(
        claim=BindingClaim(**{k: str(claim_data.get(k) or "") for k in ("path", "symbol", "command", "kind", "source")}),
        status=str(item.get("status") or "unresolved"),
        diagnostics=[str(x) for x in item.get("diagnostics", []) if x],
        resolved_path=str(item.get("resolved_path") or ""),
    )


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



def _patch_materializes(candidate: CandidateGenome | None, relative_path: str, symbol: str = "") -> bool:
    if candidate is None:
        return False
    patch_result = getattr(candidate, "patch_application_result", None)
    applied_files = []
    if isinstance(patch_result, dict):
        applied_files = [str(item).replace("\\", "/") for item in patch_result.get("applied_files", []) if item]
    path_ok = relative_path in applied_files or not applied_files
    artifact = getattr(candidate, "artifact", "")
    if isinstance(artifact, dict):
        text = "\n".join(str(v) for v in artifact.values())
    else:
        text = str(artifact or "")
    if relative_path and relative_path not in text and not path_ok:
        return False
    if symbol and symbol not in text:
        return False
    return bool(path_ok or relative_path in text)

def _python_symbol_exists(path: Path, symbol: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return True
    return False


def _command_allowed(command: str) -> bool:
    head = str(command or "").strip().split(" ", 1)[0]
    return head in {"python", "python3", "pytest", "ruff", "mypy", "npm"}


__all__ = [
    "BindingClaim",
    "BindingResolution",
    "SourceBindingManifest",
    "annotate_candidate_source_bindings",
    "candidate_admission_route",
    "candidate_source_binding_class",
    "final_candidate_source_bindings_allowed",
    "resolve_candidate_source_bindings",
]
