"""Deterministic candidate fingerprints used by search-kernel consumers."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.obligations import formal_signature

_RUNTIME_KEYS = {
    "id",
    "round",
    "created_at",
    "runtime",
    "runtime_ns",
    "runtime_ms",
    "duration",
    "elapsed",
    "latency",
    "timestamp",
    "wall_time",
}


@dataclass(frozen=True)
class CandidateFingerprint:
    artifact_type: str
    semantic_signature: str
    artifact_signature: str
    grounded_signature: str
    descriptor_tokens: tuple[str, ...]
    lineage_root: str
    failure_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _NameNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast API
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802 - ast API
        node.arg = "ARG"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802 - ast API
        node.name = "FUNC"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802 - ast API
        node.name = "FUNC"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802 - ast API
        node.name = "CLASS"
        self.generic_visit(node)
        return node


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802 - ast API
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802 - ast API
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802 - ast API
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node


def candidate_fingerprint(candidate: CandidateGenome) -> CandidateFingerprint:
    artifact_sig = _artifact_signature(_candidate_artifact_value(candidate))
    grounded = _grounded_signature(candidate, artifact_sig=artifact_sig)
    tokens = tuple(sorted(candidate_descriptor_tokens(candidate)))
    lineage_root = str(candidate.lineage[0] if candidate.lineage else candidate.id)
    failure_sig = _failure_signature(candidate)
    return CandidateFingerprint(
        artifact_type=str(candidate.artifact_type or "answer").lower(),
        semantic_signature=candidate_semantic_signature(candidate),
        artifact_signature=artifact_sig,
        grounded_signature=grounded,
        descriptor_tokens=tokens,
        lineage_root=lineage_root,
        failure_signature=failure_sig,
    )



def base_mechanism_family(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    search_space = coerce_dict(getattr(candidate, "search_space", None) or metadata.get("search_space"))
    for value in (
        search_space.get("family_id"),
        search_space.get("plane_id"),
        candidate.niche_memberships[0] if candidate.niche_memberships else "",
        candidate.novelty_descriptors[0] if candidate.novelty_descriptors else "",
        candidate.core_mechanism,
        candidate.artifact_type,
        candidate.lineage[0] if candidate.lineage else candidate.id,
    ):
        token = normalize_token(value)
        if token:
            return token[:80]
    return "general"


def candidate_semantic_signature(candidate: CandidateGenome) -> str:
    mechanism = normalize_token(candidate.core_mechanism or "")
    claim = normalize_text(candidate.concise_claim or "")
    artifact = _artifact_signature(_candidate_artifact_value(candidate))
    if artifact:
        artifact = artifact[:320]
    proof_sig = formal_signature(candidate)
    descriptors = ",".join(sorted(normalize_token(item) for item in (candidate.niche_memberships + candidate.novelty_descriptors) if item))
    return "|".join([str(candidate.artifact_type or "answer").lower(), mechanism, claim, artifact, proof_sig, descriptors])


def candidate_descriptor_tokens(candidate: CandidateGenome) -> set[str]:
    values: list[Any] = [
        candidate.id,
        candidate.artifact_type,
        candidate.core_mechanism,
        candidate.concise_claim,
        *candidate.niche_memberships,
        *candidate.novelty_descriptors,
        *candidate.edge_knowledge_seeds,
    ]
    metadata = coerce_dict(candidate.metadata)
    for key in ("seed_type", "exploration_source", "operator", "failure_class", "verifier_modality", "descriptor_cell"):
        values.append(metadata.get(key))
    for trace in candidate.verification_trace:
        if isinstance(trace, dict):
            values.extend([trace.get("modality"), trace.get("oracle_kind"), trace.get("verifier_fingerprint")])
    out: set[str] = set()
    for value in values:
        text = normalize_token(value)
        if not text:
            continue
        out.add(text)
        out.update(part for part in text.split("_") if part)
    return out


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", str(value or "").strip().lower())).strip()


def normalized_ast_signature(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    tree = _DocstringStripper().visit(tree)
    tree = _NameNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return stable_hash(ast.dump(tree, include_attributes=False))


def _candidate_artifact_value(candidate: CandidateGenome) -> Any:
    patch_set = getattr(candidate, "patch_set", None)
    if patch_set:
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in patch_set]
    return candidate.artifact


def _artifact_semantic_text(artifact: Any) -> str:
    if isinstance(artifact, str):
        return normalize_text(artifact)
    return normalize_text(_stable_json(_strip_runtime_fields(artifact)))


def _artifact_signature(artifact: Any) -> str:
    if isinstance(artifact, str):
        ast_sig = normalized_ast_signature(artifact)
        if ast_sig:
            return "pyast:" + ast_sig
        return "text:" + stable_hash(normalize_text(artifact)[:4000])
    return "json:" + stable_hash(_strip_runtime_fields(artifact))


def _grounded_signature(candidate: CandidateGenome, *, artifact_sig: str) -> str:
    if artifact_sig.startswith(("pyast:", "json:")):
        return artifact_sig
    passed: list[str] = []
    failed: list[str] = []
    for item in list(candidate.verification_trace or []) + list(candidate.proof_obligations or []):
        if not isinstance(item, dict):
            continue
        obligation_id = str(item.get("obligation_id") or item.get("id") or item.get("verifier_fingerprint") or "").strip()
        if not obligation_id:
            continue
        if item.get("passed") is True or str(item.get("status") or "").lower() in {"passed", "pass", "ok"}:
            passed.append(obligation_id)
        elif item.get("passed") is False or str(item.get("status") or "").lower() in {"failed", "fail", "error"}:
            failed.append(obligation_id)
    if not passed and not failed:
        return "UNDEFINED"
    return "obl:" + stable_hash({"passed": sorted(set(passed)), "failed": sorted(set(failed))})


def _failure_signature(candidate: CandidateGenome) -> str:
    lessons = [normalize_text(item)[:160] for item in candidate.failure_lessons if str(item or "").strip()]
    metadata = coerce_dict(candidate.metadata)
    failure = coerce_dict(metadata.get("failure_classification") or metadata.get("failure_class"))
    if not lessons and not failure:
        return ""
    return stable_hash({"lessons": lessons[:8], "failure": failure})


def _strip_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _RUNTIME_KEYS:
                continue
            out[str(key)] = _strip_runtime_fields(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_strip_runtime_fields(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return repr(value)


__all__ = [
    "CandidateFingerprint",
    "base_mechanism_family",
    "candidate_descriptor_tokens",
    "candidate_fingerprint",
    "candidate_semantic_signature",
    "normalize_text",
    "normalize_token",
    "normalized_ast_signature",
]
