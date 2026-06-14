"""Proof/progress obligation utilities for hard open-ended tasks.

This module is intentionally deterministic.  It does not try to prove the
theorem; it enforces a runtime invariant: for proof-like objectives, narrative
deepening is not progress unless the candidate carries a concrete formal object
and a ledger delta against named obligations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, stable_hash


FORMAL_ARTIFACT_KINDS = {
    "equation_set",
    "construction",
    "inequality_proof",
    "case_analysis",
    "witness",
    "counterexample",
    "lemma_ref",
    "derivation",
    "proof_step",
    "assertion_set",
    "verification_witness",
}

PROGRESS_DELTA_KEYS = {
    "discharged",
    "closed",
    "decomposed",
    "refuted",
    "introduced",
    "blocked",
    "targeted",
}

EVIDENCE_REF_KINDS = {
    "formal_artifact",
    "source_file",
    "schema_field",
    "test",
    "patch",
    "checkpoint",
    "event",
    "verification",
    "command",
    "artifact",
}

HARD_PROOF_FAILURES = {
    "proof_object_absent",
    "proof_object_structurally_weak",
    "ledger_non_progressing",
    "duplicate_formal_signature",
    "blocking_obligation_not_targeted",
}

HARD_EVIDENCE_FAILURES = {
    "obligation_delta_absent",
    "evidence_ref_absent",
    "evidence_ref_unverified",
    "source_binding_absent",
    "source_binding_missing_path",
    "patch_target_missing",
}

PROOF_TASK_TOKENS = (
    "prove",
    "proof",
    "theorem",
    "lemma",
    "conjecture",
    "derive",
    "derivation",
    "equation",
    "inequality",
    "bound",
    "lower bound",
    "upper bound",
    "counterexample",
    "证明",
    "证法",
    "定理",
    "引理",
    "猜想",
    "推导",
    "方程",
    "不等式",
    "上界",
    "下界",
    "反例",
)

FORMAL_TEXT_PATTERN = re.compile(
    r"(=|≤|>=|<=|\bforall\b|\bexists\b|∀|∃|\bsum\b|\bprod\b|\bmod\b|\bgraph\b|\bpolynomial\b|\bequation\b|\binequality\b|[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))",
    re.IGNORECASE,
)

PLACEHOLDER_PATTERN = re.compile(r"\b(todo|tbd|placeholder|needs? to|should derive|must derive|not specified|unknown|缺少|待定)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProofProgressSummary:
    candidate_id: str
    required: bool
    artifact_count: int = 0
    structural_ok: bool = False
    progress_action_count: int = 0
    formal_signature: str = ""
    duplicate_formal_signature: bool = False
    targets_blocking: bool = True
    rank_eligible: bool = True
    final_eligible: bool = True
    diagnostics: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceObligationSummary:
    candidate_id: str
    required: bool
    obligation_delta_count: int = 0
    evidence_ref_count: int = 0
    verified_evidence_ref_count: int = 0
    source_binding_count: int = 0
    rank_eligible: bool = True
    final_eligible: bool = True
    diagnostics: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def requires_proof_progress(contract: Any | None = None, world: Any | None = None) -> bool:
    """Heuristic proof-task detector used only to enable stricter gates."""

    text_parts: list[str] = []
    for source in (contract, world):
        if source is None:
            continue
        if isinstance(source, dict):
            values = source.values()
        else:
            values = [
                getattr(source, "original_user_goal", ""),
                getattr(source, "normalized_goal", ""),
                getattr(source, "expected_output_forms", ""),
                getattr(source, "verification_preferences", ""),
                getattr(source, "success_dimensions", ""),
                getattr(source, "raw_text", ""),
                getattr(source, "kind", ""),
            ]
        for value in values:
            if isinstance(value, (list, tuple, set)):
                text_parts.extend(str(item) for item in value)
            else:
                text_parts.append(str(value))
    text = " ".join(text_parts).lower()
    return any(token in text for token in PROOF_TASK_TOKENS)


def requires_source_grounding(contract: Any | None = None, world: Any | None = None, candidate: CandidateGenome | None = None) -> bool:
    """Return true for project/code/schema tasks where proposals need exact sources."""

    if candidate is not None:
        artifact_type = str(getattr(candidate, "artifact_type", "") or "").lower()
        if artifact_type in {"project_patch", "patch", "code_patch"} or getattr(candidate, "patch_set", None) or getattr(candidate, "touched_files", None):
            return True
    text_parts: list[str] = []
    for source in (contract, world):
        if source is None:
            continue
        if isinstance(source, dict):
            values = source.values()
        else:
            values = [
                getattr(source, "kind", ""),
                getattr(source, "project_summary", ""),
                getattr(source, "normalized_goal", ""),
                getattr(source, "expected_output_forms", ""),
                getattr(source, "verification_preferences", ""),
                getattr(source, "implementation_files", ""),
                getattr(source, "test_contracts", ""),
                getattr(source, "allowed_patch_scope", ""),
            ]
        for value in values:
            if isinstance(value, (list, tuple, set)):
                text_parts.extend(str(item) for item in value)
            else:
                text_parts.append(str(value))
    text = " ".join(text_parts).lower()
    return any(token in text for token in ("project", "patch", "code", "test", "schema", "file", "pytest", "source"))


def evidence_obligation_summary(
    candidate: CandidateGenome,
    *,
    contract: Any | None = None,
    world: Any | None = None,
) -> EvidenceObligationSummary:
    """Check that named obligations are bound to runtime-verifiable evidence."""

    proof_required = requires_proof_progress(contract, world)
    source_required = requires_source_grounding(contract, world, candidate)
    required = proof_required or source_required or bool(candidate.obligation_delta)
    if not required:
        return EvidenceObligationSummary(candidate_id=candidate.id, required=False, score=1.0)

    obligation_delta = candidate_obligation_delta(candidate)
    obligation_delta_count = _count_progress_actions(obligation_delta, candidate.proof_obligations)
    evidence_refs = candidate_evidence_refs(candidate)
    verified_refs = [item for item in evidence_refs if evidence_ref_verified(item)]
    source_bindings = candidate_source_bindings(candidate)

    diagnostics: list[str] = []
    if obligation_delta_count <= 0:
        diagnostics.append("obligation_delta_absent")
    if not evidence_refs:
        diagnostics.append("evidence_ref_absent")
    if evidence_refs and not verified_refs:
        diagnostics.append("evidence_ref_unverified")
    if source_required and not source_bindings:
        diagnostics.append("source_binding_absent")

    score = 0.0
    if obligation_delta_count:
        score += min(0.35, 0.20 + 0.05 * obligation_delta_count)
    if evidence_refs:
        score += min(0.30, 0.12 + 0.04 * len(evidence_refs))
    if verified_refs:
        score += min(0.20, 0.08 + 0.04 * len(verified_refs))
    if source_bindings:
        score += min(0.15, 0.06 + 0.03 * len(source_bindings))
    score = max(0.0, min(1.0, score))
    passed = not any(item in HARD_EVIDENCE_FAILURES for item in diagnostics)
    return EvidenceObligationSummary(
        candidate_id=candidate.id,
        required=True,
        obligation_delta_count=obligation_delta_count,
        evidence_ref_count=len(evidence_refs),
        verified_evidence_ref_count=len(verified_refs),
        source_binding_count=len(source_bindings),
        rank_eligible=passed,
        final_eligible=passed,
        diagnostics=diagnostics,
        score=score,
    )


def proof_progress_summary(
    candidate: CandidateGenome,
    *,
    contract: Any | None = None,
    world: Any | None = None,
    existing_signatures: set[str] | None = None,
    blocking_obligation_ids: list[str] | None = None,
) -> ProofProgressSummary:
    required = requires_proof_progress(contract, world)
    if not required:
        return ProofProgressSummary(candidate_id=candidate.id, required=False, score=1.0)

    artifacts = candidate_formal_artifacts(candidate)
    structural_ok = any(looks_like_formal_artifact(item) for item in artifacts)
    signature = formal_signature(candidate)
    duplicate = bool(signature and existing_signatures is not None and signature in existing_signatures)
    delta = candidate_obligation_delta(candidate)
    progress_actions = _count_progress_actions(delta, candidate.proof_obligations)
    blocking_ids = [item for item in coerce_str_list(blocking_obligation_ids) if item]
    targets_blocking = True if not blocking_ids else candidate_targets_blocking(candidate, blocking_ids)

    diagnostics: list[str] = []
    if not artifacts:
        diagnostics.append("proof_object_absent")
    elif not structural_ok:
        diagnostics.append("proof_object_structurally_weak")
    if progress_actions <= 0:
        diagnostics.append("ledger_non_progressing")
    if duplicate:
        diagnostics.append("duplicate_formal_signature")
    if not targets_blocking:
        diagnostics.append("blocking_obligation_not_targeted")

    score = 0.0
    if artifacts:
        score += 0.35
    if structural_ok:
        score += 0.25
    if progress_actions > 0:
        score += min(0.25, 0.08 * progress_actions + 0.09)
    if targets_blocking:
        score += 0.10
    if duplicate:
        score -= 0.25
    score = max(0.0, min(1.0, score))
    passed = not any(item in HARD_PROOF_FAILURES for item in diagnostics)
    return ProofProgressSummary(
        candidate_id=candidate.id,
        required=True,
        artifact_count=len(artifacts),
        structural_ok=structural_ok,
        progress_action_count=progress_actions,
        formal_signature=signature,
        duplicate_formal_signature=duplicate,
        targets_blocking=targets_blocking,
        rank_eligible=passed,
        final_eligible=passed,
        diagnostics=diagnostics,
        score=score,
    )


def candidate_formal_artifacts(candidate: CandidateGenome) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for item in getattr(candidate, "formal_artifacts", []) or []:
        if isinstance(item, dict):
            artifacts.append(coerce_dict(item))
    metadata = coerce_dict(candidate.metadata)
    for key in ("proof_objects", "formal_artifacts"):
        value = metadata.get(key)
        if isinstance(value, list):
            artifacts.extend(coerce_dict(item) for item in value if isinstance(item, dict))
    if isinstance(candidate.artifact, dict):
        artifact = coerce_dict(candidate.artifact)
        kind = formal_artifact_kind(artifact)
        if kind in FORMAL_ARTIFACT_KINDS or any(key in artifact for key in ("equations", "inequalities", "construction", "witness", "counterexample", "lemma", "cases", "derivation", "assertions", "checks", "invariants", "test_cases")):
            artifacts.append(artifact)
    elif str(candidate.artifact_type or "").strip().lower() in FORMAL_ARTIFACT_KINDS:
        artifacts.append({"kind": str(candidate.artifact_type), "content": str(candidate.artifact or "")})
    return artifacts


def candidate_obligation_delta(candidate: CandidateGenome) -> dict[str, Any]:
    delta = coerce_dict(getattr(candidate, "obligation_delta", {}))
    if not delta:
        metadata = coerce_dict(candidate.metadata)
        delta = coerce_dict(metadata.get("obligation_delta"))
    return delta


def candidate_has_obligation_or_evidence_delta(candidate: CandidateGenome) -> bool:
    if _count_progress_actions(candidate_obligation_delta(candidate), candidate.proof_obligations) > 0:
        return True
    delta = coerce_dict(getattr(candidate, "evidence_delta", {}))
    return any(bool(value) for value in delta.values())


def candidate_evidence_refs(candidate: CandidateGenome) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in getattr(candidate, "evidence_refs", []) or []:
        if isinstance(item, dict):
            refs.append(normalize_evidence_ref(item))
    metadata = coerce_dict(candidate.metadata)
    value = metadata.get("evidence_refs")
    if isinstance(value, list):
        refs.extend(normalize_evidence_ref(item) for item in value if isinstance(item, dict))
    for artifact in candidate_formal_artifacts(candidate):
        refs.append(
            normalize_evidence_ref(
                {
                    "id": artifact.get("id") or artifact.get("target_obligation_id") or formal_artifact_ref_id(artifact),
                    "kind": "formal_artifact",
                    "status": "verified" if looks_like_formal_artifact(artifact) else "unverified",
                    "source_hash": stable_hash(artifact),
                }
            )
        )
    for item in list(getattr(candidate, "tool_results", []) or []) + list(getattr(candidate, "verification_trace", []) or []):
        if isinstance(item, dict) and item.get("status") in {"ok", "passed", "applied", "verified"}:
            refs.append(
                normalize_evidence_ref(
                    {
                        "id": item.get("tool_id") or item.get("id") or stable_hash(item)[:16],
                        "kind": "verification",
                        "status": "verified",
                        "source_hash": stable_hash(item),
                    }
                )
            )
    return _dedupe_dicts(refs, key_fields=("id", "kind", "source_hash"))


def candidate_source_bindings(candidate: CandidateGenome) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for item in getattr(candidate, "source_bindings", []) or []:
        if isinstance(item, dict):
            bindings.append(coerce_dict(item))
    for path in coerce_str_list(getattr(candidate, "touched_files", [])):
        bindings.append({"path": path, "kind": "touched_file", "required": True})
    for op in getattr(candidate, "patch_set", []) or []:
        path = getattr(op, "path", None) or (op.get("path") if isinstance(op, dict) else "")
        if path:
            bindings.append({"path": str(path), "kind": "patch_operation", "required": True})
    metadata = coerce_dict(candidate.metadata)
    value = metadata.get("source_bindings")
    if isinstance(value, list):
        bindings.extend(coerce_dict(item) for item in value if isinstance(item, dict))
    return _dedupe_dicts(bindings, key_fields=("path", "symbol", "kind"))


def evidence_ref_verified(ref: dict[str, Any]) -> bool:
    status = str(ref.get("status") or ref.get("verification_status") or "").strip().lower()
    return status in {"verified", "passed", "ok", "applied", "committed"}


def normalize_evidence_ref(ref: dict[str, Any]) -> dict[str, Any]:
    out = coerce_dict(ref)
    kind = str(out.get("kind") or out.get("type") or out.get("ref_type") or "artifact").strip().lower()
    out["kind"] = kind if kind in EVIDENCE_REF_KINDS else "artifact"
    out["id"] = str(out.get("id") or out.get("ref") or out.get("path") or out.get("test_name") or stable_hash(out)[:16])
    if "status" in out:
        out["status"] = str(out.get("status") or "")
    return out


def formal_artifact_ref_id(artifact: dict[str, Any]) -> str:
    return "formal_" + stable_hash(artifact)[:12]


def candidate_targets_blocking(candidate: CandidateGenome, blocking_ids: list[str]) -> bool:
    targets = set(coerce_str_list(candidate_obligation_delta(candidate).get("targeted")))
    metadata = coerce_dict(candidate.metadata)
    targets.update(coerce_str_list(metadata.get("target_obligation_ids")))
    targets.update(coerce_str_list(metadata.get("blocking_obligation_ids")))
    for artifact in candidate_formal_artifacts(candidate):
        targets.update(coerce_str_list(artifact.get("target_obligation_ids")))
        if artifact.get("target_obligation_id"):
            targets.add(str(artifact.get("target_obligation_id")))
    return bool(targets.intersection(blocking_ids))


def formal_artifact_kind(artifact: dict[str, Any]) -> str:
    """Return the normalized formal artifact kind.

    Model schemas historically used ``kind`` while several model outputs used
    JSON-schema-style ``type``.  Treating ``type`` as an alias prevents valid
    assertion_set/verification_witness objects from being misclassified as
    structurally weak solely because of the field name.
    """

    return str(
        artifact.get("kind")
        or artifact.get("artifact_kind")
        or artifact.get("artifact_type")
        or artifact.get("type")
        or ""
    ).strip().lower()


def looks_like_formal_artifact(artifact: dict[str, Any]) -> bool:
    kind = formal_artifact_kind(artifact)
    if kind not in FORMAL_ARTIFACT_KINDS:
        return False
    content_fields = [
        artifact.get("content"),
        artifact.get("statement"),
        artifact.get("expression"),
        artifact.get("construction"),
        artifact.get("witness"),
        artifact.get("counterexample"),
        artifact.get("lemma"),
        artifact.get("derivation"),
        artifact.get("equations"),
        artifact.get("inequalities"),
        artifact.get("cases"),
        artifact.get("object"),
        artifact.get("assertions"),
        artifact.get("checks"),
        artifact.get("invariants"),
        artifact.get("test_cases"),
        artifact.get("expected_results"),
    ]
    text = normalize_formal_text(content_fields)
    if len(text) < 8 or PLACEHOLDER_PATTERN.search(text):
        return False
    return bool(FORMAL_TEXT_PATTERN.search(text))


def formal_signature(candidate: CandidateGenome) -> str:
    artifacts = [artifact for artifact in candidate_formal_artifacts(candidate) if looks_like_formal_artifact(artifact)]
    if not artifacts:
        return ""
    payload = [
        {
            "kind": formal_artifact_kind(artifact),
            "text": normalize_formal_text(artifact),
        }
        for artifact in artifacts
    ]
    return stable_hash(payload)


def normalize_formal_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = [str(key) + ":" + normalize_formal_text(val) for key, val in sorted(value.items())]
        text = " ".join(parts)
    elif isinstance(value, (list, tuple, set)):
        text = " ".join(normalize_formal_text(item) for item in value)
    else:
        text = str(value or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def blocking_obligations_from_history(history: list[dict[str, Any]] | None, *, threshold: int = 3) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for entry in history or []:
        for candidate in _candidate_dicts_from_history_entry(entry):
            for missing in coerce_str_list(candidate.get("missing_parts")):
                key = _obligation_key(missing)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
                labels.setdefault(key, missing)
    return [
        {"id": key, "description": labels[key], "status": "blocking", "seen_count": count}
        for key, count in counts.items()
        if count >= threshold
    ]


def repeated_proof_failure_counts(candidates: list[CandidateGenome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        result = coerce_dict(getattr(candidate, "verification_result", {}))
        diagnostics = coerce_str_list(result.get("diagnostics"))
        proof = coerce_dict(result.get("proof_progress"))
        diagnostics.extend(coerce_str_list(proof.get("diagnostics")))
        for item in diagnostics:
            if item in HARD_PROOF_FAILURES:
                counts[item] = counts.get(item, 0) + 1
    return counts


def _count_progress_actions(delta: dict[str, Any], obligations: list[dict[str, Any]]) -> int:
    count = 0
    for key in PROGRESS_DELTA_KEYS:
        value = delta.get(key)
        if isinstance(value, list):
            count += len([item for item in value if item])
        elif value:
            count += 1
    for obligation in obligations or []:
        if not isinstance(obligation, dict):
            continue
        status = str(obligation.get("status") or "").strip().lower()
        if status in {"discharged", "closed", "decomposed", "refuted", "introduced", "blocked"}:
            count += 1
    return count


def _dedupe_dicts(items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _candidate_dicts_from_history_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    population = entry.get("population")
    if isinstance(population, dict):
        for item in population.get("candidates", []) or []:
            if isinstance(item, dict):
                out.append(item)
    return out


def _obligation_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", str(text or "").strip().lower()).strip("_")
    if not normalized:
        return ""
    return "obl_" + stable_hash(normalized)[:12]


__all__ = [
    "EVIDENCE_REF_KINDS",
    "FORMAL_ARTIFACT_KINDS",
    "HARD_EVIDENCE_FAILURES",
    "HARD_PROOF_FAILURES",
    "EvidenceObligationSummary",
    "ProofProgressSummary",
    "blocking_obligations_from_history",
    "candidate_evidence_refs",
    "candidate_has_obligation_or_evidence_delta",
    "candidate_formal_artifacts",
    "candidate_obligation_delta",
    "candidate_source_bindings",
    "candidate_targets_blocking",
    "evidence_obligation_summary",
    "formal_artifact_kind",
    "formal_signature",
    "looks_like_formal_artifact",
    "proof_progress_summary",
    "repeated_proof_failure_counts",
    "requires_proof_progress",
    "requires_source_grounding",
]
