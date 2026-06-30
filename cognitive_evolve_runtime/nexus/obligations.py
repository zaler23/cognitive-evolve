"""Proof/progress obligation utilities for hard open-ended tasks.

This module is intentionally deterministic.  It does not try to prove the
theorem; it enforces a runtime invariant: for proof-like objectives, narrative
deepening is not progress unless the candidate carries a concrete formal object
and a ledger delta against named obligations.
"""
from __future__ import annotations

import re
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

FORMAL_TEXT_PATTERN = re.compile(
    r"(=|≤|>=|<=|\bforall\b|\bexists\b|∀|∃|\bsum\b|\bprod\b|\bmod\b|\bgraph\b|\bpolynomial\b|\bequation\b|\binequality\b|[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))",
    re.IGNORECASE,
)

PLACEHOLDER_PATTERN = re.compile(r"\b(todo|tbd|placeholder|needs? to|should derive|must derive|not specified|unknown|缺少|待定)\b", re.IGNORECASE)


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
    "blocking_obligations_from_history",
    "candidate_evidence_refs",
    "candidate_has_obligation_or_evidence_delta",
    "candidate_formal_artifacts",
    "candidate_obligation_delta",
    "candidate_source_bindings",
    "candidate_targets_blocking",
    "formal_artifact_kind",
    "formal_signature",
    "looks_like_formal_artifact",
]
