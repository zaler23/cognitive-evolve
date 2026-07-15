"""Hash-locked claim grounding against one frozen input dossier."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .evidence import EvidenceRecord, apply_evidence_record


CLAIM_STATUSES = frozenset({"supported", "contradicted", "unknown"})


def frozen_dossier_hash(dossier_text: str) -> str:
    return "sha256:" + hashlib.sha256(str(dossier_text).encode("utf-8")).hexdigest()


def ground_frozen_dossier_claims(
    candidate: Any,
    *,
    dossier_text: str,
    dossier_hash: str,
    claims: Iterable[dict[str, Any]],
) -> EvidenceRecord:
    actual_hash = frozen_dossier_hash(dossier_text)
    if dossier_hash != actual_hash:
        raise ValueError(f"dossier hash mismatch: {dossier_hash} != {actual_hash}")

    results = [_ground_claim(item, dossier_text=dossier_text, dossier_hash=dossier_hash) for item in claims]
    supported = sum(item["status"] == "supported" for item in results)
    contradicted = sum(item["status"] == "contradicted" for item in results)
    unknown = sum(item["status"] == "unknown" for item in results)
    total = len(results)
    record = EvidenceRecord(
        candidate_id=str(getattr(candidate, "id", "")),
        source="frozen_dossier_grounding",
        stage="dossier_grounding",
        score=supported / max(1, total),
        confidence=1.0,
        final_blocked=True,
        diagnostics=[
            f"{item['claim_id']}:{item['status']}:{item['reason']}"
            for item in results
            if item["status"] != "supported"
        ],
        metadata={
            "authority": "frozen_dossier",
            "dossier_hash": dossier_hash,
            "claim_grounding": results,
            "classification_counts": {
                "supported": supported,
                "contradicted": contradicted,
                "unknown": unknown,
            },
        },
    )
    apply_evidence_record(candidate, record)
    return record


def _ground_claim(raw: dict[str, Any], *, dossier_text: str, dossier_hash: str) -> dict[str, Any]:
    claim_id = str(raw.get("claim_id") or "")
    claim = str(raw.get("claim") or "")
    proposed = str(raw.get("status") or "unknown").lower()
    source = str(raw.get("source") or "dossier")
    external = source not in {"dossier", "frozen_dossier"} or bool(raw.get("external_refs"))
    spans = _valid_spans(raw.get("evidence_spans"), dossier_text)
    if external:
        status, spans, reason = "unknown", [], "external_reference"
    elif proposed == "unknown":
        status, spans, reason = "unknown", [], "model_unknown_preserved"
    elif proposed not in CLAIM_STATUSES:
        status, spans, reason = "unknown", [], "invalid_status"
    elif not spans:
        status, reason = "unknown", "missing_or_invalid_dossier_span"
    else:
        status, reason = proposed, "grounded_dossier_span"
    return {
        "claim_id": claim_id,
        "claim": claim,
        "status": status,
        "evidence_spans": spans,
        "dossier_hash": dossier_hash,
        "reason": reason,
    }


def _valid_spans(value: Any, dossier_text: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    spans: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(dossier_text):
            continue
        text = dossier_text[start:end]
        if item.get("text") is not None and str(item.get("text")) != text:
            continue
        spans.append({"start": start, "end": end, "text": text})
    return spans


__all__ = ["CLAIM_STATUSES", "frozen_dossier_hash", "ground_frozen_dossier_claims"]
