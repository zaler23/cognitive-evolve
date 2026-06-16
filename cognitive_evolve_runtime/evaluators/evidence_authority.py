"""Central EvidenceRecord authority aggregation helpers.

The Evidence Control Plane stores extensible authority hints in record.metadata,
but callers must not scatter raw metadata lookups across the runtime.  This
module is the only place that interprets authority, artifact hash transfer, and
final revocation semantics.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus._serde import coerce_dict

EVIDENCE_AUTHORITY_ORDER: dict[str, int] = {
    "probe": 10,
    "verifier": 20,
    "frontier": 30,
    "final": 40,
    "certificate": 50,
    "human_review": 60,
}
HIGH_FINAL_AUTHORITIES = {"final", "certificate", "human_review"}


def evidence_authority(record: Any) -> str:
    metadata = coerce_dict(getattr(record, "metadata", {}))
    raw = str(metadata.get("authority") or getattr(record, "stage", "") or "probe").strip().lower()
    aliases = {
        "l4": "final",
        "certification": "certificate",
        "certified": "certificate",
        "external_evaluator": "verifier",
        "progressive_evaluator": "probe",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in EVIDENCE_AUTHORITY_ORDER else "probe"


def evidence_authority_rank(record: Any) -> int:
    return EVIDENCE_AUTHORITY_ORDER[evidence_authority(record)]


def evidence_artifact_hash(record: Any) -> str:
    metadata = coerce_dict(getattr(record, "metadata", {}))
    for key in ("artifact_identity_hash", "artifact_hash"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    artifact_state = coerce_dict(metadata.get("artifact_state"))
    artifact_policy = coerce_dict(metadata.get("artifact_policy"))
    for key in ("artifact_identity_hash", "artifact_hash"):
        value = str(artifact_state.get(key) or "").strip()
        if value:
            return value
    artifact = artifact_state.get("normalized_artifact")
    if artifact is not None:
        return stable_artifact_identity_hash(artifact_state, artifact_policy=artifact_policy)
    for key in ("normalized_artifact_hash",):
        value = str(metadata.get(key) or artifact_state.get(key) or "").strip()
        if value:
            return value
    return ""


def evidence_revokes_final(record: Any) -> bool:
    metadata = coerce_dict(getattr(record, "metadata", {}))
    return bool(metadata.get("revokes_final"))


def aggregate_evidence_state(records: list[Any]) -> dict[str, Any]:
    if not records:
        return {
            "search_score": 0.0,
            "final_score": 0.0,
            "final_blocked": False,
            "parent_blocked": False,
            "terminal_reject": False,
            "repair_value": 0.0,
            "continuation_value": 0.0,
            "target_challenge_ids": [],
            "resolved_challenge_ids": [],
            "emitted_challenge_ids": [],
            "challenge_resolution": 0.0,
            "confidence": 0.0,
            "authority": "probe",
            "artifact_hash": "",
        }
    ordered = list(records)
    latest = ordered[-1]
    targets = _unique(item for record in ordered for item in getattr(record, "target_challenge_ids", []) or [])
    resolved = _unique(item for record in ordered for item in getattr(record, "resolved_challenge_ids", []) or [])
    emitted = _unique(item for record in ordered for item in getattr(record, "emitted_challenge_ids", []) or [])
    latest_hash = evidence_artifact_hash(latest)
    latest_rank = evidence_authority_rank(latest)
    terminal_reject = any(bool(getattr(record, "terminal_reject", False)) for record in ordered)
    for record in ordered:
        if evidence_revokes_final(record) and evidence_authority_rank(record) >= EVIDENCE_AUTHORITY_ORDER["final"]:
            terminal_reject = bool(getattr(record, "terminal_reject", terminal_reject))
    final_records = []
    blockers = []
    for record in ordered:
        authority = evidence_authority(record)
        rank = evidence_authority_rank(record)
        record_hash = evidence_artifact_hash(record)
        same_artifact = not latest_hash or not record_hash or record_hash == latest_hash
        if not same_artifact and authority in HIGH_FINAL_AUTHORITIES:
            continue
        if evidence_revokes_final(record) and rank >= EVIDENCE_AUTHORITY_ORDER["final"]:
            blockers.append(record)
            continue
        if not bool(getattr(record, "final_blocked", True)) and authority in HIGH_FINAL_AUTHORITIES and same_artifact:
            final_records.append(record)
        elif bool(getattr(record, "final_blocked", False)) and rank >= EVIDENCE_AUTHORITY_ORDER["final"] and same_artifact:
            blockers.append(record)
    final_blocked = bool(getattr(latest, "final_blocked", True))
    if final_records and not blockers:
        final_blocked = False
    if blockers:
        final_blocked = True
    if latest_rank < EVIDENCE_AUTHORITY_ORDER["final"] and final_records and not blockers:
        # A fresh probe cannot downgrade an existing same-artifact certificate.
        final_blocked = False
    challenge_resolution = len(set(resolved) & set(targets)) / max(1, len(set(targets))) if targets else bounded_score(len(resolved) / max(1, len(emitted)))
    top_authority = max(ordered, key=evidence_authority_rank)
    return {
        "search_score": max(bounded_score(getattr(record, "score", 0.0)) for record in ordered),
        "final_score": max([bounded_score(getattr(record, "score", 0.0) * getattr(record, "confidence", 0.0)) for record in final_records] or [0.0]),
        "final_blocked": bool(final_blocked),
        "parent_blocked": bool(getattr(latest, "parent_blocked", False)),
        "terminal_reject": bool(terminal_reject),
        "repair_value": max(bounded_score(getattr(record, "repair_value", 0.0)) for record in ordered),
        "continuation_value": max(bounded_score(getattr(record, "continuation_value", 0.0)) for record in ordered),
        "target_challenge_ids": targets,
        "resolved_challenge_ids": resolved,
        "emitted_challenge_ids": emitted,
        "challenge_resolution": bounded_score(challenge_resolution),
        "confidence": max(bounded_score(getattr(record, "confidence", 0.0)) for record in ordered),
        "stage": str(getattr(latest, "stage", "probe") or "probe"),
        "source": str(getattr(latest, "source", "") or ""),
        "authority": evidence_authority(top_authority),
        "artifact_hash": latest_hash,
    }


def stable_artifact_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return "artifact:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def artifact_identity_payload(artifact_state: dict[str, Any], *, artifact_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    state = coerce_dict(artifact_state)
    policy = coerce_dict(artifact_policy)
    return {
        "artifact_type": str(state.get("artifact_type") or policy.get("artifact_type") or ""),
        "normalized_artifact": state.get("normalized_artifact"),
        "schema_status": str(state.get("status") or ""),
        "final_eligible": bool(state.get("final_eligible", False)),
        "missing_required_fields": sorted(str(item) for item in state.get("missing_required_fields", []) if str(item or "").strip()) if isinstance(state.get("missing_required_fields"), list) else [],
        "policy": {
            "artifact_type": str(policy.get("artifact_type") or ""),
            "required_fields": sorted(str(item) for item in policy.get("required_fields", []) if str(item or "").strip()) if isinstance(policy.get("required_fields"), list) else [],
            "artifact_type_aliases": dict(sorted((str(k), str(v)) for k, v in coerce_dict(policy.get("artifact_type_aliases")).items())),
            "field_aliases": dict(sorted((str(k), str(v)) for k, v in coerce_dict(policy.get("field_aliases")).items())),
            "machine_readable_required": bool(policy.get("machine_readable_required") or policy.get("machine_artifact_required")),
            "allow_refold_for_final": bool(policy.get("allow_refold_for_final")),
            "final_requires_clean_schema": bool(policy.get("final_requires_clean_schema", True)),
            "schema_version": str(policy.get("schema_version") or policy.get("version") or ""),
        },
    }


def stable_artifact_identity_hash(artifact_state: dict[str, Any], *, artifact_policy: dict[str, Any] | None = None) -> str:
    return stable_artifact_hash(artifact_identity_payload(artifact_state, artifact_policy=artifact_policy))


def _unique(items: Any) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "EVIDENCE_AUTHORITY_ORDER",
    "aggregate_evidence_state",
    "artifact_identity_payload",
    "evidence_artifact_hash",
    "evidence_authority",
    "evidence_authority_rank",
    "evidence_revokes_final",
    "stable_artifact_hash",
    "stable_artifact_identity_hash",
]
