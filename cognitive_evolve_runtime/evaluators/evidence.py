"""Evidence Control Plane primitives for Nexus evolution.

This module is the single public boundary for candidate evidence.  It keeps the
runtime domain-neutral: evaluators may inspect patches, prompts, workflows,
mathematical objects, or machine artifacts, but Nexus consumes only normalized
control signals for artifact policy, search value, challenge pressure, and final
projection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now


@dataclass(frozen=True)
class ArtifactPolicy:
    """Runtime boundary for whether an artifact may be probed or finalized."""

    machine_readable_required: bool = False
    allow_text_fallback: bool = True
    allow_refold_for_probe: bool = True
    allow_refold_for_final: bool = False
    final_requires_certificate: bool = False
    projection_required: bool = True
    artifact_type: str = ""
    artifact_type_aliases: dict[str, str] = field(default_factory=dict)
    field_aliases: dict[str, str] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    final_requires_clean_schema: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ArtifactPolicy":
        cfg = coerce_dict(data)
        evidence = coerce_dict(cfg.get("evidence"))
        merged = {**cfg, **evidence}
        if "machine_artifact_required" in merged and "machine_readable_required" not in merged:
            merged["machine_readable_required"] = merged.get("machine_artifact_required")
        metadata = {k: v for k, v in coerce_dict(merged.get("metadata")).items() if not _sensitive_key(k)}
        for key in (
            "domain_vocabulary",
            "allowed_terms",
            "allowed_domain_terms",
            "forbidden_semantic_terms",
            "semantic_drift_terms",
        ):
            if key in merged and key not in metadata and not _sensitive_key(key):
                metadata[key] = merged[key]
        return cls(
            machine_readable_required=_bool(merged.get("machine_readable_required"), default=False),
            allow_text_fallback=_bool(merged.get("allow_text_fallback"), default=True),
            allow_refold_for_probe=_bool(merged.get("allow_refold_for_probe"), default=True),
            allow_refold_for_final=_bool(merged.get("allow_refold_for_final"), default=False),
            final_requires_certificate=_bool(merged.get("final_requires_certificate"), default=False),
            projection_required=_bool(merged.get("projection_required"), default=True),
            artifact_type=str(merged.get("artifact_type") or ""),
            artifact_type_aliases={str(k): str(v) for k, v in coerce_dict(merged.get("artifact_type_aliases")).items() if str(k or "").strip() and str(v or "").strip()},
            field_aliases={str(k): str(v) for k, v in coerce_dict(merged.get("field_aliases")).items() if str(k or "").strip() and str(v or "").strip()},
            required_fields=_str_list(merged.get("required_fields")),
            final_requires_clean_schema=_bool(merged.get("final_requires_clean_schema"), default=True),
            metadata=metadata,
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """Canonical candidate evidence record written by evaluators."""

    candidate_id: str
    source: str = "progressive_evaluator"
    stage: str = "probe"
    score: float = 0.0
    confidence: float = 0.5
    cost: dict[str, Any] = field(default_factory=dict)
    final_blocked: bool = True
    parent_blocked: bool = False
    terminal_reject: bool = False
    repair_value: float = 0.0
    continuation_value: float = 0.0
    novelty_value: float = 0.0
    target_challenge_ids: list[str] = field(default_factory=list)
    resolved_challenge_ids: list[str] = field(default_factory=list)
    emitted_challenge_ids: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = bounded_score(self.score)
        payload["confidence"] = bounded_score(self.confidence)
        payload["repair_value"] = bounded_score(self.repair_value)
        payload["continuation_value"] = bounded_score(self.continuation_value)
        payload["novelty_value"] = bounded_score(self.novelty_value)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvidenceRecord | None":
        if not isinstance(data, dict):
            return None
        candidate_id = str(data.get("candidate_id") or "")
        if not candidate_id:
            return None
        return cls(
            candidate_id=candidate_id,
            source=str(data.get("source") or "progressive_evaluator"),
            stage=str(data.get("stage") or data.get("level") or "probe"),
            score=bounded_score(data.get("score", 0.0)),
            confidence=bounded_score(data.get("confidence", 0.5)),
            cost=coerce_dict(data.get("cost")),
            final_blocked=bool(data.get("final_blocked", not bool(data.get("final_eligible")))),
            parent_blocked=bool(data.get("parent_blocked", data.get("hard_reject", False))),
            terminal_reject=bool(data.get("terminal_reject", data.get("hard_reject", False))),
            repair_value=bounded_score(data.get("repair_value", 0.0)),
            continuation_value=bounded_score(data.get("continuation_value", data.get("repair_value", 0.0))),
            novelty_value=bounded_score(data.get("novelty_value", 0.0)),
            target_challenge_ids=_str_list(data.get("target_challenge_ids")),
            resolved_challenge_ids=_str_list(data.get("resolved_challenge_ids")),
            emitted_challenge_ids=_str_list(data.get("emitted_challenge_ids")),
            diagnostics=_str_list(data.get("diagnostics")),
            hints=_str_list(data.get("hints") or data.get("repair_hints")),
            metadata=_safe_metadata(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_now()),
        )


@dataclass(frozen=True)
class SearchPressure:
    """Compiled search pressure consumed by mutation planning and selection."""

    id: str
    parent_id: str | None = None
    scope: str = "global"
    target_challenge_ids: list[str] = field(default_factory=list)
    avoid_challenge_ids: list[str] = field(default_factory=list)
    preserve_refs: list[dict[str, Any]] = field(default_factory=list)
    mutable_refs: list[dict[str, Any]] = field(default_factory=list)
    artifact_requirements: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    challenge_weights: dict[str, float] = field(default_factory=dict)
    selection_advisory: dict[str, float] = field(default_factory=dict)
    mutation_instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_parts(
        cls,
        *,
        parent_id: str | None = None,
        scope: str = "global",
        target_challenge_ids: list[str] | None = None,
        avoid_challenge_ids: list[str] | None = None,
        artifact_requirements: dict[str, Any] | None = None,
        success_criteria: list[dict[str, Any]] | None = None,
        challenge_weights: dict[str, float] | None = None,
        selection_advisory: dict[str, float] | None = None,
        mutation_instruction: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "SearchPressure":
        targets = list(dict.fromkeys(_str_list(target_challenge_ids)))
        avoids = list(dict.fromkeys(_str_list(avoid_challenge_ids)))
        pressure_id = _stable_id({"parent_id": parent_id or "", "scope": scope, "targets": targets, "avoids": avoids, "instruction": mutation_instruction, "metadata": coerce_dict(metadata)})
        return cls(
            id=pressure_id,
            parent_id=parent_id,
            scope=str(scope or "global"),
            target_challenge_ids=targets,
            avoid_challenge_ids=avoids,
            artifact_requirements=coerce_dict(artifact_requirements),
            success_criteria=[dict(item) for item in success_criteria or [] if isinstance(item, dict)],
            challenge_weights={str(k): bounded_score(v) for k, v in coerce_dict(challenge_weights).items()},
            selection_advisory={str(k): bounded_score(v) for k, v in coerce_dict(selection_advisory).items()},
            mutation_instruction=str(mutation_instruction or ""),
            metadata=_safe_metadata(metadata),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SearchPressure":
        payload = coerce_dict(data)
        legacy_advisory = coerce_dict(payload.get("selection_advisory"))
        challenge_weights = coerce_dict(payload.get("challenge_weights"))
        if not challenge_weights and legacy_advisory and _looks_like_challenge_map(legacy_advisory):
            challenge_weights = legacy_advisory
            legacy_advisory = {}
            metadata = coerce_dict(payload.get("metadata"))
            metadata["legacy_selection_advisory_migrated_to_challenge_weights"] = True
            payload["metadata"] = metadata
        return cls(
            id=str(payload.get("id") or _stable_id(payload)),
            parent_id=str(payload.get("parent_id") or "") or None,
            scope=str(payload.get("scope") or "global"),
            target_challenge_ids=_str_list(payload.get("target_challenge_ids")),
            avoid_challenge_ids=_str_list(payload.get("avoid_challenge_ids")),
            preserve_refs=[dict(item) for item in payload.get("preserve_refs", []) if isinstance(item, dict)],
            mutable_refs=[dict(item) for item in payload.get("mutable_refs", []) if isinstance(item, dict)],
            artifact_requirements=coerce_dict(payload.get("artifact_requirements")),
            success_criteria=[dict(item) for item in payload.get("success_criteria", []) if isinstance(item, dict)],
            challenge_weights={str(k): bounded_score(v) for k, v in challenge_weights.items()},
            selection_advisory={str(k): bounded_score(v) for k, v in legacy_advisory.items()},
            mutation_instruction=str(payload.get("mutation_instruction") or ""),
            metadata=_safe_metadata(payload.get("metadata")),
        )


def apply_evidence_record(candidate: Any, record: EvidenceRecord) -> None:
    """Write canonical evidence state to a CandidateGenome-like object."""

    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    migrated = _legacy_records_from_metadata(metadata, fallback_candidate_id=getattr(candidate, "id", ""))
    existing = [item for item in (EvidenceRecord.from_dict(raw) for raw in metadata.get("evidence_records", [])) if item is not None]
    records = [*migrated, *existing, record]
    records = _dedupe_records(records)[-20:]
    metadata.pop("progressive_evidence", None)
    metadata.pop("challenge_failures", None)
    metadata["evidence_records"] = [item.to_dict() for item in records]
    metadata["evidence_state"] = evidence_state_from_records(records)
    metadata["target_challenge_ids"] = list(metadata["evidence_state"].get("target_challenge_ids", []))
    metadata["resolved_challenge_ids"] = list(metadata["evidence_state"].get("resolved_challenge_ids", []))
    metadata["terminal_failure"] = bool(metadata["evidence_state"].get("terminal_reject"))
    metadata["repair_value"] = bounded_score(metadata["evidence_state"].get("repair_value", 0.0))
    candidate.metadata = metadata

    scores = coerce_dict(getattr(candidate, "multihead_scores", {}))
    state = metadata["evidence_state"]
    scores["frontier_score"] = bounded_score(state.get("search_score", 0.0))
    scores["repair_value"] = bounded_score(state.get("repair_value", 0.0))
    scores["continuation_value"] = bounded_score(state.get("continuation_value", 0.0))
    scores["final_confidence"] = bounded_score(state.get("final_score", 0.0))
    scores["challenge_resolution"] = bounded_score(state.get("challenge_resolution", 0.0))
    artifact_state = coerce_dict(record.metadata.get("artifact_state"))
    scores["schema_cleanliness"] = bounded_score(artifact_state.get("schema_cleanliness", scores.get("schema_cleanliness", 0.0)))
    scores["evaluator_score"] = bounded_score(record.metadata.get("evaluator_score", record.score))
    scores["challenge_pass_rate"] = bounded_score(record.metadata.get("challenge_pass_rate", scores.get("challenge_pass_rate", 1.0 if not record.final_blocked else 0.0)))
    scores["final_verification"] = 0.0 if state.get("final_blocked") else bounded_score(state.get("final_score", 0.0))
    candidate.multihead_scores = scores


def evidence_records(candidate: Any) -> list[EvidenceRecord]:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    records = [item for item in (EvidenceRecord.from_dict(raw) for raw in metadata.get("evidence_records", [])) if item is not None]
    if records:
        return records
    return _legacy_records_from_metadata(metadata, fallback_candidate_id=getattr(candidate, "id", ""))


def latest_evidence_record(candidate: Any) -> EvidenceRecord | None:
    records = evidence_records(candidate)
    return records[-1] if records else None


def evidence_state(candidate: Any) -> dict[str, Any]:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    state = coerce_dict(metadata.get("evidence_state"))
    if state:
        return state
    return evidence_state_from_records(evidence_records(candidate))


def evidence_state_from_records(records: list[EvidenceRecord]) -> dict[str, Any]:
    from cognitive_evolve_runtime.evaluators.evidence_authority import aggregate_evidence_state

    return aggregate_evidence_state(records)


def evidence_terminal_reject(candidate: Any) -> bool:
    return bool(evidence_state(candidate).get("terminal_reject"))


def evidence_final_blocked(candidate: Any) -> bool:
    return bool(evidence_state(candidate).get("final_blocked"))


def evidence_parent_blocked(candidate: Any) -> bool:
    return bool(evidence_state(candidate).get("parent_blocked"))


def evidence_repair_value(candidate: Any) -> float:
    return bounded_score(evidence_state(candidate).get("repair_value", 0.0))


def evidence_search_score(candidate: Any) -> float:
    return bounded_score(evidence_state(candidate).get("search_score", 0.0))


def has_repair_value(candidate: Any) -> bool:
    return evidence_repair_value(candidate) > 0.0


def evidence_advisory_features(candidates: list[Any]) -> dict[str, Any]:
    """Build ParentSelector-compatible advisory objects from canonical evidence state."""

    out: dict[str, Any] = {}
    for candidate in candidates:
        state = evidence_state(candidate)
        if not state:
            continue
        records = evidence_records(candidate)
        latest = records[-1] if records else None
        category_counts = _record_category_counts(latest)
        resolved = len(state.get("resolved_challenge_ids") or [])
        targets = len(state.get("target_challenge_ids") or [])
        unresolved_penalty = max(0.0, targets - resolved) / max(1, targets) if targets else 0.0
        semantic_or_behavior_pressure = category_counts.get("semantic_drift", 0) + category_counts.get("behavior_score_failure", 0)
        schema_pressure = (
            category_counts.get("contract_artifact_policy_conflict", 0)
            + category_counts.get("artifact_type_mismatch", 0)
            + category_counts.get("missing_required_field", 0)
            + category_counts.get("machine_parse_failure", 0)
            + category_counts.get("field_alias", 0)
        )
        resolved_bonus = 0.15 * resolved
        clean_schema_bonus = 0.1 if bounded_score(getattr(candidate, "multihead_scores", {}).get("schema_cleanliness", 0.0) if isinstance(getattr(candidate, "multihead_scores", None), dict) else 0.0) >= 1.0 else 0.0
        risk = 1.0 if state.get("terminal_reject") else (0.5 if state.get("parent_blocked") else 0.25 * unresolved_penalty)
        risk = bounded_score(risk + min(0.35, 0.10 * schema_pressure + 0.08 * semantic_or_behavior_pressure))
        out[getattr(candidate, "id", "")] = {
            "rank_prior": bounded_score(state.get("search_score", 0.0)),
            "plan_value": bounded_score(float(state.get("repair_value", 0.0) or 0.0) + resolved_bonus + clean_schema_bonus),
            "diversity": bounded_score(getattr(candidate, "multihead_scores", {}).get("novelty", 0.0) if isinstance(getattr(candidate, "multihead_scores", None), dict) else 0.0),
            "risk": risk,
        }
    return out


def repair_value_from_record(record: EvidenceRecord) -> float:
    if record.terminal_reject:
        return 0.0
    if not record.final_blocked:
        return 0.1
    challenge_count = len(record.emitted_challenge_ids)
    hint_count = len(record.hints)
    return bounded_score(max(record.repair_value, min(1.0, 0.2 + 0.12 * challenge_count + 0.08 * hint_count), record.score * 0.6))


def _legacy_records_from_metadata(metadata: dict[str, Any], *, fallback_candidate_id: str = "") -> list[EvidenceRecord]:
    """Private one-way reader for older snapshots; callers must not write legacy keys."""

    raw = coerce_dict(metadata.get("progressive_evidence"))
    if not raw:
        return []
    challenge_items = [coerce_dict(item) for item in raw.get("challenge_cases", []) if isinstance(item, dict)]
    emitted = [str(item.get("id")) for item in challenge_items if item.get("id")]
    artifact_state = coerce_dict(raw.get("artifact_view"))
    record = EvidenceRecord(
        candidate_id=str(raw.get("candidate_id") or fallback_candidate_id),
        source="legacy_progressive_evidence_migration",
        stage=str(raw.get("stage") or raw.get("level") or "probe"),
        score=bounded_score(raw.get("score", 0.0)),
        confidence=0.5,
        final_blocked=not bool(raw.get("final_eligible")) or not bool(raw.get("passed")),
        parent_blocked=bool(raw.get("hard_reject")),
        terminal_reject=bool(raw.get("hard_reject")),
        repair_value=bounded_score(metadata.get("repair_value", 0.0)),
        continuation_value=bounded_score(metadata.get("repair_value", 0.0)),
        target_challenge_ids=_str_list(metadata.get("target_challenge_ids")),
        resolved_challenge_ids=_str_list(raw.get("resolved_challenge_ids") or metadata.get("resolved_challenge_ids")),
        emitted_challenge_ids=emitted,
        diagnostics=_str_list(raw.get("diagnostics")),
        hints=_str_list(raw.get("repair_hints")),
        metadata={
            "artifact_state": artifact_state,
            "challenge_items": challenge_items,
            "legacy_status": str(raw.get("status") or ""),
            "metrics": coerce_dict(raw.get("metrics")),
        },
        created_at=str(raw.get("created_at") or utc_now()),
    )
    return [record]


def _looks_like_challenge_map(value: dict[str, Any]) -> bool:
    if not value:
        return False
    keys = [str(key) for key in value.keys()]
    return sum(1 for key in keys if key.startswith("case-") or key.startswith("challenge-")) >= max(1, len(keys) // 2)


def _record_category_counts(record: EvidenceRecord | None) -> dict[str, int]:
    if record is None:
        return {}
    try:
        from cognitive_evolve_runtime.evaluators.challenge_memory import classify_diagnostic
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for diagnostic in record.diagnostics:
        category = classify_diagnostic(diagnostic)
        counts[category] = counts.get(category, 0) + 1
    challenge_items = record.metadata.get("challenge_items", [])
    if not isinstance(challenge_items, list):
        challenge_items = []
    for item in challenge_items:
        if isinstance(item, dict):
            category = str(coerce_dict(item.get("metadata")).get("category") or classify_diagnostic(str(item.get("summary") or "")))
            counts[category] = counts.get(category, 0) + 1
    return counts

def _dedupe_records(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    out: list[EvidenceRecord] = []
    seen: set[str] = set()
    for record in records:
        key = _stable_id({"candidate_id": record.candidate_id, "source": record.source, "stage": record.stage, "created_at": record.created_at, "diagnostics": record.diagnostics})
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return "pressure-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_metadata(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in coerce_dict(value).items():
        if _sensitive_key(key):
            continue
        out[str(key)] = item
    return out


def _sensitive_key(key: Any) -> bool:
    return any(token in str(key).lower() for token in ("key", "secret", "token", "password", "prompt"))


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "required"}


__all__ = [
    "ArtifactPolicy",
    "EvidenceRecord",
    "SearchPressure",
    "apply_evidence_record",
    "evidence_advisory_features",
    "evidence_final_blocked",
    "evidence_parent_blocked",
    "evidence_records",
    "evidence_repair_value",
    "evidence_search_score",
    "evidence_state",
    "evidence_state_from_records",
    "evidence_terminal_reject",
    "has_repair_value",
    "latest_evidence_record",
    "repair_value_from_record",
]
