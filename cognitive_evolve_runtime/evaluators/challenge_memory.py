"""Domain-neutral challenge memory for the Evidence Control Plane."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure
from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now
from cognitive_evolve_runtime.core.scalars import bounded_score


SCHEMA_CHALLENGE_CATEGORIES = {"artifact_type_mismatch", "missing_required_field", "machine_parse_failure", "field_alias"}


@dataclass
class ChallengeMemoryItem:
    id: str
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    priority: float = 0.5
    confidence: float = 0.5
    cost_to_retest: dict[str, Any] = field(default_factory=dict)
    emitted_by_candidate_ids: list[str] = field(default_factory=list)
    targeted_by_candidate_ids: list[str] = field(default_factory=list)
    resolved_by_candidate_ids: list[str] = field(default_factory=list)
    affected_lineages: list[str] = field(default_factory=list)
    affected_regions: list[str] = field(default_factory=list)
    first_seen_round: int = 0
    last_seen_round: int = 0
    kill_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["priority"] = bounded_score(self.priority)
        payload["confidence"] = bounded_score(self.confidence)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChallengeMemoryItem | None":
        if not isinstance(data, dict):
            return None
        case_id = str(data.get("id") or "")
        if not case_id:
            return None
        emitted = _str_list(data.get("emitted_by_candidate_ids"))
        legacy_candidate = coerce_dict(data.get("payload")).get("candidate_id")
        if legacy_candidate and legacy_candidate not in emitted:
            emitted.append(str(legacy_candidate))
        return cls(
            id=case_id,
            payload=coerce_dict(data.get("payload")),
            summary=str(data.get("summary") or data.get("kind") or case_id)[:240],
            priority=bounded_score(data.get("priority", 0.5)),
            confidence=bounded_score(data.get("confidence", 0.5)),
            cost_to_retest=coerce_dict(data.get("cost_to_retest")),
            emitted_by_candidate_ids=emitted,
            targeted_by_candidate_ids=_str_list(data.get("targeted_by_candidate_ids")),
            resolved_by_candidate_ids=_str_list(data.get("resolved_by_candidate_ids")),
            affected_lineages=_str_list(data.get("affected_lineages") or data.get("lineage_ids")),
            affected_regions=_str_list(data.get("affected_regions") or data.get("region_ids")),
            first_seen_round=_int(data.get("first_seen_round"), 0),
            last_seen_round=_int(data.get("last_seen_round"), 0),
            kill_count=max(1, _int(data.get("kill_count"), 1)),
            metadata=coerce_dict(data.get("metadata")),
        )


@dataclass
class ChallengeMemory:
    version: str = "challenge-memory/v1"
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def ingest(self, record: EvidenceRecord, *, round_index: int = 0, candidate_fate: str = "", lineage_id: str = "", region_id: str = "") -> list[ChallengeMemoryItem]:
        stored: list[ChallengeMemoryItem] = []
        for raw in _challenge_items_from_record(record):
            item = ChallengeMemoryItem.from_dict(raw)
            if item is None:
                continue
            current = ChallengeMemoryItem.from_dict(self.items.get(item.id))
            if current is None:
                current = item
                current.first_seen_round = int(round_index or item.first_seen_round or 0)
                current.kill_count = max(1, current.kill_count)
            else:
                current.kill_count += 1
            current.last_seen_round = int(round_index or item.last_seen_round or current.last_seen_round or 0)
            current.priority = max(current.priority, item.priority)
            current.confidence = max(current.confidence, item.confidence)
            current.emitted_by_candidate_ids = _append_unique(current.emitted_by_candidate_ids, record.candidate_id)
            if lineage_id:
                current.affected_lineages = _append_unique(current.affected_lineages, lineage_id)
            if region_id:
                current.affected_regions = _append_unique(current.affected_regions, region_id)
            if str(candidate_fate).lower() == "elite":
                current.priority = max(current.priority, 0.8)
            self.items[item.id] = current.to_dict()
            stored.append(current)
        if stored:
            self.updated_at = utc_now()
        return stored

    def mark_targeted(self, candidate_id: str, challenge_ids: list[str]) -> None:
        changed = False
        for challenge_id_ in challenge_ids:
            item = ChallengeMemoryItem.from_dict(self.items.get(challenge_id_))
            if item is None:
                continue
            item.targeted_by_candidate_ids = _append_unique(item.targeted_by_candidate_ids, candidate_id)
            self.items[item.id] = item.to_dict()
            changed = True
        if changed:
            self.updated_at = utc_now()

    def mark_resolved(self, candidate_id: str, challenge_ids: list[str]) -> None:
        changed = False
        for challenge_id_ in challenge_ids:
            item = ChallengeMemoryItem.from_dict(self.items.get(challenge_id_))
            if item is None:
                continue
            item.resolved_by_candidate_ids = _append_unique(item.resolved_by_candidate_ids, candidate_id)
            self.items[item.id] = item.to_dict()
            changed = True
        if changed:
            self.updated_at = utc_now()

    def mark_schema_resolved_from_record(self, record: EvidenceRecord) -> list[str]:
        """Conservatively resolve targeted schema challenges no longer present."""

        target_ids = list(record.target_challenge_ids or [])
        if not target_ids:
            return []
        current_categories = {classify_diagnostic(diagnostic) for diagnostic in record.diagnostics}
        metrics = coerce_dict(record.metadata.get("metrics"))
        artifact_state = coerce_dict(record.metadata.get("artifact_state"))
        schema_clean = _float(metrics.get("schema_cleanliness"), 0.0) >= 1.0 or _float(artifact_state.get("schema_cleanliness"), 0.0) >= 1.0
        evaluator_passed = str(record.metadata.get("status") or "").strip().lower() == "passed" or bool(metrics.get("correctness") is True)
        if not (schema_clean or evaluator_passed):
            return []
        resolved: list[str] = []
        for challenge_id_ in target_ids:
            item = ChallengeMemoryItem.from_dict(self.items.get(challenge_id_))
            if item is None:
                continue
            category = str(item.metadata.get("category") or classify_diagnostic(item.summary))
            if category not in SCHEMA_CHALLENGE_CATEGORIES:
                continue
            if category in current_categories:
                continue
            item.resolved_by_candidate_ids = _append_unique(item.resolved_by_candidate_ids, record.candidate_id)
            self.items[item.id] = item.to_dict()
            resolved.append(item.id)
        if resolved:
            self.updated_at = utc_now()
        return resolved

    def compile_search_pressure(self, *, parent_id: str | None = None, scope: str = "global", limit: int = 3, artifact_requirements: dict[str, Any] | None = None) -> SearchPressure | None:
        unresolved = [item for item in (ChallengeMemoryItem.from_dict(raw) for raw in self.items.values()) if item is not None and not item.resolved_by_candidate_ids]
        if not unresolved:
            return None
        ranked = sorted(unresolved, key=_pressure_score, reverse=True)[: max(1, int(limit or 1))]
        target_ids = [item.id for item in ranked]
        success = [{"kind": "resolve_challenge", "challenge_id": item.id, "summary": item.summary} for item in ranked]
        schema_focus = any(str(item.metadata.get("category") or classify_diagnostic(item.summary)) in SCHEMA_CHALLENGE_CATEGORIES for item in ranked)
        return SearchPressure.from_parts(
            parent_id=parent_id,
            scope=scope,
            target_challenge_ids=target_ids,
            artifact_requirements=artifact_requirements or {},
            success_criteria=success,
            challenge_weights={item.id: _pressure_score(item) for item in ranked},
            selection_advisory={},
            mutation_instruction=_mutation_instruction(ranked, artifact_requirements or {}, schema_focus=schema_focus),
            metadata={"challenge_count": len(self.items), "selected_count": len(target_ids), "schema_repair_focus": schema_focus},
        )

    def targeted_resolution_rate(self) -> float:
        targeted = [item for item in (ChallengeMemoryItem.from_dict(raw) for raw in self.items.values()) if item is not None and item.targeted_by_candidate_ids]
        if not targeted:
            return 0.0
        resolved = [item for item in targeted if item.resolved_by_candidate_ids]
        return bounded_score(len(resolved) / max(1, len(targeted)))

    def summary(self, *, limit: int = 12) -> dict[str, Any]:
        ranked = sorted((item for item in (ChallengeMemoryItem.from_dict(raw) for raw in self.items.values()) if item is not None), key=_pressure_score, reverse=True)[: max(0, int(limit or 0))]
        return {
            "version": self.version,
            "case_count": len(self.items),
            "targeted_count": len([item for item in (ChallengeMemoryItem.from_dict(raw) for raw in self.items.values()) if item is not None and item.targeted_by_candidate_ids]),
            "resolved_count": len([item for item in (ChallengeMemoryItem.from_dict(raw) for raw in self.items.values()) if item is not None and item.resolved_by_candidate_ids]),
            "targeted_resolution_rate": self.targeted_resolution_rate(),
            "top_cases": [
                {
                    "id": item.id,
                    "summary": item.summary[:240],
                    "priority": bounded_score(item.priority),
                    "confidence": bounded_score(item.confidence),
                    "kill_count": int(item.kill_count or 0),
                    "targeted_count": len(item.targeted_by_candidate_ids),
                    "resolved_count": len(item.resolved_by_candidate_ids),
                }
                for item in ranked
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChallengeMemory":
        if not isinstance(data, dict):
            return cls()
        raw_items = data.get("items") if isinstance(data.get("items"), dict) else data.get("cases")
        items = {}
        for key, raw in coerce_dict(raw_items).items():
            item = ChallengeMemoryItem.from_dict(raw)
            if item is not None:
                items[item.id or str(key)] = item.to_dict()
        return cls(version="challenge-memory/v1", items=items, updated_at=str(data.get("updated_at") or utc_now()))


def challenge_id(*, source: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"source": source, "payload": payload}, ensure_ascii=False, sort_keys=True, default=str)
    return "case-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def challenge_from_diagnostic(*, candidate_id: str, source: str, diagnostic: str, round_index: int = 0, priority: float = 0.5) -> dict[str, Any]:
    summary = str(diagnostic or source or "challenge").strip()[:240] or "challenge"
    payload = {"diagnostic": summary, "candidate_id": str(candidate_id or "")}
    case_id = challenge_id(source=source, payload=payload)
    category = classify_diagnostic(summary)
    return ChallengeMemoryItem(
        id=case_id,
        payload=payload,
        summary=summary,
        priority=priority,
        confidence=0.5,
        first_seen_round=int(round_index or 0),
        last_seen_round=int(round_index or 0),
        emitted_by_candidate_ids=[str(candidate_id or "")] if candidate_id else [],
        metadata={"source": source, "category": category},
    ).to_dict()


def _challenge_items_from_record(record: EvidenceRecord) -> list[dict[str, Any]]:
    raw = record.metadata.get("challenge_items")
    if isinstance(raw, list):
        items = [dict(item) for item in raw if isinstance(item, dict)]
        if items:
            return items
    items: list[dict[str, Any]] = []
    for diagnostic in record.diagnostics[:8] or record.hints[:3]:
        items.append(challenge_from_diagnostic(candidate_id=record.candidate_id, source=record.source, diagnostic=diagnostic, priority=0.7 if record.score >= 0.65 else 0.5))
    return items


def _pressure_score(item: ChallengeMemoryItem) -> float:
    cost = _cost_value(item.cost_to_retest)
    reuse = 0.05 * len(item.targeted_by_candidate_ids)
    return bounded_score((item.priority * max(0.1, item.confidence)) / max(1.0, cost) + min(0.2, reuse) + min(0.2, 0.03 * item.kill_count))


def classify_diagnostic(diagnostic: str) -> str:
    text = str(diagnostic or "").strip().lower()
    if "artifact_type_alias_normalized" in text or "artifact_type_mismatch" in text or "artifact_type must be" in text or "artifact type" in text:
        return "artifact_type_mismatch"
    if "missing_required_fields" in text or "missing required" in text or "missing required cache policy sections" in text:
        return "missing_required_field"
    if "machine_parse_failure" in text or "not_machine_parseable" in text or "must be a json object" in text or "artifact must be a json object" in text:
        return "machine_parse_failure"
    if "field_alias_normalized" in text or "forbidden alias" in text or "eviction_scoring" in text or "state_update" in text:
        return "field_alias"
    return "generic"


def _mutation_instruction(items: list[ChallengeMemoryItem], artifact_requirements: dict[str, Any], *, schema_focus: bool) -> str:
    summaries = "; ".join(item.summary for item in items if item.summary)[:600]
    if not schema_focus:
        return "Generate a child candidate that directly addresses these unresolved challenges: " + summaries
    required_type = str(artifact_requirements.get("artifact_type") or "").strip()
    required_fields = _str_list(artifact_requirements.get("required_fields"))
    artifact_type_aliases = coerce_dict(artifact_requirements.get("artifact_type_aliases"))
    field_aliases = coerce_dict(artifact_requirements.get("field_aliases"))
    forbidden_type_aliases = ", ".join(str(key) for key in artifact_type_aliases.keys() if str(key).strip())
    forbidden_field_aliases = ", ".join(str(key) for key in field_aliases.keys() if str(key).strip())
    parts = [
        "Schema repair has priority over semantic innovation for this child candidate.",
        "Emit a clean machine-readable artifact, not a string-wrapped JSON object.",
    ]
    if required_type:
        parts.append(f"Use exactly artifact_type={required_type}.")
    if required_fields:
        parts.append("Include exactly these required top-level fields: " + ", ".join(required_fields) + ".")
    if forbidden_type_aliases:
        parts.append("Do not use artifact_type aliases: " + forbidden_type_aliases + ".")
    if forbidden_field_aliases:
        parts.append("Do not use field aliases: " + forbidden_field_aliases + ".")
    parts.append("Resolve these targeted schema challenges: " + summaries)
    return " ".join(parts)


def _cost_value(cost: dict[str, Any]) -> float:
    for key in ("seconds", "tool_seconds", "tokens", "estimated"):
        try:
            value = float(cost.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 1.0


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_unique(items: list[str], value: str) -> list[str]:
    out = list(items or [])
    if value and value not in out:
        out.append(value)
    return out


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ChallengeMemory", "ChallengeMemoryItem", "challenge_from_diagnostic", "challenge_id", "classify_diagnostic"]
