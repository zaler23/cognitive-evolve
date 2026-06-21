"""Durable append-only ledger for M5.1 latent-objective evidence.

The event log is the source of truth.  Snapshots are caches that can be rebuilt
by replaying evidence events from the initial latent problem state.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.durable.file_lock import atomic_write_json
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.outcomes.latent import PreferenceEvidence
from cognitive_evolve_runtime.persistence.event_store import EventStore


EVIDENCE_ADDED = "evidence_added"
EVIDENCE_DEDUPLICATED = "evidence_deduplicated"
EVIDENCE_REJECTED = "evidence_rejected"
EVIDENCE_RETRACTED = "evidence_retracted"
EVIDENCE_SUPERSEDED = "evidence_superseded"
POSTERIOR_UPDATED = "posterior_updated"
POSTERIOR_SNAPSHOT_MATERIALIZED = "posterior_snapshot_materialized"
DECISION_BOUND_TO_POSTERIOR = "decision_bound_to_posterior"

LEDGER_EVENT_TYPES = {
    EVIDENCE_ADDED,
    EVIDENCE_DEDUPLICATED,
    EVIDENCE_REJECTED,
    EVIDENCE_RETRACTED,
    EVIDENCE_SUPERSEDED,
    POSTERIOR_UPDATED,
    POSTERIOR_SNAPSHOT_MATERIALIZED,
    DECISION_BOUND_TO_POSTERIOR,
}


@dataclass(frozen=True)
class LatentLedgerEvent:
    sequence: int
    event_type: str
    event_id: str = ""
    evidence_id: str = ""
    idempotency_key: str = ""
    target_evidence_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    created_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        event_type = str(self.event_type or "")
        if event_type not in LEDGER_EVENT_TYPES:
            raise ValueError(f"unknown latent ledger event type: {event_type}")
        object.__setattr__(self, "sequence", max(1, int(self.sequence or 1)))
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "payload", coerce_dict(self.payload))
        if not self.event_id:
            object.__setattr__(self, "event_id", latent_event_id(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentLedgerEvent":
        return cls(
            sequence=int(data.get("sequence") or 1),
            event_type=str(data.get("event_type") or ""),
            event_id=str(data.get("event_id") or ""),
            evidence_id=str(data.get("evidence_id") or ""),
            idempotency_key=str(data.get("idempotency_key") or ""),
            target_evidence_id=str(data.get("target_evidence_id") or ""),
            payload=coerce_dict(data.get("payload")),
            reason=str(data.get("reason") or ""),
            created_at_utc=str(data.get("created_at_utc") or utc_now()),
        )

    def replay_payload(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("created_at_utc", None)
        data.pop("event_id", None)
        return data


@dataclass(frozen=True)
class LatentLedgerReplay:
    cursor: int
    active_evidence: tuple[PreferenceEvidence, ...] = ()
    active_evidence_ids: tuple[str, ...] = ()
    active_idempotency_keys: tuple[str, ...] = ()
    rejected_events: tuple[dict[str, Any], ...] = ()
    retracted_evidence_ids: tuple[str, ...] = ()
    superseded_evidence_ids: tuple[str, ...] = ()
    ledger_replay_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "active_evidence": [item.to_dict() for item in self.active_evidence],
            "active_evidence_ids": list(self.active_evidence_ids),
            "active_idempotency_keys": list(self.active_idempotency_keys),
            "rejected_events": [dict(item) for item in self.rejected_events],
            "retracted_evidence_ids": list(self.retracted_evidence_ids),
            "superseded_evidence_ids": list(self.superseded_evidence_ids),
            "ledger_replay_hash": self.ledger_replay_hash,
        }


@dataclass
class LatentLedger:
    ledger_id: str = "latent-ledger:v1"
    events: list[LatentLedgerEvent] = field(default_factory=list)
    created_at_utc: str = field(default_factory=utc_now)
    version: str = "latent-ledger/v1"

    @property
    def cursor(self) -> int:
        return max((event.sequence for event in self.events), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ledger_id": self.ledger_id,
            "created_at_utc": self.created_at_utc,
            "events": [event.to_dict() for event in sorted(self.events, key=lambda item: item.sequence)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "LatentLedger" | None) -> "LatentLedger":
        if isinstance(data, LatentLedger):
            return data
        raw = coerce_dict(data)
        return cls(
            ledger_id=str(raw.get("ledger_id") or "latent-ledger:v1"),
            events=[
                LatentLedgerEvent.from_dict(item)
                for item in raw.get("events", [])
                if isinstance(item, dict) and item.get("event_type")
            ],
            created_at_utc=str(raw.get("created_at_utc") or utc_now()),
            version=str(raw.get("version") or "latent-ledger/v1"),
        )

    def append_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        evidence_id: str = "",
        idempotency_key: str = "",
        target_evidence_id: str = "",
        reason: str = "",
    ) -> LatentLedgerEvent:
        event = LatentLedgerEvent(
            sequence=self.cursor + 1,
            event_type=event_type,
            evidence_id=str(evidence_id or ""),
            idempotency_key=str(idempotency_key or ""),
            target_evidence_id=str(target_evidence_id or ""),
            payload=coerce_dict(payload),
            reason=str(reason or ""),
        )
        self.events.append(event)
        return event

    def add_evidence(self, evidence: PreferenceEvidence | dict[str, Any], *, idempotency_key: str | None = None) -> LatentLedgerEvent:
        item = preference_evidence_from_any(evidence)
        if item is None:
            return self.reject_evidence(evidence, reason="malformed_preference_evidence")
        evidence_id = preference_evidence_id(item)
        key = str(idempotency_key or item.evidence_ref or evidence_id)
        replay = self.replay()
        if key in replay.active_idempotency_keys or evidence_id in replay.active_evidence_ids:
            return self.append_event(
                EVIDENCE_DEDUPLICATED,
                evidence_id=evidence_id,
                idempotency_key=key,
                payload={"evidence": item.to_dict(), "duplicate_of": evidence_id},
                reason="duplicate_idempotency_key_or_evidence_id",
            )
        return self.append_event(
            EVIDENCE_ADDED,
            evidence_id=evidence_id,
            idempotency_key=key,
            payload={"evidence": item.to_dict()},
        )

    def reject_evidence(self, raw: Any, *, reason: str, source_type: str = "", provenance_ref: str = "") -> LatentLedgerEvent:
        payload = {
            "raw_hash": stable_hash(raw),
            "raw": _small_raw(raw),
            "source_type": str(source_type or ""),
            "provenance_ref": str(provenance_ref or ""),
        }
        return self.append_event(EVIDENCE_REJECTED, payload=payload, reason=reason)

    def retract_evidence(self, evidence_id_or_key: str, *, reason: str = "evidence_retracted") -> LatentLedgerEvent:
        target = self._resolve_evidence_id(evidence_id_or_key)
        return self.append_event(
            EVIDENCE_RETRACTED,
            target_evidence_id=target or str(evidence_id_or_key or ""),
            reason=reason,
        )

    def supersede_evidence(
        self,
        target_evidence_id_or_key: str,
        evidence: PreferenceEvidence | dict[str, Any],
        *,
        idempotency_key: str | None = None,
        reason: str = "evidence_superseded",
    ) -> LatentLedgerEvent:
        item = preference_evidence_from_any(evidence)
        if item is None:
            return self.reject_evidence(evidence, reason="malformed_superseding_preference_evidence")
        evidence_id = preference_evidence_id(item)
        key = str(idempotency_key or item.evidence_ref or evidence_id)
        return self.append_event(
            EVIDENCE_SUPERSEDED,
            evidence_id=evidence_id,
            idempotency_key=key,
            target_evidence_id=self._resolve_evidence_id(target_evidence_id_or_key) or str(target_evidence_id_or_key or ""),
            payload={"evidence": item.to_dict()},
            reason=reason,
        )

    def record_posterior_updated(self, snapshot: Any) -> LatentLedgerEvent:
        payload = _snapshot_payload(snapshot)
        return self.append_event(
            POSTERIOR_UPDATED,
            payload=payload,
            reason="posterior_recomputed_from_active_evidence",
        )

    def record_posterior_snapshot_materialized(self, snapshot: Any) -> LatentLedgerEvent:
        payload = _snapshot_payload(snapshot)
        return self.append_event(
            POSTERIOR_SNAPSHOT_MATERIALIZED,
            payload=payload,
            reason="posterior_snapshot_cached",
        )

    def record_decision_bound(self, *, decision_type: str, snapshot: Any, decision_payload: dict[str, Any] | None = None) -> LatentLedgerEvent:
        payload = {
            "decision_type": str(decision_type or "latent_decision"),
            "snapshot": _snapshot_payload(snapshot),
            "decision": coerce_dict(decision_payload),
        }
        return self.append_event(
            DECISION_BOUND_TO_POSTERIOR,
            payload=payload,
            reason="latent_informed_decision_bound_to_pinned_posterior",
        )

    def replay(self, *, cursor: int | None = None) -> LatentLedgerReplay:
        active: dict[str, PreferenceEvidence] = {}
        key_by_id: dict[str, str] = {}
        id_by_key: dict[str, str] = {}
        rejected: list[dict[str, Any]] = []
        retracted: list[str] = []
        superseded: list[str] = []
        selected_events = [event for event in sorted(self.events, key=lambda item: item.sequence) if cursor is None or event.sequence <= cursor]
        for event in selected_events:
            if event.event_type == EVIDENCE_ADDED:
                evidence = preference_evidence_from_any(event.payload.get("evidence"))
                if evidence is None:
                    rejected.append({"event_id": event.event_id, "reason": "malformed_added_evidence"})
                    continue
                evidence_id = event.evidence_id or preference_evidence_id(evidence)
                key = event.idempotency_key or evidence.evidence_ref or evidence_id
                if evidence_id in active or key in id_by_key:
                    continue
                active[evidence_id] = evidence
                key_by_id[evidence_id] = key
                id_by_key[key] = evidence_id
            elif event.event_type == EVIDENCE_RETRACTED:
                target = self._resolve_evidence_id(event.target_evidence_id, active=active, id_by_key=id_by_key) or event.target_evidence_id
                if target in active:
                    key = key_by_id.pop(target, "")
                    if key:
                        id_by_key.pop(key, None)
                    active.pop(target, None)
                if target:
                    retracted.append(target)
            elif event.event_type == EVIDENCE_SUPERSEDED:
                target = self._resolve_evidence_id(event.target_evidence_id, active=active, id_by_key=id_by_key) or event.target_evidence_id
                if target in active:
                    key = key_by_id.pop(target, "")
                    if key:
                        id_by_key.pop(key, None)
                    active.pop(target, None)
                if target:
                    superseded.append(target)
                evidence = preference_evidence_from_any(event.payload.get("evidence"))
                if evidence is None:
                    rejected.append({"event_id": event.event_id, "reason": "malformed_superseding_evidence"})
                    continue
                evidence_id = event.evidence_id or preference_evidence_id(evidence)
                key = event.idempotency_key or evidence.evidence_ref or evidence_id
                if evidence_id not in active and key not in id_by_key:
                    active[evidence_id] = evidence
                    key_by_id[evidence_id] = key
                    id_by_key[key] = evidence_id
            elif event.event_type == EVIDENCE_REJECTED:
                rejected.append({"event_id": event.event_id, "sequence": event.sequence, "reason": event.reason, **event.payload})
        evidence_ids = tuple(active.keys())
        replay_hash = stable_hash([event.replay_payload() for event in selected_events])
        return LatentLedgerReplay(
            cursor=max((event.sequence for event in selected_events), default=0),
            active_evidence=tuple(active[evidence_id] for evidence_id in evidence_ids),
            active_evidence_ids=evidence_ids,
            active_idempotency_keys=tuple(key_by_id[evidence_id] for evidence_id in evidence_ids),
            rejected_events=tuple(rejected),
            retracted_evidence_ids=tuple(dict.fromkeys(retracted)),
            superseded_evidence_ids=tuple(dict.fromkeys(superseded)),
            ledger_replay_hash=replay_hash,
        )

    def compaction_snapshot(self, *, cursor: int | None = None, rejected_tail: int = 20) -> dict[str, Any]:
        """Return a deterministic compacted replay anchor for long-lived ledgers.

        This does not mutate the append-only event log.  It materializes the
        active evidence set at a pinned cursor so a cache or archived segment can
        be restored without treating old event frequency as desirability.
        """

        replay = self.replay(cursor=cursor)
        rejected = list(replay.rejected_events)[-max(0, int(rejected_tail or 0)) :]
        payload = {
            "version": "latent-ledger-compaction/v1",
            "source_ledger_id": self.ledger_id,
            "source_cursor": int(replay.cursor),
            "source_replay_hash": replay.ledger_replay_hash,
            "active_evidence": [item.to_dict() for item in replay.active_evidence],
            "active_evidence_ids": list(replay.active_evidence_ids),
            "active_idempotency_keys": list(replay.active_idempotency_keys),
            "rejected_tail": rejected,
            "retracted_evidence_ids": list(replay.retracted_evidence_ids),
            "superseded_evidence_ids": list(replay.superseded_evidence_ids),
        }
        payload["compaction_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_compaction_snapshot(cls, snapshot: dict[str, Any] | None) -> "LatentLedger":
        """Rebuild a compact ledger from ``compaction_snapshot`` output."""

        data = coerce_dict(snapshot)
        ledger = cls(ledger_id=str(data.get("source_ledger_id") or "latent-ledger:v1"))
        evidence = [
            preference_evidence_from_any(item)
            for item in data.get("active_evidence", [])
            if isinstance(item, dict)
        ]
        keys = [str(item) for item in data.get("active_idempotency_keys", [])]
        for index, item in enumerate(evidence):
            if item is None:
                continue
            key = keys[index] if index < len(keys) and keys[index] else item.evidence_ref or preference_evidence_id(item)
            ledger.add_evidence(item, idempotency_key=key)
        return ledger

    def _resolve_evidence_id(
        self,
        evidence_id_or_key: str,
        *,
        active: dict[str, PreferenceEvidence] | None = None,
        id_by_key: dict[str, str] | None = None,
    ) -> str:
        value = str(evidence_id_or_key or "")
        if not value:
            return ""
        if active is not None and value in active:
            return value
        if id_by_key is not None and value in id_by_key:
            return id_by_key[value]
        replay = self.replay() if active is None or id_by_key is None else None
        if replay is not None:
            if value in replay.active_evidence_ids:
                return value
            for key, evidence_id in zip(replay.active_idempotency_keys, replay.active_evidence_ids, strict=False):
                if value == key:
                    return evidence_id
        return value


class LatentLedgerStore:
    """File-level JSONL persistence for latent ledger events and snapshots."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.event_store = EventStore(self.root / "latent-events.jsonl")
        self.snapshot_path = self.root / "latent-posterior-snapshot.json"
        self.ledger_cache_path = self.root / "latent-ledger.json"
        self.compaction_path = self.root / "latent-ledger-compaction.json"

    def persist_ledger(self, ledger: LatentLedger) -> dict[str, Any]:
        events = [
            {"type": "latent_ledger_event", "event_id": event.event_id, "sequence": event.sequence, "event": event.to_dict()}
            for event in sorted(ledger.events, key=lambda item: item.sequence)
        ]
        appended = self.event_store.append_many_once(events, identity_keys=("type", "event_id"))
        ledger_payload = ledger.to_dict()
        atomic_write_json(self.ledger_cache_path, ledger_payload, sort_keys=True)
        try:
            read_back = json.loads(self.ledger_cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"latent ledger sidecar is not readable JSON: {self.ledger_cache_path}") from exc
        if not isinstance(read_back, dict):
            raise ValueError(f"latent ledger sidecar must be a JSON object: {self.ledger_cache_path}")
        ledger_hash = stable_hash(read_back)
        cursor = max((int(event.sequence or 0) for event in ledger.events), default=0)
        return {
            "sidecar_schema": "latent-ledger-sidecar/v1",
            "path": str(self.ledger_cache_path),
            "latent_events_path": str(self.event_store.path),
            "latent_ledger_cache_path": str(self.ledger_cache_path),
            "sha256": ledger_hash,
            "ledger_hash": ledger_hash,
            "cursor": cursor,
            "ledger_cursor": cursor,
            "events_total": len(ledger.events),
            "events_appended": len(appended),
        }

    def load_ledger(self) -> LatentLedger:
        events: list[LatentLedgerEvent] = []
        for row in self.event_store.replay():
            data = coerce_dict(row)
            event_data = coerce_dict(data.get("event"))
            if not event_data:
                continue
            try:
                events.append(LatentLedgerEvent.from_dict(event_data))
            except Exception:
                continue
        if not events and self.ledger_cache_path.exists():
            try:
                data = json.loads(self.ledger_cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            return LatentLedger.from_dict(data if isinstance(data, dict) else {})
        return LatentLedger(events=events)

    def persist_snapshot(self, snapshot: Any) -> dict[str, Any]:
        payload = _snapshot_payload(snapshot)
        atomic_write_json(self.snapshot_path, payload, sort_keys=True)
        self.event_store.append_once(
            {
                "type": "latent_posterior_snapshot",
                "snapshot_hash": payload.get("snapshot_hash") or stable_hash(payload),
                "ledger_cursor": payload.get("ledger_cursor"),
                "snapshot": payload,
            },
            identity_keys=("type", "snapshot_hash", "ledger_cursor"),
        )
        return {"latent_snapshot_path": str(self.snapshot_path), "snapshot_hash": payload.get("snapshot_hash") or stable_hash(payload)}

    def persist_compaction_snapshot(self, ledger: LatentLedger, *, cursor: int | None = None) -> dict[str, Any]:
        payload = ledger.compaction_snapshot(cursor=cursor)
        atomic_write_json(self.compaction_path, payload, sort_keys=True)
        self.event_store.append_once(
            {
                "type": "latent_ledger_compaction_snapshot",
                "source_cursor": payload.get("source_cursor"),
                "compaction_hash": payload.get("compaction_hash"),
                "snapshot": payload,
            },
            identity_keys=("type", "source_cursor", "compaction_hash"),
        )
        return {
            "latent_compaction_path": str(self.compaction_path),
            "source_cursor": int(payload.get("source_cursor") or 0),
            "compaction_hash": str(payload.get("compaction_hash") or ""),
        }

    def load_compaction_snapshot(self) -> dict[str, Any]:
        if not self.compaction_path.exists():
            return {}
        try:
            data = json.loads(self.compaction_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return coerce_dict(data)


def preference_evidence_id(evidence: PreferenceEvidence | dict[str, Any]) -> str:
    item = preference_evidence_from_any(evidence)
    payload = item.to_dict() if item is not None else coerce_dict(evidence)
    return "pe:" + stable_hash(payload)


def preference_evidence_from_any(raw: PreferenceEvidence | dict[str, Any] | None) -> PreferenceEvidence | None:
    if isinstance(raw, PreferenceEvidence):
        return raw
    data = coerce_dict(raw)
    intent_id = str(data.get("intent_id") or "").strip()
    if not intent_id:
        return None
    try:
        return PreferenceEvidence(
            intent_id=intent_id,
            support=_nonnegative_float(data.get("support"), default=0.0),
            contradiction=_nonnegative_float(data.get("contradiction"), default=0.0),
            weight=_nonnegative_float(data.get("weight"), default=1.0),
            evidence_ref=str(data.get("evidence_ref") or ""),
            source_type=str(data.get("source_type") or "unknown"),
            provenance_ref=str(data.get("provenance_ref") or ""),
            confidence=max(0.0, min(1.0, _nonnegative_float(data.get("confidence"), default=1.0))),
            calibration=str(data.get("calibration") or "uncalibrated"),
            metadata=coerce_dict(data.get("metadata")),
        )
    except (TypeError, ValueError):
        return None


def latent_event_id(event: LatentLedgerEvent) -> str:
    return "le:" + stable_hash(event.replay_payload())


def _nonnegative_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return max(0.0, parsed)


def _small_raw(raw: Any) -> Any:
    if isinstance(raw, PreferenceEvidence):
        return raw.to_dict()
    if isinstance(raw, dict):
        return {str(key): value for key, value in list(raw.items())[:20]}
    text = str(raw)
    return text[:1000]


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    if hasattr(snapshot, "to_dict"):
        data = snapshot.to_dict()
    else:
        data = coerce_dict(snapshot)
    if hasattr(snapshot, "snapshot_hash"):
        data.setdefault("snapshot_hash", snapshot.snapshot_hash())
    elif data:
        data.setdefault("snapshot_hash", stable_hash(data))
    return data


__all__ = [
    "DECISION_BOUND_TO_POSTERIOR",
    "EVIDENCE_ADDED",
    "EVIDENCE_DEDUPLICATED",
    "EVIDENCE_REJECTED",
    "EVIDENCE_RETRACTED",
    "EVIDENCE_SUPERSEDED",
    "LEDGER_EVENT_TYPES",
    "POSTERIOR_SNAPSHOT_MATERIALIZED",
    "POSTERIOR_UPDATED",
    "LatentLedger",
    "LatentLedgerEvent",
    "LatentLedgerReplay",
    "LatentLedgerStore",
    "latent_event_id",
    "preference_evidence_from_any",
    "preference_evidence_id",
]
