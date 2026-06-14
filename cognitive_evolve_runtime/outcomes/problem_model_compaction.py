"""Append-only compaction records for M6 problem-model snapshots.

The compacted record keeps the full ``ProblemModelSnapshot`` payload plus the
hashes needed to verify that payload.  It intentionally omits raw ledger/event
history; that history can be archived separately while the compacted record
remains sufficient to round-trip the snapshot and verify model lineage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.outcomes.problem_model import (
    PROBLEM_MODEL_EVOLUTION_VERSION,
    ProblemModelHypothesis,
    ProblemModelSnapshot,
)


PROBLEM_MODEL_COMPACTION_RECORD_VERSION = "problem-model-compaction-record/v1"
PROBLEM_MODEL_COMPACTION_RECORD_TYPE = "problem_model_snapshot_compaction"
PROBLEM_MODEL_COMPACTION_FILE = "problem-model-compactions.jsonl"


class ProblemModelCompactionError(ValueError):
    """Raised when a problem-model compaction record cannot be trusted."""


@dataclass(frozen=True)
class ProblemModelCompactionVerification:
    ok: bool
    record_count: int
    head_record_hash: str = ""
    head_snapshot_hash: str = ""
    known_model_hashes: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProblemModelCompactionRecord:
    """Content-addressed compacted ``ProblemModelSnapshot`` record."""

    snapshot: ProblemModelSnapshot
    parent_record_hash: str = ""
    source_snapshot_hash: str | None = None
    source_model_space_hash: str | None = None
    source_ledger_cursor: int | None = None
    source_ledger_replay_hash: str | None = None
    retained_hashes: dict[str, Any] | None = None
    loss_policy: dict[str, Any] | None = None
    record_hash: str = ""
    record_type: str = PROBLEM_MODEL_COMPACTION_RECORD_TYPE
    version: str = PROBLEM_MODEL_COMPACTION_RECORD_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_record_hash", str(self.parent_record_hash or ""))
        snapshot_hash = self.snapshot.snapshot_hash() if self.source_snapshot_hash is None else str(self.source_snapshot_hash)
        model_space_hash = self.snapshot.model_space_hash() if self.source_model_space_hash is None else str(self.source_model_space_hash)
        ledger_cursor = self.snapshot.ledger_cursor if self.source_ledger_cursor is None else int(self.source_ledger_cursor)
        ledger_replay_hash = self.snapshot.ledger_replay_hash if self.source_ledger_replay_hash is None else str(self.source_ledger_replay_hash)
        object.__setattr__(self, "source_snapshot_hash", snapshot_hash)
        object.__setattr__(self, "source_model_space_hash", model_space_hash)
        object.__setattr__(self, "source_ledger_cursor", int(ledger_cursor))
        object.__setattr__(self, "source_ledger_replay_hash", ledger_replay_hash)
        object.__setattr__(
            self,
            "retained_hashes",
            sufficient_problem_model_hashes(self.snapshot) if self.retained_hashes is None else coerce_dict(self.retained_hashes),
        )
        object.__setattr__(
            self,
            "loss_policy",
            _default_loss_policy() if self.loss_policy is None else coerce_dict(self.loss_policy),
        )
        object.__setattr__(self, "record_type", str(self.record_type or PROBLEM_MODEL_COMPACTION_RECORD_TYPE))
        object.__setattr__(self, "version", str(self.version or PROBLEM_MODEL_COMPACTION_RECORD_VERSION))
        if not self.record_hash:
            object.__setattr__(self, "record_hash", problem_model_compaction_record_hash(self.stable_payload()))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ProblemModelSnapshot,
        *,
        parent_record_hash: str = "",
        retained_hashes: dict[str, Any] | None = None,
        loss_policy: dict[str, Any] | None = None,
    ) -> "ProblemModelCompactionRecord":
        return cls(
            snapshot=snapshot,
            parent_record_hash=parent_record_hash,
            retained_hashes=retained_hashes or sufficient_problem_model_hashes(snapshot),
            loss_policy=loss_policy or _default_loss_policy(),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | "ProblemModelCompactionRecord") -> "ProblemModelCompactionRecord":
        if isinstance(raw, ProblemModelCompactionRecord):
            return raw
        data = coerce_dict(raw)
        snapshot = problem_model_snapshot_from_dict(data.get("snapshot"))
        return cls(
            snapshot=snapshot,
            parent_record_hash=str(data.get("parent_record_hash") or ""),
            source_snapshot_hash=data.get("source_snapshot_hash") if "source_snapshot_hash" in data else None,
            source_model_space_hash=data.get("source_model_space_hash") if "source_model_space_hash" in data else None,
            source_ledger_cursor=data.get("source_ledger_cursor") if "source_ledger_cursor" in data else None,
            source_ledger_replay_hash=data.get("source_ledger_replay_hash") if "source_ledger_replay_hash" in data else None,
            retained_hashes=coerce_dict(data.get("retained_hashes")) if "retained_hashes" in data else None,
            loss_policy=coerce_dict(data.get("loss_policy")) if "loss_policy" in data else None,
            record_hash=str(data.get("record_hash") or ""),
            record_type=str(data.get("record_type") or PROBLEM_MODEL_COMPACTION_RECORD_TYPE),
            version=str(data.get("version") or PROBLEM_MODEL_COMPACTION_RECORD_VERSION),
        )

    def stable_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "record_type": self.record_type,
            "parent_record_hash": self.parent_record_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "source_model_space_hash": self.source_model_space_hash,
            "source_ledger_cursor": int(self.source_ledger_cursor),
            "source_ledger_replay_hash": self.source_ledger_replay_hash,
            "retained_hashes": self.retained_hashes,
            "loss_policy": self.loss_policy,
            "snapshot": problem_model_snapshot_payload(self.snapshot),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.stable_payload()
        payload["record_hash"] = self.record_hash
        return payload


class ProblemModelCompactionStore:
    """JSONL store for content-addressed problem-model compaction records."""

    def __init__(self, path: str | Path) -> None:
        base = Path(path)
        self.path = base if base.suffix == ".jsonl" else base / PROBLEM_MODEL_COMPACTION_FILE

    def append_snapshot(
        self,
        snapshot: ProblemModelSnapshot,
        *,
        parent_record_hash: str | None = None,
    ) -> ProblemModelCompactionRecord:
        records = self.load_records(verify=True) if self.path.exists() else ()
        parent = records[-1].record_hash if parent_record_hash is None and records else str(parent_record_hash or "")
        record = ProblemModelCompactionRecord.from_snapshot(snapshot, parent_record_hash=parent)
        for existing in records:
            if existing.record_hash == record.record_hash:
                return existing
        verify_problem_model_compaction_chain((*records, record), strict=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, default=str, allow_nan=False) + "\n")
        return record

    def load_records(self, *, verify: bool = True) -> tuple[ProblemModelCompactionRecord, ...]:
        if not self.path.exists():
            return ()
        records: list[ProblemModelCompactionRecord] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProblemModelCompactionError(f"invalid_jsonl_record:{line_number}") from exc
            if not isinstance(row, dict):
                raise ProblemModelCompactionError(f"non_object_record:{line_number}")
            if verify:
                _verify_raw_record_hash(row, line_number=line_number)
            records.append(ProblemModelCompactionRecord.from_dict(row))
        if verify:
            verify_problem_model_compaction_chain(tuple(records), strict=True)
        return tuple(records)

    def load_latest_snapshot(self, *, verify: bool = True) -> ProblemModelSnapshot | None:
        records = self.load_records(verify=verify)
        return records[-1].snapshot if records else None

    def verify(self) -> ProblemModelCompactionVerification:
        return verify_problem_model_compaction_chain(self.load_records(verify=True), strict=True)


def problem_model_compaction_record_hash(payload: dict[str, Any]) -> str:
    data = dict(payload)
    data.pop("record_hash", None)
    return "pmc:" + stable_hash(data)


def problem_model_snapshot_payload(snapshot: ProblemModelSnapshot) -> dict[str, Any]:
    payload = snapshot.to_dict()
    payload["snapshot_hash"] = snapshot.snapshot_hash()
    return payload


def problem_model_snapshot_from_dict(raw: Any) -> ProblemModelSnapshot:
    if isinstance(raw, ProblemModelSnapshot):
        return raw
    data = coerce_dict(raw)
    models = tuple(
        model
        for model in (ProblemModelHypothesis.from_dict(item) for item in data.get("active_models", []))
        if model is not None
    )
    return ProblemModelSnapshot(
        active_models=models,
        ledger_cursor=int(data.get("ledger_cursor") or 0),
        active_model_hashes=tuple(_str_list(data.get("active_model_hashes"))),
        promoted_model_hashes=tuple(_str_list(data.get("promoted_model_hashes"))),
        ledger_replay_hash=str(data.get("ledger_replay_hash") or ""),
        update_model_version=str(data.get("update_model_version") or PROBLEM_MODEL_EVOLUTION_VERSION),
        materialized_at_utc=str(data.get("materialized_at_utc") or ""),
        version=str(data.get("version") or "problem-model-snapshot/v1"),
    )


def sufficient_problem_model_hashes(snapshot: ProblemModelSnapshot) -> dict[str, Any]:
    model_hashes = [model.model_hash() for model in snapshot.active_models]
    return {
        "snapshot_hash": snapshot.snapshot_hash(),
        "model_space_hash": snapshot.model_space_hash(),
        "ledger_cursor": int(snapshot.ledger_cursor),
        "ledger_replay_hash": snapshot.ledger_replay_hash,
        "active_model_hashes": list(snapshot.active_model_hashes),
        "promoted_model_hashes": list(snapshot.promoted_model_hashes),
        "model_hashes": model_hashes,
        "model_payload_hashes": {model.model_hash(): stable_hash(model.stable_payload()) for model in snapshot.active_models},
        "model_parent_hashes": {model.model_hash(): list(model.parent_model_hashes) for model in snapshot.active_models},
        "model_evidence_basis_hashes": {model.model_hash(): list(model.evidence_basis_hashes) for model in snapshot.active_models},
    }


def verify_problem_model_compaction_chain(
    records: tuple[ProblemModelCompactionRecord, ...] | list[ProblemModelCompactionRecord],
    *,
    strict: bool = True,
) -> ProblemModelCompactionVerification:
    failures: list[str] = []
    previous_record_hash = ""
    known_model_hashes: set[str] = set()
    typed_records = tuple(ProblemModelCompactionRecord.from_dict(record) for record in records)

    for index, record in enumerate(typed_records):
        expected_record_hash = problem_model_compaction_record_hash(record.stable_payload())
        if record.record_hash != expected_record_hash:
            failures.append(f"record_hash_mismatch:{index + 1}")
        if index == 0:
            if record.parent_record_hash:
                failures.append(f"root_parent_record_hash_not_empty:{index + 1}")
        elif record.parent_record_hash != previous_record_hash:
            failures.append(f"parent_record_hash_mismatch:{index + 1}")

        snapshot = record.snapshot
        snapshot_hash = snapshot.snapshot_hash()
        model_space_hash = snapshot.model_space_hash()
        computed_model_hashes = tuple(model.model_hash() for model in snapshot.active_models)
        if record.source_snapshot_hash != snapshot_hash:
            failures.append(f"source_snapshot_hash_mismatch:{index + 1}")
        if record.source_model_space_hash != model_space_hash:
            failures.append(f"source_model_space_hash_mismatch:{index + 1}")
        if record.source_ledger_cursor != int(snapshot.ledger_cursor):
            failures.append(f"source_ledger_cursor_mismatch:{index + 1}")
        if record.source_ledger_replay_hash != snapshot.ledger_replay_hash:
            failures.append(f"source_ledger_replay_hash_mismatch:{index + 1}")
        if tuple(snapshot.active_model_hashes) != computed_model_hashes:
            failures.append(f"active_model_hashes_mismatch:{index + 1}")

        retained_failures = _retained_hash_failures(record, computed_model_hashes)
        failures.extend(f"{failure}:{index + 1}" for failure in retained_failures)

        current_model_hashes = set(computed_model_hashes)
        for model in snapshot.active_models:
            missing_parents = [
                parent
                for parent in model.parent_model_hashes
                if parent and parent not in current_model_hashes and parent not in known_model_hashes
            ]
            if missing_parents:
                failures.append(f"missing_model_parent:{model.model_hash()}:{','.join(missing_parents)}")

        known_model_hashes.update(current_model_hashes)
        previous_record_hash = record.record_hash

    result = ProblemModelCompactionVerification(
        ok=not failures,
        record_count=len(typed_records),
        head_record_hash=typed_records[-1].record_hash if typed_records else "",
        head_snapshot_hash=typed_records[-1].source_snapshot_hash if typed_records else "",
        known_model_hashes=tuple(sorted(known_model_hashes)),
        failures=tuple(failures),
    )
    if strict and failures:
        raise ProblemModelCompactionError("; ".join(failures))
    return result


def _verify_raw_record_hash(row: dict[str, Any], *, line_number: int) -> None:
    expected = problem_model_compaction_record_hash(row)
    actual = str(row.get("record_hash") or "")
    if actual != expected:
        raise ProblemModelCompactionError(f"record_hash_mismatch:{line_number}")


def _retained_hash_failures(record: ProblemModelCompactionRecord, computed_model_hashes: tuple[str, ...]) -> tuple[str, ...]:
    retained = coerce_dict(record.retained_hashes)
    snapshot = record.snapshot
    failures: list[str] = []
    expected = sufficient_problem_model_hashes(snapshot)
    for key in (
        "snapshot_hash",
        "model_space_hash",
        "ledger_cursor",
        "ledger_replay_hash",
        "active_model_hashes",
        "promoted_model_hashes",
        "model_hashes",
        "model_payload_hashes",
        "model_parent_hashes",
        "model_evidence_basis_hashes",
    ):
        if retained.get(key) != expected[key]:
            failures.append(f"retained_{key}_mismatch")
    if tuple(retained.get("model_hashes") or ()) != computed_model_hashes:
        failures.append("retained_model_hash_order_mismatch")
    if not coerce_dict(record.loss_policy).get("sufficient_hashes_retained"):
        failures.append("sufficient_hashes_not_retained")
    return tuple(dict.fromkeys(failures))


def _default_loss_policy() -> dict[str, Any]:
    return {
        "lossy": True,
        "omitted": ["raw_problem_model_ledger_events"],
        "sufficient_hashes_retained": True,
        "lossless_for": [
            "problem_model_snapshot_payload",
            "snapshot_hash",
            "model_space_hash",
            "active_model_hashes",
            "promoted_model_hashes",
            "model_payload_hashes",
            "model_parent_hashes",
            "model_evidence_basis_hashes",
            "ledger_cursor",
            "ledger_replay_hash",
        ],
    }


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


__all__ = [
    "PROBLEM_MODEL_COMPACTION_FILE",
    "PROBLEM_MODEL_COMPACTION_RECORD_TYPE",
    "PROBLEM_MODEL_COMPACTION_RECORD_VERSION",
    "ProblemModelCompactionError",
    "ProblemModelCompactionRecord",
    "ProblemModelCompactionStore",
    "ProblemModelCompactionVerification",
    "problem_model_compaction_record_hash",
    "problem_model_snapshot_from_dict",
    "problem_model_snapshot_payload",
    "sufficient_problem_model_hashes",
    "verify_problem_model_compaction_chain",
]
