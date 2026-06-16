"""Append-only campaign trace ledger for concept effects."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.nexus._serde import json_ready, stable_hash, utc_now


@dataclass(frozen=True)
class TraceEntry:
    round: int
    concept_id: str
    consumed_refs: list[str] = field(default_factory=list)
    produced_effects: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    decision_changed: bool = False
    replay_hash: str = ""
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceLedger:
    def __init__(self, path: str | Path | None = None, *, entries: list[dict[str, Any]] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: list[dict[str, Any]] = [dict(item) for item in entries or [] if isinstance(item, dict)]

    def record(
        self,
        *,
        round_index: int,
        concept_id: str,
        consumed_refs: list[str] | None = None,
        produced_effects: dict[str, Any] | None = None,
        cost: dict[str, Any] | None = None,
        decision_changed: bool = False,
        replay_hash: str = "",
    ) -> dict[str, Any]:
        payload = json_ready(produced_effects or {})
        entry = TraceEntry(
            round=int(round_index or 0),
            concept_id=str(concept_id or ""),
            consumed_refs=[str(item) for item in consumed_refs or [] if str(item or "").strip()],
            produced_effects=payload if isinstance(payload, dict) else {"value": payload},
            cost=dict(cost or {}),
            decision_changed=bool(decision_changed),
            replay_hash=str(replay_hash or "trace-" + stable_hash(payload)[:16]),
        ).to_dict()
        self.entries.append(entry)
        self._append(entry)
        return entry

    def record_violation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.record(
            round_index=int(payload.get("round") or 0),
            concept_id=str(payload.get("concept_id") or "guard"),
            produced_effects={"guard_violation": payload},
            decision_changed=False,
            replay_hash="guard-violation",
        )

    def to_jsonl(self) -> str:
        return "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) + "\n" for item in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": list(self.entries), "entry_count": len(self.entries)}

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]] | None, *, path: str | Path | None = None) -> "TraceLedger":
        return cls(path=path, entries=entries)

    def _append(self, entry: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")


__all__ = ["TraceEntry", "TraceLedger"]
