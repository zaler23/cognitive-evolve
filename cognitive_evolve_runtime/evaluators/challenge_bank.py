"""Domain-neutral challenge memory for progressive evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import ChallengeCase, EvidenceResult
from cognitive_evolve_runtime.nexus._serde import utc_now


@dataclass
class ChallengeBank:
    version: str = "challenge-bank/v1"
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def ingest(self, result: EvidenceResult, *, round_index: int = 0, candidate_fate: str = "") -> list[ChallengeCase]:
        stored: list[ChallengeCase] = []
        for case in result.challenge_cases:
            case_id = case.id or challenge_id(case.domain_id, case.kind, case.payload or {"summary": case.summary})
            current = self.cases.get(case_id)
            if current:
                merged = ChallengeCase.from_dict(current) or case
                payload = merged.to_dict()
                payload["last_seen_round"] = int(round_index or case.last_seen_round or payload.get("last_seen_round") or 0)
                payload["kill_count"] = int(payload.get("kill_count") or 0) + 1
                if str(candidate_fate).lower() == "elite":
                    payload["elite_kill_count"] = int(payload.get("elite_kill_count") or 0) + 1
                if result.score >= 0.65:
                    payload["frontier_kill_count"] = int(payload.get("frontier_kill_count") or 0) + 1
                self.cases[case_id] = payload
            else:
                payload = case.to_dict()
                payload["id"] = case_id
                payload["first_seen_round"] = int(round_index or case.first_seen_round or 0)
                payload["last_seen_round"] = int(round_index or case.last_seen_round or 0)
                payload["kill_count"] = max(1, int(payload.get("kill_count") or 1))
                if str(candidate_fate).lower() == "elite":
                    payload["elite_kill_count"] = max(1, int(payload.get("elite_kill_count") or 0))
                if result.score >= 0.65:
                    payload["frontier_kill_count"] = max(1, int(payload.get("frontier_kill_count") or 0))
                self.cases[case_id] = payload
            restored = ChallengeCase.from_dict(self.cases[case_id])
            if restored is not None:
                stored.append(restored)
        if stored:
            self.updated_at = utc_now()
        return stored

    def mark_resolved(self, candidate_id: str, challenge_ids: list[str]) -> None:
        for case_id in challenge_ids:
            payload = self.cases.get(case_id)
            if not payload:
                continue
            resolved = list(payload.get("resolved_by_candidate_ids") or [])
            if candidate_id not in resolved:
                resolved.append(candidate_id)
            payload["resolved_by_candidate_ids"] = resolved
        if challenge_ids:
            self.updated_at = utc_now()

    def summary(self, *, limit: int = 12) -> dict[str, Any]:
        cases = sorted(self.cases.values(), key=lambda item: (float(item.get("priority") or 0.0), int(item.get("kill_count") or 0)), reverse=True)[: max(0, int(limit or 0))]
        return {
            "version": self.version,
            "case_count": len(self.cases),
            "top_cases": [
                {
                    "id": str(item.get("id") or ""),
                    "kind": str(item.get("kind") or ""),
                    "summary": str(item.get("summary") or "")[:240],
                    "kill_count": int(item.get("kill_count") or 0),
                    "priority": float(item.get("priority") or 0.0),
                }
                for item in cases
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChallengeBank":
        if not isinstance(data, dict):
            return cls()
        return cls(version=str(data.get("version") or "challenge-bank/v1"), cases={str(k): dict(v) for k, v in dict(data.get("cases") or {}).items() if isinstance(v, dict)}, updated_at=str(data.get("updated_at") or utc_now()))


def challenge_id(domain_id: str, kind: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"domain_id": domain_id, "kind": kind, "payload": payload}, ensure_ascii=False, sort_keys=True, default=str)
    return "case-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def challenge_from_diagnostic(*, candidate_id: str, domain_id: str, diagnostic: str, kind: str = "evaluator_failure", round_index: int = 0) -> ChallengeCase:
    summary = str(diagnostic or kind).strip()[:240] or kind
    payload = {"diagnostic": summary, "candidate_id": str(candidate_id or "")}
    return ChallengeCase(
        id=challenge_id(domain_id, kind, payload),
        domain_id=domain_id,
        kind=kind,
        payload=payload,
        summary=summary,
        first_seen_round=int(round_index or 0),
        last_seen_round=int(round_index or 0),
        priority=0.7 if kind in {"counterexample", "regression", "boundary"} else 0.5,
    )


__all__ = ["ChallengeBank", "challenge_from_diagnostic", "challenge_id"]
