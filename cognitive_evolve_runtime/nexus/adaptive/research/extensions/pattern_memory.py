"""Reusable pattern memory extension."""
from __future__ import annotations

import hashlib
from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import ArchiveDirective

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import SearchPressure, evidence_state
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class PatternMemoryExtension:
    extension_id = "pattern_memory"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.patterns: dict[str, dict[str, Any]] = {}

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        learned = 0
        quarantined = 0
        directives: list[ArchiveDirective] = []
        for candidate in ctx.candidates:
            state = evidence_state(candidate)
            resolved = state.get("resolved_challenge_ids") or []
            if resolved:
                pattern = _pattern_from_candidate(candidate)
                current = dict(self.patterns.get(pattern["id"]) or pattern)
                current["success_count"] = int(current.get("success_count") or 0) + len(resolved)
                current["source_candidate_ids"] = list(dict.fromkeys([*(current.get("source_candidate_ids") or []), candidate.id]))
                current["resolved_challenge_ids"] = list(dict.fromkeys([*(current.get("resolved_challenge_ids") or []), *resolved]))
                current["weight"] = bounded_score(0.2 + 0.1 * current["success_count"] - 0.15 * int(current.get("failure_count") or 0))
                self.patterns[current["id"]] = current
                directives.append(ArchiveDirective(kind="add_descriptor", descriptor=("pattern", current["id"]), payload={"weight": current["weight"], "source_candidate_ids": current["source_candidate_ids"], "resolved_challenge_ids": current["resolved_challenge_ids"], "descriptor_token": current.get("token", "")}))
                learned += 1
            elif state.get("terminal_reject"):
                pattern = _pattern_from_candidate(candidate)
                current = dict(self.patterns.get(pattern["id"]) or pattern)
                current["failure_count"] = int(current.get("failure_count") or 0) + 1
                current["weight"] = bounded_score(float(current.get("weight") or 0.0) - 0.2)
                current["quarantined"] = current["failure_count"] >= 2
                self.patterns[current["id"]] = current
                quarantined += 1 if current["quarantined"] else 0
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, archive_directives=directives, metrics={"pattern_count": len(self.patterns), "pattern_learned_count": learned, "pattern_quarantine_count": quarantined})

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        advisory = {}
        good = [p for p in self.patterns.values() if p.get("weight", 0.0) > 0.2 and not p.get("quarantined")]
        for candidate in ctx.candidates:
            text = _candidate_text(candidate).lower()
            hits = sum(1 for pattern in good if str(pattern.get("token") or "") in text)
            if hits:
                advisory[candidate.id] = {"plan_value": bounded_score(0.1 * hits), "rank_prior": bounded_score(0.05 * hits), "diversity": 0.0, "risk": 0.0}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory, metrics={"pattern_advisory_hits": len(advisory)})

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        useful = sorted((p for p in self.patterns.values() if p.get("weight", 0.0) > 0.2 and not p.get("quarantined")), key=lambda p: float(p.get("weight") or 0.0), reverse=True)[:3]
        if not useful or ctx.parent is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        instruction = "Reuse only evidence-backed patterns if they help targeted challenges: " + "; ".join(str(p.get("guidance") or p.get("token")) for p in useful)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction=instruction, metadata={"source_extension": self.extension_id, "pattern_ids": [p["id"] for p in useful]})
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure], metrics={"pattern_pressure_count": 1})

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"patterns": dict(self.patterns)} if self.patterns else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.patterns = {str(k): dict(v) for k, v in ((state or {}).get("patterns") or {}).items() if isinstance(v, dict)}


def _candidate_text(candidate: Any) -> str:
    return " ".join(str(getattr(candidate, key, "") or "") for key in ("artifact_type", "concise_claim", "core_mechanism", "artifact"))[:4000]


def _pattern_from_candidate(candidate: Any) -> dict[str, Any]:
    text = _candidate_text(candidate).lower()
    token = next((part for part in text.replace("_", " ").split() if len(part) > 5), str(getattr(candidate, "artifact_type", "pattern")))
    pid = "pattern-" + hashlib.sha256((str(getattr(candidate, "artifact_type", "")) + token).encode("utf-8")).hexdigest()[:12]
    return {"id": pid, "kind": "artifact_text", "token": token, "guidance": f"preserve useful pattern token `{token}` when it is evidence-backed", "source_candidate_ids": [], "resolved_challenge_ids": [], "success_count": 0, "failure_count": 0, "weight": 0.0}


__all__ = ["PatternMemoryExtension"]
