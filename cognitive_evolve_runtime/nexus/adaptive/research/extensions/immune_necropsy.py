"""Antibody and necropsy extension for repeated terminal failures."""
from __future__ import annotations

import hashlib
from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import VerificationObligation

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import SearchPressure, evidence_records, evidence_state
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ImmuneNecropsyExtension:
    extension_id = "immune_necropsy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.rules: dict[str, dict[str, Any]] = {}
        self.necropsies: list[dict[str, Any]] = []

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        created = 0
        obligations: list[VerificationObligation] = []
        for candidate in ctx.candidates:
            state = evidence_state(candidate)
            if not state.get("terminal_reject"):
                continue
            latest = evidence_records(candidate)[-1] if evidence_records(candidate) else None
            signature = _failure_signature(latest.diagnostics if latest else [])
            rid = "antibody-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
            rule = dict(self.rules.get(rid) or {"id": rid, "signature": signature, "kind": "diagnostic", "confidence": 0.5, "severity": "warning", "false_positive_count": 0, "source_candidate_ids": []})
            rule["source_candidate_ids"] = list(dict.fromkeys([*(rule.get("source_candidate_ids") or []), candidate.id]))
            count = len(rule["source_candidate_ids"])
            rule["confidence"] = bounded_score(0.45 + 0.1 * count)
            rule["severity"] = "reject" if count >= int(self.config.get("hard_reject_after", 3) or 3) else "warning"
            self.rules[rid] = rule
            if count >= 2:
                self.necropsies.append({"candidate_id": candidate.id, "rule_id": rid, "common_failure_signature": signature, "recommended_reseed": "repair_from_non_terminal_parent", "contract_mutated": False})
            obligations.append(VerificationObligation(id=rid, verifier_fingerprint="immune:" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16], must_pass=count >= int(self.config.get("hard_reject_after", 3) or 3), strength_contribution=1 if count >= 2 else 0, replayable=True, origin=self.extension_id))
            created += 1
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, verification_obligations=obligations, metrics={"antibody_rule_count": len(self.rules), "necropsy_report_count": len(self.necropsies), "immune_observed_failures": created})

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        advisory = {}
        for candidate in ctx.candidates:
            text = _candidate_text(candidate)
            risk = 0.0
            for rule in self.rules.values():
                if str(rule.get("signature") or "") and str(rule.get("signature")) in text:
                    risk = max(risk, float(rule.get("confidence") or 0.0))
            if risk:
                advisory[candidate.id] = {"risk": bounded_score(risk), "plan_value": 0.0, "rank_prior": 0.0, "diversity": 0.0}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory, metrics={"immune_risk_candidates": len(advisory)})

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        if not self.rules or ctx.parent is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        rules = sorted(self.rules.values(), key=lambda r: float(r.get("confidence") or 0.0), reverse=True)[:3]
        instruction = "Avoid repeated failure signatures learned by the immune system: " + "; ".join(str(r.get("signature")) for r in rules)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction=instruction, metadata={"source_extension": self.extension_id, "rule_ids": [r["id"] for r in rules]})
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure])

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, final_gate_directives=[{"kind": "immune_necropsy_report", "report_count": len(self.necropsies), "contract_mutation_allowed": False}]) if self.necropsies else ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"rules": self.rules, "necropsies": self.necropsies[-50:]} if self.rules or self.necropsies else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.rules = {str(k): dict(v) for k, v in ((state or {}).get("rules") or {}).items() if isinstance(v, dict)}
        self.necropsies = [dict(item) for item in ((state or {}).get("necropsies") or []) if isinstance(item, dict)]


def _failure_signature(diagnostics: list[str]) -> str:
    text = next((str(item) for item in diagnostics if item), "terminal_reject")
    return text[:160]


def _candidate_text(candidate: Any) -> str:
    return " ".join(str(item) for item in [getattr(candidate, "artifact", ""), getattr(candidate, "core_mechanism", ""), getattr(candidate, "concise_claim", "")])[:4000]


__all__ = ["ImmuneNecropsyExtension"]
