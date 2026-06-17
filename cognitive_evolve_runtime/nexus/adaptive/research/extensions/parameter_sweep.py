"""Bounded parameter sweep extension."""
from __future__ import annotations

from itertools import product
from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import CandidateTransform

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ParameterSweepExtension:
    extension_id = "parameter_sweep"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.sweeps: dict[str, dict[str, Any]] = {}

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        records: list[EvidenceRecord] = []
        transforms: list[CandidateTransform] = []
        max_combinations = int(self.config.get("max_combinations", 32) or 32)
        for candidate in ctx.candidates:
            space = candidate.metadata.get("parameter_space") if isinstance(candidate.metadata, dict) else None
            if not isinstance(space, dict) or not space:
                continue
            combos = _combinations(space, max_combinations=max_combinations)
            collapsed = combos[0] if combos else {}
            slots = candidate.metadata.get("parameter_slots") if isinstance(candidate.metadata, dict) else None
            has_slots = isinstance(slots, dict) and bool(slots)
            self.sweeps[candidate.id] = {"combination_count": len(combos), "collapsed_assignment": collapsed, "final_eligible": False, "has_parameter_slots": has_slots}
            if has_slots:
                transforms.append(CandidateTransform(candidate_id=candidate.id, kind="collapse_params", payload={"assignment": collapsed, "combination_count": len(combos), "parameter_slots": dict(slots)}, preserve_score_within=float(self.config.get("score_epsilon", 0.0) or 0.0)))
            records.append(EvidenceRecord(candidate_id=candidate.id, source=self.extension_id, stage="probe", score=bounded_score(min(1.0, len(combos) / max(1, max_combinations))), final_blocked=True, parent_blocked=False, repair_value=0.4, continuation_value=0.6, diagnostics=["parametric_candidate_must_collapse_before_final" if has_slots else "parametric_candidate_needs_explicit_parameter_slots_before_transform"], metadata={"authority": "probe", "parameter_sweep": self.sweeps[candidate.id]}))
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, evidence_records=records, candidate_transforms=transforms, metrics={"parameter_sweep_candidate_count": len(self.sweeps)})

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        advisory = {cid: {"plan_value": 0.25, "rank_prior": 0.0, "diversity": 0.1, "risk": 0.1} for cid in self.sweeps}
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, selection_advisory=advisory)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        if ctx.parent is None or ctx.parent.id not in self.sweeps:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction="Collapse the parametric candidate to one concrete artifact assignment before any final claim; do not leave template placeholders in final output.", metadata={"source_extension": self.extension_id, "collapsed_assignment": self.sweeps[ctx.parent.id].get("collapsed_assignment")})
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure])

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        directives = [{"kind": "parametric_candidate_not_final", "candidate_id": cid} for cid in self.sweeps]
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, final_gate_directives=directives) if directives else ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"sweeps": self.sweeps} if self.sweeps else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.sweeps = {str(k): dict(v) for k, v in ((state or {}).get("sweeps") or {}).items() if isinstance(v, dict)}


def _combinations(space: dict[str, Any], *, max_combinations: int) -> list[dict[str, Any]]:
    keys = [str(k) for k, v in space.items() if isinstance(v, list) and v]
    values = [space[k] for k in keys]
    out = []
    for combo in product(*values):
        out.append({key: value for key, value in zip(keys, combo)})
        if len(out) >= max_combinations:
            break
    return out


__all__ = ["ParameterSweepExtension"]
