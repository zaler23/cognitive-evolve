"""Contract refinement proposal extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ContractRefinementExtension:
    extension_id = "contract_refinement"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.proposals: list[dict[str, Any]] = []

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        memory = ctx.challenge_memory
        generic_count = 0
        if memory is not None:
            for raw in memory.items.values():
                metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
                if metadata.get("category") == "generic":
                    generic_count += 1
        threshold = int(self.config.get("generic_challenge_threshold", 5) or 5)
        if generic_count >= threshold:
            proposal = {"kind": "contract_refinement_proposal", "generic_challenge_count": generic_count, "silent_mutation_allowed": False, "requires_user_decision": True}
            self.proposals.append(proposal)
            return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics={"contract_refinement_proposal_count": len(self.proposals)})
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        directives = [{**proposal, "final_projection_status": "needs_user_decision"} for proposal in self.proposals[-3:]]
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, final_gate_directives=directives) if directives else ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"proposals": self.proposals[-20:]} if self.proposals else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.proposals = [dict(item) for item in ((state or {}).get("proposals") or []) if isinstance(item, dict)]


__all__ = ["ContractRefinementExtension"]
