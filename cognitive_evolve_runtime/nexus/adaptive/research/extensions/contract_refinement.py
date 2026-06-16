"""Contract refinement proposal extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import ContractDeltaProposal
from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ContractRefinementExtension:
    extension_id = "contract_refinement"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
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
            before = _objective_hash(ctx)
            change = f"review_generic_challenge_contract_boundary:{generic_count}"
            proposal = ContractDeltaProposal(
                delta_id="contract-delta-" + stable_hash({"source": self.extension_id, "round": ctx.round_index, "generic_count": generic_count, "before": before})[:16],
                proposed_change=change,
                reason="generic challenge accumulation suggests the objective boundary may be underspecified; approval is required before any new success criterion is adopted",
                objective_hash_before=before,
                objective_hash_after="objective-" + stable_hash({"before": before, "proposed_change": change})[:16],
                requires_approval=True,
            ).to_dict()
            self.proposals.append(proposal)
            return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, metrics={"contract_refinement_proposal_count": len(self.proposals)})
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        proposals = [{**proposal, "final_projection_status": "proposal_queue_only"} for proposal in self.proposals[-3:]]
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, contract_delta_proposals=proposals) if proposals else ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"proposals": self.proposals[-20:]} if self.proposals else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.proposals = [dict(item) for item in ((state or {}).get("proposals") or []) if isinstance(item, dict)]


def _objective_hash(ctx: ResearchContext) -> str:
    contract = ctx.contract
    if contract is not None and hasattr(contract, "contract_hash"):
        try:
            return str(contract.contract_hash())
        except Exception:
            pass
    if contract is not None and hasattr(contract, "to_dict"):
        try:
            return "objective-" + stable_hash(contract.to_dict())[:16]
        except Exception:
            pass
    return "objective-" + stable_hash({"final_certificate": ctx.final_certificate, "round": ctx.round_index})[:16]


__all__ = ["ContractRefinementExtension"]
