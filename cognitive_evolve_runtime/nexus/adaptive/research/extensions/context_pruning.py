"""Evidence-aware context pruning advisory extension."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.concepts.contract import contract_for
from cognitive_evolve_runtime.concepts.effects import ContextTransform

from cognitive_evolve_runtime.evaluators.evidence import SearchPressure
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ContextPruningExtension:
    extension_id = "context_pruning"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.contract = contract_for(self.extension_id)
        self.saved_tokens = 0

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        if ctx.parent is None:
            return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)
        self.saved_tokens += int(self.config.get("estimated_tokens_saved", 128) or 128)
        pressure = SearchPressure.from_parts(parent_id=ctx.parent.id, scope="candidate", mutation_instruction="Prune mutation context to active target challenges, evidence-backed patterns, and current artifact policy; do not include stale resolved challenges or full archive dumps.", metadata={"source_extension": self.extension_id, "resume_critical_state_preserved": True})
        transform = ContextTransform(protect_refs=["problem_spec", "verification_plan", "honesty_invariant"], drop_refs=["stale_resolved_challenges", "full_archive_dump"], view_hash=f"context-prune-{ctx.round_index}-{ctx.parent.id}")
        return ResearchSignal(source=self.extension_id, round_index=ctx.round_index, search_pressures=[pressure], context_transforms=[transform], metrics={"context_tokens_saved": self.saved_tokens})

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {"context_tokens_saved": self.saved_tokens} if self.saved_tokens else {}

    def restore(self, state: dict[str, Any]) -> None:
        self.saved_tokens = int((state or {}).get("context_tokens_saved") or 0)


__all__ = ["ContextPruningExtension"]
