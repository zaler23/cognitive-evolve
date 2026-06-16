"""Experimental adaptive research extension protocol.

This is intentionally scoped under ``nexus.adaptive``.  It is not a second
runtime and does not own candidate fate, challenge truth, or final authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal
from cognitive_evolve_runtime.concepts.contract import ConceptContract, contract_for


@dataclass(frozen=True)
class ResearchContext:
    round_index: int = 0
    candidates: list[CandidateGenome] = field(default_factory=list)
    population: CandidatePopulation | None = None
    challenge_memory: ChallengeMemory | None = None
    adaptive_state: Any | None = None
    spatial_state: Any | None = None
    config: dict[str, Any] = field(default_factory=dict)
    policy: Any | None = None
    contract: Any | None = None
    world: Any | None = None
    parent: CandidateGenome | None = None
    final_certificate: dict[str, Any] = field(default_factory=dict)


class ResearchExtension(Protocol):
    extension_id: str
    contract: ConceptContract

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal: ...
    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal: ...
    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal: ...
    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal: ...
    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: dict[str, Any]) -> None: ...


class NoOpResearchExtension:
    extension_id = "noop"

    def __init__(self) -> None:
        self.contract = contract_for(self.extension_id)

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return ResearchSignal.empty(source=self.extension_id, round_index=ctx.round_index)

    def snapshot(self) -> dict[str, Any]:
        return {}

    def restore(self, state: dict[str, Any]) -> None:
        return None


__all__ = ["NoOpResearchExtension", "ResearchContext", "ResearchExtension"]
