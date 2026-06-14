"""Public Nexus engine result."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus.state_contract import (
    EXTERNAL_QUESTIONS_ALLOWED,
    FINAL_ANSWER_MAY_REQUEST_CLARIFICATION,
    INTERACTION_MODE,
    RUNTIME_PATH,
    RUNTIME_VERSION,
)
from cognitive_evolve_runtime.nexus.state import nexus_evolution_summary, nexus_search_state, nexus_verification_results


@dataclass
class NexusEngineResult:
    """Stable public result returned by the Nexus orchestrator."""

    prompt: str
    mode: str
    contract: dict[str, Any]
    policy: dict[str, Any]
    world: dict[str, Any]
    evolution: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    pipeline_events: list[dict[str, Any]] = field(default_factory=list)
    context_protocol: dict[str, Any] = field(default_factory=dict)
    verification_summaries: list[dict[str, Any]] = field(default_factory=list)
    verification_results: dict[str, Any] | None = None

    @property
    def final_answer(self) -> str:
        synthesis = self.evolution.get("synthesis") if isinstance(self.evolution.get("synthesis"), dict) else {}
        return str(synthesis.get("final_answer") or "")

    @property
    def nexus_evolution(self) -> dict[str, Any]:
        return nexus_evolution_summary({"evolution": self.evolution})

    @property
    def nexus_search(self) -> dict[str, Any]:
        return nexus_search_state({"evolution": self.evolution})

    def verification_payload(self) -> dict[str, Any]:
        payload = dict(self.verification_results or {})
        derived = nexus_verification_results({"evolution": self.evolution, "verification_summaries": self.verification_summaries})
        for key, value in derived.items():
            payload.setdefault(key, value)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": RUNTIME_VERSION,
            "runtime_path": RUNTIME_PATH,
            "runtime_architecture": "nexus",
            "interaction_mode": INTERACTION_MODE,
            "external_questions_allowed": EXTERNAL_QUESTIONS_ALLOWED,
            "final_answer_may_request_clarification": FINAL_ANSWER_MAY_REQUEST_CLARIFICATION,
            "prompt": self.prompt,
            "mode": self.mode,
            "contract": self.contract,
            "policy": self.policy,
            "world": self.world,
            "evolution": self.evolution,
            "artifacts": self.artifacts,
            "pipeline_events": self.pipeline_events,
            "context_protocol": self.context_protocol,
            "verification_summaries": self.verification_summaries,
            "nexus_evolution": self.nexus_evolution,
            "nexus_search": self.nexus_search,
            "final_answer": self.final_answer,
            "verification_results": self.verification_payload(),
        }


__all__ = ["NexusEngineResult"]
