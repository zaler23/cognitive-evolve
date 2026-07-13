"""Typed model boundary for the Nexus runtime.

The runtime still accepts lightweight test doubles, but the expected model
surface is now explicit.  Call sites may use ``hasattr`` for optional
capabilities, while IDEs/type-checkers can see the contract shared by
``StructuredModelAdapter`` and compatible custom adapters.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class NexusContractModelProtocol(Protocol):
    def build_text_world_model(self, *, packet: Any) -> dict[str, Any]: ...

    def build_objective_contract(self, *, user_goal: str, world: Any) -> dict[str, Any]: ...

    def build_project_objective_contract(self, *, user_goal: str, snapshot: Any, world: Any | None = None) -> dict[str, Any]: ...


@runtime_checkable
class NexusPolicyModelProtocol(Protocol):
    def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]: ...


@runtime_checkable
class NexusSeedModelProtocol(Protocol):
    def seed_population(self, *, contract: Any, world: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class NexusPoolPreprocessModelProtocol(Protocol):
    def preprocess_candidate_pool(self, *, contract: Any, policy: Any, coverage_report: dict[str, Any], clusters: list[dict[str, Any]], representatives: list[dict[str, Any]], instructions: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]: ...


@runtime_checkable
class NexusRankingModelProtocol(Protocol):
    def relative_rank(self, *, candidates: list[Any], contract: Any, policy: Any, archives: Any) -> dict[str, Any]: ...


@runtime_checkable
class NexusCritiqueModelProtocol(Protocol):
    def critique_candidates(self, *, candidates: list[Any], round_index: int, contract: Any, policy: Any, archives: Any) -> list[dict[str, Any]]: ...


@runtime_checkable
class NexusDiagnosisModelProtocol(Protocol):
    def diagnose_search_state(self, *, population: list[Any], archives: Any, history: list[dict[str, Any]], contract: Any, policy: Any) -> dict[str, Any]: ...

    def update_policy(self, *, policy: Any, diagnosis: Any) -> dict[str, Any]: ...


@runtime_checkable
class NexusContextModelProtocol(Protocol):
    def request_context(self, *, contract: Any, world: Any, parents: list[Any], archives: Any, mutation_instruction: str = "") -> dict[str, Any]: ...


@runtime_checkable
class NexusMutationPlannerModelProtocol(Protocol):
    def plan_mutations(self, *, parents: list[Any], actions: list[str], archives: Any, diagnosis: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class NexusOffspringModelProtocol(Protocol):
    def generate_offspring(self, *, plans: list[Any], parents: list[Any], world: Any, contract: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class NexusMutationModelProtocol(NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol, Protocol):
    pass


@runtime_checkable
class NexusSynthesisModelProtocol(Protocol):
    def synthesize_result(self, *, population: list[Any], archives: Any, contract: Any, world: Any) -> dict[str, Any]: ...


@runtime_checkable
class NexusStopModelProtocol(Protocol):
    def should_stop(self, *, budget: Any, diagnosis: Any, best_answer_id: str, population: list[Any]) -> dict[str, Any] | bool: ...


@runtime_checkable
class NexusModelProtocol(
    NexusContractModelProtocol,
    NexusPolicyModelProtocol,
    NexusPoolPreprocessModelProtocol,
    NexusSeedModelProtocol,
    NexusRankingModelProtocol,
    NexusCritiqueModelProtocol,
    NexusDiagnosisModelProtocol,
    NexusContextModelProtocol,
    NexusMutationModelProtocol,
    NexusSynthesisModelProtocol,
    NexusStopModelProtocol,
    Protocol,
):
    """Full adapter protocol implemented by ``StructuredModelAdapter``.

    Runtime entry points accept lightweight partial models, so most call sites
    should annotate with ``NexusModelLike`` and narrow to the relevant small
    protocol only at the boundary where a capability is invoked.
    """

    metadata: dict[str, Any]


@runtime_checkable
class NexusClassifierProtocol(Protocol):
    def classify_task(self, *, prompt: str) -> dict[str, Any]: ...


NexusModelLike: TypeAlias = (
    NexusModelProtocol
    | NexusContractModelProtocol
    | NexusPolicyModelProtocol
    | NexusPoolPreprocessModelProtocol
    | NexusSeedModelProtocol
    | NexusRankingModelProtocol
    | NexusCritiqueModelProtocol
    | NexusDiagnosisModelProtocol
    | NexusContextModelProtocol
    | NexusMutationModelProtocol
    | NexusSynthesisModelProtocol
    | NexusStopModelProtocol
    | NexusClassifierProtocol
)


__all__ = [
    "NexusClassifierProtocol",
    "NexusContextModelProtocol",
    "NexusContractModelProtocol",
    "NexusCritiqueModelProtocol",
    "NexusDiagnosisModelProtocol",
    "NexusModelLike",
    "NexusModelProtocol",
    "NexusMutationModelProtocol",
    "NexusMutationPlannerModelProtocol",
    "NexusOffspringModelProtocol",
    "NexusPolicyModelProtocol",
    "NexusPoolPreprocessModelProtocol",
    "NexusRankingModelProtocol",
    "NexusSeedModelProtocol",
    "NexusStopModelProtocol",
    "NexusSynthesisModelProtocol",
]
