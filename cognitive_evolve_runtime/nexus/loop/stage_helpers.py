"""Stage eligibility and parent-repair helpers for Nexus rounds."""
from __future__ import annotations

from typing import Any, Callable

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.theory import TheoryConfig

def _error_progress_event(previous_event: dict[str, Any], current_round: int) -> dict[str, Any]:
    """Return progress metadata that is safe to store inside an error checkpoint."""

    event = dict(previous_event or {})
    previous_round = event.get("round")
    event["round"] = int(current_round or 0)
    event.setdefault("type", "evolution_progress")
    metadata = dict(event.get("metadata") or {})
    try:
        previous_round_int = int(previous_round or 0)
    except (TypeError, ValueError):
        previous_round_int = -1
    if previous_round is not None and previous_round_int != int(current_round or 0):
        metadata["previous_progress_round"] = previous_round
        metadata["error_checkpoint_round"] = int(current_round or 0)
        metadata["round_reconciled_for_error_checkpoint"] = True
    event["metadata"] = metadata
    return event


def _eligibility_policy(policy: EvolutionPolicy | None) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("eligibility_policy") or metadata.get("stage_policy")
    return dict(raw) if isinstance(raw, dict) else {}


def _theory_config_from_policy(policy: EvolutionPolicy | None) -> TheoryConfig:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    if not isinstance(metadata, dict):
        return TheoryConfig()
    raw = metadata.get("theory") or metadata.get("theory_config") or {}
    return TheoryConfig.from_mapping(raw if isinstance(raw, dict) else {})


def _raise_if_cancelled(cancellation_callback: Callable[[], bool] | None) -> None:
    if cancellation_callback is not None and cancellation_callback():
        raise InterruptedError("nexus evolution cancellation requested")


def _notify_observer(
    observer: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    round_index: int,
    population: CandidatePopulation,
    archives: ArchiveManager,
    policy: EvolutionPolicy,
    diagnosis: SearchDiagnosis,
    progress_event: dict[str, Any],
    budget_history: list[dict[str, Any]],
    elo_state: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    adaptive_state: dict[str, Any] | None = None,
    fabric_state: dict[str, Any] | None = None,
) -> None:
    if observer is None:
        return
    observer(
        {
            "phase": phase,
            "round": round_index,
            "population": population,
            "archives": archives,
            "policy": policy,
            "diagnosis": diagnosis,
            "progress_event": progress_event,
            "budget_history": list(budget_history),
            "elo": dict(elo_state or {}),
            "error": error,
            "adaptive_state": dict(adaptive_state or {}),
            "fabric": dict(fabric_state or {}),
        }
    )


__all__ = ["_eligibility_policy", "_error_progress_event", "_notify_observer", "_raise_if_cancelled", "_theory_config_from_policy"]
