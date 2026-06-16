"""Checkpoint store tying population, archives, policy, contract, and progress together."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json
from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now


@dataclass
class NexusCheckpoint:
    round: int
    max_rounds: int
    population: dict[str, Any]
    archives: dict[str, Any]
    policy: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    progress_event: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    budget_history: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    adaptive_state: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusCheckpoint":
        return cls(
            round=int(data.get("round") or 0),
            max_rounds=int(data.get("max_rounds") or 0),
            population=coerce_dict(data.get("population")),
            archives=coerce_dict(data.get("archives")),
            policy=coerce_dict(data.get("policy")),
            diagnosis=coerce_dict(data.get("diagnosis")),
            progress_event=coerce_dict(data.get("progress_event")),
            contract=coerce_dict(data.get("contract")),
            world=coerce_dict(data.get("world")),
            mode=str(data.get("mode") or ""),
            budget_history=[dict(item) for item in data.get("budget_history", []) if isinstance(item, dict)],
            budget=coerce_dict(data.get("budget")),
            adaptive_state=coerce_dict(data.get("adaptive_state")),
            created_at=str(data.get("created_at") or utc_now()),
        )


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, checkpoint: NexusCheckpoint, *, allow_progress_round_repair: bool = False) -> None:
        checkpoint = _repair_progress_event_round(checkpoint) if allow_progress_round_repair else checkpoint
        if checkpoint.progress_event:
            event_round = checkpoint.progress_event.get("round")
            if event_round is not None and int(event_round) != checkpoint.round:
                raise ValueError("checkpoint round and progress event round differ")
        atomic_write_json(self.path, checkpoint.to_dict(), sort_keys=True)

    def load(self) -> NexusCheckpoint | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"checkpoint must be a JSON object: {self.path}")
        return NexusCheckpoint.from_dict(data)

    def restore_state(self) -> dict[str, Any] | None:
        checkpoint = self.load()
        if checkpoint is None:
            return None
        from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
        from cognitive_evolve_runtime.nexus.generation_plan import validate_generation_plan_history
        from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy

        validate_generation_plan_history(
            checkpoint.budget_history,
            archive_history=[
                dict(item)
                for item in coerce_dict(checkpoint.archives).get("history", [])
                if isinstance(item, dict)
            ],
        )
        return {
            "checkpoint": checkpoint,
            "population": CandidatePopulation.from_dict(checkpoint.population),
            "archives": ArchiveManager.from_dict(checkpoint.archives),
            "policy": EvolutionPolicy.from_dict(checkpoint.policy) if checkpoint.policy else EvolutionPolicy(),
            "diagnosis": SearchDiagnosis.from_dict(checkpoint.diagnosis) if checkpoint.diagnosis else SearchDiagnosis(),
            "budget_history": list(checkpoint.budget_history),
            "budget": dict(checkpoint.budget),
            "adaptive_state": dict(checkpoint.adaptive_state),
            "contract": checkpoint.contract,
            "world": checkpoint.world,
            "mode": checkpoint.mode,
        }

    def save_state(
        self,
        *,
        round: int,
        max_rounds: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: Any | None = None,
        diagnosis: Any | None = None,
        progress_event: dict[str, Any] | None = None,
        contract: Any | None = None,
        world: Any | None = None,
        mode: str = "",
        budget_history: list[dict[str, Any]] | None = None,
        budget: dict[str, Any] | None = None,
        adaptive_state: dict[str, Any] | None = None,
        allow_progress_round_repair: bool = False,
    ) -> NexusCheckpoint:
        checkpoint = build_checkpoint_state(
            round=round,
            max_rounds=max_rounds,
            population=population,
            archives=archives,
            policy=policy,
            diagnosis=diagnosis,
            progress_event=progress_event,
            contract=contract,
            world=world,
            mode=mode,
            budget_history=budget_history,
            budget=budget,
            adaptive_state=adaptive_state,
            allow_progress_round_repair=allow_progress_round_repair,
        )
        self.save(checkpoint, allow_progress_round_repair=allow_progress_round_repair)
        return checkpoint


def build_checkpoint_state(
    *,
    round: int,
    max_rounds: int,
    population: CandidatePopulation,
    archives: ArchiveManager,
    policy: Any | None = None,
    diagnosis: Any | None = None,
    progress_event: dict[str, Any] | None = None,
    contract: Any | None = None,
    world: Any | None = None,
    mode: str = "",
    budget_history: list[dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
    adaptive_state: dict[str, Any] | None = None,
    allow_progress_round_repair: bool = False,
) -> NexusCheckpoint:
    checkpoint = NexusCheckpoint(
        round=round,
        max_rounds=max_rounds,
        population=population.to_dict(),
        archives=archives.to_dict(),
        policy=policy.to_dict() if hasattr(policy, "to_dict") else coerce_dict(policy),
        diagnosis=diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else coerce_dict(diagnosis),
        progress_event=progress_event or {},
        contract=contract.to_dict() if hasattr(contract, "to_dict") else coerce_dict(contract),
        world=world.to_dict() if hasattr(world, "to_dict") else coerce_dict(world),
        mode=mode,
        budget_history=budget_history or [],
        budget=coerce_dict(budget),
        adaptive_state=coerce_dict(adaptive_state),
    )
    checkpoint = _repair_progress_event_round(checkpoint) if allow_progress_round_repair else checkpoint
    if checkpoint.progress_event:
        event_round = checkpoint.progress_event.get("round")
        if event_round is not None and int(event_round) != checkpoint.round:
            raise ValueError("checkpoint round and progress event round differ")
    return checkpoint


def _repair_progress_event_round(checkpoint: NexusCheckpoint) -> NexusCheckpoint:
    """Make error-path checkpoint progress metadata internally consistent.

    Normal checkpoint writes remain strict.  Error checkpoints are different:
    they can happen after the runtime has advanced to the next round but before
    a fresh progress event is available.  In that path, losing the checkpoint is
    worse than preserving the previous progress-event round verbatim, so the
    event round is reconciled and the mismatch is retained in metadata.
    """

    if not checkpoint.progress_event:
        return checkpoint
    event_round = checkpoint.progress_event.get("round")
    if event_round is None:
        return checkpoint
    try:
        parsed = int(event_round)
    except (TypeError, ValueError):
        parsed = -1
    if parsed == checkpoint.round:
        return checkpoint
    data = checkpoint.to_dict()
    progress_event = dict(data.get("progress_event") or {})
    metadata = dict(progress_event.get("metadata") or {})
    metadata["repaired_progress_event_round"] = {
        "from": event_round,
        "to": checkpoint.round,
        "reason": "error_checkpoint_round_reconciliation",
    }
    progress_event["round"] = checkpoint.round
    progress_event["metadata"] = metadata
    data["progress_event"] = progress_event
    return NexusCheckpoint.from_dict(data)


__all__ = ["CheckpointStore", "NexusCheckpoint", "build_checkpoint_state"]
