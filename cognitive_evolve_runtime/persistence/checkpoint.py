"""Checkpoint store tying population, archives, policy, contract, and progress together."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json
from cognitive_evolve_runtime.llm.call_ledger import ledger_summary
from cognitive_evolve_runtime.llm.session import current_llm_session
from cognitive_evolve_runtime.core.serialization import coerce_dict, utc_now
from cognitive_evolve_runtime.persistence.checkpoint_profile import apply_checkpoint_profile_to_history, apply_checkpoint_profile_to_population, checkpoint_profile_from_env

LATENT_LEDGER_METADATA_KEY = "latent_ledger"
LATENT_LEDGER_REF_KEYS = ("latent_ledger_ref", "latent_ledger_sidecar")


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
    trace_state: dict[str, Any] = field(default_factory=dict)
    tension_map: dict[str, Any] = field(default_factory=dict)
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    concept_snapshots: dict[str, Any] = field(default_factory=dict)
    verification_plan: dict[str, Any] = field(default_factory=dict)
    graded_output: dict[str, Any] = field(default_factory=dict)
    search_kernel: dict[str, Any] = field(default_factory=dict)
    checkpoint_profile: dict[str, Any] = field(default_factory=dict)
    call_ledger_summary: dict[str, Any] = field(default_factory=dict)
    fabric: dict[str, Any] = field(default_factory=dict)
    runtime_options: dict[str, Any] = field(default_factory=dict)
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
            trace_state=coerce_dict(data.get("trace_state")),
            tension_map=coerce_dict(data.get("tension_map")),
            cost_ledger=coerce_dict(data.get("cost_ledger")),
            concept_snapshots=coerce_dict(data.get("concept_snapshots")),
            verification_plan=coerce_dict(data.get("verification_plan")),
            graded_output=coerce_dict(data.get("graded_output")),
            search_kernel=coerce_dict(data.get("search_kernel")),
            checkpoint_profile=coerce_dict(data.get("checkpoint_profile")) or {"name": "full", "legacy_checkpoint": True},
            call_ledger_summary=coerce_dict(data.get("call_ledger_summary")),
            fabric=coerce_dict(data.get("fabric")),
            runtime_options=coerce_dict(data.get("runtime_options")),
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
        contract = _hydrate_latent_ledger_sidecar(checkpoint.contract, base_dir=self.path.parent)
        checkpoint.contract = contract
        return {
            "checkpoint": checkpoint,
            "population": CandidatePopulation.from_dict(checkpoint.population),
            "archives": ArchiveManager.from_dict(checkpoint.archives),
            "policy": EvolutionPolicy.from_dict(checkpoint.policy) if checkpoint.policy else EvolutionPolicy(),
            "diagnosis": SearchDiagnosis.from_dict(checkpoint.diagnosis) if checkpoint.diagnosis else SearchDiagnosis(),
            "budget_history": list(checkpoint.budget_history),
            "budget": dict(checkpoint.budget),
            "adaptive_state": dict(checkpoint.adaptive_state),
            "trace_state": dict(checkpoint.trace_state),
            "tension_map": dict(checkpoint.tension_map),
            "cost_ledger": dict(checkpoint.cost_ledger),
            "concept_snapshots": dict(checkpoint.concept_snapshots),
            "verification_plan": dict(checkpoint.verification_plan),
            "graded_output": dict(checkpoint.graded_output),
            "search_kernel": dict(checkpoint.search_kernel),
            "checkpoint_profile": dict(checkpoint.checkpoint_profile),
            "call_ledger_summary": dict(checkpoint.call_ledger_summary),
            "fabric": dict(checkpoint.fabric),
            "runtime_options": dict(checkpoint.runtime_options),
            "contract": contract,
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
        trace_state: dict[str, Any] | None = None,
        tension_map: dict[str, Any] | None = None,
        cost_ledger: dict[str, Any] | None = None,
        concept_snapshots: dict[str, Any] | None = None,
        verification_plan: dict[str, Any] | None = None,
        graded_output: dict[str, Any] | None = None,
        search_kernel: dict[str, Any] | None = None,
        fabric: dict[str, Any] | None = None,
        runtime_options: dict[str, Any] | None = None,
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
            trace_state=trace_state,
            tension_map=tension_map,
            cost_ledger=cost_ledger,
            concept_snapshots=concept_snapshots,
            verification_plan=verification_plan,
            graded_output=graded_output,
            search_kernel=search_kernel,
            fabric=fabric,
            runtime_options=runtime_options,
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
    trace_state: dict[str, Any] | None = None,
    tension_map: dict[str, Any] | None = None,
    cost_ledger: dict[str, Any] | None = None,
    concept_snapshots: dict[str, Any] | None = None,
    verification_plan: dict[str, Any] | None = None,
    graded_output: dict[str, Any] | None = None,
    search_kernel: dict[str, Any] | None = None,
    fabric: dict[str, Any] | None = None,
    runtime_options: dict[str, Any] | None = None,
    allow_progress_round_repair: bool = False,
) -> NexusCheckpoint:
    profile = checkpoint_profile_from_env()
    population_payload = apply_checkpoint_profile_to_population(population.to_dict(), profile)
    budget_history_payload = apply_checkpoint_profile_to_history(budget_history or [], profile)
    checkpoint = NexusCheckpoint(
        round=round,
        max_rounds=max_rounds,
        population=population_payload,
        archives=archives.to_dict(),
        policy=policy.to_dict() if hasattr(policy, "to_dict") else coerce_dict(policy),
        diagnosis=diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else coerce_dict(diagnosis),
        progress_event=progress_event or {},
        contract=contract_payload_for_persistence(contract),
        world=world.to_dict() if hasattr(world, "to_dict") else coerce_dict(world),
        mode=mode,
        budget_history=budget_history_payload,
        budget=coerce_dict(budget),
        adaptive_state=coerce_dict(adaptive_state),
        trace_state=coerce_dict(trace_state),
        tension_map=coerce_dict(tension_map),
        cost_ledger=_checkpoint_cost_ledger(cost_ledger),
        concept_snapshots=coerce_dict(concept_snapshots),
        verification_plan=coerce_dict(verification_plan),
        graded_output=coerce_dict(graded_output),
        search_kernel=coerce_dict(search_kernel),
        fabric=coerce_dict(fabric),
        runtime_options=coerce_dict(runtime_options),
        checkpoint_profile=profile.to_dict(),
        call_ledger_summary=ledger_summary(),
    )
    checkpoint = _repair_progress_event_round(checkpoint) if allow_progress_round_repair else checkpoint
    if checkpoint.progress_event:
        event_round = checkpoint.progress_event.get("round")
        if event_round is not None and int(event_round) != checkpoint.round:
            raise ValueError("checkpoint round and progress event round differ")
    return checkpoint


def contract_payload_for_persistence(contract: Any | None) -> dict[str, Any]:
    """Return a checkpoint/result-safe contract payload.

    Restore hydrates ``metadata.latent_ledger`` from the sidecar ref when needed,
    but persistence must not re-embed the hydrated ledger after a resume or final
    snapshot write.
    """

    payload = contract.to_dict() if hasattr(contract, "to_dict") else coerce_dict(contract)
    metadata = coerce_dict(payload.get("metadata"))
    if any(metadata.get(key) for key in LATENT_LEDGER_REF_KEYS):
        metadata.pop(LATENT_LEDGER_METADATA_KEY, None)
        payload["metadata"] = metadata
    return payload


def _checkpoint_cost_ledger(cost_ledger: Any | None) -> dict[str, Any]:
    """Keep provider billing telemetry in its own checkpoint namespace.

    The adaptive research-extension cost ledger is a separate accounting
    surface.  LLM provider costs are session-scoped and must not be merged into
    that extension payload, especially when parallel Nexus runs share a Python
    process.
    """

    payload = coerce_dict(cost_ledger)
    session = current_llm_session()
    events = session.snapshot()
    total = session.total_estimated_cost_usd()
    if events or total:
        provider = coerce_dict(payload.get("llm_provider"))
        provider["estimated_cost_usd"] = total
        provider["event_count"] = len(events)
        provider["source"] = "llm_session"
        payload["llm_provider"] = provider
    return payload


def _hydrate_latent_ledger_sidecar(contract: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Hydrate sidecar-only latent ledger refs during checkpoint restore.

    Legacy checkpoints may still embed ``metadata.latent_ledger``; those remain
    untouched.  New checkpoints store only a ref/hash/cursor and are hydrated
    here for runtime consumers that expect contract metadata to contain the
    materialized ledger after restore.
    """

    payload = coerce_dict(contract)
    metadata = coerce_dict(payload.get("metadata"))
    if not metadata or metadata.get("latent_ledger"):
        return payload
    ref = coerce_dict(metadata.get("latent_ledger_ref") or metadata.get("latent_ledger_sidecar"))
    raw_path = str(ref.get("path") or ref.get("latent_ledger_cache_path") or "").strip()
    if not raw_path:
        return payload
    sidecar_path = Path(raw_path)
    if not sidecar_path.is_absolute():
        sidecar_path = base_dir / sidecar_path
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        diagnostics = list(metadata.get("latent_ledger_restore_diagnostics") or [])
        diagnostics.append("latent_ledger_sidecar_unreadable")
        metadata["latent_ledger_restore_diagnostics"] = diagnostics
        payload["metadata"] = metadata
        return payload
    if not isinstance(sidecar, dict):
        return payload
    expected_hash = str(ref.get("sha256") or ref.get("ledger_hash") or "")
    if expected_hash:
        from cognitive_evolve_runtime.core.serialization import stable_hash

        actual_hash = stable_hash(sidecar)
        if actual_hash != expected_hash:
            diagnostics = list(metadata.get("latent_ledger_restore_diagnostics") or [])
            diagnostics.append("latent_ledger_sidecar_hash_mismatch")
            metadata["latent_ledger_restore_diagnostics"] = diagnostics
            payload["metadata"] = metadata
            return payload
    metadata["latent_ledger"] = sidecar
    payload["metadata"] = metadata
    return payload


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


__all__ = ["CheckpointStore", "NexusCheckpoint", "build_checkpoint_state", "contract_payload_for_persistence"]
