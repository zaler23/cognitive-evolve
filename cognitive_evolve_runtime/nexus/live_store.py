"""Live persistence for Nexus rounds and phases."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.core.serialization import json_ready
from cognitive_evolve_runtime.durable.file_lock import _fsync_dir, atomic_write_json, file_lock
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.nexus.seed_coverage import SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY, persist_seed_reservoir_sidecar
from cognitive_evolve_runtime.outcomes.latent_ledger import LatentLedger, LatentLedgerStore
from cognitive_evolve_runtime.persistence.checkpoint import build_checkpoint_state
from cognitive_evolve_runtime.persistence.event_store import EventStore
from cognitive_evolve_runtime.persistence.transactional_snapshot import NexusSnapshotTransaction, SnapshotWrite

LATENT_LEDGER_METADATA_KEY = "latent_ledger"
LATENT_POSTERIOR_SNAPSHOT_METADATA_KEY = "latent_posterior_snapshot"


@dataclass(frozen=True)
class FrozenLiveUpdate:
    phase: str
    round_index: int
    snapshot_writes: tuple[SnapshotWrite, ...]
    round_snapshot: dict[str, Any]
    journal_rows: tuple[dict[str, Any], ...]
    event: dict[str, Any]
    latent_metadata: dict[str, Any]
    checkpoint_failure: dict[str, Any] | None = None

    @property
    def serialized_bytes(self) -> int:
        payload = {
            "snapshot_writes": [item.payload for item in self.snapshot_writes],
            "round_snapshot": self.round_snapshot,
            "journal_rows": self.journal_rows,
            "event": self.event,
            "checkpoint_failure": self.checkpoint_failure,
        }
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


class LiveNexusStore:
    """Persist a recoverable Nexus state after every meaningful phase."""

    def __init__(self, output_dir: str | Path, *, mode: str, contract: Any, world: Any, max_rounds: int, budget: dict[str, Any] | None = None, runtime_options: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.contract = contract
        self.world = world
        self.max_rounds = int(max_rounds)
        self.budget = dict(budget or {"max_rounds": self.max_rounds})
        self.runtime_options = dict(runtime_options or {})
        self.event_store = EventStore(self.output_dir / "events.jsonl")
        self.latent_ledger_store = LatentLedgerStore(self.output_dir)
        self.round_dir = self.output_dir / "rounds"
        self.round_dir.mkdir(parents=True, exist_ok=True)
        self._event_hashes: set[str] = set()

    def __call__(self, update: dict[str, Any]) -> None:
        batch = self.freeze(update)
        if batch is not None:
            self.write(batch)

    def freeze(self, update: dict[str, Any]) -> FrozenLiveUpdate | None:
        """Freeze all mutable runtime state before persistence leaves the caller."""

        population = update.get("population")
        archives = update.get("archives")
        if not isinstance(population, CandidatePopulation) or not isinstance(archives, ArchiveManager):
            return None
        phase = str(update.get("phase") or "state")
        round_index = int(update.get("round") or 0)
        policy = update.get("policy")
        diagnosis = update.get("diagnosis")
        progress_event = dict(update.get("progress_event") or {})
        budget_history = [dict(item) for item in update.get("budget_history", []) if isinstance(item, dict)]
        adaptive_state = dict(update.get("adaptive_state") or {}) if isinstance(update.get("adaptive_state"), dict) else {}
        fabric_state = dict(update.get("fabric") or {}) if isinstance(update.get("fabric"), dict) else {}
        runtime_options = dict(update.get("runtime_options") or self.runtime_options) if isinstance(update.get("runtime_options") or self.runtime_options, dict) else {}
        policy_metadata = coerce_dict(getattr(policy, "metadata", None))
        search_kernel_state = dict(update.get("search_kernel") or {}) if isinstance(update.get("search_kernel"), dict) else {}
        sidecar_ref = persist_seed_reservoir_sidecar(self.output_dir, policy_metadata.get(SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY))
        if sidecar_ref:
            if isinstance(getattr(policy, "metadata", None), dict):
                policy.metadata.pop(SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY, None)
                policy.metadata["seed_reservoir_ref"] = sidecar_ref
            policy_metadata.pop(SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY, None)
            policy_metadata["seed_reservoir_ref"] = sidecar_ref
            search_kernel_state["seed_reservoir_ref"] = sidecar_ref
        for key in ("seed_coverage", "target_perturb_seed_judgment", "algorithm_efficiency", "model_parallel_efficiency", "minimal_core_ablation", "seed_active_frontier", "seed_reservoir_ref"):
            if key in policy_metadata:
                search_kernel_state.setdefault(key, policy_metadata[key])
        monitor_state = {
            "phase": phase,
            "round": round_index,
            "population_size": len(population.candidates),
            "fate_counts": _fate_counts(population),
            "search_kernel": search_kernel_state,
        }
        error = update.get("error") if isinstance(update.get("error"), dict) else None

        budget_payload = dict(self.budget)
        budget_payload["current_round"] = round_index
        budget_payload["max_rounds"] = self.max_rounds
        if progress_event.get("max_rounds"):
            budget_payload["round_limit"] = int(progress_event.get("max_rounds") or self.max_rounds)
        allow_round_repair = phase == "error_checkpoint"
        latent_metadata = self._prepare_latent_metadata()
        snapshot_writes: tuple[SnapshotWrite, ...] = ()
        checkpoint_failure: dict[str, Any] | None = None
        try:
            checkpoint = build_checkpoint_state(
                round=round_index,
                max_rounds=self.max_rounds,
                population=population,
                archives=archives,
                policy=policy,
                diagnosis=diagnosis,
                progress_event=progress_event,
                contract=self.contract,
                world=self.world,
                mode=self.mode,
                budget_history=budget_history,
                budget=budget_payload,
                adaptive_state=adaptive_state,
                fabric=fabric_state,
                search_kernel=search_kernel_state,
                runtime_options=runtime_options,
                allow_progress_round_repair=allow_round_repair,
            )
            snapshot_writes = (
                SnapshotWrite("population.json", "json", population.to_dict()),
                SnapshotWrite("archives.json", "json", archives.to_dict()),
                SnapshotWrite("checkpoint.json", "json", checkpoint.to_dict()),
            )
        except Exception as exc:
            if not allow_round_repair:
                raise
            error = dict(error or {})
            error["checkpoint_persist_error"] = f"{exc.__class__.__name__}: {exc}"
            checkpoint_failure = {
                "phase": phase,
                "round": round_index,
                "at": utc_now(),
                "error": error,
                "progress_event": progress_event,
            }
        snapshot = {
            "phase": phase,
            "round": round_index,
            "at": utc_now(),
            "error": error,
            "population": population.to_dict(),
            "archives": archives.summary(),
            "policy": policy.to_dict() if hasattr(policy, "to_dict") else {},
            "diagnosis": diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else {},
            "progress_event": progress_event,
            "adaptive_state": adaptive_state,
            "fabric": fabric_state,
            "runtime_options": runtime_options,
            "search_kernel": search_kernel_state,
            "monitor_state": monitor_state,
        }
        at = utc_now()
        journal_rows = tuple(
            {
                "at": at,
                "round": round_index,
                "phase": phase,
                "id": candidate.id,
                "generation": candidate.generation,
                "fate": candidate.current_fate,
                "parents": list(candidate.parent_ids),
                "core_mechanism": candidate.core_mechanism,
                "concise_claim": candidate.concise_claim,
                "genome_hash": candidate.genome_hash,
                "scores": dict(candidate.multihead_scores),
                "search_seed_not_final": bool(candidate.metadata.get("search_seed_not_final")),
            }
            for candidate in population.candidates
        )
        frozen_writes = tuple(
            SnapshotWrite(item.relative_path, item.kind, _freeze_json_object(item.payload), sort_keys=item.sort_keys)
            for item in snapshot_writes
        )
        return FrozenLiveUpdate(
            phase=phase,
            round_index=round_index,
            snapshot_writes=frozen_writes,
            round_snapshot=_freeze_json_object(snapshot),
            journal_rows=tuple(_freeze_json_object(row) for row in journal_rows),
            event=_freeze_json_object({"type": "nexus_live_checkpoint", "round": round_index, "phase": phase, "error": error, "population_size": len(population.candidates), "monitor_state": monitor_state}),
            latent_metadata=_freeze_json_object(latent_metadata),
            checkpoint_failure=_freeze_json_object(checkpoint_failure) if checkpoint_failure is not None else None,
        )

    def write(self, batch: FrozenLiveUpdate) -> None:
        """Write one immutable batch with full crash-atomic durability."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._persist_latent_metadata(batch.latent_metadata)
        if batch.snapshot_writes:
            NexusSnapshotTransaction(self.output_dir).commit(list(batch.snapshot_writes))
        if batch.checkpoint_failure is not None:
            atomic_write_json(
                self.output_dir / "error-checkpoint-persist-failure.json",
                batch.checkpoint_failure,
                sort_keys=True,
            )
        atomic_write_json(
            self.round_dir / f"round-{batch.round_index:04d}-{_safe_phase(batch.phase)}.json",
            batch.round_snapshot,
            sort_keys=True,
        )
        self._append_candidate_journal_rows(batch.journal_rows)
        self.append_event(batch.event)

    def append_event(self, event: dict[str, Any]) -> None:
        event_hash = stable_hash(event)
        if event_hash in self._event_hashes:
            return
        self._event_hashes.add(event_hash)
        self.event_store.append(event)

    def append_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.append_event(event)

    def _persist_latent_metadata(self, metadata: dict[str, Any]) -> None:
        ledger_raw = metadata.get(LATENT_LEDGER_METADATA_KEY)
        if ledger_raw:
            ref = self.latent_ledger_store.persist_ledger(LatentLedger.from_dict(ledger_raw))
            metadata["latent_ledger_ref"] = {
                "sidecar_schema": ref.get("sidecar_schema"),
                "path": ref.get("path") or ref.get("latent_ledger_cache_path"),
                "events_path": ref.get("latent_events_path"),
                "sha256": ref.get("sha256") or ref.get("ledger_hash"),
                "cursor": ref.get("cursor") or ref.get("ledger_cursor"),
                "events_total": ref.get("events_total"),
            }
        snapshot_raw = metadata.get(LATENT_POSTERIOR_SNAPSHOT_METADATA_KEY)
        if snapshot_raw:
            self.latent_ledger_store.persist_snapshot(snapshot_raw)

    def _prepare_latent_metadata(self) -> dict[str, Any]:
        metadata = _contract_metadata(self.contract)
        frozen = deepcopy(metadata)
        ledger_raw = frozen.get(LATENT_LEDGER_METADATA_KEY)
        if ledger_raw:
            ledger = LatentLedger.from_dict(ledger_raw)
            payload = ledger.to_dict()
            cursor = max((int(event.sequence or 0) for event in ledger.events), default=0)
            metadata["latent_ledger_ref"] = {
                "sidecar_schema": "latent-ledger-sidecar/v1",
                "path": str(self.latent_ledger_store.ledger_cache_path),
                "events_path": str(self.latent_ledger_store.event_store.path),
                "sha256": stable_hash(payload),
                "cursor": cursor,
                "events_total": len(ledger.events),
            }
        return frozen

    def _append_candidate_journal_rows(self, rows: tuple[dict[str, Any], ...]) -> None:
        if not rows:
            return
        path = self.output_dir / "candidate-journal.jsonl"
        lock_path = path.with_name(path.name + ".lock")
        with file_lock(lock_path):
            with path.open("a", encoding="utf-8") as handle:
                for payload in rows:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_dir(path.parent)


def _safe_phase(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[:80] or "state"


def _freeze_json_object(value: Any) -> dict[str, Any]:
    frozen = json_ready(value)
    if not isinstance(frozen, dict):
        raise TypeError("live persistence payload must serialize to a JSON object")
    return frozen


def _contract_metadata(contract: Any | None) -> dict[str, Any]:
    if contract is None:
        return {}
    if isinstance(contract, dict):
        metadata = contract.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            contract["metadata"] = metadata
        return metadata
    metadata = getattr(contract, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    if hasattr(contract, "to_dict"):
        return coerce_dict(contract.to_dict().get("metadata"))
    return {}


def _fate_counts(population: CandidatePopulation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in population.candidates:
        fate = str(candidate.current_fate or "Unknown")
        counts[fate] = counts.get(fate, 0) + 1
    return counts


__all__ = ["FrozenLiveUpdate", "LiveNexusStore"]
