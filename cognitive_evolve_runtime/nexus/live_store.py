"""Live persistence for Nexus rounds and phases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json, file_lock
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.outcomes.latent_ledger import LatentLedger, LatentLedgerStore
from cognitive_evolve_runtime.persistence.archive_store import ArchiveStore
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.persistence.event_store import EventStore
from cognitive_evolve_runtime.persistence.population_store import PopulationStore

LATENT_LEDGER_METADATA_KEY = "latent_ledger"
LATENT_POSTERIOR_SNAPSHOT_METADATA_KEY = "latent_posterior_snapshot"


class LiveNexusStore:
    """Persist a recoverable Nexus state after every meaningful phase."""

    def __init__(self, output_dir: str | Path, *, mode: str, contract: Any, world: Any, max_rounds: int, budget: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.contract = contract
        self.world = world
        self.max_rounds = int(max_rounds)
        self.budget = dict(budget or {"max_rounds": self.max_rounds})
        self.population_store = PopulationStore(self.output_dir / "population.json")
        self.archive_store = ArchiveStore(self.output_dir / "archives.json")
        self.event_store = EventStore(self.output_dir / "events.jsonl")
        self.latent_ledger_store = LatentLedgerStore(self.output_dir)
        self.checkpoint_store = CheckpointStore(self.output_dir / "checkpoint.json")
        self.round_dir = self.output_dir / "rounds"
        self.round_dir.mkdir(parents=True, exist_ok=True)
        self._event_hashes: set[str] = set()

    def __call__(self, update: dict[str, Any]) -> None:
        population = update.get("population")
        archives = update.get("archives")
        if not isinstance(population, CandidatePopulation) or not isinstance(archives, ArchiveManager):
            return
        phase = str(update.get("phase") or "state")
        round_index = int(update.get("round") or 0)
        policy = update.get("policy")
        diagnosis = update.get("diagnosis")
        progress_event = dict(update.get("progress_event") or {})
        budget_history = [dict(item) for item in update.get("budget_history", []) if isinstance(item, dict)]
        adaptive_state = dict(update.get("adaptive_state") or {}) if isinstance(update.get("adaptive_state"), dict) else {}
        error = update.get("error") if isinstance(update.get("error"), dict) else None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.population_store.save(population)
        self.archive_store.save(archives)
        budget_payload = dict(self.budget)
        budget_payload["current_round"] = round_index
        budget_payload["max_rounds"] = self.max_rounds
        if progress_event.get("max_rounds"):
            budget_payload["round_limit"] = int(progress_event.get("max_rounds") or self.max_rounds)
        allow_round_repair = phase == "error_checkpoint"
        checkpoint_written = False
        try:
            self.checkpoint_store.save_state(
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
                allow_progress_round_repair=allow_round_repair,
            )
            checkpoint_written = True
        except Exception as exc:
            if not allow_round_repair:
                raise
            error = dict(error or {})
            error["checkpoint_persist_error"] = f"{exc.__class__.__name__}: {exc}"
            atomic_write_json(
                self.output_dir / "error-checkpoint-persist-failure.json",
                {
                    "phase": phase,
                    "round": round_index,
                    "at": utc_now(),
                    "error": error,
                    "progress_event": progress_event,
                },
                sort_keys=True,
            )
        if checkpoint_written:
            self._persist_latent_metadata()
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
        }
        atomic_write_json(self.round_dir / f"round-{round_index:04d}-{_safe_phase(phase)}.json", snapshot, sort_keys=True)
        self._append_candidate_journal(round_index, phase, population)
        self.append_event({"type": "nexus_live_checkpoint", "round": round_index, "phase": phase, "error": error, "population_size": len(population.candidates)})

    def append_event(self, event: dict[str, Any]) -> None:
        event_hash = stable_hash(event)
        if event_hash in self._event_hashes:
            return
        self._event_hashes.add(event_hash)
        self.event_store.append(event)

    def append_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.append_event(event)

    def _persist_latent_metadata(self) -> None:
        metadata = _contract_metadata(self.contract)
        ledger_raw = metadata.get(LATENT_LEDGER_METADATA_KEY)
        if ledger_raw:
            self.latent_ledger_store.persist_ledger(LatentLedger.from_dict(ledger_raw))
        snapshot_raw = metadata.get(LATENT_POSTERIOR_SNAPSHOT_METADATA_KEY)
        if snapshot_raw:
            self.latent_ledger_store.persist_snapshot(snapshot_raw)

    def _append_candidate_journal(self, round_index: int, phase: str, population: CandidatePopulation) -> None:
        path = self.output_dir / "candidate-journal.jsonl"
        lock_path = path.with_name(path.name + ".lock")
        with file_lock(lock_path):
            with path.open("a", encoding="utf-8") as handle:
                for candidate in population.candidates:
                    payload = {
                        "at": utc_now(),
                        "round": round_index,
                        "phase": phase,
                        "id": candidate.id,
                        "generation": candidate.generation,
                        "fate": candidate.current_fate,
                        "parents": candidate.parent_ids,
                        "core_mechanism": candidate.core_mechanism,
                        "concise_claim": candidate.concise_claim,
                        "genome_hash": candidate.genome_hash,
                        "scores": candidate.multihead_scores,
                        "search_seed_not_final": bool(candidate.metadata.get("search_seed_not_final")),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _safe_phase(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[:80] or "state"


def _contract_metadata(contract: Any | None) -> dict[str, Any]:
    if contract is None:
        return {}
    if isinstance(contract, dict):
        return coerce_dict(contract.get("metadata"))
    metadata = getattr(contract, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    if hasattr(contract, "to_dict"):
        return coerce_dict(contract.to_dict().get("metadata"))
    return {}


__all__ = ["LiveNexusStore"]
