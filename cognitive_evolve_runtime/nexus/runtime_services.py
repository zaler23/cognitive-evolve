"""Side-effect services used by :mod:`nexus.runtime`.

The Nexus runtime remains the orchestration authority, but persistence and
project verification are intentionally isolated here so they can be tested and
reasoned about without reading the whole runtime entrypoint.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.consistency import assert_runtime_consistency
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopResult
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.project_verification import ProjectCandidateVerifier, ProjectVerificationSummary
from cognitive_evolve_runtime.persistence.checkpoint import build_checkpoint_state
from cognitive_evolve_runtime.persistence.event_store import EventStore
from cognitive_evolve_runtime.persistence.transactional_snapshot import NexusSnapshotTransaction, SnapshotWrite


class NexusProjectVerificationService:
    """Run project patch verification without making ``NexusRuntime`` own it."""

    def __init__(self, *, output_dir: Path | None) -> None:
        self.output_dir = output_dir

    def verify_population(
        self,
        snapshot: ProjectSnapshot,
        candidates: list[Any],
        *,
        include_tests: bool = False,
    ) -> list[ProjectVerificationSummary]:
        source_root = Path(snapshot.root_path)
        sandbox_root = (self.output_dir / "patch-sandboxes") if self.output_dir is not None else Path(tempfile.mkdtemp(prefix="cogev-nexus-sandboxes-"))
        verifier = ProjectCandidateVerifier(source_root=source_root, sandbox_root=sandbox_root, include_tests=include_tests)
        patch_candidates = [candidate for candidate in candidates if has_concrete_patch_payload(candidate)]
        return verifier.verify_population(patch_candidates)


class NexusPersistenceService:
    """Persist a final or interrupted Nexus run with one coherent snapshot."""

    def __init__(self, *, output_dir: Path | None) -> None:
        self.output_dir = output_dir

    def persist(
        self,
        run: Any,
        result: EvolutionLoopResult,
        *,
        contract: Any,
        world: Any,
        budget_history: list[dict[str, Any]],
        budget: EvolutionBudget | None = None,
    ) -> dict[str, str]:
        _sync_budget_width_metadata(run.evolution, budget)
        if self.output_dir is None:
            return {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        population_path = self.output_dir / "population.json"
        archive_path = self.output_dir / "archives.json"
        event_path = self.output_dir / "events.jsonl"
        checkpoint_path = self.output_dir / "checkpoint.json"
        run_result_path = self.output_dir / "run-result.json"

        progress_event = result.progress_events[-1] if result.progress_events else {}
        if result.interrupted:
            progress_event = checkpoint_progress_event_for_interruption(progress_event, result.current_round)
        event_store = EventStore(event_path)
        fallback_events = [dict(event) for event in (run.evolution.get("fallback_events") or []) if isinstance(event, dict)]
        adaptive_state = dict(getattr(result, "adaptive_state", {}) or {})
        adaptive_events = [dict(event) for event in adaptive_state.get("events", []) if isinstance(event, dict)]
        events_to_write = list(result.pipeline_events) + list(result.progress_events) + fallback_events + adaptive_events
        if result.interrupted and progress_event and not any(
            isinstance(event, dict) and event.get("type") == "evolution_progress" and int(event.get("round") or 0) == int(progress_event.get("round") or 0)
            for event in events_to_write
        ):
            events_to_write.append(progress_event)
        event_store.append_many_once(events_to_write)

        checkpoint_round = int(result.current_round if result.interrupted else (progress_event.get("round", 0) or result.current_round or 0))
        checkpoint_max_rounds = int(progress_event.get("max_rounds", 0) or result.max_rounds or checkpoint_round)
        checkpoint = build_checkpoint_state(
            round=checkpoint_round,
            max_rounds=checkpoint_max_rounds,
            population=result.population,
            archives=result.archives,
            policy=result.policy,
            diagnosis=result.diagnosis,
            progress_event=progress_event,
            contract=contract,
            world=world,
            mode=run.mode,
            budget_history=budget_history or [],
            budget=_checkpoint_budget_payload(budget, checkpoint_round=checkpoint_round),
            adaptive_state=adaptive_state,
            allow_progress_round_repair=bool(result.interrupted),
        )
        adaptive_dir = self.output_dir / "adaptive"
        final_certificate = dict((result.synthesis.closure_certificate or {}).get("final_certificate") or adaptive_state.get("final_certificate") or {})
        final_projection: dict[str, Any] | None = None
        if adaptive_state:
            candidates = getattr(result.population, "candidates", []) if result.population is not None else []
            has_evidence = any(isinstance(getattr(candidate, "metadata", None), dict) and (candidate.metadata.get("evidence_state") or candidate.metadata.get("evidence_records")) for candidate in candidates)
            if final_certificate or has_evidence:
                final_projection = build_final_projection(
                    population=result.population,
                    synthesis=result.synthesis,
                    final_certificate=final_certificate,
                ).to_dict()

        artifacts = {
            "population": str(population_path),
            "archives": str(archive_path),
            "events": str(event_path),
            "checkpoint": str(checkpoint_path),
            "final_answer": str(self.output_dir / "final-answer.md"),
            "run_result": str(run_result_path),
            "snapshot_transaction": str(self.output_dir / "snapshot-transaction.json"),
        }
        if adaptive_state:
            artifacts.update(
                {
                    "adaptive_state": str(adaptive_dir / "adaptive-state.json"),
                    "adaptive_events": str(adaptive_dir / "adaptive-events.jsonl"),
                    "final_certificate": str(adaptive_dir / "final-certificate.json"),
                    "final_projection": str(adaptive_dir / "final-projection.json"),
                    "challenge_memory": str(self.output_dir / "challenge-memory.json"),
                    "challenge_events": str(self.output_dir / "challenge-events.jsonl"),
                }
            )
        run.artifacts = dict(artifacts)
        writes = [
            SnapshotWrite("population.json", "json", result.population.to_dict()),
            SnapshotWrite("archives.json", "json", result.archives.to_dict()),
            SnapshotWrite("checkpoint.json", "json", checkpoint.to_dict()),
            SnapshotWrite("final-answer.md", "text", final_answer_artifact_text(result) + "\n", sort_keys=False),
            SnapshotWrite("run-result.json", "json", run.to_dict()),
        ]
        if adaptive_state:
            writes.extend(_adaptive_snapshot_writes(adaptive_state, final_certificate, final_projection=final_projection))
        NexusSnapshotTransaction(self.output_dir).commit(writes)
        if adaptive_events:
            adaptive_events_path = adaptive_dir / "adaptive-events.jsonl"
            adaptive_events_path.parent.mkdir(parents=True, exist_ok=True)
            adaptive_events_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n" for event in adaptive_events),
                encoding="utf-8",
            )
        assert_runtime_consistency(checkpoint=checkpoint.to_dict(), events=event_store.read_all(), nexus_data=run.to_dict())
        return artifacts


def has_concrete_patch_payload(candidate: Any) -> bool:
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list) and patch_set:
        return True
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        return any(isinstance(artifact.get(key), str) and artifact.get(key).strip() for key in ("patch", "patch_content", "diff", "unified_diff"))
    return False


def checkpoint_progress_event_for_interruption(progress_event: dict[str, Any], current_round: int) -> dict[str, Any]:
    event = dict(progress_event or {})
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


def final_answer_artifact_text(result: EvolutionLoopResult) -> str:
    status = str(result.completion_status or result.stop_reason or "completed")
    reference_candidate_id = str(getattr(result.synthesis, "reference_candidate_id", "") or "")
    header = [
        f"# CognitiveEvolve result: {status}",
        "",
        f"- stop_reason: {result.stop_reason or 'unknown'}",
        "- result_type: candidate_final_output",
        "- correctness_verdict: external_validation_required",
        f"- continuation_available: {str(result.completion_status in {'needs_continuation', 'interrupted_checkpointed', 'paused_quota'}).lower()}",
        "- project_correctness_claim: not_claimed",
        f"- reference_candidate_id: {reference_candidate_id or 'none'}",
        "",
    ]
    final_certificate = dict((getattr(result.synthesis, "closure_certificate", {}) or {}).get("final_certificate") or {})
    population = getattr(result, "population", None)
    candidates = getattr(population, "candidates", []) if population is not None else []
    has_evidence = any(isinstance(getattr(candidate, "metadata", None), dict) and (candidate.metadata.get("evidence_state") or candidate.metadata.get("evidence_records")) for candidate in candidates)
    if population is not None and (final_certificate or has_evidence):
        projection = build_final_projection(population=population, synthesis=result.synthesis, final_certificate=final_certificate)
        return "\n".join(header) + projection.to_markdown()
    if reference_candidate_id and result.completion_status != "solved":
        header.extend(
            [
                "> The displayed candidate is the runtime's final candidate output. Correctness must be judged by a human reviewer or an external verifier; the project does not self-certify it as correct.",
                "",
            ]
        )
    return "\n".join(header) + str(result.synthesis.final_answer or "")


def _checkpoint_budget_payload(budget: EvolutionBudget | None, *, checkpoint_round: int) -> dict[str, Any]:
    payload = budget.to_dict() if budget is not None else {}
    if payload:
        payload["current_round"] = int(checkpoint_round or 0)
    return payload


def _sync_budget_width_metadata(evolution: dict[str, Any], budget: EvolutionBudget | None) -> None:
    if budget is None:
        return
    metadata = dict(evolution.get("runtime_metadata") or {})
    round_budget = dict(metadata.get("round_budget") or evolution.get("round_budget") or {})
    if round_budget:
        round_budget["mutation_branches_per_round"] = int(getattr(budget, "branch_factor", 0) or 0)
        round_budget["branch_factor_source"] = "explicit_or_policy_derived"
        metadata["round_budget"] = round_budget
        evolution["runtime_metadata"] = metadata
    runtime = dict(evolution.get("round_budget_runtime") or {})
    runtime["mutation_branches_per_round"] = int(getattr(budget, "branch_factor", 0) or 0)
    evolution["round_budget_runtime"] = runtime


def _adaptive_snapshot_writes(adaptive_state: dict[str, Any], final_certificate: dict[str, Any], *, final_projection: dict[str, Any] | None = None) -> list[SnapshotWrite]:
    spatial = dict(adaptive_state.get("spatial") or {})
    regions = dict(spatial.get("regions") or {})
    writes = [
        SnapshotWrite("adaptive/adaptive-state.json", "json", adaptive_state),
        SnapshotWrite("adaptive/final-certificate.json", "json", final_certificate or {}),
        SnapshotWrite("adaptive/final-projection.json", "json", final_projection or {}),
    ]
    challenge_memory = dict(adaptive_state.get("challenge_memory") or {})
    if challenge_memory:
        writes.append(SnapshotWrite("challenge-memory.json", "json", challenge_memory))
        writes.append(SnapshotWrite("challenge-events.jsonl", "text", _challenge_events_jsonl(challenge_memory), sort_keys=False))
    if spatial:
        writes.append(SnapshotWrite("adaptive/spatial-topology.json", "json", spatial))
        writes.append(SnapshotWrite("adaptive/spatial-regions.json", "json", regions))
    evaluator = dict(adaptive_state.get("evaluator") or {})
    if evaluator:
        writes.append(SnapshotWrite("adaptive/evaluator-results.jsonl", "text", json.dumps(evaluator, ensure_ascii=False, sort_keys=True, default=str) + "\n", sort_keys=False))
    return writes


def _challenge_events_jsonl(challenge_memory: dict[str, Any]) -> str:
    items = dict(challenge_memory.get("items") or {})
    return "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) + "\n" for item in items.values() if isinstance(item, dict))


__all__ = [
    "NexusPersistenceService",
    "NexusProjectVerificationService",
    "checkpoint_progress_event_for_interruption",
    "final_answer_artifact_text",
    "has_concrete_patch_payload",
]
