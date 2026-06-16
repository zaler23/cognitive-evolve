"""Runtime controller for the adaptive evidence control plane."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidatePopulation
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory
from cognitive_evolve_runtime.evaluators.evidence import SearchPressure, latest_evidence_record
from cognitive_evolve_runtime.nexus.adaptive.research import ResearchContext, ResearchExtensionRegistry
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.adaptive.config import AdaptiveConfig
from cognitive_evolve_runtime.nexus.adaptive.spatial_population import SpatialPopulationState, build_or_update_spatial_state
from cognitive_evolve_runtime.nexus.adaptive.state import AdaptiveRuntimeState
from cognitive_evolve_runtime.nexus.adaptive.telemetry import adaptive_event


class AdaptiveRuntimeController:
    def __init__(self, *, config: AdaptiveConfig, state: AdaptiveRuntimeState | None = None) -> None:
        self.config = config
        self.state = AdaptiveRuntimeState.from_dict(state)
        self.state.config = config.to_dict()
        self.state.enabled_features = dict(config.enabled_features)
        self._spatial_state = SpatialPopulationState.from_dict(self.state.spatial)
        self.challenge_memory = ChallengeMemory.from_dict(self.state.challenge_memory)
        self.research_registry = ResearchExtensionRegistry.from_config(config.research, restored_state=self.state.research_extensions)

    @classmethod
    def from_sources(
        cls,
        *,
        explicit: dict[str, Any] | None = None,
        restored_state: dict[str, Any] | None = None,
        contract: Any | None = None,
        policy: Any | None = None,
        world: Any | None = None,
    ) -> "AdaptiveRuntimeController":
        restored_payload = coerce_dict(restored_state)
        restored_config = coerce_dict(restored_payload.get("config"))
        if restored_config and explicit:
            restored_config = _deep_merge(restored_config, coerce_dict(explicit))
        elif explicit:
            restored_config = coerce_dict(explicit)
        config = AdaptiveConfig.from_sources(explicit=restored_config or None, contract=contract, policy=policy, world=world)
        return cls(config=config, state=AdaptiveRuntimeState.from_dict(restored_state))

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def evaluator_enabled(self) -> bool:
        return bool(self.config.enabled_features.get("external_evaluator"))

    def begin_round(self, *, round_index: int) -> AdaptiveRuntimeState:
        self.state.round_index = int(round_index or 0)
        self.state.enabled_features = dict(self.config.enabled_features)
        if self.enabled:
            self.state.record_event(adaptive_event("adaptive_round_begin", round=round_index, features=self.state.enabled_features))
            self._sync_research_state()
        return self.state

    def observe_population(self, *, population: CandidatePopulation, round_index: int) -> None:
        if not self.enabled or not self.config.spatial.enabled:
            return
        self._spatial_state = build_or_update_spatial_state(
            population.candidates,
            existing=self._spatial_state,
            round_index=round_index,
            width=self.config.spatial.width,
            height=self.config.spatial.height,
            region_size=self.config.spatial.region_size,
            neighborhood=self.config.spatial.neighborhood,
            toroidal=self.config.spatial.toroidal,
        )
        self.state.spatial = self._spatial_state.to_dict()
        self.state.metrics["spatial_region_count"] = len(self._spatial_state.regions)
        self.state.metrics["spatial_candidate_count"] = len(self._spatial_state.candidate_to_coord)
        self.state.record_event(adaptive_event("spatial_observe", round=round_index, candidate_count=len(self._spatial_state.candidate_to_coord), region_count=len(self._spatial_state.regions)))

    def record_evaluator_summary(self, *, round_index: int, evaluated: int, passed: int, failed: int, candidates: list[Any] | None = None) -> None:
        if not self.enabled:
            return
        for candidate in candidates or []:
            record = latest_evidence_record(candidate)
            if record is None:
                continue
            spatial = getattr(candidate, "metadata", {}).get("spatial", {}) if isinstance(getattr(candidate, "metadata", None), dict) else {}
            self.challenge_memory.ingest(
                record,
                round_index=round_index,
                candidate_fate=CandidateFate.normalize(getattr(candidate, "current_fate", "")),
                lineage_id=str(getattr(candidate, "lineage", [getattr(candidate, "id", "")])[-1] if getattr(candidate, "lineage", None) else getattr(candidate, "id", "")),
                region_id=str(spatial.get("region_id") or "") if isinstance(spatial, dict) else "",
            )
            if record.target_challenge_ids:
                self.challenge_memory.mark_targeted(getattr(candidate, "id", ""), record.target_challenge_ids)
                self.record_evaluated_targets(candidate_id=getattr(candidate, "id", ""), challenge_ids=record.target_challenge_ids, record=record, round_index=round_index)
            if record.resolved_challenge_ids:
                self.challenge_memory.mark_resolved(getattr(candidate, "id", ""), record.resolved_challenge_ids)
                self.record_resolved_targets(candidate_id=getattr(candidate, "id", ""), challenge_ids=record.resolved_challenge_ids, record=record, round_index=round_index)
            auto_resolved = self.challenge_memory.mark_schema_resolved_from_record(record)
            if auto_resolved:
                _attach_auto_resolved_schema_challenges(candidate, auto_resolved)
                self.record_resolved_targets(candidate_id=getattr(candidate, "id", ""), challenge_ids=auto_resolved, record=record, round_index=round_index)
        self.state.challenge_memory = self.challenge_memory.to_dict()
        challenge_summary = self.challenge_memory.summary(limit=12)
        self.state.evaluator = {
            "enabled": self.evaluator_enabled,
            "last_round": int(round_index or 0),
            "evaluated_candidates": int(evaluated or 0),
            "passed_candidates": int(passed or 0),
            "failed_candidates": int(failed or 0),
            "challenge_memory": challenge_summary,
        }
        self.state.metrics["evaluator_evaluated_candidates"] = int(evaluated or 0)
        self.state.metrics["evaluator_passed_candidates"] = int(passed or 0)
        self.state.metrics["challenge_case_count"] = int(challenge_summary.get("case_count") or 0)
        self.state.metrics["challenge_case_targeted_resolution_rate"] = float(challenge_summary.get("targeted_resolution_rate") or 0.0)
        self.state.metrics["targeted_challenge_resolution_rate"] = float(challenge_summary.get("targeted_resolution_rate") or 0.0)
        ctx = self._research_context(round_index=round_index, candidates=candidates or [])
        self.research_registry.after_evidence(ctx)
        self._sync_research_state()
        self.state.record_event(adaptive_event("external_evaluator_summary", round=round_index, evaluated=evaluated, passed=passed, failed=failed, challenge_cases=challenge_summary.get("case_count", 0), targeted_resolution_rate=challenge_summary.get("targeted_resolution_rate", 0.0)))

    def compile_search_pressure(self, *, parent_id: str | None = None, scope: str = "global", parent: Any | None = None, candidates: list[Any] | None = None) -> SearchPressure | None:
        if not self.enabled:
            return None
        requirements = dict(self.config.evidence or {})
        base = self.challenge_memory.compile_search_pressure(parent_id=parent_id, scope=scope, artifact_requirements=requirements)
        ctx = self._research_context(round_index=self.state.round_index, candidates=candidates or [], parent=parent)
        self.research_registry.before_mutation_planning(ctx)
        pressures = self.research_registry.pending_search_pressures(parent_id=parent_id)
        pressure = _merge_pressures(base, pressures, parent_id=parent_id, scope=scope)
        self._sync_research_state()
        return pressure

    def research_advisory_features(self, *, candidates: list[Any], policy: Any | None = None, contract: Any | None = None, world: Any | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        ctx = self._research_context(round_index=self.state.round_index, candidates=candidates, policy=policy, contract=contract, world=world)
        signal = self.research_registry.before_parent_selection(ctx)
        self._sync_research_state()
        return dict(signal.selection_advisory)

    def before_final_projection(self, *, candidates: list[Any], final_certificate: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        ctx = self._research_context(round_index=self.state.round_index, candidates=candidates, final_certificate=final_certificate or {})
        self.research_registry.before_final_projection(ctx)
        self._sync_research_state()

    def final_gate_directives(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        current_round = int(self.state.round_index or 0)
        return [
            dict(item)
            for item in self.research_registry.state.final_gate_directives
            if isinstance(item, dict) and _int_or_default(item.get("round_index"), -1) == current_round
        ]

    def record_generated_targets(self, *, candidate_id: str, challenge_ids: list[str], pressure_id: str, round_index: int) -> None:
        self.research_registry.record_generated_targets(candidate_id=candidate_id, challenge_ids=challenge_ids, pressure_id=pressure_id, round_index=round_index)
        self._sync_research_state()

    def record_evaluated_targets(self, *, candidate_id: str, challenge_ids: list[str], record: Any | None, round_index: int) -> None:
        self.research_registry.record_evaluated_targets(candidate_id=candidate_id, challenge_ids=challenge_ids, record=record, round_index=round_index)
        self._sync_research_state()

    def record_resolved_targets(self, *, candidate_id: str, challenge_ids: list[str], record: Any | None, round_index: int) -> None:
        self.research_registry.record_resolved_targets(candidate_id=candidate_id, challenge_ids=challenge_ids, record=record, round_index=round_index)
        self._sync_research_state()


    def _research_context(self, *, round_index: int, candidates: list[Any] | None = None, parent: Any | None = None, policy: Any | None = None, contract: Any | None = None, world: Any | None = None, final_certificate: dict[str, Any] | None = None) -> ResearchContext:
        return ResearchContext(
            round_index=int(round_index or 0),
            candidates=list(candidates or []),
            challenge_memory=self.challenge_memory,
            adaptive_state=self.state,
            spatial_state=self._spatial_state,
            config=self.config.to_dict(),
            policy=policy,
            contract=contract,
            world=world,
            parent=parent,
            final_certificate=final_certificate or {},
        )

    def _sync_research_state(self) -> None:
        snapshot = self.research_registry.snapshot()
        self.state.research_extensions = snapshot
        self.state.research_metrics = dict(snapshot.get("metrics") or {})
        self.state.research_warnings = [str(item) for item in snapshot.get("warnings", []) if item]
        self.state.metrics.update({f"research.{k}": v for k, v in self.state.research_metrics.items()})
        if self.state.research_warnings:
            self.state.warnings = list(dict.fromkeys([*self.state.warnings, *self.state.research_warnings]))[-200:]

    def attach_final_certificate(self, certificate: dict[str, Any]) -> None:
        if not self.enabled and not certificate:
            return
        self.state.final_certificate = dict(certificate)
        if certificate:
            self.state.record_event(adaptive_event("final_certificate", objective_solved=bool(certificate.get("objective_solved")), blocking_reasons=certificate.get("blocking_reasons", [])))

    def to_dict(self) -> dict[str, Any]:
        self.state.challenge_memory = self.challenge_memory.to_dict()
        self._sync_research_state()
        return self.state.to_dict()


__all__ = ["AdaptiveRuntimeController"]


def _merge_pressures(base: SearchPressure | None, extras: list[SearchPressure], *, parent_id: str | None, scope: str) -> SearchPressure | None:
    selected = [pressure for pressure in ([base] if base is not None else []) + list(extras or []) if pressure is not None]
    if not selected:
        return None
    target_ids: list[str] = []
    avoid_ids: list[str] = []
    challenge_weights: dict[str, float] = {}
    success: list[dict[str, Any]] = []
    instructions: list[str] = []
    metadata: dict[str, Any] = {"merged_pressure_count": len(selected)}
    artifact_requirements: dict[str, Any] = {}
    for pressure in selected:
        target_ids.extend(pressure.target_challenge_ids)
        avoid_ids.extend(pressure.avoid_challenge_ids)
        challenge_weights.update(pressure.challenge_weights)
        success.extend(pressure.success_criteria)
        if pressure.mutation_instruction:
            instructions.append(pressure.mutation_instruction)
        artifact_requirements.update(pressure.artifact_requirements)
        metadata.setdefault("source_pressure_ids", []).append(pressure.id)
    return SearchPressure.from_parts(
        parent_id=parent_id,
        scope=scope,
        target_challenge_ids=list(dict.fromkeys(target_ids)),
        avoid_challenge_ids=list(dict.fromkeys(avoid_ids)),
        artifact_requirements=artifact_requirements,
        success_criteria=success,
        challenge_weights=challenge_weights,
        selection_advisory={},
        mutation_instruction="\n\n".join(dict.fromkeys(instructions)),
        metadata=metadata,
    )


def _attach_auto_resolved_schema_challenges(candidate: Any, challenge_ids: list[str]) -> None:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    resolved = list(dict.fromkeys([*(metadata.get("resolved_challenge_ids") or []), *challenge_ids]))
    metadata["resolved_challenge_ids"] = resolved
    state = dict(metadata.get("evidence_state") or {})
    targets = list(dict.fromkeys(state.get("target_challenge_ids") or metadata.get("target_challenge_ids") or []))
    state["target_challenge_ids"] = targets
    state["resolved_challenge_ids"] = list(dict.fromkeys([*(state.get("resolved_challenge_ids") or []), *challenge_ids]))
    if targets:
        state["challenge_resolution"] = bounded_score(len(set(state["resolved_challenge_ids"]) & set(targets)) / max(1, len(set(targets))))
    metadata["evidence_state"] = state
    records = metadata.get("evidence_records")
    if isinstance(records, list) and records:
        latest = dict(records[-1]) if isinstance(records[-1], dict) else {}
        latest["resolved_challenge_ids"] = list(dict.fromkeys([*(latest.get("resolved_challenge_ids") or []), *challenge_ids]))
        records[-1] = latest
        metadata["evidence_records"] = records
    candidate.metadata = metadata
    scores = dict(getattr(candidate, "multihead_scores", {}) or {})
    scores["challenge_resolution"] = bounded_score(state.get("challenge_resolution", 0.0))
    candidate.multihead_scores = scores


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged
