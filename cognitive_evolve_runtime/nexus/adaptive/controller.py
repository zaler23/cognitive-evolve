"""Runtime controller for the adaptive evidence control plane."""
from __future__ import annotations

import re
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import artifact_policy_contract_conflicts
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory, challenge_from_diagnostic
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure, latest_evidence_record
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
        controller = cls(config=config, state=AdaptiveRuntimeState.from_dict(restored_state))
        controller.record_contract_artifact_policy_conflicts(contract=contract)
        return controller

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
        self.state.record_event(adaptive_event("external_evaluator_summary", round=round_index, evaluated=evaluated, passed=passed, failed=failed, challenge_cases=challenge_summary.get("case_count", 0), targeted_resolution_rate=challenge_summary.get("targeted_resolution_rate", 0.0)))

    def compile_search_pressure(self, *, parent_id: str | None = None, scope: str = "global", parent: Any | None = None, candidates: list[Any] | None = None) -> SearchPressure | None:
        if not self.enabled:
            return None
        requirements = dict(self.config.evidence or {})
        base = self.challenge_memory.compile_search_pressure(parent_id=parent_id, scope=scope, artifact_requirements=requirements)
        if base is not None:
            overlap = _challenge_candidate_overlap(base, parent=parent, candidates=candidates or [])
            if overlap:
                base.metadata["challenge_candidate_overlap"] = overlap
        return base

    def record_contract_artifact_policy_conflicts(self, *, contract: Any | None) -> None:
        if not self.enabled or contract is None or not self.config.evidence:
            return
        diagnostics = list(artifact_policy_contract_conflicts(self.config.evidence, contract))
        metadata = getattr(contract, "metadata", None)
        if isinstance(metadata, dict):
            diagnostics.extend(str(item) for item in metadata.get("contract_artifact_policy_conflict_diagnostics", []) if item)
        diagnostics = list(dict.fromkeys(str(item) for item in diagnostics if str(item).strip()))
        if not diagnostics:
            return
        challenge_items = [
            challenge_from_diagnostic(
                candidate_id="__contract__",
                source="artifact_contract_policy",
                diagnostic=diagnostic,
                round_index=int(self.state.round_index or 0),
                priority=0.95,
            )
            for diagnostic in diagnostics[:8]
        ]
        record = EvidenceRecord(
            candidate_id="__contract__",
            source="artifact_contract_policy",
            stage="contract",
            score=0.0,
            confidence=0.9,
            final_blocked=True,
            parent_blocked=False,
            terminal_reject=False,
            repair_value=0.8,
            continuation_value=0.8,
            emitted_challenge_ids=[str(item.get("id")) for item in challenge_items if item.get("id")],
            diagnostics=diagnostics,
            hints=["align the model-generated artifact with the explicit ArtifactPolicy dynamic contract"],
            metadata={"challenge_items": challenge_items, "category": "contract_artifact_policy_conflict"},
        )
        self.challenge_memory.ingest(record, round_index=int(self.state.round_index or 0))
        self.state.challenge_memory = self.challenge_memory.to_dict()
        self.state.metrics["contract_artifact_policy_conflict_count"] = len(diagnostics)

    def selection_advisory_features(self, *, candidates: list[Any], policy: Any | None = None, contract: Any | None = None, world: Any | None = None) -> dict[str, Any]:
        return {}

    def before_final_projection(self, *, candidates: list[Any], final_certificate: dict[str, Any] | None = None) -> None:
        return None

    def set_verification_plan(self, plan: Any | None) -> None:
        if plan is None:
            return
        if hasattr(plan, "to_dict"):
            data = plan.to_dict()
        elif isinstance(plan, dict):
            data = dict(plan)
        else:
            return
        self.state.verification_plan = data
        if self.enabled:
            self.state.record_event({"event": "verification_plan_set"})

    def verification_plan_dict(self) -> dict[str, Any]:
        return dict(self.state.verification_plan or {})

    def verification_cache(self) -> dict[str, dict[str, Any]]:
        return self.state.verification_cache

    def update_verification_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        self.state.verification_cache = {str(k): dict(v) for k, v in dict(cache or {}).items() if isinstance(v, dict)}

    def record_verification_plan_resynthesized(self, *, reason: str = "") -> None:
        if self.enabled:
            self.state.record_event({"event": "verification_plan_resynthesized", "reason": str(reason or "")})

    def final_gate_directives(self) -> list[dict[str, Any]]:
        return []

    def budget_directive_features(self) -> list[dict[str, Any]]:
        return []

    def archive_directive_features(self) -> list[dict[str, Any]]:
        return []

    def context_transform_features(self) -> list[dict[str, Any]]:
        return []

    def candidate_transform_features(self) -> list[dict[str, Any]]:
        return []

    def verification_obligation_features(self) -> list[dict[str, Any]]:
        return []

    def effect_consumed(self, channel: str, item: Any, *, key: str | None = None) -> bool:
        return True

    def record_effect_application(self, *, channel: str, item: Any, changed: bool, consumer: str, reason: str = "", result: dict[str, Any] | None = None, key: str | None = None, consume: bool = True) -> dict[str, Any]:
        if not self.enabled:
            return {}
        record = {"channel": str(channel), "changed": bool(changed), "consumer": str(consumer), "reason": str(reason or "")}
        if result:
            record["result"] = dict(result)
        self.state.effect_applications.append(record)
        self.state.effect_applications = self.state.effect_applications[-200:]
        return record

    def record_generated_targets(self, *, candidate_id: str, challenge_ids: list[str], pressure_id: str, round_index: int) -> None:
        return None

    def record_evaluated_targets(self, *, candidate_id: str, challenge_ids: list[str], record: Any | None, round_index: int) -> None:
        return None

    def record_resolved_targets(self, *, candidate_id: str, challenge_ids: list[str], record: Any | None, round_index: int) -> None:
        return None

    def record_honesty_control_signal(self, signal: Any, *, history_limit: int) -> None:
        data = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal or {})
        if not data:
            return
        item = {
            "signal_id": str(data.get("signal_id") or ""),
            "round_index": int(self.state.round_index or 0),
            "sample_count": int(data.get("sample_count") or 0),
            "error_vector": coerce_dict(data.get("error_vector")),
            "pressure": coerce_dict(data.get("pressure")),
        }
        self.state.honesty_error_history.append(item)
        self.state.honesty_error_history = self.state.honesty_error_history[-max(1, int(history_limit or 1)) :]
        self.state.metrics["honesty_control_sample_count"] = item["sample_count"]
        self.state.metrics["honesty_control_frontier_pressure"] = float(item["pressure"].get("frontier_exploration_pressure", 0.0) or 0.0)
        self.state.record_event(adaptive_event("honesty_control_signal", **item))

    def record_cell_activation_map(self, activation_map: dict[str, Any], *, round_index: int, history_limit: int) -> None:
        data = coerce_dict(activation_map)
        if not data:
            return
        item = {
            "round_index": int(round_index or 0),
            "cell_count": len(data),
            "cell_activation_map": data,
        }
        self.state.cell_activation_history.append(item)
        self.state.cell_activation_history = self.state.cell_activation_history[-max(1, int(history_limit or 1)) :]
        self.state.metrics["cell_activation_cell_count"] = len(data)
        self.state.record_event(adaptive_event("cell_activation_map", round=round_index, cell_count=len(data)))


    def record_canonical_family_metrics(self, metrics: dict[str, Any], *, round_index: int) -> None:
        data = coerce_dict(metrics)
        if not data:
            return
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                # Strip a self-describing canonical_family_ prefix so the namespaced
                # key never doubles up into canonical_family_canonical_family_*.
                metric_key = key[len("canonical_family_"):] if key.startswith("canonical_family_") else key
                self.state.metrics[f"canonical_family_{metric_key}"] = value
        self.state.record_event(adaptive_event("canonical_family_metrics", round=round_index, **data))


    def attach_final_certificate(self, certificate: dict[str, Any]) -> None:
        if not self.enabled and not certificate:
            return
        self.state.final_certificate = dict(certificate)
        if certificate:
            self.state.record_event(adaptive_event("final_certificate", objective_solved=bool(certificate.get("objective_solved")), blocking_reasons=certificate.get("blocking_reasons", [])))

    def to_dict(self) -> dict[str, Any]:
        self.state.challenge_memory = self.challenge_memory.to_dict()
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


def _challenge_candidate_overlap(pressure: SearchPressure, *, parent: Any | None, candidates: list[Any]) -> dict[str, Any]:
    challenge_terms = _overlap_terms(
        " ".join([*(str(item.get("summary") or item.get("challenge_id") or "") for item in pressure.success_criteria), pressure.mutation_instruction])
    )
    selected = [item for item in [parent, *list(candidates or [])] if item is not None]
    if not challenge_terms or not selected:
        return {}
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in selected:
        candidate_id = str(getattr(candidate, "id", "") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        shared = sorted(challenge_terms & _candidate_overlap_terms(candidate))
        matches.append({"candidate_id": candidate_id, "shared_terms": shared[:12], "overlap_score": bounded_score(len(shared) / max(1, len(challenge_terms)))})
    return {"schema": "challenge-candidate-overlap/v1", "advisory_only": True, "matches": matches}


def _candidate_overlap_terms(candidate: Any) -> set[str]:
    return _overlap_terms(
        " ".join(
            str(value or "")
            for value in (
                getattr(candidate, "id", ""),
                getattr(candidate, "artifact_type", ""),
                getattr(candidate, "concise_claim", ""),
                getattr(candidate, "core_mechanism", ""),
                getattr(candidate, "artifact", ""),
            )
        )
    )


def _overlap_terms(value: Any) -> set[str]:
    return {item.lower() for item in re.findall(r"[\w\-]{3,}", str(value or ""), flags=re.UNICODE)}


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
