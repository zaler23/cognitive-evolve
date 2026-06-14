"""Runtime controller for the adaptive evidence layer."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.nexus.adaptive.config import AdaptiveConfig
from cognitive_evolve_runtime.nexus.adaptive.spatial_population import SpatialPopulationState, build_or_update_spatial_state
from cognitive_evolve_runtime.nexus.adaptive.state import AdaptiveRuntimeState
from cognitive_evolve_runtime.nexus.adaptive.telemetry import adaptive_event


class AdaptiveRuntimeController:
    def __init__(self, *, config: AdaptiveConfig, state: AdaptiveRuntimeState | None = None) -> None:
        self.config = config
        self.state = AdaptiveRuntimeState.from_dict(state)
        self.state.enabled_features = dict(config.enabled_features)
        self._spatial_state = SpatialPopulationState.from_dict(self.state.spatial)

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
        config = AdaptiveConfig.from_sources(explicit=explicit, contract=contract, policy=policy, world=world)
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

    def record_evaluator_summary(self, *, round_index: int, evaluated: int, passed: int, failed: int) -> None:
        if not self.enabled:
            return
        self.state.evaluator = {
            "enabled": self.evaluator_enabled,
            "last_round": int(round_index or 0),
            "evaluated_candidates": int(evaluated or 0),
            "passed_candidates": int(passed or 0),
            "failed_candidates": int(failed or 0),
        }
        self.state.metrics["evaluator_evaluated_candidates"] = int(evaluated or 0)
        self.state.metrics["evaluator_passed_candidates"] = int(passed or 0)
        self.state.record_event(adaptive_event("external_evaluator_summary", round=round_index, evaluated=evaluated, passed=passed, failed=failed))

    def attach_final_certificate(self, certificate: dict[str, Any]) -> None:
        if not self.enabled and not certificate:
            return
        self.state.final_certificate = dict(certificate)
        if certificate:
            self.state.record_event(adaptive_event("final_certificate", objective_solved=bool(certificate.get("objective_solved")), blocking_reasons=certificate.get("blocking_reasons", [])))

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()


__all__ = ["AdaptiveRuntimeController"]
