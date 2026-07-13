"""Safe advisory-only M6 theory layer."""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Callable

from .aggregator import aggregate_advisory_features
from .bandit import BudgetSuggestion, OperatorArmStats, suggest_budget_allocation
from .boed import produce_boed_signals
from .causal import causal_advisory_signals
from .cellular import cellular_advisory_signals
from .config import TheoryConfig
from .errors import TheoryCancelled, TheoryProducerError, TheoryTimeout
from .geometry import geometry_advisory_signals
from .mdl import produce_mdl_signals
from .observer import observe_completed_events
from .representations import CompletedEventSnapshot, PopulationRepresentation
from .signals import AdvisoryRankingFeatures, TheorySignal
from .stability import stability_advisory_signals
from .telemetry import TheoryTelemetry

Producer = Callable[[PopulationRepresentation], tuple[TheorySignal, ...]]


class TheoryLayer:
    """Run enabled theory producers with fail-closed advisory semantics.

    The live Nexus runtime currently consumes only the aggregated candidate
    sidecar from :meth:`advisory_features_for_population`.  Later theory
    packages expose explicit opt-in helper methods so they can be tested and
    used by future planners without becoming hidden gates.
    """

    def __init__(self, config: TheoryConfig | None = None, telemetry: TheoryTelemetry | None = None) -> None:
        self.config = config or TheoryConfig()
        self.telemetry = telemetry or TheoryTelemetry()
        self._cache: OrderedDict[str, dict[str, AdvisoryRankingFeatures]] = OrderedDict()

    def advisory_features_for_population(
        self,
        population: PopulationRepresentation,
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, AdvisoryRankingFeatures]:
        cfg = config or self.config
        if not cfg.enabled:
            return {}
        if cancelled is not None and cancelled():
            return {}
        cache_key = _cache_key(population, cfg)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return dict(cached)
        started = time.monotonic()
        signals: list[TheorySignal] = []
        for name, enabled, producer in (
            ("mdl", cfg.mdl_enabled, produce_mdl_signals),
            ("boed", cfg.boed_enabled, produce_boed_signals),
            ("geometry", cfg.geometry_enabled, geometry_advisory_signals),
        ):
            if not enabled:
                continue
            if time.monotonic() - started > cfg.total_timeout_seconds:
                break
            produced = self._run_population_producer(name, producer, population, cfg, cancelled=cancelled)
            signals.extend(produced)
        features = aggregate_advisory_features(signals, cfg)
        self._cache[cache_key] = dict(features)
        while cfg.cache_bound and len(self._cache) > cfg.cache_bound:
            self._cache.popitem(last=False)
        return features

    def observe_completed_events(
        self,
        events: tuple[CompletedEventSnapshot, ...],
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[TheorySignal, ...]:
        cfg = config or self.config
        if not cfg.enabled or not cfg.observer_enabled:
            return ()
        return self._run_event_producer("observer", observe_completed_events, events, cfg, cancelled=cancelled)

    def causal_advisories(
        self,
        events: tuple[CompletedEventSnapshot, ...],
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[TheorySignal, ...]:
        cfg = config or self.config
        if not cfg.enabled or not cfg.causal_enabled:
            return ()
        return self._run_event_producer("causal", causal_advisory_signals, events, cfg, cancelled=cancelled)

    def cellular_advisories(
        self,
        population: PopulationRepresentation,
        features: dict[str, AdvisoryRankingFeatures] | None = None,
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[TheorySignal, ...]:
        cfg = config or self.config
        if not cfg.enabled or not cfg.cellular_enabled:
            return ()
        if cancelled is not None and cancelled():
            return ()
        try:
            started = time.monotonic()
            signals = cellular_advisory_signals(population, features)
            if time.monotonic() - started > cfg.per_producer_timeout_seconds:
                raise TheoryTimeout("cellular")
            if cfg.telemetry_enabled:
                self.telemetry.record(cycle_id=population.cycle_id, producer="cellular", signals=signals)
            return signals
        except (TheoryCancelled, TheoryTimeout, TheoryProducerError, Exception):
            return ()

    def budget_suggestions(
        self,
        arms: tuple[OperatorArmStats, ...],
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[BudgetSuggestion, ...]:
        cfg = config or self.config
        if not cfg.enabled or not cfg.bandit_enabled:
            return ()
        if cancelled is not None and cancelled():
            return ()
        try:
            started = time.monotonic()
            suggestions = suggest_budget_allocation(arms)
            if time.monotonic() - started > cfg.per_producer_timeout_seconds:
                raise TheoryTimeout("bandit")
            return suggestions
        except (TheoryCancelled, TheoryTimeout, TheoryProducerError, Exception):
            return ()

    def stability_advisories(
        self,
        population: PopulationRepresentation,
        *,
        config: TheoryConfig | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[TheorySignal, ...]:
        cfg = config or self.config
        if not cfg.enabled or not cfg.stability_enabled:
            return ()
        return self._run_population_producer("stability", stability_advisory_signals, population, cfg, cancelled=cancelled)

    def _run_population_producer(
        self,
        name: str,
        producer: Producer,
        population: PopulationRepresentation,
        config: TheoryConfig,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[TheorySignal, ...]:
        try:
            if cancelled is not None and cancelled():
                raise TheoryCancelled(name)
            started = time.monotonic()
            signals = producer(population)
            if time.monotonic() - started > config.per_producer_timeout_seconds:
                raise TheoryTimeout(name)
            if config.telemetry_enabled:
                self.telemetry.record(cycle_id=population.cycle_id, producer=name, signals=signals)
            return signals
        except (TheoryCancelled, TheoryTimeout, TheoryProducerError, Exception):
            return ()

    def _run_event_producer(
        self,
        name: str,
        producer: Callable[[tuple[CompletedEventSnapshot, ...]], tuple[TheorySignal, ...]],
        events: tuple[CompletedEventSnapshot, ...],
        config: TheoryConfig,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[TheorySignal, ...]:
        try:
            if cancelled is not None and cancelled():
                raise TheoryCancelled(name)
            started = time.monotonic()
            signals = producer(events)
            if time.monotonic() - started > config.per_producer_timeout_seconds:
                raise TheoryTimeout(name)
            if config.telemetry_enabled:
                cycle_id = events[0].cycle_id if events else "cycle:unknown"
                self.telemetry.record(cycle_id=cycle_id, producer=name, signals=signals)
            return signals
        except (TheoryCancelled, TheoryTimeout, TheoryProducerError, Exception):
            return ()


def _cache_key(population: PopulationRepresentation, config: TheoryConfig) -> str:
    payload = {
        "population": population.to_dict(),
        "flags": {
            "enabled": config.enabled,
            "mdl_enabled": config.mdl_enabled,
            "boed_enabled": config.boed_enabled,
            "geometry_enabled": config.geometry_enabled,
            "mdl_weight": config.mdl_weight,
            "boed_weight": config.boed_weight,
            "geometry_weight": config.geometry_weight,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
    return f"{population.cycle_id}|{digest}"


__all__ = ["TheoryLayer"]
