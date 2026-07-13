"""Typed runtime configuration for v2.3 theory-strengthened search.

All tunable numeric values used by v2.3 entropy compaction, minimax verifier
budgeting, honesty control, and CA-style crossover are centralized here. Runtime
code should consume this object rather than introducing feature switches or
anonymous numeric thresholds in algorithm bodies.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash


DEFAULT_CELL_ELITE_RESERVE = 1
DEFAULT_RARE_RESERVE_PER_CELL = 1
DEFAULT_ENTROPY_GAIN_WEIGHT = 1.0
DEFAULT_FRONTIER_WEIGHT = 0.35
DEFAULT_RARITY_WEIGHT = 0.25
DEFAULT_SEARCH_QUALITY_WEIGHT = 0.25
DEFAULT_MIN_BUDGET_PER_CANDIDATE = 1
DEFAULT_PI_WINDOW = 5
DEFAULT_CONTROL_GAIN = 0.2
DEFAULT_CONTROL_INTEGRAL_GAIN = 0.05
DEFAULT_CONTROL_CLAMP = 1.0
DEFAULT_HISTORY_LIMIT = 200
DEFAULT_MIN_SHARED_DESCRIPTOR_TOKENS = 1
DEFAULT_GLOBAL_DONOR_POLICY = "highest_search_quality"
DEFAULT_DIAGNOSTIC_LIMIT = 20


@dataclass(frozen=True)
class EntropyCompactionConfig:
    cell_elite_reserve: int = DEFAULT_CELL_ELITE_RESERVE
    rare_reserve_per_cell: int = DEFAULT_RARE_RESERVE_PER_CELL
    entropy_gain_weight: float = DEFAULT_ENTROPY_GAIN_WEIGHT
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT
    rarity_weight: float = DEFAULT_RARITY_WEIGHT
    search_quality_weight: float = DEFAULT_SEARCH_QUALITY_WEIGHT

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EntropyCompactionConfig":
        payload = coerce_dict(data)
        return cls(
            cell_elite_reserve=_positive_int(payload.get("cell_elite_reserve"), DEFAULT_CELL_ELITE_RESERVE),
            rare_reserve_per_cell=_nonnegative_int(payload.get("rare_reserve_per_cell"), DEFAULT_RARE_RESERVE_PER_CELL),
            entropy_gain_weight=_float(payload.get("entropy_gain_weight"), DEFAULT_ENTROPY_GAIN_WEIGHT),
            frontier_weight=_float(payload.get("frontier_weight"), DEFAULT_FRONTIER_WEIGHT),
            rarity_weight=_float(payload.get("rarity_weight"), DEFAULT_RARITY_WEIGHT),
            search_quality_weight=_float(payload.get("search_quality_weight"), DEFAULT_SEARCH_QUALITY_WEIGHT),
        )


@dataclass(frozen=True)
class MinimaxBudgetConfig:
    min_budget_per_candidate: int = DEFAULT_MIN_BUDGET_PER_CANDIDATE
    diagnostics_limit: int = DEFAULT_DIAGNOSTIC_LIMIT

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MinimaxBudgetConfig":
        payload = coerce_dict(data)
        return cls(
            min_budget_per_candidate=_nonnegative_int(payload.get("min_budget_per_candidate"), DEFAULT_MIN_BUDGET_PER_CANDIDATE),
            diagnostics_limit=_positive_int(payload.get("diagnostics_limit"), DEFAULT_DIAGNOSTIC_LIMIT),
        )


@dataclass(frozen=True)
class HonestyControlConfig:
    window: int = DEFAULT_PI_WINDOW
    proportional_gain: dict[str, float] = field(default_factory=lambda: _gain_dict(DEFAULT_CONTROL_GAIN))
    integral_gain: dict[str, float] = field(default_factory=lambda: _gain_dict(DEFAULT_CONTROL_INTEGRAL_GAIN))
    clamp_abs: float = DEFAULT_CONTROL_CLAMP
    history_limit: int = DEFAULT_HISTORY_LIMIT

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HonestyControlConfig":
        payload = coerce_dict(data)
        return cls(
            window=_positive_int(payload.get("window"), DEFAULT_PI_WINDOW),
            proportional_gain=_coerce_gain(payload.get("proportional_gain"), DEFAULT_CONTROL_GAIN),
            integral_gain=_coerce_gain(payload.get("integral_gain"), DEFAULT_CONTROL_INTEGRAL_GAIN),
            clamp_abs=max(0.0, _float(payload.get("clamp_abs"), DEFAULT_CONTROL_CLAMP)),
            history_limit=_positive_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT),
        )


@dataclass(frozen=True)
class CACrossoverConfig:
    min_shared_descriptor_tokens: int = DEFAULT_MIN_SHARED_DESCRIPTOR_TOKENS
    global_donor_policy: str = DEFAULT_GLOBAL_DONOR_POLICY
    activation_history_limit: int = DEFAULT_HISTORY_LIMIT

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CACrossoverConfig":
        payload = coerce_dict(data)
        return cls(
            min_shared_descriptor_tokens=_positive_int(payload.get("min_shared_descriptor_tokens"), DEFAULT_MIN_SHARED_DESCRIPTOR_TOKENS),
            global_donor_policy=str(payload.get("global_donor_policy") or DEFAULT_GLOBAL_DONOR_POLICY),
            activation_history_limit=_positive_int(payload.get("activation_history_limit"), DEFAULT_HISTORY_LIMIT),
        )


@dataclass(frozen=True)
class V23TheoryRuntimeConfig:
    entropy: EntropyCompactionConfig = field(default_factory=EntropyCompactionConfig)
    minimax_budget: MinimaxBudgetConfig = field(default_factory=MinimaxBudgetConfig)
    honesty_control: HonestyControlConfig = field(default_factory=HonestyControlConfig)
    ca_crossover: CACrossoverConfig = field(default_factory=CACrossoverConfig)
    diagnostics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "V23TheoryRuntimeConfig":
        payload = coerce_dict(data)
        return cls(
            entropy=EntropyCompactionConfig.from_dict(payload.get("entropy")),
            minimax_budget=MinimaxBudgetConfig.from_dict(payload.get("minimax_budget")),
            honesty_control=HonestyControlConfig.from_dict(payload.get("honesty_control")),
            ca_crossover=CACrossoverConfig.from_dict(payload.get("ca_crossover")),
            diagnostics=[str(item) for item in payload.get("diagnostics", []) if item] if isinstance(payload.get("diagnostics"), list) else [],
        )

    @classmethod
    def from_runtime_context(cls, *, policy: Any | None = None, contract: Any | None = None, branch_factor: int | None = None, population_size: int | None = None) -> "V23TheoryRuntimeConfig":
        policy_metadata = coerce_dict(getattr(policy, "metadata", {}) if policy is not None else {})
        contract_metadata = coerce_dict(getattr(contract, "metadata", {}) if contract is not None else {})
        configured = coerce_dict(policy_metadata.get("v23_theory_runtime")) or coerce_dict(contract_metadata.get("v23_theory_runtime"))
        base = cls.from_dict(configured)
        diagnostics = list(base.diagnostics)
        ignored = _deprecated_switch_diagnostics(policy_metadata, contract_metadata)
        diagnostics.extend(ignored)
        diagnostics.extend(_deprecated_switch_env_diagnostics())
        return cls(
            entropy=base.entropy,
            minimax_budget=base.minimax_budget,
            honesty_control=base.honesty_control,
            ca_crossover=base.ca_crossover,
            diagnostics=list(dict.fromkeys(diagnostics)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return "v23-theory-" + stable_hash(self.to_dict())[:16]


def _deprecated_switch_diagnostics(*dicts: dict[str, Any]) -> list[str]:
    names = {
        "COGEV_COMPACT_MODE",
        "COGEV_DYNAMIC_ADVERSARIAL_BUDGET",
        "COGEV_HONESTY_CONTROL",
        "COGEV_CROSSOVER_MODE",
    }
    out: list[str] = []
    for payload in dicts:
        for name in names:
            if name in payload:
                out.append(f"ignored_deprecated_v23_switch:{name}")
    return out


def _deprecated_switch_env_diagnostics() -> list[str]:
    names = (
        "COGEV_COMPACT_MODE",
        "COGEV_DYNAMIC_ADVERSARIAL_BUDGET",
        "COGEV_HONESTY_CONTROL",
        "COGEV_CROSSOVER_MODE",
    )
    return [f"ignored_deprecated_v23_env_switch:{name}" for name in names if name in os.environ]


def _gain_dict(value: float) -> dict[str, float]:
    return {dimension: float(value) for dimension in ("exogeneity", "variety", "falsification", "replay")}


def _coerce_gain(value: Any, default: float) -> dict[str, float]:
    payload = coerce_dict(value)
    base = _gain_dict(default)
    for key in base:
        if key in payload:
            base[key] = _float(payload.get(key), default)
    return base


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "CACrossoverConfig",
    "EntropyCompactionConfig",
    "HonestyControlConfig",
    "MinimaxBudgetConfig",
    "V23TheoryRuntimeConfig",
]
