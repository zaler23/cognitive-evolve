"""Adaptive-internal research extension registry configuration."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.concepts.contract import CONTRACTS
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import NoOpResearchExtension, ResearchExtension


@dataclass(frozen=True)
class ResearchConfig:
    enabled: bool = False
    mode: str = "observe"
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace_enabled: bool = True
    ablation_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ResearchConfig":
        raw = coerce_dict(data)
        mode = str(raw.get("mode") or "observe").strip().lower() or "observe"
        if mode not in {"observe", "advisory", "active"}:
            mode = "observe"
        extensions = {str(k): coerce_dict(v) for k, v in coerce_dict(raw.get("extensions")).items()}
        for key in _KNOWN_EXTENSION_IDS:
            if key in raw and isinstance(raw.get(key), dict):
                extensions[key] = coerce_dict(raw.get(key))
        default_enabled = any(_bool(cfg.get("enabled"), default=False) for cfg in extensions.values())
        if os.environ.get("COGEV_HERMETIC_TEST") or os.environ.get("COGEV_RESEARCH_DEFAULT_ENABLED") in {"0", "false", "False", "off"}:
            default_enabled = False
        enabled = _bool(raw.get("enabled"), default=default_enabled)
        return cls(
            enabled=enabled,
            mode=mode,
            extensions=extensions,
            trace_enabled=_bool(raw.get("trace_enabled"), default=True),
            ablation_enabled=_bool(raw.get("ablation_enabled"), default=True),
        )

    def extension_config(self, extension_id: str) -> dict[str, Any]:
        return coerce_dict(self.extensions.get(extension_id))

    def extension_enabled(self, extension_id: str) -> bool:
        return self.enabled and _bool(self.extension_config(extension_id).get("enabled"), default=False)


_KNOWN_EXTENSION_IDS = {
    "spatial_selection",
    "pattern_memory",
    "immune_necropsy",
    "budget_backpressure",
    "mdl_compression",
    "parameter_sweep",
    "chaos",
    "bft_quorum",
    "context_pruning",
    "contract_refinement",
    "noop",
}


def build_research_extensions(config: ResearchConfig) -> list[ResearchExtension]:
    extensions: list[ResearchExtension] = []
    for extension_id in sorted(config.extensions):
        if not config.extension_enabled(extension_id):
            continue
        factory = _extension_factories().get(extension_id)
        if factory is None:
            continue
        extension = factory(config.extension_config(extension_id))
        _validate_extension_contract(extension_id, extension)
        extensions.append(extension)
    if "noop" in config.extensions and config.extension_enabled("noop"):
        noop = NoOpResearchExtension()
        _validate_extension_contract("noop", noop)
        extensions.append(noop)
    return extensions


def _extension_factories() -> dict[str, Callable[[dict[str, Any]], ResearchExtension]]:
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.bft_quorum import BFTQuorumExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.budget_backpressure import BudgetBackpressureExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.chaos import ChaosExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.context_pruning import ContextPruningExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.contract_refinement import ContractRefinementExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.immune_necropsy import ImmuneNecropsyExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.mdl_compression import MDLCompressionExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.parameter_sweep import ParameterSweepExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.pattern_memory import PatternMemoryExtension
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.spatial_selection import SpatialSelectionExtension

    return {
        "spatial_selection": SpatialSelectionExtension,
        "pattern_memory": PatternMemoryExtension,
        "immune_necropsy": ImmuneNecropsyExtension,
        "budget_backpressure": BudgetBackpressureExtension,
        "mdl_compression": MDLCompressionExtension,
        "parameter_sweep": ParameterSweepExtension,
        "chaos": ChaosExtension,
        "bft_quorum": BFTQuorumExtension,
        "context_pruning": ContextPruningExtension,
        "contract_refinement": ContractRefinementExtension,
    }


def _validate_extension_contract(extension_id: str, extension: ResearchExtension) -> None:
    expected = CONTRACTS.get(extension_id)
    contract = getattr(extension, "contract", None)
    if expected is None or contract is None or getattr(contract, "concept_id", None) != expected.concept_id:
        raise ValueError(f"research extension {extension_id} is missing a valid ConceptContract")


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "active", "advisory", "observe"}


__all__ = ["ResearchConfig", "build_research_extensions"]
