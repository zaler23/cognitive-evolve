"""Adaptive-internal research extension registry configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import NoOpResearchExtension, ResearchExtension


@dataclass(frozen=True)
class ResearchConfig:
    enabled: bool = False
    mode: str = "observe"
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        enabled = _bool(raw.get("enabled"), default=any(_bool(cfg.get("enabled"), default=False) for cfg in extensions.values()))
        return cls(enabled=enabled, mode=mode, extensions=extensions)

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
        extensions.append(factory(config.extension_config(extension_id)))
    if "noop" in config.extensions and config.extension_enabled("noop"):
        extensions.append(NoOpResearchExtension())
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


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "active", "advisory", "observe"}


__all__ = ["ResearchConfig", "build_research_extensions"]
