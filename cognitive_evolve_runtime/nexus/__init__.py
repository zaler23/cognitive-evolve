"""Nexus runtime: model-driven offline evolution primitives.

The package uses lazy public exports so importing an internal submodule such as
``cognitive_evolve_runtime.nexus._serde`` does not eagerly import the full Nexus
runtime and create circular dependencies.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "EvolutionPolicy": ("cognitive_evolve_runtime.nexus.policy", "EvolutionPolicy"),
    "SearchDiagnosis": ("cognitive_evolve_runtime.nexus.diagnosis", "SearchDiagnosis"),
    "SearchStateDiagnoser": ("cognitive_evolve_runtime.nexus.diagnosis", "SearchStateDiagnoser"),
    "PolicyUpdater": ("cognitive_evolve_runtime.nexus.diagnosis", "PolicyUpdater"),
    "NexusRuntime": ("cognitive_evolve_runtime.nexus.runtime", "NexusRuntime"),
    "NexusRunResult": ("cognitive_evolve_runtime.nexus.runtime", "NexusRunResult"),
    "EvolutionBudget": ("cognitive_evolve_runtime.nexus.loop", "EvolutionBudget"),
    "EvolutionLoopResult": ("cognitive_evolve_runtime.nexus.loop", "EvolutionLoopResult"),
    "evolve_once": ("cognitive_evolve_runtime.nexus.loop", "evolve_once"),
    "seed_population": ("cognitive_evolve_runtime.nexus.loop", "seed_population"),
    "StructuredModelAdapter": ("cognitive_evolve_runtime.nexus.model_adapter", "StructuredModelAdapter"),
    "ContextOrchestrator": ("cognitive_evolve_runtime.nexus.context_protocol", "ContextOrchestrator"),
    "ContextProtocolResult": ("cognitive_evolve_runtime.nexus.context_protocol", "ContextProtocolResult"),
    "ProjectCandidateVerifier": ("cognitive_evolve_runtime.nexus.project_verification", "ProjectCandidateVerifier"),
    "ProjectVerificationSummary": ("cognitive_evolve_runtime.nexus.project_verification", "ProjectVerificationSummary"),
    "NexusModelLike": ("cognitive_evolve_runtime.nexus.protocols", "NexusModelLike"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
