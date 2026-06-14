"""Canonical Nexus pipeline metadata.

This module is intentionally small and data-oriented. It records the single
runtime path used by service and CLI entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineStage:
    """One externally visible stage in the Nexus runtime pipeline."""

    id: str
    owner: str
    purpose: str


@dataclass(frozen=True)
class EvolutionPipeline:
    """Authoritative evolution pipeline declaration.

    NexusRuntime is the single execution authority.
    """

    source_of_truth: str = "NexusRuntime"
    public_entrypoint: str = "cognitive_evolve_runtime.engine.orchestrator.EngineOrchestrator"
    route_order: tuple[str, ...] = (
        "input_packet",
        "world_model",
        "objective_contract",
        "evolution_policy",
        "candidate_population",
        "local_verification",
        "relative_ranking",
        "archives_and_diagnosis",
        "nexus_synthesis",
    )
    stages: tuple[PipelineStage, ...] = field(
        default_factory=lambda: (
            PipelineStage("input_packet", "NexusRuntime", "Normalize text or project input without treating projects as one prompt."),
            PipelineStage("world_model", "NexusRuntime", "Build the task or project world model."),
            PipelineStage("objective_contract", "NexusRuntime", "Create and freeze the task objective contract."),
            PipelineStage("evolution_policy", "NexusRuntime", "Create model-driven niches, axes, operators, and archive policy."),
            PipelineStage("candidate_population", "NexusRuntime", "Seed and mutate structured candidate genomes."),
            PipelineStage("local_verification", "VerifierStack", "Run local tools and attach structured feedback."),
            PipelineStage("relative_ranking", "ranking", "Apply relative rating and multihead scoring."),
            PipelineStage("archives_and_diagnosis", "archives", "Update archives and diagnose stagnation/control actions."),
            PipelineStage("nexus_synthesis", "NexusRuntime", "Synthesize final answer, patch, report, and Nexus artifacts."),
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_of_truth": self.source_of_truth,
            "public_entrypoint": self.public_entrypoint,
            "route_order": list(self.route_order),
            "stages": [stage.__dict__.copy() for stage in self.stages],
        }


DEFAULT_PIPELINE = EvolutionPipeline()


__all__ = ["PipelineStage", "EvolutionPipeline", "DEFAULT_PIPELINE"]
