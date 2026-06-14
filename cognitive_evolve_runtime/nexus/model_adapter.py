"""Public facade for the structured Nexus model adapter.

Implementation details live in focused core, schema, repair, and protocol-facet
modules.  This module intentionally remains a small compatibility boundary for
``StructuredModelAdapter`` imports.
"""
from __future__ import annotations

from cognitive_evolve_runtime.nexus.model_adapter_core import (
    JsonCaller,
    ModelResponseSchemaError,
    StructuredModelAdapterCore,
)
from cognitive_evolve_runtime.nexus.model_adapter_facets import (
    ClassificationFacet,
    ContextFacet,
    ContractsFacet,
    CritiqueFacet,
    DiagnosisFacet,
    MutationFacet,
    PolicyFacet,
    PopulationFacet,
    RankingFacet,
    StopFacet,
    SynthesisFacet,
)


class StructuredModelAdapter(
    ContractsFacet,
    ClassificationFacet,
    PolicyFacet,
    PopulationFacet,
    RankingFacet,
    CritiqueFacet,
    DiagnosisFacet,
    ContextFacet,
    MutationFacet,
    SynthesisFacet,
    StopFacet,
    StructuredModelAdapterCore,
):
    """Opt-in structured model adapter with deterministic tests and no implicit API-key use."""


__all__ = ["JsonCaller", "ModelResponseSchemaError", "StructuredModelAdapter"]
