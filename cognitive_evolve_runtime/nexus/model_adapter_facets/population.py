"""PopulationFacet methods for StructuredModelAdapter."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import relative_rater_schema

from cognitive_evolve_runtime.nexus.model_adapter_schemas import (
    _candidate_critiques_schema,
    _candidate_population_schema,
    _context_request_schema,
    _evolution_policy_schema,
    _mutation_plan_schema,
    _objective_contract_schema,
    _offspring_population_schema,
    _search_diagnosis_schema,
    _stop_decision_schema,
    _synthesis_schema,
    _task_classification_schema,
    _text_world_model_schema,
)



class PopulationFacet:
    def seed_population(self, *, contract: Any, world: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        schema = _candidate_population_schema()
        payload = {"contract": contract, "world": world, "policy": policy}
        if provided_context:
            payload["source_context"] = provided_context
        data = self._call("nexus_seed_population", payload, schema)
        return [dict(item) for item in data.get("candidates", []) if isinstance(item, dict)]


__all__ = ["PopulationFacet"]
