"""DiagnosisFacet methods for StructuredModelAdapter."""
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



class DiagnosisFacet:
    def diagnose_search_state(self, *, population: list[CandidateGenome], archives: Any, history: list[dict[str, Any]], contract: Any, policy: Any) -> dict[str, Any]:
        schema = _search_diagnosis_schema()
        return self._call(
            "nexus_diagnose_search_state",
            {"population": population, "archives": archives, "history": history, "contract": contract, "policy": policy},
            schema,
        )
    
    def update_policy(self, *, policy: EvolutionPolicy, diagnosis: SearchDiagnosis) -> dict[str, Any]:
        schema = _evolution_policy_schema()
        return self._call("nexus_update_policy", {"policy": policy, "diagnosis": diagnosis}, schema)


__all__ = ["DiagnosisFacet"]
