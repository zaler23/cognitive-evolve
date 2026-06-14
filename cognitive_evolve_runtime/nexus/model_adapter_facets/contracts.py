"""ContractsFacet methods for StructuredModelAdapter."""
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



class ContractsFacet:
    def build_objective_contract(self, *, user_goal: str, world: Any) -> dict[str, Any]:
        schema = _objective_contract_schema(project=False)
        return self._call("nexus_build_objective_contract", {"user_goal": user_goal, "world": world}, schema)
    
    def build_text_world_model(self, *, packet: Any) -> dict[str, Any]:
        schema = _text_world_model_schema()
        return self._call("nexus_build_text_world_model", {"packet": packet}, schema)
    
    def build_project_objective_contract(self, *, user_goal: str, snapshot: Any, world: Any | None = None) -> dict[str, Any]:
        schema = _objective_contract_schema(project=True)
        return self._call("nexus_build_project_objective_contract", {"user_goal": user_goal, "snapshot": snapshot, "world": world}, schema)


__all__ = ["ContractsFacet"]
