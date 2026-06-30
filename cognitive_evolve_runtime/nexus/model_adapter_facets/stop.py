"""StopFacet methods for StructuredModelAdapter."""
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



class StopFacet:
    def should_stop(self, *, budget: Any, diagnosis: Any, best_answer_id: str, population: list[Any]) -> dict[str, Any]:
        schema = _stop_decision_schema()
        return self._call(
            "nexus_should_stop",
            {
                "budget": budget,
                "diagnosis": diagnosis,
                "best_answer_id": best_answer_id,
                "population": population,
                "instruction": (
                    "Return stop=true only for one of these terminal choices: "
                    "diminishing_returns_checkpoint or objective_solved. "
                    "Use candidate_ready_for_external_review when the run has a clear direct answer; use objective_solved only when an external/user-owned solved claim is explicitly warranted. "
                    "Use diminishing_returns_checkpoint when more rounds have low expected marginal value. "
                    "Safety checkpoints are not completion, and solved=true means answer produced, not externally verified."
                ),
            },
            schema,
        )


__all__ = ["StopFacet"]
