"""Single-round ranking, diagnosis, allocation, and reproduction pipeline."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlanner
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.critique import CritiqueEngine
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater
from cognitive_evolve_runtime.evaluators import ExternalEvaluatorRunner
from cognitive_evolve_runtime.theory import TheoryLayer
from cognitive_evolve_runtime.verification.cache import check_with_cache
from cognitive_evolve_runtime.verification.factory import verifier_from_plan
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.strength import measured_strength_from_result
from cognitive_evolve_runtime.verification.types import VerificationPlan, VerificationResult

from .budget import EvolutionBudget
from .evaluate_stage import EvaluateStage
from .reproduce_stage import (
    ReproduceStage,
    _attach_branch_allocation_to_plans,
    _canonical_family_metrics,
    _cell_activation_map,
)
from .round_context import RoundEvaluation


class EvolutionRound(EvaluateStage, ReproduceStage):
    """Thin facade over the evaluation and reproduction stages."""

    def __init__(self, *, model: NexusModelLike | None, budget: EvolutionBudget, adaptive: AdaptiveRuntimeController | None = None) -> None:
        self.model = model
        self.budget = budget
        self.adaptive = adaptive or AdaptiveRuntimeController.from_sources()
        self.rater = RelativeRater(model=model)
        self.elo = MultiHeadElo()
        self.diagnoser = SearchStateDiagnoser(model=model)
        self.updater = PolicyUpdater()
        self.selector = ParentSelector()
        self.mutation_planner = MutationPlanner()
        self.mutation_engine = MutationEngine()
        self.critique_engine = CritiqueEngine(model=model)
        self.evaluator_runner = ExternalEvaluatorRunner()
        self.stop_decider = StopDecisionEngine()
        self.theory_layer = TheoryLayer()
        self.last_generation_plan: dict[str, Any] = {}
        self.last_completed_stage_ops: list[str] = []
        self.last_offspring_harvest_outcome: dict[str, Any] = {}

    def _run_synthesized_verifier(self, candidates: list[CandidateGenome], *, current_round: int) -> list[Any]:
        plan_data = self.adaptive.verification_plan_dict()
        if not plan_data:
            return []
        plan = VerificationPlan.from_dict(plan_data)
        verifier = verifier_from_plan(plan)
        if verifier is None:
            return []
        cache = self.adaptive.verification_cache()
        results: list[Any] = []
        viable = [
            candidate
            for candidate in candidates
            if CandidateFate.normalize(candidate.current_fate) in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
        ]
        max_checks = max(1, min(len(viable), self._branch_limit() if self.budget.adaptive else max(self._branch_limit(), 4)))
        for candidate in viable[:max_checks]:
            result, cache_key, cache_hit = check_with_cache(candidate, verifier, cache)
            trace_item = result.to_dict()
            trace_item.setdefault("metadata", {})
            if isinstance(trace_item["metadata"], dict):
                trace_item["metadata"].update({"round_index": current_round, "cache_key": cache_key, "cache_hit": cache_hit})
            candidate.verification_trace = [*candidate.verification_trace, trace_item]
            measured_strength = measured_strength_from_result(result)
            if result.passed and result.replayable and measured_strength > VerificationStrength.NONE:
                current = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
                current_strength = measured_strength_from_result(VerificationResult.from_dict(current)) if current else measured_strength_from_result(None)
                if measured_strength >= current_strength:
                    candidate.verification_result = trace_item
            results.append(result)
        self.adaptive.update_verification_cache(cache)
        return results



__all__ = [
    "EvolutionRound",
    "RoundEvaluation",
    "_attach_branch_allocation_to_plans",
    "_canonical_family_metrics",
    "_cell_activation_map",
]
