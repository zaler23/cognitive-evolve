"""Evaluation half of one Nexus evolution round."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ProgressiveEvaluator, apply_evidence_record, latest_evidence_record
from cognitive_evolve_runtime.evaluators.evidence import select_preliminary_incumbent
from cognitive_evolve_runtime.llm.session import current_llm_session, logical_llm_call
from cognitive_evolve_runtime.nexus.critique import CandidateCritique
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.honesty_control import compute_honesty_control_signal
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings
from cognitive_evolve_runtime.nexus.receipts import record_transfer_receipts
from cognitive_evolve_runtime.nexus.stop_reasons import stop_reason_class
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    annotate_candidates_with_latent_signals,
    ingest_latent_feedback,
    ingest_runtime_trial_feedback,
)
from cognitive_evolve_runtime.verification.information_gain import population_information_gain_report
from cognitive_evolve_runtime.nexus.v23_theory_config import V23TheoryRuntimeConfig
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import productive_outcomes

from .offspring import _best_auxiliary_id
from .stage_helpers import _eligibility_policy

from .round_context import RoundEvaluation

_RELATIVE_RANK_PROMPT_TEMPLATE_VERSION = "nexus-relative-rank/v1"


class EvaluateStage:
    def evaluate(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> RoundEvaluation:
        self.adaptive.begin_round(round_index=current_round)
        self._sync_model_context_controls()
        self.adaptive.observe_population(population=population, round_index=current_round)
        evaluator_config = dict(self.adaptive.config.evaluator or {})
        if self.adaptive.config.evidence:
            evaluator_config.setdefault("evidence", dict(self.adaptive.config.evidence))
        evaluator_spec = EvaluatorSpec.from_mapping(evaluator_config)
        evaluator_led = self._evaluator_led(evaluator_spec)
        previously_evaluated_ids = {
            candidate.id
            for candidate in population.candidates
            if isinstance(candidate.metadata, dict) and isinstance(candidate.metadata.get("evaluator"), dict)
        }
        configured_admission_limit = int(policy.metadata.get("pre_rank_admission_limit") or len(population.candidates))
        evaluation_candidates, admission_audit = _pre_rank_admission_view(
            population.candidates,
            limit=configured_admission_limit if evaluator_spec.enabled else len(population.candidates),
            history=self.budget.history,
            uncertainty_floor=max(1, int(policy.metadata.get("pre_rank_uncertainty_floor") or 1)),
        )
        evaluator_results = self.evaluator_runner.evaluate_population_if_configured(
            evaluation_candidates,
            spec=evaluator_spec,
            round_index=current_round,
        )
        newly_evaluated = [
            candidate
            for candidate in evaluation_candidates
            if candidate.id not in previously_evaluated_ids
            and isinstance(candidate.metadata, dict)
            and isinstance(candidate.metadata.get("evaluator"), dict)
        ]
        metric_directions = {item.name: item.direction for item in evaluator_spec.metrics}
        newly_evaluated_ids = {candidate.id for candidate in newly_evaluated}
        grounded_outcomes = tuple(
            outcome
            for outcome in productive_outcomes(
                population.candidates,
                metric_directions=metric_directions,
            )
            if outcome.candidate_id in newly_evaluated_ids
        )
        direction_aware_gain = {
            "schema": "cogev.direction_aware_gain.v1",
            "total_gain": round(sum(max(0.0, item.reward) for item in grounded_outcomes), 12),
            "sample_count": len(grounded_outcomes),
            "candidate_ids": [item.candidate_id for item in grounded_outcomes],
            "outcomes": [item.to_dict() for item in grounded_outcomes],
            "authority": "post_evaluator_observation_only",
        }
        admission_audit.update(
            {
                "evaluator_enabled": evaluator_spec.enabled,
                "evaluator_result_count": len(evaluator_results),
                "newly_evaluated_candidate_ids": [candidate.id for candidate in newly_evaluated],
            }
        )
        if not evaluator_results and self.adaptive.enabled:
            progressive = ProgressiveEvaluator()
            for candidate in population.candidates:
                apply_evidence_record(candidate, progressive.evaluate_result(candidate, None, spec=evaluator_spec, round_index=current_round))
        critiques, verification_results = self.critique_and_verify(
            current_round=current_round,
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            use_model_critique=not evaluator_led,
        )
        if self.adaptive.enabled:
            if evaluator_results:
                passed = len([item for item in evaluator_results if item.passed])
                evaluated = len(evaluator_results)
            else:
                evidence_items = [candidate.metadata.get("evidence_state") for candidate in population.candidates if isinstance(candidate.metadata, dict) and candidate.metadata.get("evidence_state")]
                passed = len([item for item in evidence_items if isinstance(item, dict) and not item.get("final_blocked")])
                evaluated = len(evidence_items)
            self.adaptive.record_evaluator_summary(
                round_index=current_round,
                evaluated=evaluated,
                passed=passed,
                failed=max(0, evaluated - passed),
                candidates=population.candidates,
            )
        rankings = self.rank(population=population, archives=archives, policy=policy, contract=contract, current_round=current_round)
        self.last_generation_plan["pre_rank_admission"] = admission_audit
        self.last_generation_plan["direction_aware_gain"] = direction_aware_gain
        self.adaptive.observe_population(population=population, round_index=current_round)
        plan = GenerationPlan.from_dict(self.last_generation_plan)
        completed_stage_ops = list(self.last_completed_stage_ops)
        repair_parent_candidates = list(population.candidates)
        assert_stage_ready(plan, "compact", completed_stage_ops)
        ranking_compaction = compact_live_population(
            population,
            archives,
            policy,
            branch_factor=self.budget.branch_factor,
            round_index=current_round,
        )
        completed_stage_ops.append("compact")
        assert_stage_ready(plan, "diagnose", completed_stage_ops)
        diagnosis, updated_policy = self.diagnose_and_update(population=population, archives=archives, policy=policy, contract=contract)
        completed_stage_ops.append("diagnose")
        best_aux = _best_auxiliary_id(population.candidates)
        best_answer = rankings.best_final_answer_id
        assert_stage_ready(plan, "stop_check", completed_stage_ops)
        stop_reason = self.stop_decider.stop_reason_after_round(
            budget=self.budget,
            completed_round=current_round,
            diagnosis=diagnosis,
            best_answer_id=best_answer,
            population=population,
            model=None if evaluator_led else self.model,
            contract=contract,
            evolution_policy=updated_policy,
        )
        stop_class = stop_reason_class(stop_reason)
        completed_stage_ops.append("stop_check")
        generation_plan = dict(self.last_generation_plan)
        generation_plan["completed_stage_ops"] = list(completed_stage_ops)
        self.last_generation_plan = dict(generation_plan)
        self.last_completed_stage_ops = list(completed_stage_ops)
        progress_event = EvolutionProgressEvent(
            round=current_round,
            max_rounds=self.budget.round_limit,
            population_size=len(population.candidates),
            active_candidates=len([c for c in population.candidates if CandidateFate.normalize(c.current_fate) == CandidateFate.ACTIVE.value]),
            dormant_candidates=len(archives.dormant_archive.candidates),
            archive_elites=len(archives.answer_archive),
            tool_calls=sum(len(c.tool_results) for c in population.candidates),
            best_answer_candidate=best_answer,
            best_auxiliary_candidate=best_aux,
            search_diagnosis=diagnosis.stagnation_type,
            next_action=stop_reason or (diagnosis.recommended_actions[0] if diagnosis.recommended_actions else "continue"),
            metadata={
                "stop_policy": self.budget.stop_policy,
                "stop_reason": stop_reason,
                "stop_reason_class": stop_class,
                "stop_decision": {
                    "stop": bool(stop_reason),
                    "reason": stop_reason,
                    "reason_class": stop_class,
                    "best_candidate_id": best_answer,
                },
                "adaptive": self.budget.adaptive,
                "round_safety_limit": self.budget.round_limit if self.budget.adaptive else 0,
                "completion_requires_stop_signal": self.budget.completion_requires_stop_signal,
                "progress_semantics": "open_ended_safety_checkpoint" if self.budget.adaptive else "fixed_round_budget",
                "current_round": current_round,
                "budget_current_round": self.budget.current_round,
                "round_limit": self.budget.round_limit,
                "generation_plan_id": generation_plan.get("plan_id", ""),
                "incubating_candidates": len([c for c in population.candidates if CandidateFate.normalize(c.current_fate) == CandidateFate.INCUBATING.value]),
                "population_vitality": vitality_snapshot(population.candidates, branch_factor=self.budget.branch_factor).to_dict(),
                "adaptive_features": dict(self.adaptive.state.enabled_features),
                "search_phase": self.budget.search_phase,
                "pre_rank_admission": admission_audit,
                "direction_aware_gain": direction_aware_gain,
            },
        ).to_dict()
        if generation_plan.get("representation_shadow"):
            progress_event["metadata"]["representation_shadow"] = dict(generation_plan["representation_shadow"])
        stage_count = self.budget.round_limit
        pipeline_event = PipelineProgressEvent(
            stage="candidate_population",
            stage_index=min(current_round, stage_count),
            stage_count=stage_count,
            stage_progress=0.0 if self.budget.adaptive else current_round / max(1, stage_count),
            metadata={
                "adaptive": self.budget.adaptive,
                "progress_semantics": "open_ended_no_percent_complete" if self.budget.adaptive else "fixed_percent_complete",
                "current_round": current_round,
                "round_limit": self.budget.round_limit,
            },
        ).to_dict()
        return RoundEvaluation(
            rankings=rankings,
            policy=updated_policy,
            diagnosis=diagnosis,
            critiques=critiques,
            verification_results=verification_results,
            progress_event=progress_event,
            pipeline_event=pipeline_event,
            stop_reason=stop_reason,
            population_compaction=ranking_compaction.to_dict(),
            repair_parent_candidates=repair_parent_candidates,
            generation_plan=generation_plan,
        )

    def rank(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        current_round: int,
    ) -> RelativeRankingResult:
        self.last_generation_plan = {}
        self.last_completed_stage_ops = []
        run_id = str(current_llm_session().run_id or "run")
        with logical_llm_call(
            f"{run_id}/round-{current_round}/relative-rank/logical-pass",
            template_version=_RELATIVE_RANK_PROMPT_TEMPLATE_VERSION,
        ):
            preliminary_incumbent = select_preliminary_incumbent(population.candidates)
            if preliminary_incumbent is not None:
                rankings = RelativeRater(model=None).rank(
                    candidates=population.candidates,
                    contract=contract,
                    policy=policy,
                    archives=archives,
                )
                rankings.best_final_answer_id = preliminary_incumbent.id
                rankings.raw_notes = (
                    rankings.raw_notes + "; " if rankings.raw_notes else ""
                ) + "preliminary_evaluator_is_selection_authority"
            else:
                rankings = self.rater.rank(candidates=population.candidates, contract=contract, policy=policy, archives=archives)
        self.elo.update_from_relative(rankings)
        self.elo.apply_to_candidates(population.candidates, axes=list(policy.fitness_axes or []))
        latent_ranking_summary = annotate_candidates_with_latent_signals(population.candidates, contract)
        assignments = archives.assign_by_policy(
            population.candidates,
            rankings,
            current_round=current_round,
            round_limit=self.budget.round_limit,
            branch_factor=self.budget.branch_factor,
            eligibility_policy=_eligibility_policy(policy),
        )
        representation_shadow = (
            self.representation_shadow.observe(
                round_index=current_round,
                candidates=population.candidates,
                fate_assignments=assignments,
            )
            if self.representation_shadow is not None
            else None
        )
        generation_plan = build_generation_plan(
            round_index=current_round,
            candidates=population.candidates,
            fate_assignments=assignments,
            ranking=rankings,
            stage_graph=[
                {"op": "critique_and_verify"},
                {"op": "rank"},
                {"op": "archive_assign"},
                {"op": "generation_plan_validate"},
            {"op": "archive_update"},
            {"op": "compact"},
            {"op": "diagnose"},
            {"op": "stop_check"},
            {"op": "select_parents"},
            {"op": "plan_mutations"},
            {"op": "generate_offspring"},
            {"op": "verify_offspring"},
        ],
            representation_shadow=representation_shadow,
            source="runtime_rank_archive_transition",
        )
        if latent_ranking_summary:
            generation_plan.ranking_summary["latent_ranking"] = latent_ranking_summary
        generation_plan_data = generation_plan.to_dict()
        record_transfer_receipts(generation_plan_data, population.candidates)
        generation_plan = GenerationPlan.from_dict(generation_plan_data)
        object.__setattr__(generation_plan, "plan_id", expected_generation_plan_id(generation_plan))
        apply_generation_plan(generation_plan, population.candidates, archives)
        self.last_generation_plan = generation_plan.to_dict()
        self.last_completed_stage_ops = ["critique_and_verify", "rank", "archive_assign", "generation_plan_validate", "archive_update"]
        return rankings

    def critique_and_verify(
        self,
        *,
        current_round: int,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
        use_model_critique: bool = True,
    ) -> tuple[list[CandidateCritique], list[Any]]:
        critiques: list[CandidateCritique] = []
        if use_model_critique:
            critiques = self.critique_engine.critique(
                candidates=population.candidates,
                round_index=current_round,
                contract=contract,
                policy=policy,
                archives=archives,
            )
            self.critique_engine.apply(candidates=population.candidates, critiques=critiques)
        for candidate in population.candidates:
            try:
                annotate_candidate_source_bindings(candidate, project_root=getattr(archives, "project_root", None) or None)
            except Exception:
                if isinstance(candidate.metadata, dict):
                    candidate.metadata.setdefault("source_binding_manifest", {"binding_class": "no_binding", "admission_route": "repair_only", "diagnostics": ["source_binding_annotation_failed"]})
        verification_results: list[Any] = []
        verification_results.extend(self._run_synthesized_verifier(population.candidates, current_round=current_round))
        ingest_latent_feedback(
            contract=contract,
            critiques=critiques,
            verifier_results=verification_results,
        )
        ingest_runtime_trial_feedback(contract=contract, candidates=population.candidates)
        return critiques, verification_results

    def diagnose_and_update(
        self,
        *,
        population: CandidatePopulation,
        archives: ArchiveManager,
        policy: EvolutionPolicy,
        contract: NexusObjectiveContract,
    ) -> tuple[SearchDiagnosis, EvolutionPolicy]:
        gain_report = population_information_gain_report(population.candidates, self.budget.history)
        policy.metadata.setdefault("engine_grounded_information_gain", gain_report)
        policy.metadata["engine_grounded_information_gain"] = gain_report
        control_model = None if self._evaluator_led() else self.model
        diagnoser = SearchStateDiagnoser(model=control_model) if control_model is None else self.diagnoser
        diagnosis = diagnoser.diagnose(population=population.candidates, archives=archives, history=self.budget.history, contract=contract, policy=policy)
        diagnosis.grounded_information_gain = gain_report
        v23_config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy, contract=contract, branch_factor=self.budget.branch_factor, population_size=len(population.candidates))
        signal = compute_honesty_control_signal(
            candidates=population.candidates,
            config=v23_config.honesty_control,
            history=self.adaptive.state.honesty_error_history,
        )
        diagnosis.metadata["honesty_control"] = signal.to_dict()
        diagnosis.metadata["v23_theory_config_hash"] = v23_config.config_hash
        if v23_config.diagnostics:
            diagnosis.metadata["v23_theory_config_diagnostics"] = list(v23_config.diagnostics)
        self.adaptive.record_honesty_control_signal(signal, history_limit=v23_config.honesty_control.history_limit)
        return diagnosis, self.updater.update(policy, diagnosis, model=control_model, archives=archives)


def _pre_rank_admission_view(
    candidates: list[CandidateGenome],
    *,
    limit: int,
    history: list[dict[str, Any]],
    uncertainty_floor: int,
) -> tuple[list[CandidateGenome], dict[str, Any]]:
    deferred_counts: dict[str, int] = {}
    for record in history:
        if not isinstance(record, dict):
            continue
        plan = record.get("generation_plan") if isinstance(record.get("generation_plan"), dict) else {}
        audit = plan.get("pre_rank_admission") if isinstance(plan.get("pre_rank_admission"), dict) else {}
        for candidate_id in audit.get("deferred_candidate_ids", []):
            candidate_id = str(candidate_id or "")
            if candidate_id:
                deferred_counts[candidate_id] = deferred_counts.get(candidate_id, 0) + 1

    signals = {candidate.id: _admission_signals(candidate, deferred_counts.get(candidate.id, 0)) for candidate in candidates}

    def priority(candidate: CandidateGenome) -> tuple[float, ...]:
        signal = signals[candidate.id]
        return (
            float(not signal["evaluated"]),
            float(signal["deferred_rounds"]),
            float(signal["uncertainty"]),
            float(signal["value"]),
            -float(signal["confidence"]),
        )

    floor_reasons: dict[str, set[str]] = {}

    def reserve(candidate: CandidateGenome, reason: str) -> None:
        floor_reasons.setdefault(candidate.id, set()).add(reason)

    axes: dict[str, list[CandidateGenome]] = {}
    families: dict[str, list[CandidateGenome]] = {}
    for candidate in candidates:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        search_space = metadata.get("search_space") if isinstance(metadata.get("search_space"), dict) else {}
        axis = str(search_space.get("seed_axis") or "").strip()
        family = str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()
        if axis:
            axes.setdefault(axis, []).append(candidate)
        if family:
            families.setdefault(family, []).append(candidate)
    for axis, group in axes.items():
        reserve(max(group, key=lambda candidate: (*priority(candidate), candidate.id)), f"axis:{axis}")
    for family, group in families.items():
        reserve(max(group, key=lambda candidate: (*priority(candidate), candidate.id)), f"family:{family}")
    for candidate in sorted(
        candidates,
        key=lambda item: (float(signals[item.id]["uncertainty"]), *priority(item), item.id),
        reverse=True,
    )[: max(0, int(uncertainty_floor))]:
        reserve(candidate, "uncertainty")

    protected_ids = set(floor_reasons)
    effective_limit = min(len(candidates), max(max(0, int(limit)), len(protected_ids)))
    ordered = sorted(candidates, key=lambda candidate: (*priority(candidate), candidate.id), reverse=True)
    admitted = [candidate for candidate in ordered if candidate.id in protected_ids]
    admitted.extend(candidate for candidate in ordered if candidate.id not in protected_ids and len(admitted) < effective_limit)
    admitted_ids = {candidate.id for candidate in admitted}
    deferred = [candidate for candidate in ordered if candidate.id not in admitted_ids]

    decisions = []
    for candidate in [*admitted, *deferred]:
        signal = signals[candidate.id]
        if candidate.id in floor_reasons:
            reason = "floor:" + ",".join(sorted(floor_reasons[candidate.id]))
        elif candidate.id in admitted_ids and signal["deferred_rounds"]:
            reason = "deferred_reentry"
        elif candidate.id in admitted_ids:
            reason = "high_value_or_uncertainty"
        elif signal["confidence"] >= 0.8 and signal["value"] < 0.5:
            reason = "high_confidence_low_value"
        else:
            reason = "budget_deferred"
        decisions.append(
            {
                "candidate_id": candidate.id,
                "decision": "admitted" if candidate.id in admitted_ids else "deferred",
                "reason": reason,
                **signal,
            }
        )
    axis_floor_ids = sorted(candidate_id for candidate_id, reasons in floor_reasons.items() if any(reason.startswith("axis:") for reason in reasons))
    family_floor_ids = sorted(candidate_id for candidate_id, reasons in floor_reasons.items() if any(reason.startswith("family:") for reason in reasons))
    uncertainty_floor_ids = sorted(candidate_id for candidate_id, reasons in floor_reasons.items() if "uncertainty" in reasons)
    audit = {
        "schema": "cogev.pre_rank_admission.v1",
        "requested_limit": max(0, int(limit)),
        "effective_limit": effective_limit,
        "admitted_candidate_ids": [candidate.id for candidate in admitted],
        "deferred_candidate_ids": [candidate.id for candidate in deferred],
        "floor_candidate_ids": sorted(protected_ids),
        "axis_floor_candidate_ids": axis_floor_ids,
        "family_floor_candidate_ids": family_floor_ids,
        "uncertainty_floor_candidate_ids": uncertainty_floor_ids,
        "decisions": decisions,
        "effect": "evaluation_order_and_round_only_candidates_preserved_evaluator_authority_unchanged",
    }
    return admitted, audit


def _admission_signals(candidate: CandidateGenome, deferred_rounds: int) -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    evaluator = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
    evidence = latest_evidence_record(candidate)
    confidence = float(evidence.confidence) if evidence is not None else (1.0 if evaluator else 0.0)
    uncertainty_items = len(candidate.uncertainty_notes) + len(candidate.missing_parts)
    uncertainty = min(1.0, uncertainty_items / max(1.0, uncertainty_items + 2.0))
    scores = candidate.multihead_scores if isinstance(candidate.multihead_scores, dict) else {}
    value = max(
        float(scores.get("answer_likelihood", 0.0) or 0.0),
        float(scores.get("objective_alignment", 0.0) or 0.0),
        float(scores.get("objective_score", 0.0) or 0.0),
    )
    return {
        "deferred_rounds": max(0, int(deferred_rounds)),
        "evaluated": bool(evaluator),
        "uncertainty": round(uncertainty, 6),
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "value": round(max(0.0, min(1.0, value)), 6),
    }



__all__ = ["EvaluateStage"]
