"""Runtime bridge between Nexus evolution and M5/M5.1 outcome kernels; Chinese markers are multilingual triggers.

The outcome kernels are intentionally domain-neutral.  This bridge keeps their
runtime integration small: contracts can carry latent objective state, candidate
selection can consume posterior/Pareto signals, and solved claims can be checked
against verified M5 certificates without replacing the existing Nexus loop.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.outcomes.improvement import ImprovementCertificate, OutcomeContract, TrialObservation, certificate_from_dict, compare_outcomes
from cognitive_evolve_runtime.outcomes.evidence_feedback import adapt_latent_feedback
from cognitive_evolve_runtime.outcomes.latent import (
    ExplorationAction,
    FrontierCandidate,
    IntentHypothesis,
    LatentProblemState,
    assess_convergence,
    freeze_outcome_contract,
    pareto_frontier,
    rank_candidates,
    select_exploration_action,
)
from cognitive_evolve_runtime.outcomes.latent_ledger import LatentLedger
from cognitive_evolve_runtime.outcomes.posterior_update import (
    LatentPosteriorSnapshot,
    materialize_posterior_snapshot,
)
from cognitive_evolve_runtime.outcomes.anytime_valid import verify_anytime_valid_certificate
from cognitive_evolve_runtime.outcomes.calibration import CalibrationPolicy, calibration_block_reasons
from cognitive_evolve_runtime.outcomes.closure_bundle import audit_structural_replay_bundle, structural_replay_bundle_hash
from cognitive_evolve_runtime.outcomes.falsification import audit_falsification_gauntlet, falsification_bundle_hash
from cognitive_evolve_runtime.outcomes.problem_model import (
    ModelDiscriminationAction,
    ProblemModelHypothesis,
    ProblemModelLedger,
    ProblemModelPrediction,
    ProblemModelSnapshot,
    detect_problem_residuals,
    initial_problem_model_from_latent_state,
    materialize_problem_model_snapshot,
    propose_structural_models,
    select_model_discrimination_action,
)

LATENT_METADATA_KEY = "latent_problem_state"
LATENT_SUMMARY_KEY = "latent_problem_state_summary"
LATENT_HASH_KEY = "latent_problem_state_hash"
LATENT_LEDGER_KEY = "latent_ledger"
LATENT_POSTERIOR_SNAPSHOT_KEY = "latent_posterior_snapshot"
LATENT_POSTERIOR_SUMMARY_KEY = "latent_posterior_summary"
LATENT_POSTERIOR_HASH_KEY = "latent_posterior_state_hash"
LATENT_DECISION_TRACE_KEY = "latent_decision_trace"
PROBLEM_MODEL_METADATA_KEY = "problem_model_state"
PROBLEM_MODEL_SUMMARY_KEY = "problem_model_summary"
PROBLEM_MODEL_HASH_KEY = "problem_model_hash"
PROBLEM_MODEL_LEDGER_KEY = "problem_model_ledger"
PROBLEM_MODEL_SNAPSHOT_KEY = "problem_model_snapshot"
PROBLEM_MODEL_DECISION_TRACE_KEY = "problem_model_decision_trace"
RUNTIME_TRIAL_PAIR_MIN_EFFECT = 0.05
TRUSTED_TRIAL_PAIR_SOURCES = frozenset({"runtime_verifier", "verifier_result", "tool_verifier", "m5_verifier", "verified_trial"})
TRUSTED_TRIAL_PAIR_CONTAINER_PREFIXES = ("candidate.verification_result",)


def attach_latent_state_if_needed(contract: Any, world: Any | None = None) -> LatentProblemState | None:
    """Attach a serializable latent problem state to a Nexus contract when needed.

    Model-authored state wins.  The local fallback is deliberately generic and
    only triggers for open-ended/ambiguous objectives; it is an initialization
    aid, not a domain classifier or final authority.
    """

    metadata = _contract_metadata(contract)
    existing = latent_state_from_contract(contract)
    if existing is None and _should_initialize_latent_state(contract, world):
        existing = _fallback_latent_state(contract, world)
    if existing is None:
        return None
    summary = latent_state_summary(existing)
    metadata[LATENT_METADATA_KEY] = existing.to_dict()
    metadata[LATENT_SUMMARY_KEY] = summary
    metadata[LATENT_HASH_KEY] = existing.state_hash()
    metadata.setdefault(LATENT_LEDGER_KEY, LatentLedger().to_dict())
    metadata.setdefault("latent_objective_source", "model_or_runtime_initialized")
    _set_contract_metadata(contract, metadata)
    attach_problem_model_if_needed(contract)
    return existing


def latent_state_from_contract(contract: Any | None) -> LatentProblemState | None:
    metadata = _contract_metadata(contract)
    for raw in (
        metadata.get(LATENT_METADATA_KEY),
        metadata.get("latent_state"),
        coerce_dict(getattr(contract, "outcome_policy", {})).get(LATENT_METADATA_KEY) if contract is not None else None,
        coerce_dict(getattr(contract, "outcome_policy", {})).get("latent_state") if contract is not None else None,
    ):
        state = latent_state_from_dict(raw)
        if state is not None:
            return state
    return None


def latent_state_from_dict(raw: Any) -> LatentProblemState | None:
    if isinstance(raw, LatentProblemState):
        return raw
    data = coerce_dict(raw)
    intents_raw = data.get("intents")
    if not isinstance(intents_raw, list) or not intents_raw:
        return None
    intents: list[IntentHypothesis] = []
    for item in intents_raw:
        item_data = coerce_dict(item)
        intent_id = str(item_data.get("id") or item_data.get("name") or "").strip()
        statement = str(item_data.get("statement") or item_data.get("description") or intent_id).strip()
        if not intent_id or not statement:
            continue
        intents.append(
            IntentHypothesis(
                id=intent_id,
                statement=statement,
                posterior=_float(item_data.get("posterior"), default=1.0),
                utility_dimensions=tuple(_str_list(item_data.get("utility_dimensions")) or [intent_id]),
                hard_constraints=tuple(_str_list(item_data.get("hard_constraints"))),
                representation_refs=tuple(_str_list(item_data.get("representation_refs"))),
                evaluator_refs=tuple(_str_list(item_data.get("evaluator_refs"))),
                uncertainty=_float(item_data.get("uncertainty"), default=0.5),
            )
        )
    if not intents:
        return None
    frontier_items = [_frontier_from_dict(item) for item in data.get("frontier_candidates", [])]
    action_items = [_action_from_dict(item) for item in data.get("actions", [])]
    frontier = tuple(item for item in frontier_items if item is not None)
    actions = tuple(item for item in action_items if item is not None)
    return LatentProblemState(
        intents=tuple(intents),
        frontier_candidates=frontier,
        actions=actions,
        evidence_refs=tuple(_str_list(data.get("evidence_refs"))),
    )


def latent_state_summary(state: LatentProblemState) -> dict[str, Any]:
    top = state.top_intent()
    return {
        "version": state.version,
        "state_hash": state.state_hash(),
        "intent_count": len(state.intents),
        "top_intent_id": top.id,
        "top_intent_posterior": top.posterior,
        "posterior_entropy": state.posterior_entropy(),
        "frontier_candidate_count": len(state.frontier_candidates),
        "exploration_action_count": len(state.actions),
    }


def latent_ledger_from_contract(contract: Any | None) -> LatentLedger:
    metadata = _contract_metadata(contract)
    return LatentLedger.from_dict(metadata.get(LATENT_LEDGER_KEY))


def materialize_contract_latent_posterior(
    contract: Any | None,
    *,
    force: bool = False,
    record_events: bool = False,
) -> LatentPosteriorSnapshot | None:
    """Return the pinned posterior snapshot for latent-informed decisions."""

    if contract is None:
        return None
    metadata = _contract_metadata(contract)
    if not force:
        existing = _posterior_snapshot_from_dict(metadata.get(LATENT_POSTERIOR_SNAPSHOT_KEY))
        if existing is not None:
            return existing
    base_state = latent_state_from_contract(contract)
    if base_state is None:
        return None
    ledger = latent_ledger_from_contract(contract)
    snapshot = materialize_posterior_snapshot(base_state, ledger)
    if record_events:
        ledger.record_posterior_updated(snapshot)
        ledger.record_posterior_snapshot_materialized(snapshot)
    metadata[LATENT_LEDGER_KEY] = ledger.to_dict()
    metadata[LATENT_POSTERIOR_SNAPSHOT_KEY] = snapshot.to_dict()
    metadata[LATENT_POSTERIOR_SUMMARY_KEY] = latent_state_summary(snapshot.state)
    metadata[LATENT_POSTERIOR_HASH_KEY] = snapshot.state.state_hash()
    _set_contract_metadata(contract, metadata)
    return snapshot


def attach_problem_model_if_needed(contract: Any | None) -> ProblemModelHypothesis | None:
    """Attach the initial M6-alpha problem-model hypothesis as contract metadata.

    The problem model is derived from the latent posterior, but kept in its own
    ledger/snapshot lane so structural ontology changes never pollute the M5
    contract hash or let novelty count as solved.
    """

    if contract is None:
        return None
    existing = problem_model_from_contract(contract)
    if existing is not None:
        return existing
    latent_state = latent_state_from_contract(contract)
    if latent_state is None:
        return None
    model = initial_problem_model_from_latent_state(latent_state)
    ledger = ProblemModelLedger()
    ledger.add_model(model, idempotency_key=model.provenance_ref or model.model_hash())
    snapshot = materialize_problem_model_snapshot(ledger)
    metadata = _contract_metadata(contract)
    metadata[PROBLEM_MODEL_METADATA_KEY] = model.to_dict()
    metadata[PROBLEM_MODEL_SUMMARY_KEY] = problem_model_summary(model)
    metadata[PROBLEM_MODEL_HASH_KEY] = model.model_hash()
    metadata[PROBLEM_MODEL_LEDGER_KEY] = ledger.to_dict()
    metadata[PROBLEM_MODEL_SNAPSHOT_KEY] = snapshot.to_dict()
    _set_contract_metadata(contract, metadata)
    return model


def problem_model_from_contract(contract: Any | None) -> ProblemModelHypothesis | None:
    metadata = _contract_metadata(contract)
    for raw in (
        metadata.get(PROBLEM_MODEL_METADATA_KEY),
        metadata.get("problem_model"),
        coerce_dict(getattr(contract, "outcome_policy", {})).get(PROBLEM_MODEL_METADATA_KEY) if contract is not None else None,
    ):
        model = ProblemModelHypothesis.from_dict(raw)
        if model is not None:
            return model
    return None


def problem_model_ledger_from_contract(contract: Any | None) -> ProblemModelLedger:
    metadata = _contract_metadata(contract)
    return ProblemModelLedger.from_dict(metadata.get(PROBLEM_MODEL_LEDGER_KEY))


def problem_model_summary(model: ProblemModelHypothesis) -> dict[str, Any]:
    return {
        "version": model.version,
        "model_hash": model.model_hash(),
        "objective_count": len(model.objectives),
        "constraint_count": len(model.constraints),
        "mechanism_count": len(model.causal_mechanisms),
        "subproblem_count": len(model.subproblems),
        "unknown_mass": model.unknown_mass,
        "niche_id": model.niche_id,
        "complexity_score": model.complexity_score,
        "proposal_operator": model.proposal_operator,
    }


def materialize_contract_problem_model_snapshot(
    contract: Any | None,
    *,
    force: bool = False,
    record_decision: bool = False,
    decision_type: str = "problem_model_snapshot",
) -> ProblemModelSnapshot | None:
    if contract is None:
        return None
    metadata = _contract_metadata(contract)
    if not force:
        existing = _problem_model_snapshot_from_dict(metadata.get(PROBLEM_MODEL_SNAPSHOT_KEY))
        if existing is not None:
            return existing
    model = attach_problem_model_if_needed(contract)
    if model is None:
        return None
    ledger = problem_model_ledger_from_contract(contract)
    snapshot = materialize_problem_model_snapshot(ledger)
    if record_decision:
        ledger.record_decision_bound(decision_type=decision_type, snapshot=snapshot)
    metadata[PROBLEM_MODEL_LEDGER_KEY] = ledger.to_dict()
    metadata[PROBLEM_MODEL_SNAPSHOT_KEY] = snapshot.to_dict()
    metadata[PROBLEM_MODEL_SUMMARY_KEY] = problem_model_summary(model)
    metadata[PROBLEM_MODEL_HASH_KEY] = model.model_hash()
    _set_contract_metadata(contract, metadata)
    return snapshot


def propose_problem_models_for_contract(
    *,
    contract: Any | None,
    evidence: list[Any] | None = None,
    certificates: list[Any] | None = None,
    archive_observations: list[Any] | None = None,
    max_proposals: int = 3,
) -> dict[str, Any]:
    """Generate bounded structural proposals from residual evidence."""

    model = attach_problem_model_if_needed(contract)
    if model is None or contract is None:
        return {}
    residuals = detect_problem_residuals(
        model,
        evidence=evidence,
        certificates=certificates,
        archive_observations=archive_observations,
    )
    proposals = propose_structural_models(model, residuals, max_proposals=max_proposals)
    ledger = problem_model_ledger_from_contract(contract)
    added = 0
    deduplicated = 0
    for proposal in proposals:
        event = ledger.add_model(proposal.proposed_model, idempotency_key=proposal.idempotency_key)
        added += 1 if event.event_type == "problem_model_added" else 0
        deduplicated += 1 if event.event_type == "problem_model_deduplicated" else 0
    snapshot = materialize_problem_model_snapshot(ledger)
    metadata = _contract_metadata(contract)
    metadata[PROBLEM_MODEL_LEDGER_KEY] = ledger.to_dict()
    metadata[PROBLEM_MODEL_SNAPSHOT_KEY] = snapshot.to_dict()
    metadata["problem_model_proposals"] = [proposal.to_dict() for proposal in proposals]
    metadata["problem_model_residuals"] = [residual.to_dict() for residual in residuals]
    _set_contract_metadata(contract, metadata)
    return {
        "problem_model_residual_count": len(residuals),
        "problem_model_proposal_count": len(proposals),
        "problem_models_added": added,
        "problem_models_deduplicated": deduplicated,
        "problem_model_ledger_cursor": snapshot.ledger_cursor,
        "problem_model_snapshot_hash": snapshot.snapshot_hash(),
        "proposal_model_hashes": [proposal.proposed_model.model_hash() for proposal in proposals],
    }


def problem_model_discrimination_plan_for_contract(
    contract: Any | None,
    *,
    predictions: list[Any] | None = None,
    cost_by_action: dict[str, float] | None = None,
) -> dict[str, Any]:
    snapshot = materialize_contract_problem_model_snapshot(contract)
    if snapshot is None:
        return {}
    parsed_predictions = [_problem_model_prediction_from_candidate(raw) for raw in predictions or []]
    parsed = [item for item in parsed_predictions if item is not None]
    action = select_model_discrimination_action(parsed, cost_by_action=cost_by_action, snapshot=snapshot)
    if action is None:
        return {}
    return {
        "problem_model_discrimination_action": action.to_dict(),
        "problem_model_decision_trace": action.decision_trace,
        "problem_model_snapshot_hash": action.decision_trace.get("problem_model_snapshot_hash", ""),
        "problem_model_ledger_cursor": action.decision_trace.get("problem_model_ledger_cursor", 0),
    }


def evaluate_m6_closure_gate(
    *,
    contract: Any | None,
    anytime_certificate: Any | None = None,
    calibration_snapshot: Any | None = None,
    calibration_policy: CalibrationPolicy | dict[str, Any] | None = None,
    falsification_bundle: Any | None = None,
    structural_replay_bundle: Any | None = None,
    candidate_confidence: float | None = None,
) -> dict[str, Any]:
    """Evaluate the full M6 no-defer closure gate.

    This is an additive bridge hook: it records a fail-closed summary in
    contract metadata, but it does not relax any M5/M5.2 verified-certificate
    requirement or rewrite Nexus runtime control flow.
    """

    reasons: list[str] = []
    eprocess_ok = verify_anytime_valid_certificate(anytime_certificate) if anytime_certificate is not None else False
    if not eprocess_ok:
        reasons.append("m6_eprocess_certificate_not_verified")

    policy = _calibration_policy_from_any(calibration_policy)
    calibration_reasons = calibration_block_reasons(calibration_snapshot, policy, candidate_confidence=candidate_confidence)
    reasons.extend(calibration_reasons)

    falsification_audit = audit_falsification_gauntlet(falsification_bundle)
    if not falsification_audit.passed:
        reasons.extend(f"m6_falsification:{reason}" for reason in falsification_audit.failure_reasons)

    replay_audit = audit_structural_replay_bundle(structural_replay_bundle)
    if not replay_audit.passed:
        reasons.extend(f"m6_structural_replay:{reason}" for reason in replay_audit.failure_reasons)

    problem_snapshot = materialize_contract_problem_model_snapshot(contract)
    result = {
        "version": "m6-full-closure-gate/v1",
        "passed": not reasons,
        "failure_reasons": list(dict.fromkeys(reasons)),
        "eprocess_verified": eprocess_ok,
        "calibration_block_reasons": list(calibration_reasons),
        "falsification_audit": falsification_audit.to_dict(),
        "structural_replay_audit": replay_audit.to_dict(),
        "problem_model_snapshot_hash": problem_snapshot.snapshot_hash() if problem_snapshot is not None else "",
        "falsification_bundle_hash": falsification_bundle_hash(falsification_bundle),
        "structural_replay_bundle_hash": structural_replay_bundle_hash(structural_replay_bundle),
        "gate_hash": "",
    }
    result["gate_hash"] = "m6gate:" + stable_hash({key: value for key, value in result.items() if key != "gate_hash"})
    if contract is not None:
        metadata = _contract_metadata(contract)
        metadata["m6_closure_gate"] = result
        _set_contract_metadata(contract, metadata)
    return result


def ingest_latent_feedback(
    *,
    contract: Any | None,
    critiques: list[Any] | None = None,
    verifier_results: list[Any] | None = None,
    archive_observations: list[Any] | None = None,
    certificates: list[Any] | None = None,
    trial_observations: list[Any] | None = None,
) -> dict[str, Any]:
    """Append runtime evidence to the latent ledger and rebuild the posterior."""

    if contract is None:
        return {}
    snapshot = materialize_contract_latent_posterior(contract)
    state = snapshot.state if snapshot is not None else latent_state_from_contract(contract)
    if state is None:
        return {}
    output = adapt_latent_feedback(
        state=state,
        critiques=critiques,
        verifier_results=verifier_results,
        archive_observations=archive_observations,
        certificates=certificates,
        trial_observations=trial_observations,
    )
    ledger = latent_ledger_from_contract(contract)
    added = 0
    deduplicated = 0
    rejected = 0
    for item in output.evidence:
        event = ledger.add_evidence(item, idempotency_key=item.evidence_ref)
        added += 1 if event.event_type == "evidence_added" else 0
        deduplicated += 1 if event.event_type == "evidence_deduplicated" else 0
    for item in output.quarantined:
        ledger.reject_evidence(
            item.to_dict(),
            reason=item.reason,
            source_type=item.source_type,
            provenance_ref=item.provenance_ref,
        )
        rejected += 1
    metadata = _contract_metadata(contract)
    metadata[LATENT_LEDGER_KEY] = ledger.to_dict()
    _set_contract_metadata(contract, metadata)
    updated_snapshot = materialize_contract_latent_posterior(contract, force=True, record_events=True)
    return {
        "adapter_version": output.adapter_version,
        "evidence_added": added,
        "evidence_deduplicated": deduplicated,
        "evidence_rejected": rejected,
        "latent_ledger_cursor": updated_snapshot.ledger_cursor if updated_snapshot is not None else ledger.cursor,
        "latent_posterior_snapshot_hash": updated_snapshot.snapshot_hash() if updated_snapshot is not None else "",
        "latent_update_model_version": updated_snapshot.update_model_version if updated_snapshot is not None else "",
        "active_evidence_ids": list(updated_snapshot.active_evidence_ids) if updated_snapshot is not None else [],
    }


def latent_exploration_plan_for_contract(contract: Any | None, *, limit: int = 1) -> dict[str, Any]:
    """Choose high-information latent exploration actions for mutation planning."""

    snapshot = materialize_contract_latent_posterior(contract)
    state = snapshot.state if snapshot is not None else latent_state_from_contract(contract)
    if state is None:
        return {}
    remaining = list(state.actions)
    selected: list[ExplorationAction] = []
    for _ in range(max(0, int(limit or 0))):
        scoped = LatentProblemState(
            intents=state.intents,
            frontier_candidates=state.frontier_candidates,
            actions=tuple(action for action in remaining if action.action_id not in {item.action_id for item in selected}),
            evidence_refs=state.evidence_refs,
            version=state.version,
        )
        action = select_exploration_action(scoped)
        if action is None:
            break
        selected.append(action)
    if not selected:
        return {}
    trace = snapshot.decision_trace(decision_type="exploration_planning") if snapshot is not None else {}
    return {
        "latent_exploration_actions": [action.to_dict() for action in selected],
        "mutation_actions": [_mutation_action_for_exploration(action) for action in selected],
        "latent_decision_trace": trace,
        "latent_posterior_snapshot_hash": trace.get("latent_posterior_snapshot_hash", ""),
        "latent_ledger_cursor": trace.get("latent_ledger_cursor", 0),
    }


def apply_latent_exploration_to_mutation_plans(
    plans: list[MutationPlan],
    contract: Any | None,
    *,
    exploration: dict[str, Any] | None = None,
) -> tuple[list[MutationPlan], dict[str, Any]]:
    """Attach selected latent exploration actions to concrete mutation plans."""

    exploration = dict(exploration or latent_exploration_plan_for_contract(contract, limit=max(1, min(2, len(plans) or 1))))
    model_plan = problem_model_discrimination_plan_for_contract(
        contract,
        predictions=_problem_model_predictions_from_contract(contract),
    )
    if model_plan:
        exploration.setdefault("problem_model_discrimination", model_plan)
    actions = list(exploration.get("latent_exploration_actions") or [])
    problem_action = coerce_dict(coerce_dict(exploration.get("problem_model_discrimination")).get("problem_model_discrimination_action"))
    if not plans or (not actions and not problem_action):
        return plans, exploration
    out: list[MutationPlan] = []
    for index, plan in enumerate(plans):
        data = plan.to_dict()
        action = dict(actions[index % len(actions)]) if actions else {}
        metadata = dict(data.get("metadata") or {})
        instruction_parts = [str(data.get("instruction") or plan.instruction or "").strip()]
        if action:
            metadata["latent_exploration_action"] = action
            metadata["latent_decision_trace"] = dict(exploration.get("latent_decision_trace") or {})
            targets = ", ".join(str(item) for item in action.get("target_intent_ids", []) if item)
            action_kind = str(action.get("kind") or "latent_exploration")
            action_id = str(action.get("action_id") or action_kind)
            instruction_parts.append(
                f"Latent exploration directive {action_id}: run {action_kind}"
                + (f" for intents {targets}" if targets else "")
                + "; generate evidence that can update the latent posterior, not just a narrative preference."
            )
        if problem_action:
            metadata["problem_model_discrimination_action"] = problem_action
            metadata[PROBLEM_MODEL_DECISION_TRACE_KEY] = dict(problem_action.get("decision_trace") or {})
            metadata["problem_model_snapshot_hash"] = metadata[PROBLEM_MODEL_DECISION_TRACE_KEY].get("problem_model_snapshot_hash", "")
            metadata["problem_model_ledger_cursor"] = metadata[PROBLEM_MODEL_DECISION_TRACE_KEY].get("problem_model_ledger_cursor", 0)
            instruction_parts.append(
                "Problem-model discrimination directive "
                + str(problem_action.get("action_id") or "model_discrimination")
                + ": prefer evidence that distinguishes competing problem ontologies; structural novelty alone is not improvement."
            )
        data["instruction"] = " | ".join(part for part in instruction_parts if part)
        data["metadata"] = metadata
        out.append(MutationPlan.from_dict(data))
    return out, exploration


def freeze_improvement_certificate_from_trials(
    *,
    outcome_contract: OutcomeContract,
    baseline: TrialObservation,
    challenger: TrialObservation,
    candidate: CandidateGenome | None = None,
    intent_id: str = "",
) -> ImprovementCertificate:
    """Create and optionally attach an M5 certificate from two trial observations."""

    certificate = compare_outcomes(outcome_contract, baseline, challenger)
    payload = certificate.to_dict()
    if intent_id:
        payload["intent_id"] = intent_id
    if candidate is not None:
        candidate.metadata["improvement_certificate"] = payload
        candidate.verification_result["improvement_certificate"] = payload
        candidate.metadata["improvement_certificate_hash"] = certificate.certificate_hash()
        candidate.metadata["improvement_verified"] = bool(certificate.verified)
    return certificate


def extract_runtime_trial_observations(
    candidates: list[CandidateGenome],
    *,
    outcome_contract: OutcomeContract | None = None,
) -> dict[str, Any]:
    """Extract explicit M5 trial observations from runtime candidate evidence.

    This is intentionally conservative: complete trial observations can later
    mint certificates; partial score payloads remain weak trial evidence only.
    """

    observations: list[TrialObservation] = []
    weak_observations: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for candidate in candidates:
        for container in _candidate_trial_containers(candidate):
            for raw in _list_values(container, ("trial_observation", "m5_trial_observation", "trial_result", "metric_observation", "score_observation")):
                parsed = _trial_observation_from_any(raw, candidate=candidate, outcome_contract=outcome_contract, context=container)
                if parsed is not None:
                    observations.append(parsed)
                else:
                    weak = _weak_trial_payload(raw, candidate=candidate)
                    if weak:
                        weak_observations.append(weak)
            for raw in _list_values(container, ("trial_observations", "m5_trial_observations", "trial_results", "metric_observations", "score_observations")):
                parsed = _trial_observation_from_any(raw, candidate=candidate, outcome_contract=outcome_contract, context=container)
                if parsed is not None:
                    observations.append(parsed)
                else:
                    weak = _weak_trial_payload(raw, candidate=candidate)
                    if weak:
                        weak_observations.append(weak)
            for raw_pair in _list_values(container, ("trial_pair", "m5_trial_pair", "improvement_trial_pair")):
                pair = _trial_pair_from_any(raw_pair, candidate=candidate, outcome_contract=outcome_contract, context=container)
                if pair:
                    pairs.append(pair)
                    observations.extend([pair["baseline"], pair["challenger"]])
    return {
        "trial_observations": _dedupe_trial_observations(observations),
        "weak_trial_observations": _dedupe_weak_trials(weak_observations),
        "trial_pairs": pairs,
    }


def ingest_runtime_trial_feedback(
    *,
    contract: Any | None,
    candidates: list[CandidateGenome],
    outcome_contract: OutcomeContract | None = None,
) -> dict[str, Any]:
    """Extract runtime trials, freeze valid certificates, and ingest feedback."""

    if contract is None or not candidates:
        return {}
    extracted = extract_runtime_trial_observations(candidates, outcome_contract=outcome_contract)
    observations = list(extracted["trial_observations"]) + list(extracted["weak_trial_observations"])
    certificates: list[dict[str, Any]] = []
    for pair in extracted["trial_pairs"]:
        pair_contract = outcome_contract or _outcome_contract_for_trial_pair(contract, pair)
        if pair_contract is None:
            continue
        candidate = pair.get("candidate") if isinstance(pair.get("candidate"), CandidateGenome) else None
        intent_id = str(pair.get("intent_id") or _intent_id_from_outcome_contract(pair_contract))
        certificate = compare_outcomes(pair_contract, pair["baseline"], pair["challenger"])
        payload = certificate.to_dict() | {
            "intent_id": intent_id,
            "source_type": str(pair.get("source_type") or ""),
            "provenance_ref": str(pair.get("provenance_ref") or ""),
            "verifier_run_id": str(pair.get("verifier_run_id") or ""),
            "trial_pair_container_source": str(pair.get("container_source") or ""),
        }
        provenance_failures = _runtime_trial_pair_certificate_failures(pair)
        if provenance_failures:
            payload = _rejected_certificate_payload(payload, provenance_failures)
        if candidate is not None:
            _attach_certificate_payload(candidate, payload)
        certificates.append(payload)
    if not observations and not certificates:
        return {}
    feedback = ingest_latent_feedback(
        contract=contract,
        trial_observations=observations,
        certificates=certificates,
    )
    metadata = _contract_metadata(contract)
    metadata["latent_runtime_trial_feedback"] = {
        "trial_observations": len(observations),
        "trial_pairs": len(extracted["trial_pairs"]),
        "certificates": len(certificates),
        "verified_certificates": sum(1 for item in certificates if bool(improvement_certificate_from_any(item) and improvement_certificate_from_any(item).verified)),
        "latent_ledger_cursor": feedback.get("latent_ledger_cursor", 0),
        "latent_posterior_snapshot_hash": feedback.get("latent_posterior_snapshot_hash", ""),
    }
    _set_contract_metadata(contract, metadata)
    return feedback | metadata["latent_runtime_trial_feedback"]


def audit_latent_decision_replay(contract: Any | None, decision_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify that a latent decision trace can be replayed from ledger events."""

    trace = coerce_dict(decision_trace) or coerce_dict(_contract_metadata(contract).get(LATENT_DECISION_TRACE_KEY))
    expected_hash = str(trace.get("latent_posterior_snapshot_hash") or "")
    cursor_raw = trace.get("latent_ledger_cursor")
    has_pinned_cursor = cursor_raw not in (None, "")
    try:
        cursor = max(0, int(cursor_raw if has_pinned_cursor else 0))
    except (TypeError, ValueError):
        cursor = 0
        has_pinned_cursor = False
    base_state = latent_state_from_contract(contract)
    if base_state is None:
        return {"passed": not expected_hash, "reason": "no_latent_state"}
    ledger = latent_ledger_from_contract(contract)
    snapshot = materialize_posterior_snapshot(base_state, ledger, cursor=cursor if has_pinned_cursor else None)
    actual_hash = snapshot.snapshot_hash()
    return {
        "passed": bool(expected_hash and actual_hash == expected_hash),
        "expected_snapshot_hash": expected_hash,
        "actual_snapshot_hash": actual_hash,
        "latent_ledger_cursor": snapshot.ledger_cursor,
        "latent_update_model_version": snapshot.update_model_version,
        "active_evidence_ids": list(snapshot.active_evidence_ids),
    }


def annotate_candidates_with_latent_signals(candidates: list[CandidateGenome], contract: Any | None) -> dict[str, Any]:
    """Attach latent posterior/Pareto selection signals to candidates in-place."""

    snapshot = materialize_contract_latent_posterior(contract)
    base_state = snapshot.state if snapshot is not None else latent_state_from_contract(contract)
    if base_state is None or not candidates:
        return {}
    frontier_candidates = tuple(_frontier_from_candidate(candidate, base_state) for candidate in candidates)
    state = LatentProblemState(
        intents=base_state.intents,
        frontier_candidates=frontier_candidates,
        actions=base_state.actions,
        evidence_refs=base_state.evidence_refs,
    )
    ranked = rank_candidates(state)
    frontier_ids = {candidate.candidate_id for candidate in pareto_frontier(state)}
    raw_scores = {item.candidate_id: item for item in ranked}
    normalized = _normalize_rank_scores({item.candidate_id: item.score for item in ranked})
    decision_trace = snapshot.decision_trace(decision_type="ranking") if snapshot is not None else {}
    problem_snapshot = materialize_contract_problem_model_snapshot(contract)
    problem_trace = problem_snapshot.decision_trace(decision_type="ranking") if problem_snapshot is not None else {}
    for rank_index, item in enumerate(ranked, start=1):
        candidate = next((candidate for candidate in candidates if candidate.id == item.candidate_id), None)
        if candidate is None:
            continue
        signal = normalized.get(candidate.id, 0.0)
        candidate.multihead_scores["latent_reproductive_signal"] = signal
        candidate.multihead_scores["latent_expected_utility"] = max(0.0, min(1.0, item.expected_utility))
        candidate.multihead_scores["latent_uncertainty_penalty"] = max(0.0, min(1.0, item.uncertainty_penalty))
        candidate.multihead_scores["latent_risk_penalty"] = max(0.0, min(1.0, item.risk_penalty))
        candidate.multihead_scores["latent_cost_penalty"] = max(0.0, min(1.0, item.cost_penalty))
        candidate.metadata["latent_ranking"] = item.to_dict() | {"rank": rank_index, "normalized_signal": signal}
        candidate.metadata["latent_pareto_frontier"] = candidate.id in frontier_ids
        if decision_trace:
            candidate.metadata[LATENT_DECISION_TRACE_KEY] = dict(decision_trace)
            candidate.metadata["latent_ledger_cursor"] = decision_trace["latent_ledger_cursor"]
            candidate.metadata["latent_posterior_snapshot_hash"] = decision_trace["latent_posterior_snapshot_hash"]
            candidate.metadata["latent_update_model_version"] = decision_trace["latent_update_model_version"]
            candidate.metadata["latent_decision_trace_ref"] = decision_trace["latent_decision_trace_ref"]
        if problem_trace:
            candidate.metadata[PROBLEM_MODEL_DECISION_TRACE_KEY] = dict(problem_trace)
            candidate.metadata["problem_model_snapshot_hash"] = problem_trace["problem_model_snapshot_hash"]
            candidate.metadata["problem_model_ledger_cursor"] = problem_trace["problem_model_ledger_cursor"]
            candidate.metadata["problem_model_space_hash"] = problem_trace["problem_model_space_hash"]
        candidate.verification_result.setdefault("latent_rank_eligible", True)
    if decision_trace and contract is not None:
        ledger = latent_ledger_from_contract(contract)
        ledger.record_decision_bound(
            decision_type="ranking",
            snapshot=snapshot,
            decision_payload={
                "ranked_candidate_ids": [item.candidate_id for item in ranked],
                "pareto_frontier_ids": sorted(frontier_ids),
            },
        )
        metadata = _contract_metadata(contract)
        metadata[LATENT_LEDGER_KEY] = ledger.to_dict()
        metadata[LATENT_DECISION_TRACE_KEY] = dict(decision_trace)
        if problem_trace and problem_snapshot is not None:
            problem_ledger = problem_model_ledger_from_contract(contract)
            problem_ledger.record_decision_bound(
                decision_type="ranking",
                snapshot=problem_snapshot,
                decision_payload={
                    "ranked_candidate_ids": [item.candidate_id for item in ranked],
                    "pareto_frontier_ids": sorted(frontier_ids),
                },
            )
            metadata[PROBLEM_MODEL_LEDGER_KEY] = problem_ledger.to_dict()
            metadata[PROBLEM_MODEL_DECISION_TRACE_KEY] = dict(problem_trace)
        _set_contract_metadata(contract, metadata)
    return {
        "latent_state_hash": base_state.state_hash(),
        "latent_ledger_cursor": decision_trace.get("latent_ledger_cursor", 0),
        "latent_posterior_snapshot_hash": decision_trace.get("latent_posterior_snapshot_hash", ""),
        "latent_update_model_version": decision_trace.get("latent_update_model_version", ""),
        "latent_decision_trace_ref": decision_trace.get("latent_decision_trace_ref", ""),
        "problem_model_snapshot_hash": problem_trace.get("problem_model_snapshot_hash", ""),
        "problem_model_ledger_cursor": problem_trace.get("problem_model_ledger_cursor", 0),
        "problem_model_decision_trace_ref": problem_trace.get("problem_model_decision_trace_ref", ""),
        "ranked_candidate_ids": [item.candidate_id for item in ranked],
        "pareto_frontier_ids": sorted(frontier_ids),
    }


def best_candidate_m5_certificate(candidate: CandidateGenome | None) -> ImprovementCertificate | None:
    if candidate is None:
        return None
    for source in (
        candidate.metadata.get("improvement_certificate") if isinstance(candidate.metadata, dict) else None,
        candidate.verification_result.get("improvement_certificate") if isinstance(candidate.verification_result, dict) else None,
        candidate.obligation_delta.get("improvement_certificate") if isinstance(candidate.obligation_delta, dict) else None,
    ):
        cert = improvement_certificate_from_any(source)
        if cert is not None:
            return cert
    return None


def improvement_certificate_from_any(raw: Any) -> ImprovementCertificate | None:
    if isinstance(raw, ImprovementCertificate):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    try:
        return certificate_from_dict(data)
    except Exception:
        return None


def m5_certificate_summary(certificate: ImprovementCertificate | None) -> dict[str, Any]:
    if certificate is None:
        return {
            "improvement_certificate_hash": "",
            "improvement_verified": False,
            "baseline_id": "",
            "challenger_id": "",
            "aggregate_lcb": 0.0,
            "improvement_critical_failures": [],
        }
    return {
        "improvement_certificate_hash": certificate.certificate_hash(),
        "improvement_verified": bool(certificate.verified),
        "baseline_id": certificate.baseline_id,
        "challenger_id": certificate.challenger_id,
        "aggregate_lcb": float(certificate.aggregate_lcb),
        "improvement_critical_failures": list(certificate.critical_failures),
    }


def requires_verified_improvement(contract: Any | None) -> bool:
    policy = coerce_dict(getattr(contract, "outcome_policy", {}) if contract is not None else {})
    return bool(policy.get("requires_verified_improvement_certificate_advisory"))


def latent_completion_override(
    *,
    contract: Any | None,
    completion_status: str,
    synthesis: Any | None = None,
    improvement_certificate: Any | None = None,
) -> dict[str, Any]:
    """Return a conservative completion override for unresolved latent objectives."""

    snapshot = materialize_contract_latent_posterior(contract)
    state = snapshot.state if snapshot is not None else latent_state_from_contract(contract)
    if state is None:
        return {"completion_status": completion_status, "assessment": {}, "overridden": False}
    certificate = improvement_certificate_from_any(improvement_certificate)
    if certificate is None:
        certificate = improvement_certificate_from_any(getattr(synthesis, "improvement_certificate", None))
    assessment = assess_convergence(state, improvement_certificate=certificate)
    status = str(completion_status or "")
    if status == "solved" and not assessment.converged:
        assessment_payload = _assessment_with_trace(assessment.to_dict(), snapshot, decision_type="stop_or_completion")
        assessment_payload["answer_first_advisory"] = "latent convergence did not override completion"
        return {
            "completion_status": completion_status,
            "assessment": assessment_payload,
            "overridden": False,
            "reason": "latent_problem_space_not_converged_advisory",
        }
    return {
        "completion_status": completion_status,
        "assessment": _assessment_with_trace(assessment.to_dict(), snapshot, decision_type="stop_or_completion"),
        "overridden": False,
    }


def latent_stop_allows_solved(*, contract: Any | None, synthesis_certificate: Any | None = None) -> bool:
    return True


def _should_initialize_latent_state(contract: Any, world: Any | None) -> bool:
    metadata = _contract_metadata(contract)
    policy = coerce_dict(getattr(contract, "outcome_policy", {}))
    if metadata.get("latent_objective_enabled") is False or policy.get("latent_objective_enabled") is False:
        return False
    if any(key in metadata or key in policy for key in (LATENT_METADATA_KEY, "latent_state", "latent_intent_hypotheses", "intent_hypotheses")):
        return True
    if metadata.get("latent_objective_enabled") is True or policy.get("latent_objective_enabled") is True:
        return True
    if bool(policy.get("requires_strict_optimum")):
        return True
    goal = str(getattr(contract, "normalized_goal", "") or getattr(contract, "original_user_goal", "") or "").strip().lower()
    ambiguous_markers = (
        "best",
        "better",
        "improve",
        "optimize",
        "evolve",
        "explore",
        "discover",
        "elegant",
        "simpler",
        "more useful",
        "更好",
        "最好",
        "优化",
        "改进",
        "演化",
        "探索",
        "优雅",
        "简洁",
    )
    if any(marker in goal for marker in ambiguous_markers):
        return True
    uncertainty = _str_list(getattr(world, "uncertainty_zones", [])) if world is not None else []
    likely_types = _str_list(getattr(world, "likely_task_types", [])) if world is not None else []
    return bool(uncertainty or len(set(likely_types)) > 1)


def _fallback_latent_state(contract: Any, world: Any | None) -> LatentProblemState:
    dimensions = _latent_dimensions_from_contract(contract)
    posterior = 1.0 / max(1, len(dimensions))
    intents = tuple(
        IntentHypothesis(
            id=dimension,
            statement=f"Improve {dimension} for the declared objective without silently changing the objective.",
            posterior=posterior,
            utility_dimensions=(dimension,),
            hard_constraints=("preserve original user goal", "do not claim solved without verified improvement"),
            representation_refs=("nexus_objective_contract",),
            evaluator_refs=("runtime_relative_ranking", "m5_improvement_certificate"),
            uncertainty=0.5,
        )
        for dimension in dimensions
    )
    actions = tuple(
        ExplorationAction(
            action_id=f"probe_{dimension}",
            kind="intent_disambiguation_or_candidate_materialization",
            target_intent_ids=(dimension,),
            expected_improvement=0.05,
            information_gain=0.1,
            diversity_gain=0.05,
            cost=0.05,
            evidence_ref="runtime_initialized_latent_probe",
        )
        for dimension in dimensions[:3]
    )
    evidence_refs = ["nexus_objective_contract"]
    if world is not None and getattr(world, "input_packet_id", None):
        evidence_refs.append(str(getattr(world, "input_packet_id")))
    return LatentProblemState(intents=intents, actions=actions, evidence_refs=tuple(evidence_refs))


def _latent_dimensions_from_contract(contract: Any) -> list[str]:
    policy = coerce_dict(getattr(contract, "outcome_policy", {}))
    raw_hypotheses = policy.get("latent_intent_hypotheses") or policy.get("intent_hypotheses")
    if isinstance(raw_hypotheses, list):
        ids = []
        for item in raw_hypotheses:
            data = coerce_dict(item)
            value = str(data.get("id") or data.get("name") or "").strip()
            if value:
                ids.append(value)
        if ids:
            return list(dict.fromkeys(ids))[:5]
    dimensions = _str_list(getattr(contract, "success_dimensions", []))
    default_runtime_axes = {"objective_alignment", "verifiability", "robustness"}
    if len(dimensions) >= 2 and set(dimensions) != default_runtime_axes:
        return list(dict.fromkeys(dimensions))[:5]
    return ["clarity", "impact", "faithfulness", "specificity", "feasibility"][:5]


def _frontier_from_candidate(candidate: CandidateGenome, state: LatentProblemState) -> FrontierCandidate:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
    utility = _coerce_intent_scores(metadata.get("latent_intent_scores") or result.get("latent_intent_scores"), state)
    if not utility:
        utility = _derive_intent_scores(candidate, state)
    uncertainty = _coerce_intent_scores(metadata.get("latent_uncertainty_by_intent") or result.get("latent_uncertainty_by_intent"), state)
    default_uncertainty = _bounded_float(metadata.get("latent_uncertainty") or result.get("latent_uncertainty"), default=_candidate_uncertainty(candidate))
    if not uncertainty:
        uncertainty = {intent.id: default_uncertainty for intent in state.intents}
    return FrontierCandidate(
        candidate_id=candidate.id,
        utility_by_intent=utility,
        uncertainty_by_intent=uncertainty,
        novelty=_candidate_score(candidate, "novelty", default=0.0),
        risk=_bounded_float(metadata.get("latent_risk") or result.get("latent_risk"), default=_candidate_score(candidate, "deferral_risk", default=0.0)),
        cost=_bounded_float(metadata.get("latent_cost") or result.get("latent_cost"), default=0.0),
        evidence_refs=tuple(_candidate_evidence_refs(candidate)),
    )


def _assessment_with_trace(assessment: dict[str, Any], snapshot: LatentPosteriorSnapshot | None, *, decision_type: str) -> dict[str, Any]:
    result = dict(assessment)
    if snapshot is not None:
        result.update(snapshot.decision_trace(decision_type=decision_type))
    return result


def _problem_model_snapshot_from_dict(raw: Any) -> ProblemModelSnapshot | None:
    data = coerce_dict(raw)
    models = tuple(
        model
        for model in (ProblemModelHypothesis.from_dict(item) for item in data.get("active_models", []))
        if model is not None
    )
    if not models:
        return None
    try:
        cursor = int(data.get("ledger_cursor") or 0)
    except (TypeError, ValueError):
        cursor = 0
    return ProblemModelSnapshot(
        active_models=models,
        ledger_cursor=max(0, cursor),
        active_model_hashes=tuple(_str_list(data.get("active_model_hashes"))),
        promoted_model_hashes=tuple(_str_list(data.get("promoted_model_hashes"))),
        ledger_replay_hash=str(data.get("ledger_replay_hash") or ""),
        update_model_version=str(data.get("update_model_version") or "problem-model-evolution/v1"),
        version=str(data.get("version") or "problem-model-snapshot/v1"),
    )


def _problem_model_predictions_from_contract(contract: Any | None) -> list[ProblemModelPrediction]:
    metadata = _contract_metadata(contract)
    raw = metadata.get("problem_model_predictions")
    if isinstance(raw, list):
        return [item for item in (_problem_model_prediction_from_candidate(value) for value in raw) if item is not None]
    snapshot = materialize_contract_problem_model_snapshot(contract)
    if snapshot is None or len(snapshot.active_model_hashes) < 2:
        return []
    predictions: list[ProblemModelPrediction] = []
    for index, model_hash in enumerate(snapshot.active_model_hashes[:4]):
        predictions.append(
            ProblemModelPrediction(
                model_hash=model_hash,
                action_id="discriminate_problem_model",
                predicted_outcome=f"model_{index}_prediction",
                probability=1.0,
                evidence_ref="problem_model_snapshot",
            )
        )
    return predictions


def _problem_model_prediction_from_candidate(raw: Any) -> ProblemModelPrediction | None:
    if isinstance(raw, ProblemModelPrediction):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    model_hash = str(data.get("model_hash") or data.get("problem_model_hash") or "")
    action_id = str(data.get("action_id") or data.get("probe_action_id") or "")
    outcome = str(data.get("predicted_outcome") or data.get("outcome") or "")
    if not model_hash or not action_id or not outcome:
        return None
    return ProblemModelPrediction(
        model_hash=model_hash,
        action_id=action_id,
        predicted_outcome=outcome,
        probability=_bounded_float(data.get("probability"), default=1.0),
        evidence_ref=str(data.get("evidence_ref") or ""),
    )


def _calibration_policy_from_any(raw: CalibrationPolicy | dict[str, Any] | None) -> CalibrationPolicy:
    if isinstance(raw, CalibrationPolicy):
        return raw
    data = coerce_dict(raw)
    if not data:
        return CalibrationPolicy()
    return CalibrationPolicy(
        min_total_count=int(data.get("min_total_count") or 30),
        min_count_per_required_bin=int(data.get("min_count_per_required_bin") or 3),
        max_ece=_bounded_float(data.get("max_ece"), default=0.12),
        max_mce=_bounded_float(data.get("max_mce"), default=0.25),
        max_brier_score=_bounded_float(data.get("max_brier_score"), default=0.22),
        min_lower_confidence_coverage=_bounded_float(data.get("min_lower_confidence_coverage"), default=0.80),
        required_bins=tuple(int(item) for item in data.get("required_bins", []) if str(item).isdigit()),
    )


def _mutation_action_for_exploration(action: ExplorationAction) -> str:
    kind = str(action.kind or "").lower()
    if "intent" in kind or "disambigu" in kind or "probe" in kind:
        return "case_split"
    if "risk" in kind or "verify" in kind or "evidence" in kind:
        return "tool_ground"
    if "divers" in kind or "novel" in kind:
        return "rare_inject"
    if "improvement" in kind or "candidate" in kind:
        return "deepen"
    return "latent_exploration"


def _candidate_trial_containers(candidate: CandidateGenome) -> list[dict[str, Any]]:
    containers = [
        _trial_container(candidate.metadata, "candidate.metadata"),
        _trial_container(candidate.verification_result, "candidate.verification_result"),
        _trial_container(candidate.obligation_delta, "candidate.obligation_delta"),
    ]
    for container in list(containers):
        parent_source = str(container.get("__trial_container_source") or "")
        for nested_key in ("m5", "outcome_improvement", "verification_result"):
            nested = coerce_dict(container.get(nested_key))
            if nested:
                containers.append(_trial_container(nested, f"{parent_source}.{nested_key}" if parent_source else nested_key))
    return [container for container in containers if container]


def _trial_container(raw: Any, source: str) -> dict[str, Any]:
    data = coerce_dict(raw)
    if data:
        data.setdefault("__trial_container_source", source)
    return data


def _list_values(container: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    out: list[Any] = []
    for key in keys:
        value = container.get(key)
        if isinstance(value, list | tuple):
            out.extend(value)
        elif value not in (None, ""):
            out.append(value)
    return out


def _trial_observation_from_any(
    raw: Any,
    *,
    candidate: CandidateGenome | None = None,
    outcome_contract: OutcomeContract | None = None,
    context: dict[str, Any] | None = None,
) -> TrialObservation | None:
    if isinstance(raw, TrialObservation):
        return raw
    data = coerce_dict(raw)
    context_data = coerce_dict(context)
    scores = _trial_scores_from_data(data)
    if not scores:
        return None
    try:
        return TrialObservation(
            artifact_id=str(data.get("artifact_id") or data.get("candidate_id") or getattr(candidate, "id", "") or ""),
            contract_hash=str(data.get("contract_hash") or (outcome_contract.contract_hash() if outcome_contract is not None else "")),
            manifest_hash=str(data.get("manifest_hash") or data.get("basis_hash") or ""),
            environment_hash=str(data.get("environment_hash") or data.get("env_hash") or ""),
            evaluator_hash=str(data.get("evaluator_hash") or ""),
            scores=scores,
            uncertainty_radius={str(key): _float(value, default=0.0) for key, value in coerce_dict(data.get("uncertainty_radius")).items()},
            constraints_passed=bool(data.get("constraints_passed", True)),
            hard_constraint_failures=tuple(_str_list(data.get("hard_constraint_failures"))),
            raw_observation_ref=str(data.get("raw_observation_ref") or data.get("evidence_ref") or ""),
            resource_usage={str(key): _float(value, default=0.0) for key, value in coerce_dict(data.get("resource_usage")).items()},
            proposer_ref=str(data.get("proposer_ref") or data.get("proposer") or ""),
            verifier_ref=str(data.get("verifier_ref") or data.get("verifier") or ""),
            evidence_refs=tuple(_str_list(data.get("evidence_refs") or context_data.get("evidence_refs"))),
            seed=str(data.get("seed") or context_data.get("seed") or ""),
            source_type=str(data.get("source_type") or context_data.get("source_type") or ""),
            provenance_ref=str(data.get("provenance_ref") or context_data.get("provenance_ref") or context_data.get("run_id") or ""),
            verifier_run_id=str(data.get("verifier_run_id") or context_data.get("verifier_run_id") or context_data.get("run_id") or ""),
            raw_observation_hash=str(data.get("raw_observation_hash") or data.get("raw_hash") or ""),
        )
    except (TypeError, ValueError):
        return None


def _weak_trial_payload(raw: Any, *, candidate: CandidateGenome) -> dict[str, Any]:
    data = coerce_dict(raw)
    scores = _trial_scores_from_data(data)
    if not scores:
        return {}
    data["scores"] = scores
    data.setdefault("artifact_id", candidate.id)
    data.setdefault("candidate_id", candidate.id)
    data.setdefault("source_type", "trial_observation")
    return data


def _trial_pair_from_any(
    raw: Any,
    *,
    candidate: CandidateGenome,
    outcome_contract: OutcomeContract | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = coerce_dict(raw)
    pair_context = coerce_dict(context) | coerce_dict(data.get("provenance"))
    for key in ("source_type", "provenance_ref", "verifier_run_id", "run_id", "evidence_refs", "seed", "__trial_container_source"):
        if key in data and key not in pair_context:
            pair_context[key] = data[key]
    baseline = _trial_observation_from_any(
        data.get("baseline") or data.get("baseline_observation") or data.get("before"),
        candidate=candidate,
        outcome_contract=outcome_contract,
        context=pair_context,
    )
    challenger = _trial_observation_from_any(
        data.get("challenger") or data.get("challenger_observation") or data.get("after"),
        candidate=candidate,
        outcome_contract=outcome_contract,
        context=pair_context,
    )
    if baseline is None or challenger is None:
        return {}
    return {
        "baseline": baseline,
        "challenger": challenger,
        "candidate": candidate,
        "intent_id": str(data.get("intent_id") or data.get("latent_intent_id") or ""),
        "source_type": str(pair_context.get("source_type") or ""),
        "provenance_ref": str(pair_context.get("provenance_ref") or pair_context.get("run_id") or ""),
        "verifier_run_id": str(pair_context.get("verifier_run_id") or pair_context.get("run_id") or ""),
        "container_source": str(pair_context.get("__trial_container_source") or ""),
    }


def _runtime_trial_pair_certificate_failures(pair: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    source = str(pair.get("source_type") or "").strip()
    container_source = str(pair.get("container_source") or "").strip()
    provenance = str(pair.get("provenance_ref") or pair.get("verifier_run_id") or "").strip()
    if source not in TRUSTED_TRIAL_PAIR_SOURCES:
        failures.append("untrusted_trial_pair_source")
    if not any(container_source.startswith(prefix) for prefix in TRUSTED_TRIAL_PAIR_CONTAINER_PREFIXES):
        failures.append("untrusted_trial_pair_container")
    if not provenance:
        failures.append("missing_trial_pair_provenance")
    elif source in TRUSTED_TRIAL_PAIR_SOURCES and not _trial_pair_provenance_matches_source(source, provenance):
        failures.append("inconsistent_trial_pair_provenance")
    for role in ("baseline", "challenger"):
        observation = pair.get(role)
        if not isinstance(observation, TrialObservation):
            failures.append(f"malformed_{role}_trial_observation")
        elif not _trial_observation_has_replayable_raw_evidence(observation):
            failures.append(f"non_replayable_{role}_raw_evidence")
    return list(dict.fromkeys(failures))


def _trial_pair_provenance_matches_source(source: str, provenance: str) -> bool:
    normalized = provenance.lower()
    if source == "runtime_verifier":
        return any(token in normalized for token in ("verifier", "trial", "m5"))
    if source in {"verifier_result", "tool_verifier", "m5_verifier"}:
        return "verifier" in normalized or "m5" in normalized
    if source == "verified_trial":
        return "trial" in normalized or "verifier" in normalized
    return False


def _trial_observation_has_replayable_raw_evidence(observation: TrialObservation) -> bool:
    ref = str(getattr(observation, "raw_observation_ref", "") or "").strip()
    if not ref:
        return False
    if str(getattr(observation, "raw_observation_hash", "") or "").strip():
        return True
    if tuple(getattr(observation, "evidence_refs", ()) or ()): 
        return True
    return ref.startswith(("raw:", "verifier:", "trace:", "sha256:", "file:"))


def _rejected_certificate_payload(payload: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    out = dict(payload)
    checks = [dict(item) for item in out.get("checks", []) if isinstance(item, dict)]
    for failure in failures:
        checks.append({"check": failure, "passed": False, "detail": "runtime_trial_pair_provenance_gate"})
    existing = [str(item) for item in out.get("critical_failures", []) if str(item)]
    out["checks"] = checks
    out["critical_failures"] = list(dict.fromkeys(existing + list(failures)))
    out["status"] = "rejected"
    return out


def _attach_certificate_payload(candidate: CandidateGenome, payload: dict[str, Any]) -> None:
    candidate.metadata["improvement_certificate"] = dict(payload)
    candidate.verification_result["improvement_certificate"] = dict(payload)
    certificate = improvement_certificate_from_any(payload)
    candidate.metadata["improvement_certificate_hash"] = certificate.certificate_hash() if certificate is not None else stable_hash(payload)
    candidate.metadata["improvement_verified"] = bool(certificate and certificate.verified)


def _outcome_contract_for_trial_pair(contract: Any | None, pair: dict[str, Any]) -> OutcomeContract | None:
    state = latent_state_from_contract(contract)
    if state is None:
        return None
    intent_id = str(pair.get("intent_id") or "")
    try:
        return freeze_outcome_contract(state, intent_id=intent_id or None, min_effect=RUNTIME_TRIAL_PAIR_MIN_EFFECT)
    except ValueError:
        return None


def _intent_id_from_outcome_contract(outcome_contract: OutcomeContract) -> str:
    scope = str(outcome_contract.scope or "")
    if scope.startswith("latent-intent:"):
        return scope.split(":", 1)[1]
    return outcome_contract.metrics[0].id if outcome_contract.metrics else ""


def _trial_scores_from_data(data: dict[str, Any]) -> dict[str, float]:
    for key in ("scores", "metric_scores"):
        raw = coerce_dict(data.get(key))
        if raw:
            return {str(metric): _float(value, default=0.0) for metric, value in raw.items()}
    metrics = data.get("metrics")
    if isinstance(metrics, list | tuple):
        scores: dict[str, float] = {}
        for item in metrics:
            metric = coerce_dict(item)
            metric_id = str(metric.get("id") or metric.get("metric") or metric.get("name") or "").strip()
            if metric_id:
                scores[metric_id] = _float(metric.get("score") or metric.get("value"), default=0.0)
        if scores:
            return scores
    metric_id = str(data.get("metric_id") or data.get("metric") or data.get("intent_id") or data.get("latent_intent_id") or "").strip()
    if metric_id and any(key in data for key in ("score", "value", "quality", "confidence")):
        return {metric_id: _float(data.get("score", data.get("value", data.get("quality", data.get("confidence")))), default=0.0)}
    return {}


def _dedupe_trial_observations(observations: list[TrialObservation]) -> list[TrialObservation]:
    out: dict[str, TrialObservation] = {}
    for observation in observations:
        out.setdefault(observation.observation_hash(), observation)
    return list(out.values())


def _dedupe_weak_trials(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for observation in observations:
        out.setdefault(stable_hash(observation), observation)
    return list(out.values())


def _derive_intent_scores(candidate: CandidateGenome, state: LatentProblemState) -> dict[str, float]:
    scores: dict[str, float] = {}
    for intent in state.intents:
        values = [_candidate_score(candidate, dimension, default=-1.0) for dimension in intent.utility_dimensions]
        values = [value for value in values if value >= 0.0]
        if not values:
            values = [
                _candidate_score(candidate, "objective_alignment", default=0.0),
                _candidate_score(candidate, "answer_likelihood", default=0.0),
                _candidate_score(candidate, "core_mechanism_strength", default=0.0),
            ]
        scores[intent.id] = sum(values) / max(1, len(values))
    return scores


def _candidate_evidence_refs(candidate: CandidateGenome) -> list[str]:
    refs: list[str] = []
    for item in list(candidate.evidence_refs or []) + list(candidate.source_bindings or []):
        data = coerce_dict(item)
        value = str(data.get("ref") or data.get("path") or data.get("source") or data.get("id") or "").strip()
        if value:
            refs.append(value)
    return list(dict.fromkeys(refs))


def _candidate_uncertainty(candidate: CandidateGenome) -> float:
    uncertainty_items = len(candidate.uncertainty_notes) + len(candidate.missing_parts) + len(candidate.failure_lessons)
    return min(1.0, uncertainty_items / max(1.0, uncertainty_items + 4.0))


def _frontier_from_dict(raw: Any) -> FrontierCandidate | None:
    data = coerce_dict(raw)
    candidate_id = str(data.get("candidate_id") or data.get("id") or "").strip()
    if not candidate_id:
        return None
    return FrontierCandidate(
        candidate_id=candidate_id,
        utility_by_intent={key: _bounded_float(value, default=0.0) for key, value in coerce_dict(data.get("utility_by_intent")).items()},
        uncertainty_by_intent={key: _bounded_float(value, default=0.0) for key, value in coerce_dict(data.get("uncertainty_by_intent")).items()},
        novelty=_bounded_float(data.get("novelty"), default=0.0),
        risk=_bounded_float(data.get("risk"), default=0.0),
        cost=_bounded_float(data.get("cost"), default=0.0),
        evidence_refs=tuple(_str_list(data.get("evidence_refs"))),
    )


def _action_from_dict(raw: Any) -> ExplorationAction | None:
    data = coerce_dict(raw)
    action_id = str(data.get("action_id") or data.get("id") or "").strip()
    kind = str(data.get("kind") or "").strip()
    if not action_id or not kind:
        return None
    return ExplorationAction(
        action_id=action_id,
        kind=kind,
        target_intent_ids=tuple(_str_list(data.get("target_intent_ids"))),
        expected_improvement=_float(data.get("expected_improvement"), default=0.0),
        information_gain=_float(data.get("information_gain"), default=0.0),
        diversity_gain=_float(data.get("diversity_gain"), default=0.0),
        risk=_float(data.get("risk"), default=0.0),
        cost=_float(data.get("cost"), default=0.0),
        evidence_ref=str(data.get("evidence_ref") or ""),
    )


def _posterior_snapshot_from_dict(raw: Any) -> LatentPosteriorSnapshot | None:
    data = coerce_dict(raw)
    state = latent_state_from_dict(data.get("state"))
    if state is None:
        return None
    try:
        return LatentPosteriorSnapshot(
            state=state,
            ledger_cursor=int(data.get("ledger_cursor") or 0),
            active_evidence_ids=tuple(_str_list(data.get("active_evidence_ids"))),
            ledger_replay_hash=str(data.get("ledger_replay_hash") or ""),
            update_model_version=str(data.get("update_model_version") or "latent-posterior-update/v1"),
            update_config_hash=str(data.get("update_config_hash") or ""),
            update_trace=coerce_dict(data.get("update_trace")),
            materialized_at_utc=str(data.get("materialized_at_utc") or ""),
            version=str(data.get("version") or "latent-posterior-snapshot/v1"),
        )
    except (TypeError, ValueError):
        return None


def _coerce_intent_scores(raw: Any, state: LatentProblemState) -> dict[str, float]:
    known = {intent.id for intent in state.intents}
    return {str(key): _bounded_float(value, default=0.0) for key, value in coerce_dict(raw).items() if str(key) in known}


def _normalize_rank_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {key: 0.5 for key in scores}
    return {key: max(0.0, min(1.0, (value - lo) / (hi - lo))) for key, value in scores.items()}


def _candidate_score(candidate: CandidateGenome, key: str, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(candidate.multihead_scores.get(key, default))))
    except (TypeError, ValueError):
        return default


def _contract_metadata(contract: Any | None) -> dict[str, Any]:
    if contract is None:
        return {}
    return coerce_dict(getattr(contract, "metadata", {}))


def _set_contract_metadata(contract: Any, metadata: dict[str, Any]) -> None:
    if hasattr(contract, "metadata"):
        setattr(contract, "metadata", metadata)


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _bounded_float(value: Any, *, default: float) -> float:
    return max(0.0, min(1.0, _float(value, default=default)))


__all__ = [
    "LATENT_HASH_KEY",
    "LATENT_LEDGER_KEY",
    "LATENT_METADATA_KEY",
    "LATENT_POSTERIOR_HASH_KEY",
    "LATENT_POSTERIOR_SNAPSHOT_KEY",
    "LATENT_POSTERIOR_SUMMARY_KEY",
    "LATENT_SUMMARY_KEY",
    "annotate_candidates_with_latent_signals",
    "apply_latent_exploration_to_mutation_plans",
    "audit_latent_decision_replay",
    "attach_latent_state_if_needed",
    "best_candidate_m5_certificate",
    "extract_runtime_trial_observations",
    "freeze_improvement_certificate_from_trials",
    "improvement_certificate_from_any",
    "ingest_runtime_trial_feedback",
    "ingest_latent_feedback",
    "latent_completion_override",
    "latent_exploration_plan_for_contract",
    "latent_ledger_from_contract",
    "latent_state_from_contract",
    "latent_state_from_dict",
    "latent_state_summary",
    "latent_stop_allows_solved",
    "materialize_contract_latent_posterior",
    "m5_certificate_summary",
    "requires_verified_improvement",
]
