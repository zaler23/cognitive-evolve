from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory, challenge_from_diagnostic
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure, apply_evidence_record
from cognitive_evolve_runtime.evaluators.evidence_authority import aggregate_evidence_state
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveConfig, AdaptiveRuntimeController, AdaptiveRuntimeState, apply_research_final_gate_directives
from cognitive_evolve_runtime.nexus.adaptive.research import ResearchConfig, ResearchContext, ResearchSignal, merge_research_signals
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.spatial_selection import SpatialSelectionExtension
from cognitive_evolve_runtime.nexus.adaptive.spatial_population import build_or_update_spatial_state
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.contract_refinement import ContractRefinementExtension


def test_research_registry_disabled_is_noop() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": False}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    assert controller.to_dict()["enabled_features"]["research_extension_registry"] is False
    assert controller.research_advisory_features(candidates=[]) == {}


def test_noop_registry_snapshot_restore_is_stable() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "extensions": {"noop": {"enabled": True}}}})
    first = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    payload = first.to_dict()
    restored = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict(), restored_state=payload)
    assert restored.to_dict()["research_metrics"]["active_extensions"] == 1
    assert restored.to_dict()["research_extensions"]["version"].startswith("adaptive-research")


def test_research_enabled_without_explicit_extension_does_not_auto_enable_noop() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())

    assert controller.research_registry.enabled is False
    assert controller.to_dict()["research_metrics"]["active_extensions"] == 0


def test_research_signal_merge_is_deterministic_and_candidate_scoped() -> None:
    left = ResearchSignal(source="b", round_index=1, selection_advisory={"C1": {"plan_value": 0.2, "risk": 0.1}}, warnings=["w2"])
    right = ResearchSignal(source="a", round_index=1, selection_advisory={"C1": {"plan_value": 0.4, "risk": 0.3}}, warnings=["w1"])
    merged1 = merge_research_signals([left, right]).to_dict()
    merged2 = merge_research_signals([right, left]).to_dict()
    assert merged1 == merged2
    assert merged1["selection_advisory"]["C1"]["plan_value"] == 0.4
    assert merged1["selection_advisory"]["C1"]["risk"] == 0.3


def test_evidence_authority_preserves_certificate_against_probe() -> None:
    state = aggregate_evidence_state(
        [
            EvidenceRecord(candidate_id="C1", score=0.9, confidence=1.0, final_blocked=False, metadata={"authority": "certificate", "artifact_hash": "h1"}),
            EvidenceRecord(candidate_id="C1", score=0.1, confidence=0.2, final_blocked=True, metadata={"authority": "probe", "artifact_hash": "h1"}),
        ]
    )
    assert state["final_blocked"] is False
    assert state["final_score"] == 0.9


def test_evidence_authority_invalidates_certificate_on_artifact_hash_change() -> None:
    state = aggregate_evidence_state(
        [
            EvidenceRecord(candidate_id="C1", score=0.9, confidence=1.0, final_blocked=False, metadata={"authority": "certificate", "artifact_hash": "old"}),
            EvidenceRecord(candidate_id="C1", score=0.3, confidence=0.5, final_blocked=True, metadata={"authority": "verifier", "artifact_hash": "new"}),
        ]
    )
    assert state["final_blocked"] is True
    assert state["final_score"] == 0.0


def test_evidence_authority_invalidates_certificate_when_policy_changes() -> None:
    artifact_state = {"artifact_type": "cache_policy", "normalized_artifact": {"admission": {}}, "status": "clean", "final_eligible": True}
    state = aggregate_evidence_state(
        [
            EvidenceRecord(candidate_id="C1", score=0.9, confidence=1.0, final_blocked=False, metadata={"authority": "certificate", "artifact_state": artifact_state, "artifact_policy": {"artifact_type": "cache_policy", "required_fields": ["admission"]}}),
            EvidenceRecord(candidate_id="C1", score=0.7, confidence=0.5, final_blocked=True, metadata={"authority": "verifier", "artifact_state": artifact_state, "artifact_policy": {"artifact_type": "cache_policy", "required_fields": ["admission", "eviction"]}}),
        ]
    )

    assert state["final_blocked"] is True
    assert state["final_score"] == 0.0


def test_search_pressure_challenge_weights_migrates_legacy_advisory() -> None:
    pressure = SearchPressure.from_dict({"id": "p", "selection_advisory": {"case-123": 0.8}, "target_challenge_ids": ["case-123"]})
    assert pressure.challenge_weights == {"case-123": 0.8}
    assert pressure.selection_advisory == {}
    assert pressure.metadata["legacy_selection_advisory_migrated_to_challenge_weights"] is True


def test_challenge_memory_compile_writes_challenge_weights_only() -> None:
    memory = ChallengeMemory()
    record = EvidenceRecord(candidate_id="C1", score=0.7, final_blocked=True, emitted_challenge_ids=["case-x"], diagnostics=["missing_required_fields: admission"], metadata={"challenge_items": [challenge_from_diagnostic(candidate_id="C1", source="test", diagnostic="missing_required_fields: admission")]})
    memory.ingest(record, round_index=1)
    pressure = memory.compile_search_pressure(parent_id="C1")
    assert pressure is not None
    assert pressure.challenge_weights
    assert pressure.selection_advisory == {}


def test_target_tracking_generated_evaluated_resolved_lifecycle() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "extensions": {"noop": {"enabled": True}}}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    record = EvidenceRecord(candidate_id="C2", target_challenge_ids=["case-a"], resolved_challenge_ids=["case-a"])
    controller.record_generated_targets(candidate_id="C2", challenge_ids=["case-a"], pressure_id="p1", round_index=1)
    controller.record_evaluated_targets(candidate_id="C2", challenge_ids=["case-a"], record=record, round_index=2)
    controller.record_resolved_targets(candidate_id="C2", challenge_ids=["case-a"], record=record, round_index=2)
    controller.record_resolved_targets(candidate_id="C2", challenge_ids=["case-a"], record=record, round_index=2)
    payload = controller.to_dict()
    metrics = payload["research_metrics"]
    assert metrics["generated_targeted_count"] == 1
    assert metrics["evaluated_targeted_count"] == 1
    assert metrics["resolved_targeted_count"] == 1
    assert metrics["generated_target_evaluation_rate"] == 1.0
    assert metrics["evaluated_target_resolution_rate"] == 1.0
    assert metrics["targeted_resolution_rate"] == 1.0


def test_research_mode_observe_drops_effects_but_keeps_metrics() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "mode": "observe", "extensions": {"parameter_sweep": {"enabled": True}}}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    candidate = CandidateGenome(id="C1", artifact="x")
    candidate.metadata["parameter_space"] = {"X": [1, 2]}

    controller.record_evaluator_summary(round_index=1, evaluated=1, passed=0, failed=1, candidates=[candidate])

    payload = controller.to_dict()
    assert "evidence_records" not in candidate.metadata
    assert payload["research_metrics"]["parameter_sweep.parameter_sweep_candidate_count"] == 1
    assert "research_observe_mode_dropped_effective_signal" in payload["research_warnings"]


def test_research_mode_advisory_drops_extension_evidence_but_applies_pressure() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "mode": "advisory", "extensions": {"budget_backpressure": {"enabled": True, "pending_threshold": 0}}}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    candidate = CandidateGenome(id="C1", artifact="x")
    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", score=0.1, target_challenge_ids=["case-a"], final_blocked=True, cost={"tokens": 100}))

    controller.record_evaluator_summary(round_index=1, evaluated=1, passed=0, failed=1, candidates=[candidate])
    pressure = controller.compile_search_pressure(parent_id="C1", parent=candidate, candidates=[candidate])

    assert pressure is not None
    assert "Backpressure" in pressure.mutation_instruction


def test_spatial_extension_reuses_existing_spatial_state_model() -> None:
    candidate = CandidateGenome(id="C1", artifact="x", multihead_scores={"frontier_score": 0.6})
    spatial = build_or_update_spatial_state([candidate], existing=None, round_index=1)
    adaptive_state = AdaptiveRuntimeState(spatial=spatial.to_dict())
    signal = SpatialSelectionExtension({"enabled": True}).before_parent_selection(ResearchContext(round_index=1, candidates=[candidate], adaptive_state=adaptive_state))
    assert signal.metrics["spatial_extension_uses_existing_state"] is True
    assert "C1" in signal.selection_advisory


def test_contract_refinement_only_proposes_user_decision_not_contract_mutation() -> None:
    memory = ChallengeMemory()
    for index in range(5):
        record = EvidenceRecord(candidate_id=f"C{index}", diagnostics=[f"semantic gap {index}"], metadata={"challenge_items": [challenge_from_diagnostic(candidate_id=f"C{index}", source="test", diagnostic=f"semantic gap {index}")]})
        memory.ingest(record, round_index=1)
    ext = ContractRefinementExtension({"generic_challenge_threshold": 5})
    ext.after_evidence(ResearchContext(round_index=1, candidates=[], challenge_memory=memory))
    signal = ext.before_final_projection(ResearchContext(round_index=1, candidates=[], challenge_memory=memory))
    assert signal.final_gate_directives[0]["requires_user_decision"] is True
    assert signal.final_gate_directives[0]["silent_mutation_allowed"] is False


def test_unknown_research_signal_field_warns_not_silent() -> None:
    signal = ResearchSignal.from_dict({"source": "test", "round_index": 1, "archive_directives": [{"x": 1}]})
    assert any("unknown_research_signal_fields" in item for item in signal.warnings)


def test_active_final_gate_directive_blocks_certificate_but_report_mode_does_not() -> None:
    base = {"objective_solved": True, "candidate_id": "C1", "blocking_reasons": []}

    active = apply_research_final_gate_directives(base, [{"kind": "parametric_candidate_not_final", "candidate_id": "C1", "enforcement": "blocking"}])
    report = apply_research_final_gate_directives(base, [{"kind": "parametric_candidate_not_final", "candidate_id": "C1", "enforcement": "report"}])

    assert active["objective_solved"] is False
    assert "parametric_candidate_not_collapsed" in active["blocking_reasons"]
    assert report["objective_solved"] is True
    assert report["research_final_gate_passed"] is True


def test_active_registry_final_gate_directives_reach_certificate_path() -> None:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "mode": "active", "extensions": {"parameter_sweep": {"enabled": True}}}})
    controller = AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())
    candidate = CandidateGenome(id="C1", artifact="template {{X}}")
    candidate.metadata["parameter_space"] = {"X": [1, 2]}

    controller.record_evaluator_summary(round_index=3, evaluated=1, passed=0, failed=1, candidates=[candidate])
    controller.before_final_projection(candidates=[candidate], final_certificate={"objective_solved": True, "candidate_id": "C1"})
    certificate = apply_research_final_gate_directives({"objective_solved": True, "candidate_id": "C1", "blocking_reasons": []}, controller.final_gate_directives())

    assert certificate["objective_solved"] is False
    assert "parametric_candidate_not_collapsed" in certificate["blocking_reasons"]
