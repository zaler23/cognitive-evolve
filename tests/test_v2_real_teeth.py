from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.concepts.ablation import ConceptEffectReport
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, apply_evidence_record
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveConfig, AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.adaptive.effect_application import effect_key
from cognitive_evolve_runtime.nexus.adaptive.research.manager import ResearchExtensionRegistry
from cognitive_evolve_runtime.nexus.adaptive.research.registry import ResearchConfig
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.controller import _graded_output_for_final_state
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.strength import candidate_verification_strength
from cognitive_evolve_runtime.verification.types import VerificationResult


def _adaptive(*, mode: str = "active") -> AdaptiveRuntimeController:
    cfg = AdaptiveConfig.from_sources(explicit={"enabled": True, "research": {"enabled": True, "mode": mode, "extensions": {}}})
    return AdaptiveRuntimeController.from_sources(explicit=cfg.to_dict())


def test_v2_effect_production_is_not_decision_changed_until_consumed() -> None:
    registry = ResearchExtensionRegistry(config=ResearchConfig(enabled=True, mode="active", trace_enabled=True, ablation_enabled=True))
    registry.apply_signal(ResearchSignal(source="parameter_sweep", round_index=1, candidate_transforms=[{"candidate_id": "C1", "kind": "collapse_params"}]))

    assert registry.state.trace_entries
    assert registry.state.trace_entries[-1]["decision_changed"] is False
    assert registry.state.candidate_transforms[0]["origin"] == "parameter_sweep"
    report = ConceptEffectReport.from_trace_entries(registry.state.trace_entries).to_dict()
    assert report["concepts"].get("parameter_sweep", {}).get("decision_changed_count", 0) == 0

    registry.record_effect_application(channel="candidate_transforms", item=registry.state.candidate_transforms[0], changed=True, consumer="test")
    report = ConceptEffectReport.from_trace_entries(registry.state.trace_entries).to_dict()
    assert report["concepts"]["parameter_sweep"]["decision_changed_count"] == 1


def test_effect_key_canonicalizes_tuple_float_descriptor_stably() -> None:
    left = effect_key("archive_directives", {"kind": "rebalance", "descriptor": ("cell", 1.23000000000001)})
    right = effect_key("archive_directives", {"kind": "rebalance", "descriptor": ["cell", 1.23]})
    assert left == right


def test_budget_directive_reserves_parent_slot_once_in_active_mode() -> None:
    c1 = CandidateGenome(id="C1", artifact="a", current_fate=CandidateFate.ACTIVE.value)
    c2 = CandidateGenome(id="C2", artifact="b", current_fate=CandidateFate.ACTIVE.value)
    adaptive = _adaptive(mode="active")
    directive = {"target": "C2", "weight": 2.0, "reason": "roi", "roi_estimate": 2.0}
    adaptive.research_registry.state.budget_directives = [directive]
    rounder = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=1), adaptive=adaptive)

    selected = rounder._apply_budget_directives_to_parents([c1], population=CandidatePopulation([c1, c2]), limit=1)
    assert [candidate.id for candidate in selected] == ["C2"]
    assert adaptive.effect_consumed("budget_directives", directive) is True

    again = rounder._apply_budget_directives_to_parents([c1], population=CandidatePopulation([c1, c2]), limit=1)
    assert [candidate.id for candidate in again] == ["C1"]


def test_budget_directive_advisory_mode_only_records_noop() -> None:
    c1 = CandidateGenome(id="C1", artifact="a", current_fate=CandidateFate.ACTIVE.value)
    c2 = CandidateGenome(id="C2", artifact="b", current_fate=CandidateFate.ACTIVE.value)
    adaptive = _adaptive(mode="advisory")
    directive = {"target": "C2", "weight": 2.0, "reason": "roi", "roi_estimate": 2.0}
    adaptive.research_registry.state.budget_directives = [directive]
    rounder = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=1), adaptive=adaptive)
    selected = rounder._apply_budget_directives_to_parents([c1], population=CandidatePopulation([c1, c2]), limit=1)
    assert [candidate.id for candidate in selected] == ["C1"]
    assert adaptive.effect_consumed("budget_directives", directive) is False
    assert adaptive.research_registry.state.effect_applications[-1]["changed"] is False


def test_archive_directive_updates_qd_state_and_parent_ordering() -> None:
    target = CandidateGenome(id="C2", artifact="x", core_mechanism="topology lens", current_fate=CandidateFate.ACTIVE.value)
    other = CandidateGenome(id="C1", artifact="x", current_fate=CandidateFate.ACTIVE.value)
    archives = ArchiveManager()
    directive = {"kind": "add_descriptor", "descriptor": ["pattern", "topology"], "payload": {"source_candidate_ids": ["C2"], "descriptor_token": "topology"}}
    result = archives.apply_archive_directives([directive], [other, target])[0]
    assert result["changed"] is True
    assert archives.quality_diversity.directive_boost(target) > archives.quality_diversity.directive_boost(other)


def test_archive_directive_without_matchable_payload_is_noop() -> None:
    archives = ArchiveManager()
    result = archives.apply_archive_directives([{"kind": "add_descriptor", "descriptor": [], "payload": {}}], [CandidateGenome(id="C1")])[0]
    assert result["changed"] is False


def test_candidate_verification_strength_ignores_failed_or_non_replayable_results() -> None:
    candidate = CandidateGenome(id="C1")
    candidate.verification_trace = [
        _measured_verification_result(VerificationStrength.EMPIRICAL, replayable=False).to_dict(),
        _measured_verification_result(VerificationStrength.FORMAL, passed=False, replayable=True).to_dict(),
        _measured_verification_result(VerificationStrength.DECOMPOSED, replayable=True).to_dict(),
        VerificationResult(True, strength=VerificationStrength.FORMAL, replayable=True).to_dict(),
    ]
    assert candidate_verification_strength(candidate) == VerificationStrength.DECOMPOSED


def test_old_objective_solved_without_actual_formal_result_returns_portfolio() -> None:
    candidate = CandidateGenome(id="C1", artifact="answer")
    synthesis = SynthesizedResult(status="synthesized", final_answer="answer", best_candidate_id="C1", closure_certificate={"objective_solved": True})
    graded = _graded_output_for_final_state(population=CandidatePopulation([candidate]), synthesis=synthesis, final_certificate={"objective_solved": True, "candidate_id": "C1"}, latent_replay_audit={})
    assert graded.mode == "graded_portfolio"


def test_verified_result_requires_actual_formal_replayable_result() -> None:
    candidate = CandidateGenome(id="C1", artifact="answer")
    result = _measured_verification_result(VerificationStrength.FORMAL, evidence_ref="e1", verifier_fingerprint="vf", cache_key="ck")
    candidate.verification_trace = [result.to_dict()]
    synthesis = SynthesizedResult(status="synthesized", final_answer="answer", best_candidate_id="C1", closure_certificate={"objective_solved": True})
    graded = _graded_output_for_final_state(population=CandidatePopulation([candidate]), synthesis=synthesis, final_certificate={"objective_solved": True, "candidate_id": "C1"}, latent_replay_audit={})
    assert graded.mode == "verified_result"
    assert graded.replay_certificate["verification_cache_key"] == "ck"


def test_context_transform_changes_prompt_payload_hash_and_preserves_protected_sections() -> None:
    payload = {
        "contract": {"normalized_goal": "g"},
        "world": {"kind": "text", "summary": "w"},
        "policy": {"fitness_axes": ["a"]},
        "history": [{"round": 1}],
        "archives": ArchiveManager(),
        "_prompt_context_controls": {"protect_refs": ["problem_spec", "verification_plan"], "drop_refs": ["drop:history"], "verification_plan": {"modality": "formal"}, "view_hash": "v1"},
    }
    view = build_prompt_view("nexus_diagnose_search_state", payload, max_chars=900)
    assert "history" not in view.payload
    assert "contract" in view.payload
    assert "verification_plan" in view.payload
    assert view.metadata["context_transform_applied"] is True


def test_verification_obligations_render_as_verification_regime_section() -> None:
    payload = {
        "contract": {"normalized_goal": "g"},
        "policy": {"fitness_axes": ["a"]},
        "candidates": [CandidateGenome(id="C1", artifact="x")],
        "_prompt_context_controls": {
            "verification_regime": [
                {
                    "id": "obl1",
                    "origin": "test",
                    "must_pass": True,
                    "strength_contribution": 99,
                    "replayable": True,
                    "exogeneity_probe": {"content": "engine probe"},
                }
            ]
        },
    }
    view = build_prompt_view("nexus_diagnose_search_state", payload, max_chars=1200)
    assert "verification_regime" in view.payload
    rendered = view.payload["verification_regime"][0]
    assert rendered["id"] == "obl1"
    assert "strength_contribution" not in rendered
    assert "replayable" not in rendered


def test_verification_regime_protected_under_tight_budget() -> None:
    payload = {
        "contract": {"normalized_goal": "g"},
        "candidates": [CandidateGenome(id=f"C{i}", artifact="x" * 1000) for i in range(8)],
        "_prompt_context_controls": {
            "verification_regime": [
                {
                    "id": "must",
                    "origin": "test",
                    "must_pass": True,
                    "replay_record": {"huge": "z" * 5000},
                    "exogeneity_probe": {"context": "p" * 5000, "content": "c" * 5000},
                }
            ]
        },
    }
    view = build_prompt_view("nexus_diagnose_search_state", payload, max_chars=700)
    assert "verification_regime" in view.payload
    assert view.payload["verification_regime"][0]["id"] == "must"


def test_collapse_params_requires_parameter_slots_for_candidate_transform() -> None:
    from cognitive_evolve_runtime.nexus.adaptive.research import ResearchContext
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.parameter_sweep import ParameterSweepExtension

    candidate = CandidateGenome(id="C1", artifact={"threshold": None})
    candidate.metadata["parameter_space"] = {"THRESHOLD": [1, 2]}
    signal = ParameterSweepExtension({}).after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    assert signal.candidate_transforms == []

    candidate.metadata["parameter_slots"] = {"THRESHOLD": {"path": "threshold"}}
    signal = ParameterSweepExtension({}).after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    assert signal.candidate_transforms


def test_mutation_engine_applies_explicit_parameter_slots_and_freezes_metadata() -> None:
    parent = CandidateGenome(id="C1", artifact={"threshold": None}, metadata={"parameter_space": {"THRESHOLD": [1, 2]}})
    transform = {"candidate_id": "C1", "kind": "collapse_params", "payload": {"assignment": {"THRESHOLD": 2}, "parameter_slots": {"THRESHOLD": {"path": "threshold"}}}}
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["C1"], metadata={"candidate_transforms": [transform]})
    child = MutationEngine().mutate(parent, plan)
    assert child.artifact["threshold"] == 2
    assert child.metadata["parameter_assignment"] == {"THRESHOLD": 2}
    assert "parameter_space" not in child.metadata


def test_immune_signature_not_raw_substring_and_text_obligation_not_replayable_formal() -> None:
    from cognitive_evolve_runtime.nexus.adaptive.research import ResearchContext
    from cognitive_evolve_runtime.nexus.adaptive.research.extensions.immune_necropsy import ImmuneNecropsyExtension

    candidate = CandidateGenome(id="C1", artifact="plain text")
    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", terminal_reject=True, final_blocked=True, diagnostics=["machine_parse_failure: missing field"], metadata={"authority": "verifier"}))
    ext = ImmuneNecropsyExtension({"hard_reject_after": 1})
    signal = ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    obligation = signal.verification_obligations[0]
    assert "machine_parse_failure" not in obligation["signature"]
    assert obligation["replayable"] is False
    assert obligation["strength_contribution"] == 0


def _measured_verification_result(
    strength: VerificationStrength,
    *,
    passed: bool = True,
    replayable: bool = True,
    evidence_ref: str = "evidence",
    verifier_fingerprint: str = "vf",
    cache_key: str = "cache",
) -> VerificationResult:
    return VerificationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        strength=strength,
        evidence_ref=evidence_ref,
        replayable=replayable,
        metadata={
            "verifier_fingerprint": verifier_fingerprint,
            "cache_key": cache_key,
            "measured_strength": strength.name,
            "measured_strength_value": int(strength),
            "honesty_measurements": {
                "exogeneity_score": 1.0,
                "variety_score": 1.0,
                "falsification_score": 1.0,
                "replay_score": 1.0,
                "diagnostics": [],
            },
            "replay_record": {"frozen_artifact_hash": "artifact", "verifier_fingerprint": verifier_fingerprint},
            "diagnostics_only": False,
            "legacy": False,
        },
    )
