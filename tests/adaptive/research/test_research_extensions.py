from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, apply_evidence_record
from cognitive_evolve_runtime.nexus.adaptive.research import ResearchContext
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.bft_quorum import BFTQuorumExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.budget_backpressure import BudgetBackpressureExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.chaos import ChaosExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.context_pruning import ContextPruningExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.immune_necropsy import ImmuneNecropsyExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.mdl_compression import MDLCompressionExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.parameter_sweep import ParameterSweepExtension
from cognitive_evolve_runtime.nexus.adaptive.research.extensions.pattern_memory import PatternMemoryExtension


def _candidate(candidate_id: str = "C1") -> CandidateGenome:
    return CandidateGenome(id=candidate_id, artifact="x" * 64, core_mechanism="stable helper pattern", multihead_scores={"frontier_score": 0.6})


def test_pattern_memory_learns_from_resolved_evidence_and_emits_pressure() -> None:
    candidate = _candidate()
    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", score=0.8, target_challenge_ids=["case-a"], resolved_challenge_ids=["case-a"], final_blocked=True))
    ext = PatternMemoryExtension({"enabled": True})
    ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    signal = ext.before_mutation_planning(ResearchContext(round_index=2, candidates=[candidate], parent=candidate))
    assert signal.search_pressures
    assert "evidence-backed patterns" in signal.search_pressures[0].mutation_instruction
    assert ext.snapshot()["patterns"]


def test_immune_necropsy_warns_without_deleting_candidate_or_contract() -> None:
    candidate = _candidate()
    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", terminal_reject=True, final_blocked=True, diagnostics=["machine_parse_failure"], metadata={"authority": "verifier"}))
    ext = ImmuneNecropsyExtension({"hard_reject_after": 2})
    ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    signal = ext.before_parent_selection(ResearchContext(round_index=2, candidates=[candidate]))
    assert signal.metrics["immune_risk_candidates"] >= 0
    final = ext.before_final_projection(ResearchContext(round_index=3, candidates=[candidate]))
    assert all(item.get("contract_mutation_allowed") is not True for item in final.final_gate_directives)


def test_budget_backpressure_tracks_roi_and_can_emit_drain_pressure() -> None:
    candidate = _candidate()
    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", score=0.1, target_challenge_ids=["case-a"], final_blocked=True, cost={"tokens": 100}))
    ext = BudgetBackpressureExtension({"pending_threshold": 0})
    ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    signal = ext.before_mutation_planning(ResearchContext(round_index=2, candidates=[candidate], parent=candidate))
    assert signal.search_pressures
    assert "Backpressure" in signal.search_pressures[0].mutation_instruction


def test_mdl_compression_emits_pressure_only_for_bloated_candidate() -> None:
    candidate = CandidateGenome(id="C1", artifact="x" * 4000)
    ext = MDLCompressionExtension({"compression_threshold": 100})
    ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    signal = ext.before_mutation_planning(ResearchContext(round_index=2, candidates=[candidate], parent=candidate))
    assert signal.search_pressures
    assert "Compress" in signal.search_pressures[0].mutation_instruction


def test_parameter_sweep_marks_parametric_candidate_final_blocked_until_collapse() -> None:
    candidate = _candidate()
    candidate.metadata["parameter_space"] = {"THRESHOLD": [1, 2], "MAX_RETRIES": [1, 3]}
    ext = ParameterSweepExtension({"max_combinations": 4})
    signal = ext.after_evidence(ResearchContext(round_index=1, candidates=[candidate]))
    assert signal.evidence_records
    assert signal.evidence_records[0].final_blocked is True
    final = ext.before_final_projection(ResearchContext(round_index=2, candidates=[candidate]))
    assert final.final_gate_directives[0]["kind"] == "parametric_candidate_not_final"


def test_chaos_and_bft_are_directive_only_not_solved_authorities() -> None:
    candidate = _candidate()
    chaos = ChaosExtension({"seed": 7})
    bft = BFTQuorumExtension({"scope": "final_only"})
    chaos_signal = chaos.before_final_projection(ResearchContext(round_index=1, candidates=[candidate]))
    bft_signal = bft.before_final_projection(ResearchContext(round_index=1, candidates=[candidate]))
    assert chaos_signal.final_gate_directives[0]["replayable"] is True
    assert bft_signal.final_gate_directives[0]["objective_solved_authority"] is False


def test_context_pruning_preserves_resume_critical_state_and_only_emits_pressure() -> None:
    candidate = _candidate()
    ext = ContextPruningExtension({"estimated_tokens_saved": 256})
    signal = ext.before_mutation_planning(ResearchContext(round_index=1, candidates=[candidate], parent=candidate))
    assert signal.search_pressures
    assert signal.search_pressures[0].metadata["resume_critical_state_preserved"] is True
    assert signal.metrics["context_tokens_saved"] == 256
