from __future__ import annotations

from types import SimpleNamespace

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus import evaluation, nextgen
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.minimal_core import (
    effective_project_attachment_inventory,
    extract_failure_theorem,
    run_core_ablation,
    useful_attachment_stack,
)
from cognitive_evolve_runtime.nexus.seed_coverage import assess_seed_coverage, target_perturb_seed_judgment
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.stop_reasons import DIMINISHING_RETURNS_CHECKPOINT
from cognitive_evolve_runtime.nexus.strategy_comparison import strategy_comparison_context


def _candidate(candidate_id: str, family: str) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        concise_claim=f"claim {candidate_id}",
        core_mechanism=f"mechanism {family}",
        niche_memberships=[family],
        metadata={"search_space": {"family_id": family}},
    )


def test_best_current_payload_and_block_reason_keep_complete_semantics() -> None:
    claims = [f"support-{index}" for index in range(10)]
    candidate = CandidateGenome(
        id="C-long",
        artifact="answer",
        core_mechanism="main claim",
        metadata={"intent_binding": {"supporting_claims": claims}},
        verification_result={"diagnostics": ["D" * 500, "DIAGNOSTIC-TAIL-SENTINEL"]},
    )

    payload = nextgen.best_current_direction_payload(candidate)
    reason = nextgen.blocked_from_verified_claim_reason(candidate)

    assert payload["supporting_claims"] == claims
    assert reason.endswith("DIAGNOSTIC-TAIL-SENTINEL")
    candidate.metadata["terminal_reject_reason"] = ("R" * 500) + "REJECT-TAIL-SENTINEL"
    assert nextgen.blocked_from_verified_claim_reason(candidate).endswith("REJECT-TAIL-SENTINEL")


def test_strategy_comparison_keeps_every_model_authored_item() -> None:
    hypotheses = [f"hypothesis-{index}" for index in range(10)]
    questions = [f"question-{index}" for index in range(10)]
    candidates = [_candidate(f"C{index}", f"F{index}") for index in range(15)]
    for index, candidate in enumerate(candidates):
        candidate.metadata["strategy_observation"] = {"claim": f"observation-{index}"}
    policy = SimpleNamespace(metadata={"strategy_comparison": {"open_hypotheses": hypotheses, "decision_questions": questions}})

    context = strategy_comparison_context(policy, candidates)

    assert context["open_hypotheses"] == hypotheses
    assert context["decision_questions"] == questions
    assert [item["candidate_id"] for item in context["observations"]] == [candidate.id for candidate in candidates]


def test_seed_coverage_and_perturb_judgment_keep_complete_search_context() -> None:
    candidates = [_candidate(f"C{index}", f"F{index}") for index in range(20)]
    for index, candidate in enumerate(candidates):
        candidate.niche_memberships = [f"N{index}-{offset}" for offset in range(6)]
        candidate.metadata["best_current_direction"] = index < 8

    coverage = assess_seed_coverage(candidates)
    diagnosis = "SemanticLooping-" + ("X" * 800) + "-DIAGNOSIS-TAIL-SENTINEL"
    judgment = target_perturb_seed_judgment(candidates, current_round=10, diagnosis={"stagnation_type": diagnosis})

    assert len(coverage["top_families"]) == 20
    assert len(coverage["top_niches"]) == 120
    assert len(coverage["undercovered_family_signals"]) == 20
    assert judgment["best_direction_markers"] == [f"C{index}" for index in range(8)]
    assert judgment["diagnosis_hint"].endswith("diagnosis-tail-sentinel")


def test_minimal_core_keeps_complete_failure_and_attachment_evidence() -> None:
    mechanism = ("M" * 1_200) + "-MECHANISM-TAIL-SENTINEL"
    lesson = ("L" * 2_500) + "-LESSON-TAIL-SENTINEL"
    candidate = CandidateGenome(
        id="C-evidence",
        artifact="answer",
        core_mechanism=mechanism,
        failure_lessons=["first", "second", lesson],
    )

    theorem = extract_failure_theorem(candidate)
    stack = useful_attachment_stack(candidate, factors=[theorem])

    assert theorem is not None
    assert theorem["mechanism"].endswith("MECHANISM-TAIL-SENTINEL")
    assert theorem["theorem_claim"].endswith("LESSON-TAIL-SENTINEL")
    lesson_attachment = next(item for item in stack if item["source"] == "candidate.failure_lessons")
    assert "LESSON-TAIL-SENTINEL" in lesson_attachment["evidence"]


def test_minimal_core_inventory_does_not_drop_late_open_metadata() -> None:
    candidate = _candidate("C-inventory", "family")
    metadata = {
        f"signal_{index}": {"score": 0.5, "rationale": ("R" * 1_000) + f"-TAIL-{index}"}
        for index in range(40)
    }

    inventory = effective_project_attachment_inventory([candidate], policy=SimpleNamespace(metadata=metadata))
    by_source = {item["source"]: item for item in inventory}

    assert len([source for source in by_source if source.startswith("policy.metadata.signal_")]) == 40
    assert by_source["policy.metadata.signal_39"]["evidence"].endswith("-TAIL-39")


def test_minimal_core_report_keeps_every_extracted_failure_theorem() -> None:
    candidates = [_candidate(f"C{index}", f"F{index}") for index in range(20)]
    for index, candidate in enumerate(candidates):
        candidate.failure_lessons = [f"failure-{index}"]

    report = run_core_ablation(candidates)

    assert len(report["failure_theorems"]) == 20


def test_offline_optimization_variants_keep_the_complete_source() -> None:
    source = ("source " * 1_500) + "SOURCE-TAIL-SENTINEL"

    variants = evaluation._offline_variants(source)

    assert len(variants) == 3
    assert all("SOURCE-TAIL-SENTINEL" in item["prompt"] for item in variants)


def test_stop_decision_reads_late_diagnosis_semantics() -> None:
    diagnosis = SearchDiagnosis(notes=("N" * 2_000) + " low expected gain")

    reason = StopDecisionEngine().stop_reason_after_round(
        budget=SimpleNamespace(stop_policy="llm_after_minimum", min_rounds_before_stop=1, history=[]),
        completed_round=1,
        diagnosis=diagnosis,
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1", artifact="answer")]),
        model=None,
    )

    assert reason == DIMINISHING_RETURNS_CHECKPOINT
