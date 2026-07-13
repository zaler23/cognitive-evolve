from __future__ import annotations

from types import SimpleNamespace

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import (
    EvidenceRecord,
    apply_evidence_record,
    evidence_final_blocked,
    select_preliminary_incumbent,
)
from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import GradedOutput


def _evaluated(candidate_id: str, *, score: float, passed: bool, artifact: object | None = None) -> CandidateGenome:
    artifact = artifact if artifact is not None else {"answer": candidate_id}
    candidate = CandidateGenome(
        id=candidate_id,
        artifact=artifact,
        concise_claim=candidate_id,
        core_mechanism=candidate_id,
        multihead_scores={"objective_score": score, "correctness": 1.0 if passed else 0.0},
    )
    candidate.metadata["evaluator"] = {
        "candidate_id": candidate_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "metrics": {"score": score},
    }
    artifact_hash = stable_artifact_hash(artifact)
    apply_evidence_record(
        candidate,
        EvidenceRecord(
            candidate_id=candidate_id,
            score=score,
            final_blocked=True,
            metadata={
                "normalized_artifact_hash": artifact_hash,
                "artifact_state": {
                    "normalized_artifact": artifact,
                    "artifact_type": "answer",
                    "status": "clean",
                    "final_eligible": False,
                },
            },
        ),
    )
    return candidate


def _graded() -> GradedOutput:
    return GradedOutput(mode="graded_portfolio", verification_strength=VerificationStrength.NONE)


def test_preliminary_incumbent_prefers_pass_then_score_and_never_unevaluated() -> None:
    incumbent = _evaluated("inc", score=0.95, passed=True)
    lower = _evaluated("lower", score=0.90, passed=True)
    failed = _evaluated("failed", score=0.99, passed=False)
    unmeasured = CandidateGenome(
        id="unmeasured",
        artifact={"answer": "self scored"},
        multihead_scores={"objective_score": 1.0, "frontier_score": 1.0},
    )

    assert select_preliminary_incumbent([unmeasured, lower, failed, incumbent]) is incumbent


def test_preliminary_incumbent_keeps_operator_start_on_an_exact_tie() -> None:
    operator_start = _evaluated("operator", score=0.95, passed=True)
    operator_start.metadata["operator_provided_incumbent"] = True
    equal_child = _evaluated("child", score=0.95, passed=True)
    equal_child.generation = 1

    assert select_preliminary_incumbent([equal_child, operator_start]) is operator_start


def test_preliminary_incumbent_bypasses_model_synthesis() -> None:
    incumbent = _evaluated("inc", score=0.95, passed=True)
    unmeasured = CandidateGenome(id="new", artifact="unmeasured")

    class Model:
        def synthesize_result(self, **_: object) -> dict[str, object]:  # pragma: no cover - must not run
            raise AssertionError("evaluator-backed synthesis must not call the model")

    result = synthesize_result(
        population=CandidatePopulation([incumbent, unmeasured]),
        archives=ArchiveManager(),
        model=Model(),
    )

    assert result.best_candidate_id == "inc"
    assert result.status == "preliminary_evaluator_selected"


def test_final_projection_uses_evaluated_snapshot_and_rejects_unbound_synthesis_artifact() -> None:
    artifact = {"answer": "known-good"}
    incumbent = _evaluated("inc", score=0.95, passed=True, artifact=artifact)
    unmeasured = CandidateGenome(id="new", artifact={"answer": "unmeasured"})
    synthesis = SynthesizedResult(
        status="model_synthesized",
        final_answer="brand new unbound answer",
        best_candidate_id="new",
    )

    projection = build_final_projection(
        population=CandidatePopulation([incumbent, unmeasured]),
        synthesis=synthesis,
        graded_output=_graded(),
    )

    assert projection.candidate_id == "inc"
    assert projection.artifact == artifact
    assert projection.artifact_hash == stable_artifact_hash(artifact)
    assert projection.evaluation_bound is True


def test_compaction_never_drops_preliminary_incumbent() -> None:
    incumbent = _evaluated("inc", score=0.10, passed=True)
    incumbent.core_mechanism = "same-cell"
    incumbent.niche_memberships = ["same-cell"]
    challengers = [
        CandidateGenome(
            id=f"new-{index}",
            artifact=f"new-{index}",
            core_mechanism="same-cell",
            niche_memberships=["same-cell"],
            multihead_scores={"frontier_score": 1.0, "novelty": 1.0},
        )
        for index in range(12)
    ]
    population = CandidatePopulation([incumbent, *challengers])

    compact_live_population(
        population,
        ArchiveManager(),
        EvolutionPolicy(metadata={"quality_diversity_bin_capacity": 1, "quality_diversity_rare_reserve_per_bin": 0}),
        branch_factor=1,
    )

    assert "inc" in {candidate.id for candidate in population.candidates}


def test_round_ranking_and_parent_portfolio_keep_preliminary_incumbent() -> None:
    incumbent = _evaluated("inc", score=0.60, passed=True)
    unmeasured = CandidateGenome(
        id="unmeasured",
        artifact="model self-score only",
        concise_claim="model self-score only",
        core_mechanism="model self-score only",
        multihead_scores={"frontier_score": 1.0, "novelty": 1.0},
    )

    class Model:
        def relative_rank(self, **_: object) -> dict[str, object]:  # pragma: no cover - must not run
            raise AssertionError("external evaluator must remain the selection authority")

    population = CandidatePopulation([unmeasured, incumbent])
    archives = ArchiveManager()
    policy = EvolutionPolicy()
    round_pipeline = EvolutionRound(model=Model(), budget=EvolutionBudget(max_rounds=2, branch_factor=2))

    rankings = round_pipeline.rank(
        population=population,
        archives=archives,
        policy=policy,
        contract=None,
        current_round=1,
    )
    parents = round_pipeline._select_reproduction_parents(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=None,
        world={},
        rankings=rankings,
        diagnosis=SearchDiagnosis(),
        repair_parent_candidates=list(population.candidates),
    )

    assert rankings.best_final_answer_id == "inc"
    assert parents[0].id == "inc"


def test_preliminary_pass_can_remain_search_active_without_becoming_final_eligible() -> None:
    candidate = _evaluated("inc", score=0.95, passed=True)
    candidate.current_fate = "Active"
    archives = ArchiveManager(fates={candidate.id: "Active"})

    assert archives.is_final_answer_eligible(candidate) is False


def test_final_only_evidence_block_does_not_poison_search_fate_after_task_pass() -> None:
    passed = _evaluated("passed", score=0.95, passed=True)
    failed = _evaluated("failed", score=0.95, passed=False)
    parent_blocked = _evaluated("parent-blocked", score=0.95, passed=True)
    terminal = CandidateGenome(
        id="terminal",
        artifact={"answer": "terminal"},
        metadata={"evaluator": {"status": "passed", "passed": True}},
    )
    structural = _evaluated("structural", score=0.95, passed=True)
    passed.evidence_delta = {"task_evaluator_passed": True}
    apply_evidence_record(
        parent_blocked,
        EvidenceRecord(candidate_id=parent_blocked.id, final_blocked=True, parent_blocked=True),
    )
    apply_evidence_record(
        terminal,
        EvidenceRecord(candidate_id=terminal.id, final_blocked=True, terminal_reject=True),
    )
    structural.current_fate = "Culled"
    structural.metadata["structural_failure"] = True

    passed_assignment = ArchiveManager().assign_by_policy(
        [passed],
        SimpleNamespace(best_final_answer_id=passed.id),
    )[0]
    failed_assignment = ArchiveManager().assign_by_policy([failed])[0]
    parent_assignment = ArchiveManager().assign_by_policy([parent_blocked])[0]
    terminal_assignment = ArchiveManager().assign_by_policy([terminal])[0]
    structural_assignment = ArchiveManager().assign_by_policy([structural])[0]

    assert evidence_final_blocked(passed) is True
    assert passed_assignment.fate == "Elite"
    assert passed_assignment.failure_signature == ""
    assert failed_assignment.fate == "Dormant"
    assert failed_assignment.failure_signature
    assert parent_assignment.fate == "Dormant"
    assert parent_assignment.failure_signature
    assert terminal_assignment.fate == "Incubating"
    assert terminal_assignment.future_reactivation_condition
    assert structural_assignment.fate == "Culled"
