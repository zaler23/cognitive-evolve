from __future__ import annotations

from cognitive_evolve_runtime.evidence.ledger import ClaimRecord, EvidenceLedger, SourceRecord
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater
from cognitive_evolve_runtime.nexus.semantics import NexusRoute as Route, assess
from cognitive_evolve_runtime.candidates.genome import CandidateGenome


def test_relative_rater_rejects_empty_candidate_set_instead_of_default_winner() -> None:
    ranking = RelativeRater().rank(candidates=[])
    assert ranking.best_final_answer_id == ""
    assert ranking.mutation_worthy_ids == []


def test_failed_or_auxiliary_candidate_is_not_silent_main_answer() -> None:
    answer = CandidateGenome(id="answer", artifact="direct", concise_claim="direct", core_mechanism="direct", multihead_scores={"answer_likelihood": 0.8, "objective_alignment": 0.8})
    aux = CandidateGenome(id="aux", artifact="router", concise_claim="router", core_mechanism="routing", multihead_scores={"auxiliary_value": 1.0, "answer_likelihood": 0.1})
    ranking = RelativeRater().rank(candidates=[answer, aux])
    assert ranking.best_final_answer_id == "answer"
    assert "aux" in ranking.auxiliary_ids


def test_evidence_lexical_overlap_does_not_hard_support_decisive_claim() -> None:
    ledger = EvidenceLedger(
        sources=[
            SourceRecord(
                id="source:1",
                source_type="local_text",
                locator="fixture",
                text_digest="This theorem is proven complete for the benchmark result in the appendix.",
            )
        ]
    )
    assessed = ledger.assess_claims([ClaimRecord(id="claim:1", text="The theorem is proven complete for all production cases.", decisive=True)])
    assert assessed[0].status == "needs_semantic_review"
    assert assessed[0].status != "supported"


def test_nexus_semantics_missing_model_task_type_is_route_incomplete_not_heuristic_final() -> None:
    class BadModel:
        def classify_task(self, *, prompt: str) -> dict:
            return {"level": "L4_evolutionary", "profile": "research", "search": True, "checkmodel": True}

    assessment = assess("Refactor this architecture", model=BadModel()).to_dict()
    assert assessment["task_type"] == "route_incomplete"
    assert assessment["semantic_control"]["incomplete"] is True
    diagnostic = assessment["semantic_control"]["task_type_diagnostic"]
    assert diagnostic["task_type"] == "route_incomplete"
    assert diagnostic["source"] == "nexus_bounded_profile"


def test_route_incomplete_clamps_engine_to_single_diagnostic_round(tmp_path) -> None:
    from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator

    result = EngineOrchestrator().run(
        "Refactor this architecture",
        context={
            "task_dir": str(tmp_path),
            "rounds": 5,
            "semantic_assessment": {"task_type": "route_incomplete", "semantic_control": {"incomplete": True}},
        },
    )

    assert result.evolution["progress_events"][-1]["max_rounds"] == 1
