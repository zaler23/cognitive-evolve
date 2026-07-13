from __future__ import annotations

import json
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater


def _candidate(index: int, *, chars: int = 6_000) -> CandidateGenome:
    marker = f"EXACT-MIDDLE-{index}"
    artifact = ("L" * (chars // 2)) + marker + ("R" * (chars // 2))
    return CandidateGenome(
        id=f"C{index}",
        artifact=artifact,
        concise_claim=f"candidate {index}",
        core_mechanism=f"mechanism {index}",
        multihead_scores={
            "objective_alignment": index / 10,
            "answer_likelihood": index / 10,
            "core_mechanism_strength": index / 10,
        },
    )


def _ranking_response(payload: dict[str, Any]) -> dict[str, Any]:
    ids = [str(item["id"]) for item in payload["candidates"]]
    winner = ids[-1] if ids else ""
    return {
        "best_final_answer_id": winner,
        "strongest_mechanism_id": winner,
        "mutation_worthy_ids": ids,
        "edge_value_ids": [],
        "auxiliary_ids": [],
        "dormant_ids": [],
        "dominated_pairs": [],
        "crossover_pairs": [],
        "preserve_incomplete_ids": [],
        "pairwise_preferences": [],
        "multihead_observations": {
            candidate_id: {"objective_alignment": int(candidate_id[1:]) / 10}
            for candidate_id in ids
        },
        "raw_notes": "exact batch",
    }


def test_rank_prompt_never_replaces_exact_candidates_with_omission_markers() -> None:
    candidates = [_candidate(index) for index in range(4)]

    view = build_prompt_view(
        "nexus_relative_rank",
        {"candidates": candidates, "contract": {}, "policy": {}},
        max_chars=12_000,
    )

    assert [item["id"] for item in view.payload["candidates"]] == [candidate.id for candidate in candidates]
    assert [item["artifact"] for item in view.payload["candidates"]] == [candidate.artifact for candidate in candidates]
    assert "_omitted_items" not in json.dumps(view.payload["candidates"], ensure_ascii=False)
    assert view.metadata["protected_over_budget"] is True


def test_relative_rank_uses_exact_batches_and_merges_all_candidate_coverage(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "26000")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "26000")
    candidates = [_candidate(index) for index in range(7)]
    calls: list[dict[str, Any]] = []

    def caller(request_type: str, payload: dict[str, Any], _schema: dict[str, Any]) -> dict[str, Any]:
        assert request_type == "nexus_relative_rank"
        calls.append(payload)
        return _ranking_response(payload)

    result = StructuredModelAdapter(caller=caller).relative_rank(
        candidates=candidates,
        contract=NexusObjectiveContract(original_user_goal="rank", normalized_goal="rank"),
        policy=EvolutionPolicy(),
        archives=None,
    )

    assert len(calls) > 1
    seen: set[str] = set()
    by_id = {candidate.id: candidate for candidate in candidates}
    for payload in calls:
        for item in payload["candidates"]:
            candidate_id = str(item["id"])
            assert item["artifact"] == by_id[candidate_id].artifact
            assert "...[truncated]..." not in item["artifact"]
            seen.add(candidate_id)
    assert seen == set(by_id)
    assert set(result["multihead_observations"]) == set(by_id)
    assert result["best_final_answer_id"] == candidates[-1].id
    assert "exact_candidate_batches" in result["raw_notes"]


def test_critique_exact_batches_add_deterministic_coverage_for_model_omissions(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "26000")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "26000")
    candidates = [_candidate(index) for index in range(7)]
    calls: list[dict[str, Any]] = []

    def caller(request_type: str, payload: dict[str, Any], _schema: dict[str, Any]) -> dict[str, Any]:
        assert request_type == "nexus_critique_candidates"
        calls.append(payload)
        candidate_id = str(payload["candidates"][0]["id"])
        return {
            "critiques": [
                {
                    "candidate_id": candidate_id,
                    "round": 3,
                    "strengths": ["model covered this candidate"],
                    "flaws": [],
                    "missing_evidence": [],
                    "proposed_mutations": ["deepen"],
                    "reusable_genes": [],
                    "severity": 0.1,
                }
            ]
        }

    critiques = CritiqueEngine(model=StructuredModelAdapter(caller=caller)).critique(
        candidates=candidates,
        round_index=3,
        contract=NexusObjectiveContract(original_user_goal="critique", normalized_goal="critique"),
        policy=EvolutionPolicy(),
        archives=None,
    )

    assert len(calls) > 1
    seen = {
        str(item["id"])
        for payload in calls
        for item in payload["candidates"]
    }
    assert seen == {candidate.id for candidate in candidates}
    assert {critique.candidate_id for critique in critiques} == seen
    assert any(critique.metadata.get("coverage_source") == "deterministic_model_omission" for critique in critiques)


def test_single_oversized_candidate_gets_deterministic_rank_coverage_without_lossy_call(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "10000")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "10000")
    candidate = _candidate(9, chars=30_000)

    def forbidden(*_: Any, **__: Any) -> dict[str, Any]:  # pragma: no cover - must not run
        raise AssertionError("an oversized exact candidate must not be sent as a lossy excerpt")

    ranking = RelativeRater(model=StructuredModelAdapter(caller=forbidden)).rank(
        candidates=[candidate],
        contract=None,
        policy=None,
        archives=None,
    )

    assert candidate.id in ranking.multihead_observations
    assert "exact_candidate_exceeded_prompt_cap" in ranking.raw_notes


def test_rank_and_critique_cover_more_than_48_exact_candidates(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "500000")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "500000")
    candidates = [_candidate(index, chars=80) for index in range(55)]
    calls: dict[str, list[str]] = {}

    def caller(request_type: str, payload: dict[str, Any], _schema: dict[str, Any]) -> dict[str, Any]:
        ids = [str(item["id"]) for item in payload["candidates"]]
        calls[request_type] = ids
        if request_type == "nexus_relative_rank":
            return _ranking_response(payload)
        return {
            "critiques": [
                {
                    "candidate_id": candidate_id,
                    "round": 4,
                    "strengths": ["covered"],
                    "flaws": [],
                    "missing_evidence": [],
                    "proposed_mutations": ["deepen"],
                    "reusable_genes": [],
                    "severity": 0.1,
                }
                for candidate_id in ids
            ]
        }

    adapter = StructuredModelAdapter(caller=caller)
    ranking = adapter.relative_rank(candidates=candidates, contract=None, policy=None, archives=None)
    critiques = adapter.critique_candidates(
        candidates=candidates,
        round_index=4,
        contract=None,
        policy=None,
        archives=None,
    )

    expected = [candidate.id for candidate in candidates]
    assert calls["nexus_relative_rank"] == expected
    assert calls["nexus_critique_candidates"] == expected
    assert set(ranking["multihead_observations"]) == set(expected)
    assert {item["candidate_id"] for item in critiques} == set(expected)


def test_ranking_and_critique_outputs_are_not_locally_top_n_truncated() -> None:
    candidates = [
        CandidateGenome(
            id=f"R{index}",
            artifact=f"answer {index}",
            multihead_scores={
                "objective_alignment": index / 30,
                "answer_likelihood": index / 30,
                "core_mechanism_strength": index / 30,
                "verifiability": index / 30,
                "robustness": index / 30,
                "novelty": index / 30,
            },
        )
        for index in range(30)
    ]
    ranking = RelativeRater().rank(candidates=candidates)
    target = candidates[0]
    critique = CandidateCritique(
        candidate_id=target.id,
        round=1,
        flaws=[f"flaw-{index}" for index in range(5)],
        missing_evidence=[f"missing-{index}" for index in range(5)],
        reusable_genes=[f"gene-{index}" for index in range(5)],
    )

    CritiqueEngine().apply(candidates=[target], critiques=[critique])

    assert len(ranking.dominated_pairs) == 30 * 29 // 2
    assert len(ranking.pairwise_preferences) == 5 * 29
    assert target.failure_lessons == critique.flaws
    assert target.missing_parts == critique.missing_evidence
    assert target.inherited_genes == critique.reusable_genes
