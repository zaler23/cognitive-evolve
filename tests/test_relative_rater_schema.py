from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater, RelativeRankingResult, relative_rater_schema


def test_relative_rater_schema_contains_required_comparison_heads() -> None:
    schema = relative_rater_schema()

    for key in [
        "best_final_answer_id",
        "strongest_mechanism_id",
        "mutation_worthy_ids",
        "edge_value_ids",
        "auxiliary_ids",
        "dormant_ids",
        "dominated_pairs",
        "crossover_pairs",
        "preserve_incomplete_ids",
    ]:
        assert key in schema["required"]
        assert key in schema["properties"]


def test_relative_rater_roundtrip_and_auxiliary_detection() -> None:
    answer = CandidateGenome(id="answer", multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7})
    aux = CandidateGenome(id="aux", multihead_scores={"auxiliary_value": 0.99, "objective_alignment": 0.1})

    ranking = RelativeRater().rank(candidates=[answer, aux])
    decoded = RelativeRankingResult.from_json(ranking.to_json())

    assert decoded.best_final_answer_id == "answer"
    assert "aux" in decoded.auxiliary_ids


def test_relative_rater_repairs_non_scalar_multihead_observations() -> None:
    ranking = RelativeRankingResult.from_dict(
        {
            "best_final_answer_id": "C1",
            "strongest_mechanism_id": "C1",
            "multihead_observations": {
                "C1": {
                    "objective_alignment": [0.2, "0.8"],
                    "answer_likelihood": {"score": "0.6"},
                    "verifiability": {"a": 0.2, "b": 0.4},
                    "bad": None,
                }
            },
        }
    )

    assert ranking.multihead_observations["C1"]["objective_alignment"] == 0.5
    assert ranking.multihead_observations["C1"]["answer_likelihood"] == 0.6
    assert ranking.multihead_observations["C1"]["verifiability"] == 0.30000000000000004
    assert "ranking_schema_repair" in ranking.raw_notes


def test_relative_rater_accepts_scientific_notation_observations() -> None:
    ranking = RelativeRankingResult.from_dict(
        {
            "best_final_answer_id": "C1",
            "strongest_mechanism_id": "C1",
            "multihead_observations": {
                "C1": {
                    "objective_alignment": "1e-5",
                    "answer_likelihood": "2.3e4",
                    "verifiability": ["1.2e-3", None],
                    "bad": ["abc"],
                }
            },
        }
    )

    assert ranking.multihead_observations["C1"]["objective_alignment"] == 0.00001
    assert ranking.multihead_observations["C1"]["answer_likelihood"] == 1.0
    assert ranking.multihead_observations["C1"]["verifiability"] == 0.0012
    assert "score_dropped" in ranking.raw_notes


def test_relative_rater_malformed_multihead_values_are_dropped_not_crashing() -> None:
    ranking = RelativeRankingResult.from_dict(
        {
            "best_final_answer_id": "C1",
            "strongest_mechanism_id": "C1",
            "multihead_observations": {
                "C1": {
                    "objective_alignment": {},
                    "answer_likelihood": [],
                    "verifiability": "not-a-number",
                    "robustness": ["1.0", "bad", None],
                }
            },
        }
    )

    assert "objective_alignment" not in ranking.multihead_observations["C1"]
    assert "answer_likelihood" not in ranking.multihead_observations["C1"]
    assert "verifiability" not in ranking.multihead_observations["C1"]
    assert ranking.multihead_observations["C1"]["robustness"] == 1.0
    assert "ranking_schema_repair" in ranking.raw_notes


class BadSchemaRankModel:
    def relative_rank(self, *, candidates: list[CandidateGenome], **_: object) -> dict[str, object]:
        return {
            "best_final_answer_id": candidates[0].id,
            "strongest_mechanism_id": candidates[0].id,
            "mutation_worthy_ids": [candidates[0].id],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "multihead_observations": {candidates[0].id: {"objective_alignment": [0.25, "0.75"], "bad": {"x": []}}},
        }


def test_relative_rater_model_schema_repair_does_not_abort_evolution() -> None:
    candidate = CandidateGenome(id="C1", multihead_scores={"objective_alignment": 0.5, "answer_likelihood": 0.5})

    ranking = RelativeRater(model=BadSchemaRankModel()).rank(candidates=[candidate])

    assert ranking.best_final_answer_id == "C1"
    assert ranking.multihead_observations["C1"]["objective_alignment"] == 0.5
    assert "ranking_schema_repair" in ranking.raw_notes
