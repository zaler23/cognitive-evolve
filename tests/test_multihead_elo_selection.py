from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import reproductive_value
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater


def _candidate(candidate_id: str, *, answer: float = 0.5, novelty: float = 0.2) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        concise_claim=f"{candidate_id} claim",
        core_mechanism=f"{candidate_id} mechanism",
        current_fate=CandidateFate.ACTIVE.value,
        multihead_scores={
            "objective_alignment": answer,
            "answer_likelihood": answer,
            "core_mechanism_strength": answer,
            "verifiability": answer,
            "novelty": novelty,
        },
    )


class PairwiseRankModel:
    def __init__(self, preferences: list[dict[str, object]]) -> None:
        self.preferences = preferences

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: object) -> dict[str, object]:
        return {
            "best_final_answer_id": candidates[0].id,
            "strongest_mechanism_id": candidates[0].id,
            "mutation_worthy_ids": [candidate.id for candidate in candidates],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": self.preferences,
            "multihead_observations": {},
            "raw_notes": "model rank",
        }


def test_deterministic_relative_rater_emits_non_star_pairwise_preferences() -> None:
    candidates = [_candidate("a", answer=0.9), _candidate("b", answer=0.6), _candidate("c", answer=0.3)]

    ranking = RelativeRater().rank(candidates=candidates)

    pairs = {(item["winner"], item["loser"], item["axis"]) for item in ranking.pairwise_preferences}
    assert ("a", "b", "answer_likelihood") in pairs
    assert ("b", "c", "answer_likelihood") in pairs
    assert ("a", "c", "answer_likelihood") not in pairs


def test_relative_rater_drops_untrusted_pairwise_preferences_without_elo_effect() -> None:
    candidates = [_candidate("A"), _candidate("B")]
    preferences = [
        {"winner": "PHANTOM", "loser": "A", "axis": "answer_likelihood", "weight": 1.0},
        {"winner": "A", "loser": "A", "axis": "answer_likelihood", "weight": 1.0},
        {"winner": "A", "loser": "B", "axis": "unknown", "weight": 1.0},
        {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": "heavy"},
        {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": float("nan")},
        {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": float("inf")},
        {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": -0.1},
        {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": 1.1},
    ]

    ranking = RelativeRater(model=PairwiseRankModel(preferences)).rank(candidates=candidates)
    elo = MultiHeadElo()
    elo.update_from_relative(ranking)

    assert ranking.pairwise_preferences == []
    assert "ranking_schema_repair:pairwise_preferences_dropped:8" in ranking.raw_notes
    assert elo.ratings == {}
    assert "PHANTOM" not in elo.ratings


def test_relative_rater_keeps_valid_pairwise_preference_and_updates_elo() -> None:
    candidates = [_candidate("A"), _candidate("B")]
    preference = {"winner": "A", "loser": "B", "axis": "answer_likelihood", "weight": 1.0}

    ranking = RelativeRater(model=PairwiseRankModel([preference])).rank(candidates=candidates)
    elo = MultiHeadElo()
    elo.update_from_relative(ranking)

    assert ranking.pairwise_preferences == [preference]
    assert elo.ratings["A"]["answer_likelihood"] == 1012.0
    assert elo.ratings["B"]["answer_likelihood"] == 988.0


def test_multihead_elo_attaches_reproductive_signal_to_candidates() -> None:
    candidates = [_candidate("winner"), _candidate("loser")]
    elo = MultiHeadElo()
    ranking = RelativeRankingResult(
        pairwise_preferences=[
            {"winner": "winner", "loser": "loser", "axis": "answer_likelihood", "weight": 1.0},
            {"winner": "winner", "loser": "loser", "axis": "verifiability", "weight": 1.0},
        ]
    )

    elo.update_from_relative(ranking)
    elo.apply_to_candidates(candidates)

    assert candidates[0].multihead_scores["elo_reproductive_signal"] > candidates[1].multihead_scores["elo_reproductive_signal"]
    assert candidates[0].multihead_scores["elo_mean_rating"] > candidates[1].multihead_scores["elo_mean_rating"]


def test_multihead_elo_decay_and_old_checkpoint_restore_share_one_state() -> None:
    elo = MultiHeadElo()
    elo.update_pairwise("winner", "loser", axis="answer_likelihood")
    first_delta = elo.ratings["winner"]["answer_likelihood"] - elo.initial_rating
    previous = elo.ratings["winner"]["answer_likelihood"]
    elo.update_pairwise("winner", "loser", axis="answer_likelihood")
    second_delta = elo.ratings["winner"]["answer_likelihood"] - previous

    assert second_delta < first_delta
    assert elo.update_counts == {"answer_likelihood": 2}
    assert MultiHeadElo.from_dict({"ratings": elo.ratings}).update_counts == {}


def test_reproductive_value_uses_elo_signal_as_live_selection_pressure() -> None:
    high = _candidate("high", answer=0.4)
    low = _candidate("low", answer=0.4)
    high.multihead_scores["elo_reproductive_signal"] = 1.0
    low.multihead_scores["elo_reproductive_signal"] = 0.0
    population = [high, low]

    assert reproductive_value(high, population) > reproductive_value(low, population)


def test_evolution_round_feeds_elo_back_into_population_scores() -> None:
    population = CandidatePopulation([_candidate("a", answer=0.8), _candidate("b", answer=0.4)])
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1))

    round_stage.rank(
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        current_round=1,
    )

    assert all("elo_reproductive_signal" in candidate.multihead_scores for candidate in population.candidates)
