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


def test_deterministic_relative_rater_emits_non_star_pairwise_preferences() -> None:
    candidates = [_candidate("a", answer=0.9), _candidate("b", answer=0.6), _candidate("c", answer=0.3)]

    ranking = RelativeRater().rank(candidates=candidates)

    pairs = {(item["winner"], item["loser"], item["axis"]) for item in ranking.pairwise_preferences}
    assert ("a", "b", "answer_likelihood") in pairs
    assert ("b", "c", "answer_likelihood") in pairs
    assert ("a", "c", "answer_likelihood") not in pairs


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
