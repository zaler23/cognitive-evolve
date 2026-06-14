from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater


def test_auxiliary_candidate_not_main_winner_by_default() -> None:
    answer = CandidateGenome(
        id="answer",
        artifact="actual answer body",
        current_fate=CandidateFate.ELITE,
        multihead_scores={"objective_alignment": 0.7, "answer_likelihood": 0.7, "verifiability": 0.5},
    )
    auxiliary = CandidateGenome(
        id="aux",
        artifact="router validator framework",
        current_fate=CandidateFate.AUXILIARY,
        multihead_scores={"auxiliary_value": 1.0, "objective_alignment": 0.2, "answer_likelihood": 0.1},
    )
    archives = ArchiveManager()
    archives.update([answer, auxiliary])

    ranking = RelativeRater().rank(candidates=[answer, auxiliary])
    result = synthesize_result(population=CandidatePopulation([answer, auxiliary]), archives=archives)

    assert ranking.best_final_answer_id == "answer"
    assert result.best_candidate_id == "answer"
    assert result.best_auxiliary_candidate_id == "aux"
