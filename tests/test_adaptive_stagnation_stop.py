from __future__ import annotations

from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.adaptive_stop import adaptive_stagnation_exhausted
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.candidates.genome import CandidateGenome


def _record(best: str, actions: list[str]) -> dict[str, object]:
    return {
        "ranking": {"best_final_answer_id": best},
        "diagnosis": {"recommended_actions": actions},
    }


def test_adaptive_stagnation_requires_patience_and_all_configured_interventions() -> None:
    policy = EvolutionPolicy(
        stagnation_actions=["widen", "repair", "resurrect"],
        metadata={"adaptive_stagnation_patience": 3},
    )
    diagnosis = SearchDiagnosis(recommended_actions=["resurrect"])
    partial = [_record("best", ["widen"]), _record("best", ["repair"])]

    assert adaptive_stagnation_exhausted(
        adaptive=True,
        best_answer_id="best",
        history=partial,
        diagnosis=diagnosis,
        policy=policy,
    )
    assert not adaptive_stagnation_exhausted(
        adaptive=False,
        best_answer_id="best",
        history=partial,
        diagnosis=diagnosis,
        policy=policy,
    )
    assert not adaptive_stagnation_exhausted(
        adaptive=True,
        best_answer_id="new-best",
        history=partial,
        diagnosis=diagnosis,
        policy=policy,
    )


def test_adaptive_stagnation_is_opt_in_without_a_hardcoded_patience() -> None:
    policy = EvolutionPolicy(stagnation_actions=["repair"])

    assert not adaptive_stagnation_exhausted(
        adaptive=True,
        best_answer_id="best",
        history=[_record("best", ["repair"]) for _ in range(20)],
        diagnosis=SearchDiagnosis(recommended_actions=["repair"]),
        policy=policy,
    )


def test_adaptive_stagnation_resets_when_same_best_candidate_improves() -> None:
    policy = EvolutionPolicy(
        stagnation_actions=["repair"],
        metadata={"adaptive_stagnation_patience": 3},
    )
    history = [
        {**_record("best", ["repair"]), "best_quality_key": [1, 0.1, 0.1, 0.1]},
        {**_record("best", ["repair"]), "best_quality_key": [1, 0.1, 0.1, 0.1]},
    ]
    improved = CandidateGenome(
        id="best",
        multihead_scores={"answer_likelihood": 0.4, "objective_score": 0.4},
    )

    assert not adaptive_stagnation_exhausted(
        adaptive=True,
        best_answer_id="best",
        history=history,
        diagnosis=SearchDiagnosis(recommended_actions=["repair"]),
        policy=policy,
        candidates=[improved],
    )
