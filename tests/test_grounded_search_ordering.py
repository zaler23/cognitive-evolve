from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.reproduction import dedupe_offspring_against_population
from cognitive_evolve_runtime.ranking.novelty import novelty_distance
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector, evaluator_selection_key


def _candidate(candidate_id: str, artifact: str) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact=artifact,
        artifact_type="text",
        concise_claim=candidate_id,
        core_mechanism=candidate_id,
        novelty_descriptors=[candidate_id],
        multihead_scores={
            "objective_alignment": 1.0,
            "answer_likelihood": 1.0,
            "core_mechanism_strength": 1.0,
            "verifiability": 1.0,
        },
    )


def test_artifact_surface_similarity_grounds_soft_novelty_without_hard_dedupe() -> None:
    baseline = _candidate("baseline", "A complete mechanism with alpha beta gamma and a bounded proof.")
    surface_variant = _candidate("surface", "A complete mechanism with alpha beta gamma and a bounded proof!")
    different = _candidate("different", "Use a counterexample-guided graph search over independent constraints.")

    assert novelty_distance(baseline, surface_variant) < novelty_distance(baseline, different)
    assert dedupe_offspring_against_population([surface_variant], CandidatePopulation([baseline])) == [surface_variant]


def test_external_evaluator_forms_a_lexicographic_selection_tier() -> None:
    passed = _candidate("passed", "passed artifact")
    passed.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"score": 0.2}}
    unmeasured = _candidate("unmeasured", "unmeasured artifact")
    failed = _candidate("failed", "failed artifact")
    failed.metadata["evaluator"] = {"status": "failed", "passed": False, "metrics": {"score": 1.0}}
    failed.current_fate = CandidateFate.FAILED.value

    assert evaluator_selection_key(passed) > evaluator_selection_key(unmeasured) > evaluator_selection_key(failed)
    selected = ParentSelector().select([failed, unmeasured, passed], limit=3)

    assert [candidate.id for candidate in selected] == ["passed", "unmeasured", "failed"]
    assert failed.current_fate == CandidateFate.FAILED.value
