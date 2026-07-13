from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager, LatentParetoIntentArchive
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome


def _frontier_candidate(
    candidate_id: str,
    *,
    intent: str,
    intent_score: float,
    generation: int = 0,
    fate: CandidateFate | str = CandidateFate.INCUBATING,
) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=generation,
        current_fate=fate,
        metadata={
            "latent_pareto_frontier": True,
            "latent_intent_id": intent,
            "latent_intent_scores": {intent: intent_score},
            "latent_archive_round": generation,
        },
        multihead_scores={"latent_reproductive_signal": intent_score},
    )


def test_intent_level_quota_keeps_finite_representatives_without_frequency_weighting() -> None:
    archive = LatentParetoIntentArchive(max_representatives_per_intent=2, stale_after_rounds=99)

    archive.add(_frontier_candidate("low", intent="clarity", intent_score=0.1))
    archive.add(_frontier_candidate("high", intent="clarity", intent_score=0.9))
    archive.add(_frontier_candidate("mid", intent="clarity", intent_score=0.5))
    archive.add(_frontier_candidate("impact", intent="impact", intent_score=0.4))

    summary = archive.summary()

    assert set(archive.candidates) == {"high", "mid", "impact"}
    assert summary["intent_counts"] == {"clarity": 2, "impact": 1}
    assert summary["intent_buckets"]["clarity"] == ["high", "mid"]
    assert summary["removal_reasons"]["intent_quota"] == 1
    assert summary["intent_selection_weights"] == {"clarity": 0.5, "impact": 0.5}
    assert summary["desirability_basis"] == "intent_bucket_uniform_not_archive_frequency"


def test_survival_and_culling_are_intent_local_not_global_archive_frequency() -> None:
    archive = LatentParetoIntentArchive(max_representatives_per_intent=1, stale_after_rounds=99)

    archive.add(_frontier_candidate("clarity-a", intent="clarity", intent_score=0.3))
    archive.add(_frontier_candidate("clarity-b", intent="clarity", intent_score=0.8))
    archive.add(_frontier_candidate("impact-a", intent="impact", intent_score=0.2))

    assert set(archive.candidates) == {"clarity-b", "impact-a"}
    assert archive.summary()["intent_counts"] == {"clarity": 1, "impact": 1}
    assert archive.summary()["removed_total"] == 1


def test_stale_membership_is_removed_when_candidate_loses_frontier_signal() -> None:
    candidate = _frontier_candidate("stale", intent="clarity", intent_score=0.7)
    archives = ArchiveManager()
    archives.update([candidate])
    assert candidate.id in archives.latent_pareto_archive.candidates

    candidate.metadata.pop("latent_pareto_frontier")
    archives.update([candidate])

    governance = archives.summary()["latent_pareto_governance"]
    assert candidate.id not in archives.latent_pareto_archive.candidates
    assert governance["candidates"] == 0
    assert governance["removal_reasons"]["stale_not_frontier"] == 1


def test_aging_prune_removes_unseen_representatives() -> None:
    archive = LatentParetoIntentArchive(max_representatives_per_intent=2, stale_after_rounds=1)

    archive.add(_frontier_candidate("old", intent="clarity", intent_score=0.9, generation=0))
    archive.add(_frontier_candidate("new", intent="impact", intent_score=0.6, generation=2))

    summary = archive.summary()
    assert "old" not in archive.candidates
    assert "new" in archive.candidates
    assert summary["removal_reasons"]["stale_age"] == 1
    assert summary["latest_observed_round"] == 2


def test_persistence_restores_governance_state_and_buckets() -> None:
    archives = ArchiveManager()
    archives.latent_pareto_archive.max_representatives_per_intent = 2
    archives.latent_pareto_archive.stale_after_rounds = 9
    archives.update([
        _frontier_candidate("clarity", intent="clarity", intent_score=0.8, generation=1),
        _frontier_candidate("impact", intent="impact", intent_score=0.7, generation=1),
    ])

    reloaded = ArchiveManager.from_dict(archives.to_dict())
    governance = reloaded.summary()["latent_pareto_governance"]

    assert set(reloaded.latent_pareto_archive.candidates) == {"clarity", "impact"}
    assert governance["intent_counts"] == {"clarity": 1, "impact": 1}
    assert governance["max_representatives_per_intent"] == 2
    assert governance["stale_after_rounds"] == 9
    assert reloaded.latent_pareto_archive.candidate_observations["clarity"]["last_seen_round"] == 1


def test_latent_pareto_governance_does_not_change_final_eligibility_or_answer_archive() -> None:
    candidate = _frontier_candidate("not-final", intent="clarity", intent_score=0.9, fate=CandidateFate.INCUBATING)
    archives = ArchiveManager()

    assert archives.is_final_answer_eligible(candidate) is False
    [assignment] = archives.update([candidate])

    assert "LatentParetoIntentArchive" in assignment.archive_targets
    assert "AnswerArchive" not in assignment.archive_targets
    assert candidate.id in archives.latent_pareto_archive.candidates
    assert candidate.id not in archives.answer_archive
    assert archives.is_final_answer_eligible(candidate) is False
