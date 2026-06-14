from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome


def test_latent_pareto_archive_lane_persists_summary_and_is_not_answer_lane() -> None:
    candidate = CandidateGenome(
        id="latent-frontier",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={
            "latent_pareto_frontier": True,
            "latent_intent_scores": {"clarity": 0.9, "impact": 0.4},
        },
        multihead_scores={"latent_reproductive_signal": 0.8},
    )
    archives = ArchiveManager()

    [assignment] = archives.update([candidate])

    assert "LatentParetoIntentArchive" in assignment.archive_targets
    assert "AnswerArchive" not in assignment.archive_targets
    assert candidate.id in archives.latent_pareto_archive.candidates
    assert archives.summary()["latent_pareto_candidates"] == 1
    assert candidate.id not in archives.answer_archive
    assert archives.is_final_answer_eligible(candidate) is False

    reloaded = ArchiveManager.from_dict(archives.to_dict())

    assert reloaded.summary()["latent_pareto_candidates"] == 1
    assert reloaded.latent_pareto_archive.candidates[candidate.id]["metadata"]["latent_pareto_frontier"] is True
    assert candidate.id not in reloaded.answer_archive


def test_latent_pareto_archive_lane_removes_stale_membership() -> None:
    candidate = CandidateGenome(
        id="stale-frontier",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={"latent_pareto_frontier": True},
    )
    archives = ArchiveManager()
    archives.update([candidate])
    assert candidate.id in archives.latent_pareto_archive.candidates

    candidate.metadata.pop("latent_pareto_frontier")
    archives.update([candidate])

    assert candidate.id not in archives.latent_pareto_archive.candidates
    assert archives.summary()["latent_pareto_candidates"] == 0
