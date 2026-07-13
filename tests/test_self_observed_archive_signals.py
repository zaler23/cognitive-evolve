from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.adaptive_signals import in_top_band
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def _verified(candidate_id: str, score: float, *, niche: str = "n") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        current_fate=CandidateFate.ACTIVE.value,
        concise_claim=candidate_id,
        core_mechanism=candidate_id,
        niche_memberships=[niche],
        multihead_scores={
            "answer_likelihood": score,
            "objective_alignment": score,
            "verifiability": score,
            "evidence_progress": score,
        },
        evidence_refs=[{"id": f"ev-{candidate_id}", "status": "verified"}],
        verification_result={"passed": True, "rank_eligible": True, "final_eligible": True, "diagnostics": []},
    )


def test_archive_elite_assignment_uses_population_relative_top_band_not_global_threshold() -> None:
    low = _verified("low", 0.21)
    mid = _verified("mid", 0.31)
    top = _verified("top", 0.41)
    archives = ArchiveManager()

    assignments = archives.assign_by_policy([low, mid, top], RelativeRankingResult(best_final_answer_id=""))
    by_id = {assignment.candidate_id: assignment.fate for assignment in assignments}

    assert in_top_band(top, [low, mid, top], "answer_likelihood") is True
    assert by_id["top"] == CandidateFate.ELITE.value


def test_best_answer_candidate_considers_quality_diversity_elites() -> None:
    candidate = _verified("qd", 0.77, niche="rare-niche")
    candidate.mark_fate(CandidateFate.ELITE.value)
    archives = ArchiveManager()
    archives.fates[candidate.id] = CandidateFate.ELITE.value
    archives.quality_diversity.update(candidate)

    assert archives.best_answer_candidate([candidate]).id == "qd"
