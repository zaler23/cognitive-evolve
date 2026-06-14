from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector, reproductive_value


def _live(candidate_id: str, *, lineage_root: str = "root", niche: str = "route") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        generation=3,
        lineage=[lineage_root, candidate_id],
        core_mechanism=niche,
        niche_memberships=[niche],
        current_fate=CandidateFate.ACTIVE,
        multihead_scores={"objective_alignment": 0.6, "answer_likelihood": 0.55, "novelty": 0.4, "rarity": 0.2},
    )


def test_dead_siblings_do_not_create_live_lineage_penalty() -> None:
    candidate = _live("live")
    other = _live("other", lineage_root="other-root", niche="other")
    dead_siblings = [
        CandidateGenome(
            id=f"dead{i}",
            generation=3,
            lineage=["root", f"dead{i}"],
            core_mechanism="route",
            current_fate=CandidateFate.FAILED,
            failure_lessons=["failed"],
        )
        for i in range(12)
    ]

    baseline = reproductive_value(candidate, [candidate, other], ArchiveManager())
    with_dead = reproductive_value(candidate, [candidate, other, *dead_siblings], ArchiveManager())

    assert with_dead == baseline


def test_parent_selector_ignores_terminal_candidates_even_when_many() -> None:
    live = _live("live")
    dead = [
        CandidateGenome(
            id=f"dead{i}",
            generation=3,
            current_fate=CandidateFate.CULLED,
            core_mechanism="dead clone",
            multihead_scores={"objective_alignment": 1.0, "answer_likelihood": 1.0},
        )
        for i in range(10)
    ]

    selected = ParentSelector().select([*dead, live], ArchiveManager(), limit=3)

    assert [candidate.id for candidate in selected] == ["live"]


def test_negative_scored_elite_remains_parent_floor_when_only_live_candidate() -> None:
    elite = CandidateGenome(
        id="elite-floor",
        generation=4,
        current_fate=CandidateFate.ELITE,
        artifact_type="project_patch",
        concise_claim="best current route still needs repair",
        core_mechanism="sandbox diagnostics",
        failure_lessons=["missing local evidence", "duplicate route risk", "needs verifier-readable patch"],
        multihead_scores={"objective_alignment": 0.0, "answer_likelihood": 0.0, "verifiability": 0.0},
        metadata={"stage_eligibility": {"parent_eligible": True, "hard_reject_reason": ""}},
    )

    assert reproductive_value(elite, [elite], ArchiveManager()) < 0.0

    selected = ParentSelector().select([elite], ArchiveManager(), limit=2)

    assert [candidate.id for candidate in selected] == ["elite-floor"]


def test_parent_floor_still_respects_stage_parent_ineligible_elite() -> None:
    blocked = CandidateGenome(
        id="blocked-elite",
        current_fate=CandidateFate.ELITE,
        concise_claim="blocked current route",
        failure_lessons=["missing local evidence", "duplicate route risk", "needs verifier-readable patch"],
        multihead_scores={"objective_alignment": 0.0, "answer_likelihood": 0.0, "verifiability": 0.0},
        metadata={"stage_eligibility": {"parent_eligible": False, "hard_reject_reason": "seed_note_only_patch"}},
    )

    selected = ParentSelector().select([blocked], ArchiveManager(), limit=2)

    assert selected == []
