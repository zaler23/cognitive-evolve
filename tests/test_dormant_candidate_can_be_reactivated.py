from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome


def test_dormant_candidate_can_be_reactivated() -> None:
    candidate = CandidateGenome(id="sleepy", current_fate=CandidateFate.DORMANT, core_mechanism="latent useful gene")
    archives = ArchiveManager()
    archives.update([candidate])

    reactivated = archives.reactivate_dormant("sleepy")

    assert reactivated is not None
    assert reactivated.current_fate == CandidateFate.ACTIVE
    assert "DormantReactivation" in reactivated.mutation_history
