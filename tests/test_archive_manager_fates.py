from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result


def test_archive_manager_fates_extracts_genes_before_culling() -> None:
    elite = CandidateGenome(id="elite", core_mechanism="proof path", current_fate=CandidateFate.ELITE, multihead_scores={"answer_likelihood": 0.9})
    auxiliary = CandidateGenome(id="aux", core_mechanism="validator scaffold", current_fate=CandidateFate.AUXILIARY, multihead_scores={"auxiliary_value": 0.95})
    dormant = CandidateGenome(id="sleep", core_mechanism="edge mechanism", edge_knowledge_seeds=["obscure lemma"], current_fate=CandidateFate.DORMANT)
    culled = CandidateGenome(id="bad", core_mechanism="failed route", failure_lessons=["counterexample x"], current_fate=CandidateFate.CULLED)

    archives = ArchiveManager()
    archives.update([elite, auxiliary, dormant, culled])

    assert "elite" in archives.answer_archive
    assert "aux" in archives.auxiliary_archive.candidates
    assert "sleep" in archives.dormant_archive.candidates
    assert "obscure lemma" in archives.rarity_archive.seeds
    assert archives.failure_archive.records["bad"]["inherited_gene_summary"]
    assert archives.summary()["failure_records"] == 1


def test_failed_fate_is_terminal_and_routed_to_failure_archive() -> None:
    failed = CandidateGenome(id="failed", core_mechanism="bad patch", current_fate=CandidateFate.FAILED)
    active = CandidateGenome(id="active", current_fate=CandidateFate.ACTIVE, multihead_scores={"answer_likelihood": 0.9})

    archives = ArchiveManager()
    assignments = archives.assign_by_policy([failed, active])
    archives.update(assignments, candidates=[failed, active])

    assert failed.current_fate == CandidateFate.FAILED
    assert "failed" in archives.failure_archive.records
    assert "active" in archives.answer_archive


def test_verified_dormant_frontier_can_be_synthesized_without_deadlock() -> None:
    dormant = CandidateGenome(
        id="edge",
        artifact="rare verified answer",
        generation=3,
        core_mechanism="edge mechanism",
        edge_knowledge_seeds=["obscure lemma"],
        current_fate=CandidateFate.DORMANT,
        multihead_scores={"objective_alignment": 0.72, "answer_likelihood": 0.69, "verifiability": 0.8, "rarity": 0.9},
        verification_result={"passed": True, "rank_eligible": True, "final_eligible": True, "diagnostics": []},
    )
    archives = ArchiveManager()
    archives.update([dormant])

    synthesis = synthesize_result(population=CandidatePopulation([dormant]), archives=archives)

    assert archives.is_final_answer_eligible(dormant) is True
    assert synthesis.best_candidate_id == "edge"
    assert synthesis.final_answer == "rare verified answer"


def test_unverified_or_failed_dormant_candidate_is_not_final_eligible() -> None:
    unverified = CandidateGenome(
        id="unverified",
        generation=2,
        core_mechanism="edge mechanism",
        edge_knowledge_seeds=["rare"],
        current_fate=CandidateFate.DORMANT,
        multihead_scores={"objective_alignment": 0.95, "answer_likelihood": 0.95, "rarity": 0.9},
    )
    failed = CandidateGenome(
        id="failed-dormant",
        generation=2,
        core_mechanism="edge mechanism",
        edge_knowledge_seeds=["rare"],
        current_fate=CandidateFate.DORMANT,
        multihead_scores={"objective_alignment": 0.95, "answer_likelihood": 0.95, "rarity": 0.9},
        verification_result={"passed": False, "rank_eligible": False, "final_eligible": False, "diagnostics": ["evidence_ref_absent"]},
    )
    archives = ArchiveManager()
    archives.update([unverified, failed])

    assert archives.is_final_answer_eligible(unverified) is False
    assert archives.is_final_answer_eligible(failed) is False
