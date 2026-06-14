from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidateIdentity, CandidateState
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis


def test_candidate_genome_roundtrip() -> None:
    genome = CandidateGenome(
        id="C-roundtrip",
        parent_ids=["C-parent"],
        generation=2,
        artifact={"answer": "42"},
        concise_claim="answer claim",
        core_mechanism="mechanism",
        edge_knowledge_seeds=["rare seed"],
        failure_lessons=["lesson"],
        current_fate=CandidateFate.DORMANT,
        multihead_scores={"rarity": 0.9, "answer_likelihood": 0.4},
        contract_hash="abc",
        evidence_refs=[{"id": "test-output", "kind": "test", "status": "verified"}],
        source_bindings=[{"path": "pkg/module.py", "kind": "source_file", "required": True}],
        evidence_delta={"verified": ["test-output"]},
    )

    decoded = CandidateGenome.from_json(genome.to_json())

    assert decoded.to_dict() == genome.to_dict()
    assert decoded.lineage[-1] == "C-roundtrip"
    assert decoded.extract_inheritable_gene_summary()
    assert isinstance(decoded.identity, CandidateIdentity)
    assert isinstance(decoded.state, CandidateState)
    assert decoded.state.evidence_refs == genome.evidence_refs
    assert decoded.state.source_bindings == genome.source_bindings
    assert decoded.state.evidence_delta == genome.evidence_delta


def test_failed_fate_roundtrips_and_scores_are_bounded() -> None:
    genome = CandidateGenome(
        id="C-failed",
        current_fate=CandidateFate.FAILED,
        multihead_scores={"answer_likelihood": 9.0, "deferral_risk": -2.0},
    )

    decoded = CandidateGenome.from_dict(genome.to_dict())

    assert decoded.current_fate == CandidateFate.FAILED
    assert CandidateGenome.from_json(genome.to_json()).current_fate == CandidateFate.FAILED
    assert decoded.mark_fate(CandidateFate.FAILED).current_fate == CandidateFate.FAILED
    assert decoded.multihead_scores == {"answer_likelihood": 1.0, "deferral_risk": 0.0}


def test_project_candidate_policy_and_diagnosis_roundtrip() -> None:
    candidate = ProjectCandidateGenome(
        id="P1",
        patch_set=[PatchOperation(path="pkg/module.py", operation="replace", old_text="bad", new_text="good")],
        expected_effects=["fix behavior"],
        multihead_scores={"tool_progress": 0.3},
    )
    decoded = ProjectCandidateGenome.from_json(candidate.to_json())

    policy = EvolutionPolicy(rarity_budget=0.4)
    diagnosis = SearchDiagnosis(stagnation_detected=True, stagnation_type="AuxiliaryCollapse", recommended_actions=["core_extraction"])

    assert decoded.touched_files == ["pkg/module.py"]
    assert EvolutionPolicy.from_json(policy.to_json()).rarity_budget == 0.4
    assert SearchDiagnosis.from_json(diagnosis.to_json()).recommended_actions == ["core_extraction"]
