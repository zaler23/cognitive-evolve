from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.inputs.context_selector import ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.consistency import runtime_consistency_predicate
from cognitive_evolve_runtime.nexus.context_protocol import ContextOrchestrator
from cognitive_evolve_runtime.nexus.diagnosis import SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.loop import _attach_policy_directives_to_plans
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.parent_selection import reproductive_value
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _proof_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Prove the theorem and bind every obligation to evidence.",
        normalized_goal="prove theorem with evidence obligation ledger",
        expected_output_forms=["proof", "equation_set"],
        verification_preferences=["formal_artifact", "evidence_refs", "obligation_delta"],
    )


def test_verified_evidence_obligation_ledger_passes_candidate() -> None:
    candidate = CandidateGenome(
        id="C-ledger",
        artifact="lemma evidence",
        formal_artifacts=[{"kind": "equation_set", "target_obligation_id": "obl_eq", "equations": ["f(x)=0", "det(J_f(x)) != 0"]}],
        proof_obligations=[{"id": "obl_eq", "status": "discharged"}],
        obligation_delta={"targeted": ["obl_eq"], "discharged": ["obl_eq"]},
        evidence_refs=[{"id": "eq-check", "kind": "verification", "status": "verified", "target_obligation_id": "obl_eq"}],
        evidence_delta={"verified": ["eq-check"]},
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=_proof_contract())

    assert result.passed is True
    assert result.evidence_obligation["verified_evidence_ref_count"] >= 1
    assert candidate.multihead_scores["evidence_progress"] > 0


def test_project_candidate_without_verified_evidence_is_blocked() -> None:
    contract = NexusObjectiveContract(original_user_goal="Fix project code and tests.", normalized_goal="fix project code")
    candidate = ProjectCandidateGenome(
        id="P-no-evidence",
        patch_set=[PatchOperation(path="pkg/module.py", operation="replace", old_text="bad", new_text="good")],
        obligation_delta={"targeted": ["obl_bug"]},
        source_bindings=[{"path": "pkg/module.py", "kind": "source_file", "required": True}],
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=contract)

    assert result.passed is True
    assert "evidence_ref_absent" in result.diagnostics
    assert result.final_eligible is True


def test_archive_constraints_penalize_proposal_only_lineage_until_evidence_delta() -> None:
    archived = CandidateGenome(
        id="C-old",
        lineage=["root", "C-old"],
        current_fate=CandidateFate.DORMANT,
        failure_lessons=["missing verified evidence reference"],
        verification_result={
            "passed": False,
            "diagnostics": ["evidence_ref_absent"],
            "evidence_obligation": {"diagnostics": ["evidence_ref_absent"]},
        },
    )
    archives = ArchiveManager()
    archives.update([archived])

    proposal_only = CandidateGenome(id="C-new", lineage=["root", "C-new"], multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.8})
    with_delta = CandidateGenome(
        id="C-evidence",
        lineage=["root", "C-evidence"],
        evidence_delta={"verified": ["post-pass"]},
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.8},
    )

    assert archives.constraint_records
    assert reproductive_value(with_delta, [proposal_only, with_delta], archives) > reproductive_value(proposal_only, [proposal_only, with_delta], archives)
    assert archives.reactivate_dormant("C-old") is not None


def test_context_packet_is_obligation_targeted_and_hashes_sources(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "module.py").write_text("def f():\n    return 'bad'\n", encoding="utf-8")
    (root / "tests" / "test_module.py").write_text("from pkg.module import f\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(root)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="fix module")
    parent = ProjectCandidateGenome(
        id="P1",
        patch_set=[PatchOperation(path="pkg/module.py", operation="replace", old_text="bad", new_text="good")],
        affected_tests=["tests/test_module.py"],
        obligation_delta={"targeted": ["obl_bug"]},
        metadata={"evidence_need": "pre-fail/post-pass pytest evidence"},
    )

    result = ContextOrchestrator(selector=ContextSelector(max_file_chars=200)).build_for_parents(
        contract=NexusObjectiveContract(original_user_goal="fix module", normalized_goal="fix module"),
        snapshot=snapshot,
        world=world,
        parents=[parent],
        archives=ArchiveManager(),
    )
    packet = result.packets[0]

    assert result.requests[0].target_obligation_ids == ["obl_bug"]
    assert packet.evidence_need == "pre-fail/post-pass pytest evidence"
    assert "pkg/module.py" in packet.source_hashes
    assert packet.coverage["target_obligation_ids"] == ["obl_bug"]


def test_mutation_plan_directives_require_source_grounding() -> None:
    parent = ProjectCandidateGenome(
        id="P1",
        patch_set=[PatchOperation(path="pkg/module.py")],
        affected_tests=["tests/test_module.py"],
        obligation_delta={"targeted": ["obl_bug"]},
    )
    plan = MutationPlan(operator="Repair", parent_ids=["P1"], instruction="repair failing behavior")
    policy = EvolutionPolicy(metadata={"source_grounding_required": True, "required_evidence_kinds": ["test"]})

    patched = _attach_policy_directives_to_plans([plan], policy, parents=[parent])[0]

    assert patched.metadata["source_grounding_required"] is False
    assert patched.metadata["requires_pre_fail_post_pass"] is False
    assert patched.metadata["target_obligation_ids"] == ["obl_bug"]
    assert any(point.get("path") == "pkg/module.py" for point in patched.metadata["legacy_source_integration_points_advisory"])
    assert "Source-binding context" in patched.instruction


def test_runtime_consistency_predicate_catches_round_mismatch() -> None:
    checkpoint = {
        "round": 3,
        "max_rounds": 5,
        "progress_event": {"type": "evolution_progress", "round": 3, "max_rounds": 5},
        "budget": {"current_round": 0, "completion_status": "running"},
    }
    events = [{"type": "evolution_progress", "round": 2, "max_rounds": 5}]

    result = runtime_consistency_predicate(checkpoint=checkpoint, events=events)

    assert result.passed is False
    assert "checkpoint_event_round_mismatch:3!=2" in result.errors
    assert "checkpoint_budget_round_mismatch:3!=0" in result.errors


def test_lineage_saturation_freezes_no_evidence_family() -> None:
    population = [
        CandidateGenome(id=f"C{i}", lineage=["root", f"C{i}"], core_mechanism="same words", multihead_scores={"answer_likelihood": 0.4})
        for i in range(4)
    ]

    diagnosis = SearchStateDiagnoser().diagnose(population=population, archives=ArchiveManager(), history=[], contract=None, policy=EvolutionPolicy())

    assert diagnosis.stagnation_type == "SemanticLooping"
    assert "quarantine_lineage" in diagnosis.recommended_actions
    assert diagnosis.under_explored_families == ["new_mechanism", "edge_theory", "cross_domain_variant"]
