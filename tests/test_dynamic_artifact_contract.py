from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import (
    NexusObjectiveContract,
    NexusObjectiveContractBuilder,
    NexusProjectObjectiveContract,
    apply_artifact_policy_to_contract,
    artifact_policy_contract_conflicts,
    dynamic_artifact_contract_from_artifact_policy,
)
from cognitive_evolve_runtime.nexus.runtime import _rebase_population_contract_hashes
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.artifact_contract import (
    contract_requires_adapter,
    dynamic_artifact_contract_from,
    evaluate_candidate_against_dynamic_contract,
    validate_dynamic_artifact_contract,
)
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view, contract_prompt_view
from cognitive_evolve_runtime.nexus.source_lineage import analyze_source_lineage
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _contract(**overrides) -> NexusObjectiveContract:
    dac = {
        "objective": "Evolve the supplied artifact according to the user's objective.",
        "artifact_domain_label": "mythic_dialogue_revision_v7",
        "required_work_product": {"description": "a concrete revised artifact body"},
        "allowed_artifact_shapes": [{"name": "model_defined_shape", "required_fields": ["content"]}],
        "minimum_concrete_delta": {"observable_signal": "specific changed text or structure relative to parent"},
        "invalid_outputs": ["empty output", "meta commentary only", "restating objective without artifact"],
        "evaluation_dimensions": [{"name": "objective_fit", "measurement": "referee_or_structural_comparison"}],
        "comparison_method": {"method": "relative comparison against the frozen contract"},
        "final_gate": {"check": "structural artifact presence plus independent comparison evidence"},
        "repair_contract": {"on_missing_artifact": "produce the required work product"},
        "adapter_requirements": {},
    }
    dac.update(overrides)
    return NexusObjectiveContract(
        original_user_goal="Improve this artifact without changing the user's goal.",
        normalized_goal="improve supplied artifact",
        outcome_policy={"model_driven": True, "dynamic_artifact_contract": dac},
        expected_output_forms=["answer"],
        verification_preferences=[],
    )


def _default_generic_contract() -> dict:
    return {
        "objective": "generic artifact",
        "artifact_domain_label": "model_defined_artifact",
        "required_work_product": {"description": "generic object"},
        "allowed_artifact_shapes": [{"name": "model_defined_object", "required_fields": ["content_or_structured_object"]}],
        "minimum_concrete_delta": {"observable_signal": "changed material"},
        "invalid_outputs": ["empty output", "meta commentary only", "restating objective without artifact"],
        "evaluation_dimensions": [{"name": "objective_fit", "measurement": "structural comparison"}],
        "comparison_method": {"method": "relative comparison"},
        "final_gate": {"check": "structural artifact presence plus evidence"},
        "adapter_requirements": {},
    }


def _candidate(**overrides) -> CandidateGenome:
    data = dict(
        id="C-dac",
        artifact_type="narrative",
        concise_claim="revised artifact strengthens conflict",
        core_mechanism="the revised artifact shows a stronger conflict in the actual text",
        artifact={"content": "The revised artifact opens with conflict: Mira refuses the crown before the court can name her."},
        metadata={"artifact_delta": {"relative_to_parent": "added immediate conflict and a clearer character choice"}},
        multihead_scores={"answer_likelihood": 0.7, "verifiability": 0.6},
    )
    data.update(overrides)
    return CandidateGenome(**data)


def test_unknown_model_defined_artifact_label_is_valid_and_not_used_for_dispatch() -> None:
    contract = _contract(artifact_domain_label="unknown_future_artifact_label_42")
    dac = dynamic_artifact_contract_from(contract=contract)

    assert dac is not None
    assert dac.artifact_domain_label == "unknown_future_artifact_label_42"
    assert validate_dynamic_artifact_contract(dac).valid is True
    assert contract_requires_adapter(contract, "source") is False
    assert contract_requires_adapter(contract, "proof") is False


def test_narrative_artifact_can_be_final_when_dynamic_contract_requires_text_not_patch() -> None:
    candidate = _candidate()
    result = NexusVerifierStack().verify_candidate(candidate, contract=_contract())

    assert result.passed is True
    assert result.final_eligible is True
    diagnostics = result.diagnostics
    assert "proof_object_absent" not in diagnostics
    assert "source_binding_absent" not in diagnostics
    assert "final_artifact_type_not_publishable" not in diagnostics
    assert result.artifact_contract["required"] is True


def test_meta_commentary_without_object_level_artifact_is_rejected() -> None:
    candidate = _candidate(
        artifact="We should make the scene more tense and improve the voice later.",
        metadata={"artifact_delta": {"relative_to_parent": "describes an intent but does not provide the revised artifact"}},
    )
    result = NexusVerifierStack().verify_candidate(candidate, contract=_contract())

    assert result.passed is False
    assert "meta_commentary_only" in result.diagnostics
    assert result.final_eligible is False


def test_object_artifact_without_concrete_delta_cannot_be_final_but_remains_repairable() -> None:
    candidate = _candidate(metadata={})
    result = NexusVerifierStack().verify_candidate(candidate, contract=_contract(), current_round=5, round_limit=20)

    assert "concrete_delta_absent" in result.diagnostics
    assert result.final_eligible is False
    assert candidate.metadata["stage_eligibility"]["repair_required"] is True


def test_self_certifying_dynamic_contract_is_rejected_before_candidate_can_final() -> None:
    contract = _contract(final_gate={"check": "accept if the generator says done with confidence >= 0.8"})
    candidate = _candidate()
    result = NexusVerifierStack().verify_candidate(candidate, contract=contract)

    assert result.passed is False
    assert "final_gate_self_certifying" in result.diagnostics
    assert result.final_eligible is False


def test_source_and_patch_adapters_only_fire_when_dynamic_contract_requires_them(tmp_path: Path) -> None:
    no_source = _contract()
    patch_contract = _contract(adapter_requirements={"requires_patch": True, "requires_source_binding": True})
    candidate = _candidate(artifact_type="answer", source_bindings=[])

    no_source_result = NexusVerifierStack(project_root=tmp_path).verify_candidate(candidate, contract=no_source)
    assert "source_binding_absent" not in no_source_result.diagnostics

    patch_result = NexusVerifierStack(project_root=tmp_path).verify_candidate(_candidate(artifact_type="answer", source_bindings=[]), contract=patch_contract)
    assert "source_binding_absent" in patch_result.diagnostics


def test_source_lineage_uses_contract_materialization_scope_for_non_runtime_artifacts(tmp_path: Path) -> None:
    patch_text = """diff --git a/drafts/v2.md b/drafts/v2.md
new file mode 100644
--- /dev/null
+++ b/drafts/v2.md
@@ -0,0 +1,2 @@
+# Revised artifact
+The revised artifact body is now concrete.
"""
    candidate = _candidate(
        artifact_type="project_patch",
        artifact={"unified_diff": patch_text},
        source_bindings=[{"path": "drafts/v2.md", "declared_mode": "materialize", "kind": "artifact_file"}],
        evidence_refs=[{"kind": "verification", "path": "drafts/v2.md", "status": "verified"}],
    )
    analysis = analyze_source_lineage(candidate, project_root=tmp_path, materialization_scope=["drafts/"])

    assert analysis.required is True
    assert analysis.diagnostics == []
    assert analysis.facts[0].lineage_mode == "new_file_materialization"


def test_fallback_builder_attaches_non_vacuous_dynamic_artifact_contract() -> None:
    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Rewrite the passage into a tenser scene.", packet={"constraints": []})

    dac = dynamic_artifact_contract_from(contract=contract)
    summary = validate_dynamic_artifact_contract(dac)

    assert dac is not None
    assert summary.valid is True
    assert any(shape.get("name") == "design_candidate" for shape in dac.allowed_artifact_shapes)
    assert contract.dynamic_artifact_contract_hash()
    assert contract.to_dict()["dynamic_artifact_contract_hash"] == dac.stable_hash()


def test_artifact_policy_compiles_to_dynamic_artifact_contract() -> None:
    dac = dynamic_artifact_contract_from_artifact_policy(
        {
            "machine_artifact_required": True,
            "artifact_type": "cache_policy",
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
            "field_aliases": {"eviction_scoring": "eviction"},
            "final_requires_clean_schema": True,
        },
        objective="evolve a cache policy",
    )

    assert dac.artifact_domain_label == "cache_policy"
    assert dac.allowed_artifact_shapes == [
        {
            "name": "cache_policy",
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            "machine_readable_required": True,
            "final_eligible": True,
        }
    ]
    assert dac.final_gate["requires_clean_schema"] is True
    assert dac.adapter_requirements["machine_readable_required"] is True
    assert validate_dynamic_artifact_contract(dac).valid is True


def test_builder_overlays_model_contract_with_artifact_policy() -> None:
    class GenericContractModel:
        def build_objective_contract(self, *, user_goal, world):
            return NexusObjectiveContract(
                original_user_goal=user_goal,
                normalized_goal="generic model contract",
                dynamic_artifact_contract=_default_generic_contract(),
            ).to_dict()

    contract = NexusObjectiveContractBuilder().build_text_contract(
        user_goal="Evolve cache admission and eviction policy.",
        packet={"constraints": []},
        model=GenericContractModel(),
        artifact_policy_config={
            "machine_artifact_required": True,
            "artifact_type": "cache_policy",
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
        },
    )
    dac = dynamic_artifact_contract_from(contract=contract)

    assert dac is not None
    assert dac.artifact_domain_label == "cache_policy"
    assert dac.allowed_artifact_shapes[0]["name"] == "cache_policy"
    assert dac.allowed_artifact_shapes[0]["required_fields"] == ["admission", "eviction", "parameters", "update_or_state_update"]
    assert contract.metadata["artifact_policy_contract_overlay"]["diagnostics"]
    assert validate_dynamic_artifact_contract(dac).valid is True


def test_artifact_policy_contract_conflict_reports_missing_shape_and_fields() -> None:
    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Improve generic artifact.", packet={})

    diagnostics = artifact_policy_contract_conflicts(
        {
            "machine_artifact_required": True,
            "artifact_type": "cache_policy",
            "required_fields": ["admission", "eviction"],
        },
        contract,
    )

    assert any("artifact_type_missing expected=cache_policy" in item for item in diagnostics)
    assert any("required_fields_missing=admission,eviction" in item for item in diagnostics)


def test_resume_contract_overlay_rebases_restored_candidate_hashes() -> None:
    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Improve generic artifact.", packet={})
    previous_hash = contract.contract_hash()
    previous_dynamic_hash = contract.dynamic_artifact_contract_hash()
    candidate = CandidateGenome(id="C1", artifact={"content": "x"}, contract_hash=previous_hash)
    population = CandidatePopulation([candidate])

    apply_artifact_policy_to_contract(
        contract,
        {
            "machine_artifact_required": True,
            "artifact_type": "cache_policy",
            "required_fields": ["admission", "eviction"],
        },
        source="adaptive_state.resume",
    )
    _rebase_population_contract_hashes(
        population,
        previous_contract_hash=previous_hash,
        current_contract_hash=contract.contract_hash(),
        previous_dynamic_artifact_contract_hash=previous_dynamic_hash,
        current_dynamic_artifact_contract_hash=contract.dynamic_artifact_contract_hash(),
    )

    assert candidate.contract_hash == contract.contract_hash()
    assert candidate.metadata["contract_hash_overlay_rebased"]["previous_contract_hash"] == previous_hash


def test_contract_from_dict_lifts_nested_dynamic_artifact_contract_to_first_class_field() -> None:
    nested = {
        "objective": "Create a concrete comparison artifact.",
        "artifact_domain_label": "judge_panel_v2",
        "required_work_product": {"description": "a scored comparison table"},
        "allowed_artifact_shapes": [{"name": "table", "required_fields": ["rows"]}],
        "minimum_concrete_delta": {"observable_signal": "at least one scored row changes"},
        "invalid_outputs": ["empty output", "meta commentary only", "restating objective without artifact"],
        "evaluation_dimensions": [{"name": "decision_quality", "measurement": "row-level comparison"}],
        "comparison_method": {"method": "rubric comparison"},
        "final_gate": {"check": "schema validation plus evidence comparison"},
        "adapter_requirements": {},
    }
    contract = NexusObjectiveContract.from_dict(
        {
            "original_user_goal": "Compare the options.",
            "normalized_goal": "compare options",
            "outcome_policy": {"model_driven": True, "dynamic_artifact_contract": nested},
        }
    )

    assert contract.dynamic_artifact_contract["artifact_domain_label"] == "judge_panel_v2"
    assert contract.validate_dynamic_artifact_contract().passed is True


def test_model_adapter_repairs_legacy_contract_response_by_injecting_dynamic_artifact_contract() -> None:
    adapter = StructuredModelAdapter(caller=lambda request_type, payload, schema: {"normalized_goal": "make the draft clearer"})

    contract = adapter.build_objective_contract(user_goal="Make the draft clearer.", world={"kind": "text"})

    assert "dynamic_artifact_contract" in contract
    assert "dynamic_artifact_contract" in contract["outcome_policy"]
    dac = dynamic_artifact_contract_from(contract)
    assert dac is not None
    assert validate_dynamic_artifact_contract(dac).valid is True


def test_contract_hash_changes_when_dynamic_artifact_contract_changes() -> None:
    base = NexusObjectiveContractBuilder().build_text_contract(user_goal="Improve the artifact.", packet={})
    changed = NexusObjectiveContract.from_dict(base.to_dict())
    changed.dynamic_artifact_contract = dict(changed.dynamic_artifact_contract)
    changed.dynamic_artifact_contract["required_work_product"] = {"description": "a materially different artifact body"}

    assert base.dynamic_artifact_contract_hash() != changed.dynamic_artifact_contract_hash()
    assert base.contract_hash() != changed.contract_hash()


def test_contract_prompt_view_exposes_dynamic_artifact_contract_hash_and_summary() -> None:
    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Revise this paragraph.", packet={})
    view = contract_prompt_view(contract)

    assert view["dynamic_artifact_contract_hash"] == contract.dynamic_artifact_contract_hash()
    assert view["dynamic_artifact_contract"]["artifact_domain_label"] == "model_defined_artifact"
    assert "required_work_product" in view["dynamic_artifact_contract"]


def test_project_contract_round_trip_accepts_dynamic_artifact_contract_hash_field() -> None:
    contract = NexusProjectObjectiveContract.from_dict(
        {
            "original_user_goal": "Improve runtime.",
            "normalized_goal": "improve runtime",
            "outcome_policy": {"dynamic_artifact_contract": _contract().dynamic_artifact_contract},
            "dynamic_artifact_contract_hash": "legacy-persisted-derived-field",
            "allowed_patch_scope": ["cognitive_evolve_runtime/example.py"],
        }
    )

    assert contract.dynamic_artifact_contract
    assert contract.allowed_patch_scope == ["cognitive_evolve_runtime/example.py"]


def test_generation_prompt_requires_actual_artifact_without_domain_hardcoding() -> None:
    contract = _contract(artifact_domain_label="novel_fragment_evolution")
    view = build_prompt_view(
        "nexus_seed_population",
        {"contract": contract, "world": {"kind": "text"}, "policy": {}, "candidates": []},
    ).payload

    generation_contract = view["artifact_generation_contract"]
    assert "actual object-level artifact" in generation_contract["non_negotiable_runtime_invariant"]
    assert generation_contract["examples_are_not_domain_limits"] is True
    assert generation_contract["model_defined_required_work_product"]["description"] == "a concrete revised artifact body"


def test_structured_design_candidate_can_rank_but_never_final() -> None:
    contract = _contract(
        allowed_artifact_shapes=[
            {"name": "model_defined_shape", "required_fields": ["content"]},
            {"name": "design_candidate", "stage": "exploration_non_final"},
        ],
        repair_contract={"design_candidate_rule": "non-final exploration material"},
    )
    candidate = CandidateGenome(
        id="C-design",
        artifact_type="design_candidate",
        concise_claim="repair lane should turn dormant material into smaller executable obligations",
        core_mechanism="split dormant material into explicit obligation deltas and candidate repair tasks",
        artifact={
            "kind": "design_candidate",
            "mechanism": "obligation-guided repair queue",
            "evaluation_dimensions": ["candidate vitality", "repair throughput"],
            "design_diff": "replace all-or-nothing Dormant retention with staged repair obligations",
            "failure_conditions": ["no new repair offspring after two rounds", "same blocker repeats without delta"],
        },
        metadata={"design_delta": {"relative_to_parent": "adds a reactivation mechanism rather than only describing Dormant state"}},
    )

    summary = evaluate_candidate_against_dynamic_contract(candidate, contract=contract)
    result = NexusVerifierStack().verify_candidate(candidate, contract=contract)

    assert summary.rank_eligible is True
    assert summary.final_eligible is False
    assert "design_candidate_non_final" in summary.diagnostics
    assert "artifact_object_absent" not in summary.diagnostics
    assert "meta_commentary_only" not in summary.diagnostics
    assert result.rank_eligible is True
    assert result.final_eligible is False


def test_incomplete_design_candidate_is_not_rank_eligible() -> None:
    contract = _contract(
        allowed_artifact_shapes=[{"name": "design_candidate", "stage": "exploration_non_final"}],
        repair_contract={"design_candidate_rule": "non-final exploration material"},
    )
    candidate = CandidateGenome(
        id="C-incomplete-design",
        artifact_type="design_candidate",
        concise_claim="we should improve the run",
        core_mechanism="needs a plan",
        artifact={"kind": "design_candidate", "mechanism": "placeholder"},
    )

    summary = evaluate_candidate_against_dynamic_contract(candidate, contract=contract)

    assert summary.rank_eligible is False
    assert summary.final_eligible is False
    assert "design_candidate_incomplete" in summary.diagnostics
