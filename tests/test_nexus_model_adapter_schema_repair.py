from __future__ import annotations

from typing import Any

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract, NexusProjectObjectiveContract
from cognitive_evolve_runtime.evaluators.evidence import select_preliminary_incumbent
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.offspring import _merge_plan_metadata_into_model_offspring
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError, StructuredModelAdapter
from cognitive_evolve_runtime.nexus.prompt_view import candidate_prompt_view
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.strength import candidate_verification_strength, strongest_passed_replayable_result


def test_objective_contract_missing_required_goal_fields_is_repaired() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_constraints": "use supplied problem statement",
            "success_dimensions": "objective_alignment",
        }

    adapter = StructuredModelAdapter(caller=caller)

    contract = adapter.build_objective_contract(user_goal="  Solve the real task.  ", world={"kind": "text"})

    assert contract["original_user_goal"] == "Solve the real task."
    assert contract["normalized_goal"] == "Solve the real task."
    assert contract["input_constraints"] == ["use supplied problem statement"]
    assert contract["success_dimensions"] == ["objective_alignment"]


def test_objective_contract_records_goal_normalization_rewrite() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_user_goal": "Solve the Erdős unit-distance problem.",
            "normalized_goal": "Explain how to structure a prompt as JSON.",
        }

    contract = StructuredModelAdapter(caller=caller).build_objective_contract(
        user_goal="Solve the Erdős unit-distance problem.",
        world={"kind": "text"},
    )

    rewrite = contract["metadata"]["goal_normalization_rewrite"]
    assert rewrite["policy"] == "original_user_goal_remains_frozen_contract_boundary"
    assert rewrite["original_user_goal_sha256"] != rewrite["normalized_goal_sha256"]


def test_flat_model_artifact_schema_survives_contract_normalization() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_user_goal": "evolve permutation paths",
            "normalized_goal": "evolve permutation paths",
            "dynamic_artifact_contract": {
                "artifact_type": "permutation_path",
                "required_fields": ["representation", "permutation_path", "claimed_length"],
                "field_constraints": {"representation": {"const": "permutation_path"}},
                "forbidden_substitutions": ["partial path", "search plan"],
                "acceptance_rule": "external evaluator passes",
            },
        }

    contract = StructuredModelAdapter(caller=caller).build_objective_contract(
        user_goal="evolve permutation paths",
        world={"kind": "text"},
    )
    artifact_contract = contract["dynamic_artifact_contract"]

    assert artifact_contract["artifact_domain_label"] == "permutation_path"
    assert artifact_contract["required_work_product"]["required_fields"] == [
        "representation",
        "permutation_path",
        "claimed_length",
    ]
    assert artifact_contract["allowed_artifact_shapes"][0]["field_constraints"] == {
        "representation": {"const": "permutation_path"}
    }
    assert artifact_contract["invalid_outputs"] == ["partial path", "search plan"]
    assert artifact_contract["final_gate"] == {"acceptance_rule": "external evaluator passes"}


def test_project_contract_preserves_semantic_regions_and_dynamic_contract_extensions(tmp_path) -> None:
    source = tmp_path / "pkg" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)
    semantic_regions = ["fan-out quota allocation", "duplicate-exhaustion feedback"]
    extensions = {
        "artifact_kind": "minimal_patch_candidate",
        "required_candidate_metadata": ["search_space.family_id"],
        "exactness_invariants": ["preserve exact artifacts"],
        "combination_rule": "combine only independent loci",
        "incomplete_rule": "keep the concrete repair obligation",
    }

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_user_goal": "improve runtime",
            "normalized_goal": "improve runtime",
            "mutable_regions": semantic_regions,
            "implementation_files": ["runtime.py"],
            "allowed_patch_scope": ["runtime.py"],
            "dynamic_artifact_contract": {
                **extensions,
                "required_work_product": {"required_fields": ["locus", "minimal_diff"]},
            },
        }

    adapted = StructuredModelAdapter(caller=caller).build_project_objective_contract(
        user_goal="improve runtime",
        snapshot=snapshot,
    )
    persisted = NexusProjectObjectiveContract.from_dict(adapted).to_dict()

    assert adapted["mutable_regions"] == semantic_regions
    assert adapted["implementation_files"] == ["pkg/runtime.py"]
    assert adapted["allowed_patch_scope"] == ["pkg/runtime.py"]
    assert {key: adapted["dynamic_artifact_contract"][key] for key in extensions} == extensions
    assert {key: persisted["dynamic_artifact_contract"][key] for key in extensions} == extensions


def test_wrapped_project_objective_contract_is_unwrapped_and_repaired() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "contract": {
                "normalized_goal": "Keep public API stable",
                "allowed_patch_scope": "cognitive_evolve_runtime/**/*.py",
            },
            "provider": "fixture",
        }

    adapter = StructuredModelAdapter(caller=caller)

    contract = adapter.build_project_objective_contract(user_goal="Patch project safely", snapshot={"files": []})

    assert contract["original_user_goal"] == "Patch project safely"
    assert contract["normalized_goal"] == "Keep public API stable"
    assert contract["allowed_patch_scope"] == ["cognitive_evolve_runtime/**/*.py"]
    assert contract["provider"] == "fixture"


def test_project_contract_drops_nonexistent_model_paths_and_falls_back_to_snapshot(tmp_path) -> None:
    source = tmp_path / "cognitive_evolve_runtime" / "nexus" / "model_adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text("class StructuredModelAdapter: pass\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_nexus_model_adapter_schema_repair.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_schema_repair(): pass\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_user_goal": "Patch project safely",
            "normalized_goal": "Patch project safely",
            "implementation_files": ["model_adapter.py"],
            "test_contracts": ["test_nexus_model_adapter_schema_repair.py"],
            "allowed_patch_scope": ["nexus_model_adapter_schema_repair.py"],
        }

    adapter = StructuredModelAdapter(caller=caller)

    contract = adapter.build_project_objective_contract(user_goal="Patch project safely", snapshot=snapshot)

    assert contract["implementation_files"] == ["cognitive_evolve_runtime/nexus/model_adapter.py"]
    assert contract["test_contracts"] == ["tests/test_nexus_model_adapter_schema_repair.py"]
    assert contract["allowed_patch_scope"] == [
        "cognitive_evolve_runtime/nexus/model_adapter.py",
        "tests/test_nexus_model_adapter_schema_repair.py",
    ]
    assert any("dropped missing nexus_model_adapter_schema_repair.py" in note for note in contract["path_repair_notes"])


def test_project_contract_tests_only_scope_is_expanded_with_implementation_files(tmp_path) -> None:
    source = tmp_path / "pkg" / "runtime.py"
    source.parent.mkdir()
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_runtime.py"
    test_file.parent.mkdir()
    test_file.write_text("from pkg.runtime import value\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_user_goal": "Patch project safely",
            "normalized_goal": "Patch runtime and tests",
            "allowed_patch_scope": ["tests/test_runtime.py"],
        }

    contract = StructuredModelAdapter(caller=caller).build_project_objective_contract(
        user_goal="Patch project safely",
        snapshot=snapshot,
    )

    assert contract["allowed_patch_scope"] == ["pkg/runtime.py", "tests/test_runtime.py"]
    assert "allowed_patch_scope:added_snapshot_implementation_files" in contract["path_repair_notes"]


def test_offspring_response_does_not_invent_artifact_from_claim() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "id": "O1",
                    "artifact_type": "answer",
                    "concise_claim": "offspring under alias",
                    "core_mechanism": "alias repair",
                    "parent_ids": ["P1"],
                }
            ]
        }

    adapter = StructuredModelAdapter(caller=caller)

    with pytest.raises(ModelResponseSchemaError, match="artifact"):
        adapter.generate_offspring(plans=[], parents=[], world={}, contract={}, policy={})


def test_bare_frozen_task_artifact_is_preserved_in_runtime_envelope() -> None:
    calls = 0
    artifact = {
        "format": "full_file",
        "path": "atomic_store.py",
        "content": "class AtomicKV:\n    pass\n",
    }

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert request_type == "nexus_generate_offspring"
        return dict(artifact)

    parent = CandidateGenome(id="P1", artifact="parent")
    plan = MutationPlan(
        operator="Repair",
        parent_ids=[parent.id],
        instruction="return the complete task artifact",
        metadata={
            "plan_id": "plan-1",
            "plan_source": "runtime_lineage_envelope",
            "completion_mode": "complete_task_artifact_only",
        },
    )
    contract = NexusObjectiveContract(
        original_user_goal="produce atomic_store.py",
        normalized_goal="produce atomic_store.py",
        frozen_spec={"problem_text": "Output {format,path,content}."},
    )

    offspring = StructuredModelAdapter(caller=caller).generate_offspring(
        plans=[plan],
        parents=[parent],
        world={},
        contract=contract,
        policy={},
    )

    assert calls == 1
    assert offspring[0]["artifact"] == artifact
    assert offspring[0]["parent_ids"] == [parent.id]
    assert offspring[0]["metadata"]["plan_id"] == "plan-1"
    assert offspring[0]["metadata"]["runtime_artifact_ingestion"] == {
        "source": "bare_model_task_artifact",
        "authority": "external_evaluator+contract.frozen_spec",
        "artifact_preserved": True,
        "parent_binding": "single_mutation_plan",
    }
    child = candidate_from_dict(offspring[0])
    _merge_plan_metadata_into_model_offspring([child], [plan], [parent])
    assert child.artifact == artifact
    assert child.generation == 1
    assert child.lineage == [parent.id, child.id]


def test_bare_task_artifact_without_frozen_authority_remains_schema_error() -> None:
    calls = 0

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"format": "full_file", "path": "atomic_store.py", "content": "pass\n"}

    parent = CandidateGenome(id="P1", artifact="parent")
    plan = MutationPlan(operator="Repair", parent_ids=[parent.id], instruction="repair")

    with pytest.raises(ModelResponseSchemaError, match="offspring"):
        StructuredModelAdapter(caller=caller).generate_offspring(
            plans=[plan],
            parents=[parent],
            world={},
            contract=NexusObjectiveContract(original_user_goal="task", normalized_goal="task"),
            policy={},
        )
    assert calls == 1


def test_exploratory_lineage_envelope_cannot_ingest_bare_frozen_artifact() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {"format": "full_file", "path": "atomic_store.py", "content": "pass\n"}

    parent = CandidateGenome(id="P1", artifact="parent")
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        metadata={
            "plan_source": "runtime_lineage_envelope",
            "completion_mode": "concrete_progress_allowed",
        },
    )
    contract = NexusObjectiveContract(
        original_user_goal="produce atomic_store.py",
        normalized_goal="produce atomic_store.py",
        frozen_spec={"problem_text": "Output {format,path,content}."},
    )

    with pytest.raises(ModelResponseSchemaError, match="offspring"):
        StructuredModelAdapter(caller=caller).generate_offspring(
            plans=[plan],
            parents=[parent],
            world={},
            contract=contract,
            policy={},
        )


def test_bare_artifact_must_satisfy_declared_fields_even_with_frozen_spec() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {"analysis": "not the required artifact"}

    parent = CandidateGenome(id="P1", artifact={"answer": "parent"})
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        metadata={"plan_source": "runtime_lineage_envelope"},
    )
    contract = NexusObjectiveContract(
        original_user_goal="return answer",
        normalized_goal="return answer",
        frozen_spec={"problem_text": "Return an answer object."},
        dynamic_artifact_contract={
            "adapter_requirements": {"required_fields": ["answer"]},
        },
    )

    with pytest.raises(ModelResponseSchemaError, match="offspring"):
        StructuredModelAdapter(caller=caller).generate_offspring(
            plans=[plan],
            parents=[parent],
            world={},
            contract=contract,
            policy={},
        )


def test_offspring_requested_count_is_explicit_in_model_payload() -> None:
    captured: dict[str, Any] = {}

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {
            "offspring": [
                {
                    "id": "O1",
                    "artifact": {"answer": 1},
                    "artifact_type": "machine",
                    "concise_claim": "one direct child",
                    "core_mechanism": "direct evolution",
                    "parent_ids": ["P1"],
                }
            ]
        }

    StructuredModelAdapter(caller=caller).generate_offspring(
        plans=[MutationPlan(operator="ModelDirected", parent_ids=["P1"])],
        parents=[CandidateGenome(id="P1", artifact={"answer": 0})],
        world={},
        contract={},
        policy={"metadata": {"requested_candidate_count": 3}},
    )

    assert captured["requested_candidate_count"] == 3


def test_seed_population_missing_candidates_triggers_exactly_one_schema_repair_retry() -> None:
    calls: list[dict[str, Any]] = []

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(payload))
        if len(calls) == 1:
            return {"diagnostic": "missing candidates"}
        assert payload["_schema_repair_retry"]["max_retries"] == 1
        return {
            "candidates": [
                {
                    "id": "S1",
                    "artifact": "seed",
                    "artifact_type": "answer",
                    "concise_claim": "seed claim",
                    "core_mechanism": "seed mechanism",
                    "assumptions": [],
                    "missing_parts": [],
                    "uncertainty_notes": [],
                }
            ]
        }

    adapter = StructuredModelAdapter(caller=caller)

    seeds = adapter.seed_population(contract={}, world={}, policy={})

    assert [seed["id"] for seed in seeds] == ["S1"]
    assert len(calls) == 2
    repairs = adapter.metadata["schema_repair_events"]
    assert [event["repair"] for event in repairs].count("schema_repair_retry") == 1


def test_bare_seed_artifact_with_declared_schema_is_preserved_without_retry() -> None:
    calls = 0
    artifact = {
        "representation": "permutation_path",
        "permutation_path": ["123456"],
        "claimed_length": 6,
    }

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert request_type == "nexus_seed_population"
        return dict(artifact)

    seeds = StructuredModelAdapter(caller=caller).seed_population(
        contract={
            "dynamic_artifact_contract": {
                "artifact_domain_label": "permutation_path",
                "required_work_product": {
                    "required_fields": [
                        "representation",
                        "permutation_path",
                        "claimed_length",
                    ]
                },
                "allowed_artifact_shapes": [],
            }
        },
        world={},
        policy={},
    )

    assert calls == 1
    assert len(seeds) == 1
    assert seeds[0]["artifact"] == artifact
    assert seeds[0]["artifact_type"] == "permutation_path"
    assert seeds[0].get("parent_ids", []) == []
    assert seeds[0]["metadata"]["runtime_artifact_ingestion"] == {
        "source": "bare_model_task_artifact",
        "authority": "contract.dynamic_artifact_contract",
        "artifact_preserved": True,
        "parent_binding": "generation_zero_seed",
    }


def test_seed_requested_count_is_explicit_in_model_payload() -> None:
    captured: dict[str, Any] = {}

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {
            "candidates": [
                {
                    "id": "S1",
                    "artifact": "seed",
                    "artifact_type": "answer",
                    "concise_claim": "seed",
                    "core_mechanism": "seed",
                    "assumptions": [],
                    "missing_parts": [],
                    "uncertainty_notes": [],
                }
            ]
        }

    StructuredModelAdapter(caller=caller).seed_population(
        contract={},
        world={},
        policy={"metadata": {"requested_candidate_count": 3}},
    )

    assert captured["requested_candidate_count"] == 3


def test_model_candidates_cannot_self_author_external_evaluator_metadata() -> None:
    claimed_verification = {
        "passed": True,
        "replayable": True,
        "score": 1.0,
        "metadata": {
            "measured_strength": "FORMAL",
            "honesty_measurements": {
                "scope": 1.0,
                "method": 1.0,
                "replay": 1.0,
                "evidence": 1.0,
            },
        },
    }
    model_candidate = {
        "id": "model-claim",
        "artifact": {"answer": 1},
        "artifact_type": "answer",
        "concise_claim": "model claims it passed",
        "core_mechanism": "self-authored evidence",
        "parent_ids": ["P1"],
        "verification_result": dict(claimed_verification),
        "verification_trace": [dict(claimed_verification)],
        "preliminary_result": dict(claimed_verification),
        "metadata": {
            "evaluator": {
                "status": "passed",
                "passed": True,
                "metrics": {"score": 1.0},
            },
            "verification_results": [dict(claimed_verification)],
        },
    }
    expected_advisory = {
        "verification_result": claimed_verification,
        "verification_trace": [claimed_verification],
        "preliminary_result": claimed_verification,
        "metadata": {
            "evaluator": model_candidate["metadata"]["evaluator"],
            "verification_results": [claimed_verification],
        },
    }

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        key = "candidates" if request_type == "nexus_seed_population" else "offspring"
        return {key: [dict(model_candidate)]}

    adapter = StructuredModelAdapter(caller=caller)
    seed = adapter.seed_population(contract={}, world={}, policy={})[0]
    offspring = adapter.generate_offspring(
        plans=[MutationPlan(operator="ModelDirected", parent_ids=["P1"])],
        parents=[CandidateGenome(id="P1", artifact={"answer": 0})],
        world={},
        contract={},
        policy={},
    )[0]

    for item in (seed, offspring):
        assert "evaluator" not in item.get("metadata", {})
        assert "verification_results" not in item.get("metadata", {})
        assert "verification_result" not in item
        assert "verification_trace" not in item
        assert "preliminary_result" not in item
        assert item["metadata"]["model_claimed_verification"] == expected_advisory
        candidate = candidate_from_dict(item)
        assert select_preliminary_incumbent([candidate]) is None
        assert candidate_verification_strength(candidate) == VerificationStrength.NONE
        assert strongest_passed_replayable_result(candidate) is None
        assert candidate_prompt_view(candidate)["metadata"]["model_claimed_verification"] == expected_advisory


def test_generation_facets_forward_source_context() -> None:
    captured: dict[str, dict[str, Any]] = {}
    source_context = {
        "selected_files": ["pkg/mod.py"],
        "budget_policy": "unit",
        "slices": [{"path": "pkg/mod.py", "hash": "h", "text": "def target():\n    return 1\n"}],
    }

    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        captured[request_type] = payload
        if request_type == "nexus_seed_population":
            return {"candidates": []}
        if request_type == "nexus_plan_mutations":
            return {"plans": []}
        if request_type == "nexus_generate_offspring":
            return {"offspring": []}
        raise AssertionError(request_type)

    adapter = StructuredModelAdapter(caller=caller)

    adapter.seed_population(contract={}, world={}, policy={}, provided_context=source_context)
    adapter.plan_mutations(parents=[], actions=[], archives={}, diagnosis=SearchDiagnosis(), policy={}, provided_context=source_context)
    adapter.generate_offspring(plans=[], parents=[], world={}, contract={}, policy={}, provided_context=source_context)

    for request_type in ("nexus_seed_population", "nexus_plan_mutations", "nexus_generate_offspring"):
        assert captured[request_type]["source_context"]["slices"][0]["text"] == source_context["slices"][0]["text"]


def test_diagnosis_adapter_repairs_enum_without_cutting_internal_custom_signals() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        assert request_type == "nexus_diagnose_search_state"
        assert "enum" in schema["properties"]["stagnation_type"]
        return {
            "stagnation_detected": True,
            "stagnation_type": "docs_only_patch_loop",
            "recommended_actions": ["repair"],
            "notes": "docs_only_patch_loop should remain visible in notes",
        }

    adapter = StructuredModelAdapter(caller=caller)

    repaired = adapter.diagnose_search_state(population=[], archives={}, history=[], contract={}, policy={})
    internal = SearchDiagnosis(stagnation_detected=True, stagnation_type="docs_only_patch_loop")

    assert repaired["stagnation_type"] == "RouteIncomplete"
    assert repaired["metadata"]["raw_stagnation_type"] == "docs_only_patch_loop"
    assert "docs_only_patch_loop" in repaired["notes"]
    assert internal.stagnation_type == "docs_only_patch_loop"


def test_offspring_schema_repairs_structured_fields_from_patch_headers() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        assert "touched_files" in schema["properties"]["offspring"]["items"]["required"]
        return {
            "offspring": [
                {
                    "id": "O2",
                    "artifact_type": "code_patch",
                    "concise_claim": "patch transport retry",
                    "core_mechanism": "increase long call output budget",
                    "parent_ids": ["P1"],
                    "artifact": {
                        "unified_diff": (
                            "diff --git a/cognitive_evolve_runtime/llm/transport.py b/cognitive_evolve_runtime/llm/transport.py\n"
                            "--- a/cognitive_evolve_runtime/llm/transport.py\n"
                            "+++ b/cognitive_evolve_runtime/llm/transport.py\n"
                            "@@ -1 +1 @@\n"
                            "-old\n"
                            "+new\n"
                        )
                    },
                    "multihead_scores": {"verifiability": 0.8},
                }
            ]
        }

    offspring = StructuredModelAdapter(caller=caller).generate_offspring(plans=[], parents=[], world={}, contract={}, policy={})

    assert offspring[0]["touched_files"] == ["cognitive_evolve_runtime/llm/transport.py"]
    assert offspring[0]["source_bindings"][0]["path"] == "cognitive_evolve_runtime/llm/transport.py"
    assert offspring[0]["evaluation_dimensions"] == ["verifiability"]


def test_evolution_policy_schema_accepts_model_driven_eligibility_metadata() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        assert request_type == "nexus_build_evolution_policy"
        assert "metadata" in schema["properties"]
        return {
            "candidate_niches": ["direct"],
            "fitness_axes": ["objective_alignment"],
            "mutation_operators": ["Deepen"],
            "archive_schema": {"AnswerArchive": {"enabled": True}},
            "metadata": {
                "eligibility_policy": {
                    "source": "model",
                    "stage_fractions": {"early_until": 0.2, "middle_until": 0.8, "late_until": 0.95},
                    "active_floor": {"enabled": True, "branch_multiplier": "auto"},
                }
            },
        }

    adapter = StructuredModelAdapter(caller=caller)

    policy = adapter.build_evolution_policy(contract={}, world={})

    assert policy["metadata"]["eligibility_policy"]["source"] == "model"
