from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter


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


def test_offspring_response_accepts_candidate_alias() -> None:
    def caller(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "id": "O1",
                    "artifact_type": "answer",
                    "concise_claim": "offspring under alias",
                    "core_mechanism": "alias repair",
                }
            ]
        }

    adapter = StructuredModelAdapter(caller=caller)

    offspring = adapter.generate_offspring(plans=[], parents=[], world={}, contract={}, policy={})

    assert offspring[0]["id"] == "O1"
    assert offspring[0]["artifact"] == "offspring under alias"
    assert offspring[0]["touched_files"] == []
    assert offspring[0]["source_bindings"] == []
    assert offspring[0]["evidence_refs"] == []
    assert offspring[0]["evaluation_dimensions"] == []
    assert offspring[0]["final_gate"] == {}
    assert set(offspring[0]["metadata"]["schema_repair_fields"]) >= {
        "touched_files",
        "evidence_refs",
        "evaluation_dimensions",
        "final_gate",
    }


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
