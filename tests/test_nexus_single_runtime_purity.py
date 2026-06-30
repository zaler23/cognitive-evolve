from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cognitive_evolve_runtime.runtime import runtime_run

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_run_has_no_runtime_selector() -> None:
    import inspect

    signature = inspect.signature(runtime_run)
    assert "runtime" not in signature.parameters


def test_runtime_default_writes_nexus_state(monkeypatch, tmp_path: Path) -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "classify", lambda prompt: SimpleNamespace(to_dict=lambda: {"route": "nexus_test"}, semantic={}))
    monkeypatch.setattr(runtime_module, "required_capabilities", lambda prompt, route=None: [])
    monkeypatch.setattr(runtime_module, "ensure_enhanced_task_contract", lambda *args, **kwargs: {})

    assert runtime_run(str(tmp_path), "solve with nexus", rounds=1, offline=True) == 0
    state = json.loads((tmp_path / "runtime-state.json").read_text(encoding="utf-8"))
    assert state["runtime_architecture"] == "nexus"
    assert state["runtime_path"] == "nexus"
    assert state["single_runtime"]["enforced"] is True
    assert (tmp_path / "nexus-runtime" / "run-result.json").exists()
    assert not (tmp_path / "internal-evolution").exists()


def test_runtime_explicit_rounds_are_not_clamped_by_route_incomplete_classifier() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    budget = runtime_module._runtime_round_budget(
        route_profile="deep",
        route_semantic={"task_type": "route_incomplete"},
        rounds=2,
    )

    assert budget.source == "explicit_request"
    assert budget.explicit_override == 2
    assert budget.max_rounds == 2
    assert budget.stop_policy != "route_incomplete_single_diagnostic"


def test_runtime_default_route_incomplete_uses_model_difficulty_budget_not_single_diagnostic() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    budget = runtime_module._runtime_round_budget(
        route_profile="deep",
        route_semantic={"task_type": "route_incomplete"},
        rounds=None,
    )

    assert budget.source == "adaptive_evolution_profile"
    assert budget.max_rounds > 1
    assert budget.stop_policy != "route_incomplete_single_diagnostic"


def test_runtime_model_difficulty_uses_generalized_hard_algorithm_baseline() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    route = SimpleNamespace(
        level="L3_comparative",
        profile="balanced",
        semantic={
            "task_type": "algorithm_problem",
            "model_route_available": True,
            "difficulty": "CCF-A hard",
            "model_self_assessment": {"capability_tier": "standard", "capability_score": 0.55},
        },
    )
    difficulty = runtime_module._runtime_entry_difficulty(route)
    budget = runtime_module._runtime_round_budget(
        route_profile=runtime_module._runtime_profile_from_difficulty(route.profile, difficulty),
        route_semantic=route.semantic,
        rounds=None,
        difficulty_assessment=difficulty,
    )

    assert difficulty["difficulty"] == "hard"
    assert difficulty["round_estimate"]["round_band"]["checkpoint_rounds"] == 48
    assert difficulty["round_estimate"]["semantics"] == "selected_rounds is an initial evolution checkpoint, not a correctness or solve guarantee"
    assert difficulty["round_estimate"]["selected_rounds"] == 48
    assert budget.profile == "deep"
    assert budget.source == "model_difficulty_round_estimate"
    assert budget.max_rounds == 48


def test_runtime_research_problem_uses_open_checkpoint_not_fixed_solution_count() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    route = SimpleNamespace(
        level="L4_evolutionary",
        profile="balanced",
        semantic={
            "task_type": "novel_algorithm_design",
            "model_route_available": True,
            "difficulty": "research",
            "suggested_rounds": 8,
            "model_self_assessment": {"capability_tier": "standard", "capability_score": 0.55},
        },
    )

    difficulty = runtime_module._runtime_entry_difficulty(route)
    budget = runtime_module._runtime_round_budget(
        route_profile=runtime_module._runtime_profile_from_difficulty(route.profile, difficulty),
        route_semantic=route.semantic,
        rounds=None,
        difficulty_assessment=difficulty,
    )

    assert difficulty["difficulty"] == "research"
    assert difficulty["round_estimate"]["round_band"]["lower_bound_rounds"] == 36
    assert difficulty["round_estimate"]["round_band"]["checkpoint_rounds"] == 120
    assert difficulty["round_estimate"]["model_suggested_rounds"] == 8
    assert difficulty["round_estimate"]["selected_rounds"] == 120
    assert budget.profile == "exhaustive"
    assert budget.max_rounds == 120


def test_runtime_model_self_assessed_limited_capability_increases_rounds() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    route = SimpleNamespace(
        level="L3_comparative",
        profile="balanced",
        semantic={
            "task_type": "algorithm_problem",
            "model_route_available": True,
            "difficulty": "hard",
            "model_self_assessment": {"capability_tier": "limited", "capability_score": 0.35},
        },
    )

    difficulty = runtime_module._runtime_entry_difficulty(route)
    budget = runtime_module._runtime_round_budget(
        route_profile=runtime_module._runtime_profile_from_difficulty(route.profile, difficulty),
        route_semantic=route.semantic,
        rounds=None,
        difficulty_assessment=difficulty,
    )

    assert difficulty["round_estimate"]["round_band"]["checkpoint_rounds"] == 48
    assert difficulty["round_estimate"]["model_capability"]["round_multiplier"] == 1.35
    assert difficulty["round_estimate"]["selected_rounds"] == 65
    assert budget.max_rounds == 65


def test_runtime_complex_research_direction_can_be_hundreds_of_rounds() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    route = SimpleNamespace(
        level="L4_evolutionary",
        profile="balanced",
        semantic={
            "task_type": "novel_algorithm_design",
            "model_route_available": True,
            "difficulty": "research",
            "model_self_assessment": {
                "capability_tier": "limited",
                "capability_score": 0.35,
                "target_output_level": "direction",
                "effort_class": "open_research",
                "complexity_dimensions": {
                    "novelty": 0.9,
                    "proof_burden": 0.85,
                    "search_space_width": 0.8,
                    "verification_difficulty": 0.75,
                },
            },
        },
    )

    difficulty = runtime_module._runtime_entry_difficulty(route)
    budget = runtime_module._runtime_round_budget(
        route_profile=runtime_module._runtime_profile_from_difficulty(route.profile, difficulty),
        route_semantic=route.semantic,
        rounds=None,
        difficulty_assessment=difficulty,
    )

    assert difficulty["round_estimate"]["target_output_level"] == "direction"
    assert difficulty["round_estimate"]["effort_class"] == "open_research"
    assert difficulty["round_estimate"]["selected_rounds"] >= 250
    assert budget.max_rounds == difficulty["round_estimate"]["selected_rounds"]


def test_runtime_explicit_rounds_override_model_difficulty_estimate() -> None:
    import cognitive_evolve_runtime.runtime as runtime_module

    budget = runtime_module._runtime_round_budget(
        route_profile="exhaustive",
        route_semantic={"task_type": "novel_algorithm_design"},
        rounds=6,
        difficulty_assessment={
            "difficulty": "research",
            "suggested_rounds": 8,
            "model_self_assessment": {"capability_tier": "limited", "capability_score": 0.35},
        },
    )

    assert budget.source == "explicit_request"
    assert budget.explicit_override == 6
    assert budget.max_rounds == 6


def test_final_answer_artifact_includes_human_candidate_table() -> None:
    from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
    from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
    from cognitive_evolve_runtime.nexus.runtime_services import candidates_markdown, final_answer_artifact_text

    candidate = ProjectCandidateGenome(
        id="C1",
        concise_claim="update parser",
        touched_files=["pkg/parser.py"],
        patch_application_result={"status": "applied", "applied_files": ["pkg/parser.py"]},
        multihead_scores={"objective_alignment": 0.8},
    )
    result = SimpleNamespace(
        completion_status="completed",
        stop_reason="max_rounds",
        synthesis=SimpleNamespace(best_candidate_id="C1", final_answer="\nAnswer body", closure_certificate={}),
        population=CandidatePopulation([candidate]),
        graded_output={},
    )

    text = final_answer_artifact_text(result)
    candidates = candidates_markdown(result)

    assert "## Candidate portfolio summary" in text
    assert "update parser" in text
    assert "| rank | candidate | direction | rank score | target files | patch status |" in candidates
    assert "pkg/parser.py" in candidates
    assert "| 1 | `C1` | update parser" in candidates
    assert "Mechanism family" not in candidates


def test_final_answer_candidate_portfolio_groups_by_canonical_family() -> None:
    from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
    from cognitive_evolve_runtime.nexus.runtime_services import final_answer_artifact_text

    alpha = CandidateGenome(
        id="A1",
        concise_claim="alpha direction",
        missing_parts=["alpha gap", "second gap"],
        uncertainty_notes=["alpha uncertainty", "second note"],
        metadata={"nextgen": {"canonical_mechanism_family_id": "alpha#m1"}},
        multihead_scores={"rank_score": 0.9},
    )
    beta = CandidateGenome(
        id="B1",
        concise_claim="beta direction",
        metadata={"nextgen": {"canonical_mechanism_family_id": "beta#m2"}},
        multihead_scores={"rank_score": 0.8},
    )
    result = SimpleNamespace(
        completion_status="completed",
        stop_reason="max_rounds",
        synthesis=SimpleNamespace(best_candidate_id="A1", final_answer="\nAnswer body", closure_certificate={}),
        population=CandidatePopulation([alpha, beta]),
        graded_output={},
    )

    text = final_answer_artifact_text(result)

    assert "### Mechanism family: alpha#m1" in text
    assert "### Mechanism family: beta#m2" in text
    assert "| rank | direction | verification status | uncertainty |" in text
    assert "missing: ['alpha gap']; notes: ['alpha uncertainty']" in text
    assert "second gap" not in text


def test_final_answer_candidate_portfolio_missing_canonical_uses_unrecorded() -> None:
    from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
    from cognitive_evolve_runtime.nexus.runtime_services import final_answer_artifact_text

    result = SimpleNamespace(
        completion_status="completed",
        stop_reason="max_rounds",
        synthesis=SimpleNamespace(best_candidate_id="C1", final_answer="\nAnswer body", closure_certificate={}),
        population=CandidatePopulation(
            [
                CandidateGenome(id="C1", concise_claim="first", multihead_scores={"rank_score": 0.9}),
                CandidateGenome(id="C2", concise_claim="second", multihead_scores={"rank_score": 0.8}),
            ]
        ),
        graded_output={},
    )

    text = final_answer_artifact_text(result)

    assert text.count("### Mechanism family: unrecorded") == 1
    assert "### Mechanism family: C1" not in text
    assert "### Mechanism family: C2" not in text


def test_final_answer_candidate_status_is_answer_first_not_local_verified() -> None:
    from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
    from cognitive_evolve_runtime.nexus.runtime_services import final_answer_artifact_text

    result = SimpleNamespace(
        completion_status="completed",
        stop_reason="max_rounds",
        synthesis=SimpleNamespace(best_candidate_id="C1", final_answer="\nAnswer body", closure_certificate={}),
        population=CandidatePopulation(
            [
                CandidateGenome(
                    id="C1",
                    concise_claim="local verified only",
                    verification_result={"passed": True},
                    multihead_scores={"rank_score": 0.9},
                )
            ]
        ),
        graded_output={},
    )

    text = final_answer_artifact_text(result)

    assert "correctness_verdict: external_validation_required" in text
    assert "project_correctness_claim: not_claimed" in text
    assert "| 1 | local verified only | advisory |" in text


def test_final_answer_artifact_defers_correctness_to_external_review() -> None:
    from cognitive_evolve_runtime.nexus.runtime_services import final_answer_artifact_text

    text = final_answer_artifact_text(
        SimpleNamespace(
            completion_status="completed",
            stop_reason="max_rounds",
            synthesis=SimpleNamespace(best_candidate_id="C1", final_answer="\nAnswer body"),
        )
    )

    assert "correctness_verdict: external_validation_required" in text
    assert "project_correctness_claim: not_claimed" in text
    assert "final_gate:" not in text
    assert "objective_solved:" not in text


def test_runtime_run_requires_llm_unless_explicit_offline(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv("COGEV_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("COGEV_LLM_FIXTURE", raising=False)
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    assert runtime_run(str(tmp_path), "solve with nexus", rounds=1) == 2
    err = capsys.readouterr().err
    assert "LLM configuration required" in err
    assert "--offline" in err


def test_no_package_json_control_plane() -> None:
    assert not Path("package.json").exists()


def test_removed_parallel_namespaces_are_absent() -> None:
    for relative in ["cognitive_evolve_runtime/archive", "cognitive_evolve_runtime/optimizer", "cognitive_evolve_runtime/adaptive_engine.py", "cognitive_evolve_runtime/candidate_search.py", "cognitive_evolve_runtime/multi_agent_optimizer.py"]:
        assert not (ROOT / relative).exists(), relative
