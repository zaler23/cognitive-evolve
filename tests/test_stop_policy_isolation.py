from __future__ import annotations

import os
from pathlib import Path

from cognitive_evolve_runtime.configuration import load_layered_config
from cognitive_evolve_runtime.nexus.budgeting import resolve_nexus_round_budget
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


def test_stop_policy_isolation_ignores_home_config_in_hermetic_mode(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-root"
    home_cfg = runtime_root / ".cogev" / "config.yaml"
    home_cfg.parent.mkdir(parents=True)
    home_cfg.write_text("evolution:\n  stop_policy: aggressive-home-policy\n", encoding="utf-8")

    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.delenv("COGEV_CONFIG_FILE", raising=False)
    monkeypatch.delenv("COGEV_STOP_POLICY", raising=False)

    assert load_layered_config() is None
    assert os.environ.get("COGEV_STOP_POLICY") is None


def test_stop_policy_can_be_loaded_from_explicit_tmp_config(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("evolution:\n  stop_policy: fixture-only\n", encoding="utf-8")
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_CONFIG_FILE", str(cfg))
    monkeypatch.delenv("COGEV_STOP_POLICY", raising=False)

    assert load_layered_config() == cfg.resolve()
    assert os.environ["COGEV_STOP_POLICY"] == "fixture-only"


def test_stop_policy_is_carried_into_budget_checkpoint_and_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_STOP_POLICY", "convergence_or_max_rounds")
    monkeypatch.setenv("COGEV_MIN_ROUNDS_BEFORE_STOP", "1")

    budget = resolve_nexus_round_budget({"rounds": 3})
    assert budget.stop_policy == "convergence_or_max_rounds"
    assert budget.min_rounds_before_stop == 1

    result = NexusRuntime(output_dir=tmp_path).run_text(
        "Small task",
        max_rounds=3,
        stop_policy=budget.stop_policy,
        min_rounds_before_stop=budget.min_rounds_before_stop,
    )

    assert result.evolution["stop_reason"] in {"converged_after_minimum", "max_rounds"}
    checkpoint = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    assert '"stop_policy": "convergence_or_max_rounds"' in checkpoint


def test_runtime_accepts_single_evolution_budget_object(tmp_path: Path) -> None:
    budget = EvolutionBudget(max_rounds=1, branch_factor=3, initial_candidate_count=11, stop_policy="max_rounds")

    result = NexusRuntime(output_dir=tmp_path).run_text("Budget object task", budget=budget)

    assert result.evolution["max_rounds"] == 1
    assert len(result.evolution["population"]["candidates"]) >= 11
    checkpoint = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    assert '"branch_factor": 3' in checkpoint
    assert '"initial_candidate_count": 11' in checkpoint


def test_legacy_profile_rounds_and_candidates_do_not_override_adaptive_defaults(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS", "60")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_CANDIDATES", "24")
    monkeypatch.delenv("COGEV_ACCEPT_LEGACY_PROFILE_ROUNDS", raising=False)
    monkeypatch.delenv("COGEV_ACCEPT_LEGACY_PROFILE_CANDIDATES", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_SAFETY_MAX_ROUNDS", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_MIN_CANDIDATES", raising=False)

    budget = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})

    assert budget.adaptive is True
    assert budget.round_safety_limit == 240
    assert budget.max_rounds == 240
    assert budget.initial_candidate_count == 0
    assert "COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS_ignored_use_COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS" in budget.to_dict()["config_warnings"]
    assert "COGEV_NEXUS_PROFILE_EXHAUSTIVE_CANDIDATES_ignored_use_COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES" in budget.to_dict()["config_warnings"]


def test_new_safety_and_candidate_floor_env_names_are_honored(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS", "60")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_CANDIDATES", "24")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "180")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES", "9")

    budget = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})

    assert budget.round_safety_limit == 180
    assert budget.initial_candidate_count == 9


def test_budget_guarded_stop_policy_is_adaptive_alias(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_STOP_POLICY", "budget_guarded")

    budget = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})

    assert budget.stop_policy == "adaptive_until_solved"


def test_invalid_stop_policy_falls_back_to_selected_profile_default(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_STOP_POLICY", "not-a-policy")

    exhaustive = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})
    balanced = resolve_nexus_round_budget({"evolution_profile": "balanced"})

    assert exhaustive.stop_policy == "adaptive_until_solved"
    assert balanced.stop_policy == "llm_after_minimum"


def test_legacy_branch_and_active_pool_env_do_not_shape_adaptive_budget(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MUTATION_BRANCH_FACTOR", "1")
    monkeypatch.setenv("COGEV_ACTIVE_POOL_LIMIT", "6")
    monkeypatch.delenv("COGEV_ACCEPT_LEGACY_MUTATION_BRANCH_FACTOR", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_BRANCH_FACTOR", raising=False)

    budget = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})
    warnings = budget.to_dict()["config_warnings"]

    assert budget.mutation_branches_per_round == 0
    assert "COGEV_MUTATION_BRANCH_FACTOR_ignored_use_COGEV_NEXUS_BRANCH_FACTOR" in warnings
    assert "COGEV_ACTIVE_POOL_LIMIT_ignored_active_pool_is_policy_internal" in warnings


def test_new_global_branch_factor_env_is_honored(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MUTATION_BRANCH_FACTOR", "1")
    monkeypatch.setenv("COGEV_NEXUS_BRANCH_FACTOR", "7")

    budget = resolve_nexus_round_budget({"evolution_profile": "exhaustive"})

    assert budget.mutation_branches_per_round == 7
