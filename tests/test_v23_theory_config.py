from __future__ import annotations

from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.v23_theory_config import V23TheoryRuntimeConfig


class _Contract:
    def __init__(self, metadata):
        self.metadata = metadata


def test_policy_metadata_drives_typed_v23_config() -> None:
    policy = EvolutionPolicy(
        metadata={
            "v23_theory_runtime": {
                "entropy": {"cell_elite_reserve": 2, "rarity_weight": 0.7},
                "minimax_budget": {"min_budget_per_candidate": 2},
                "honesty_control": {"window": 3, "proportional_gain": {"variety": 0.4}},
                "ca_crossover": {"global_donor_policy": "highest_final_quality"},
            }
        }
    )
    config = V23TheoryRuntimeConfig.from_runtime_context(policy=policy)

    assert config.entropy.cell_elite_reserve == 2
    assert config.entropy.rarity_weight == 0.7
    assert config.minimax_budget.min_budget_per_candidate == 2
    assert config.honesty_control.window == 3
    assert config.honesty_control.proportional_gain["variety"] == 0.4
    assert config.ca_crossover.global_donor_policy == "highest_final_quality"
    assert config.config_hash.startswith("v23-theory-")


def test_contract_metadata_config_is_used_when_policy_absent() -> None:
    contract = _Contract({"v23_theory_runtime": {"ca_crossover": {"min_shared_descriptor_tokens": 2}}})
    config = V23TheoryRuntimeConfig.from_runtime_context(contract=contract)

    assert config.ca_crossover.min_shared_descriptor_tokens == 2


def test_deprecated_env_switches_are_diagnostics_only(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_COMPACT_MODE", "entropy")
    policy = EvolutionPolicy(metadata={"COGEV_HONESTY_CONTROL": "1"})
    before = V23TheoryRuntimeConfig.from_runtime_context(policy=EvolutionPolicy()).to_dict()
    after = V23TheoryRuntimeConfig.from_runtime_context(policy=policy).to_dict()

    assert before["entropy"] == after["entropy"]
    assert before["honesty_control"] == after["honesty_control"]
    assert any("ignored_deprecated_v23_env_switch:COGEV_COMPACT_MODE" == item for item in after["diagnostics"])
    assert any("ignored_deprecated_v23_switch:COGEV_HONESTY_CONTROL" == item for item in after["diagnostics"])
