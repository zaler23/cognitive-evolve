from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import _enable_project_theory_advisory_pressure
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def test_enabled_theory_policy_adds_sidecar_without_changing_eligibility() -> None:
    short = CandidateGenome(id="short", current_fate="Active", concise_claim="small", multihead_scores={"objective_alignment": 0.5})
    long = CandidateGenome(id="long", current_fate="Active", concise_claim="large " * 200, multihead_scores={"objective_alignment": 0.5})
    dormant = CandidateGenome(id="dormant", current_fate="Dormant", concise_claim="tiny", multihead_scores={"objective_alignment": 1.0})
    policy = EvolutionPolicy(
        metadata={
            "theory": {
                "enabled": True,
                "producers": {"mdl": True},
                "weights": {"mdl": 3.0},
            }
        }
    )
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=2))

    features = round_stage._theory_advisory_features(policy=policy, candidates=[long, short, dormant], current_round=1)
    selected = ParentSelector().select([long, short, dormant], limit=2, advisory_features=features)

    assert set(features) == {"dormant", "long", "short"}
    assert selected[0].id == "short"
    assert {candidate.id for candidate in selected} == {"short", "dormant"}
    assert "dormant" in {candidate.id for candidate in selected}


def test_disabled_theory_policy_keeps_original_parent_order() -> None:
    a = CandidateGenome(id="a", current_fate="Active", concise_claim="short", multihead_scores={"objective_alignment": 0.6})
    b = CandidateGenome(id="b", current_fate="Active", concise_claim="long " * 100, multihead_scores={"objective_alignment": 0.4})
    policy = EvolutionPolicy()
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=2))

    features = round_stage._theory_advisory_features(policy=policy, candidates=[a, b], current_round=1)
    selected = ParentSelector().select([a, b], limit=2, advisory_features=features)

    assert features == {}
    assert [candidate.id for candidate in selected] == ["a", "b"]


def test_project_runtime_theory_defaults_feed_soft_parent_pressure() -> None:
    long = CandidateGenome(id="long", current_fate="Active", concise_claim="large " * 200, multihead_scores={"objective_alignment": 0.5})
    short = CandidateGenome(id="short", current_fate="Active", concise_claim="small", multihead_scores={"objective_alignment": 0.5})
    policy = EvolutionPolicy()
    _enable_project_theory_advisory_pressure(policy)
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=2))

    features = round_stage._theory_advisory_features(policy=policy, candidates=[long, short], current_round=1)
    selected = ParentSelector().select([long, short], limit=2, advisory_features=features)

    assert policy.metadata["theory"]["enabled"] is True
    assert policy.metadata["theory"]["producers"] == {"mdl": True, "boed": True, "geometry": True}
    assert set(features) == {"long", "short"}
    assert [candidate.id for candidate in selected] == ["short", "long"]
    assert {candidate.id for candidate in selected} == {"long", "short"}


def test_policy_update_preserves_theory_advisory_metadata() -> None:
    policy = EvolutionPolicy()
    _enable_project_theory_advisory_pressure(policy)

    updated = PolicyUpdater().update(policy, SearchDiagnosis(stagnation_type="healthy"), model=None)

    assert updated.metadata["theory"] == policy.metadata["theory"]
