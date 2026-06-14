from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.theory import AdvisoryRankingFeatures, build_population_representation


def test_sidecar_identity_uses_stable_candidate_id_after_roundtrip() -> None:
    candidate = CandidateGenome(id="C-stable", concise_claim="claim", multihead_scores={"objective_alignment": 0.5})
    restored = CandidateGenome.from_dict(candidate.to_dict())

    representation = build_population_representation([restored], cycle_id="round:1")

    assert representation.candidates[0].candidate_id == "C-stable"


def test_advisory_features_reorder_but_do_not_change_parent_eligibility() -> None:
    weak = CandidateGenome(id="weak", current_fate="Active", multihead_scores={"objective_alignment": 0.1})
    strong = CandidateGenome(id="strong", current_fate="Active", multihead_scores={"objective_alignment": 0.9})
    dormant = CandidateGenome(id="dormant", current_fate="Dormant", multihead_scores={"objective_alignment": 1.0})
    features = {
        "weak": AdvisoryRankingFeatures(candidate_id="weak", rank_prior=1.0, plan_value=1.0, diversity=1.0),
        "dormant": AdvisoryRankingFeatures(candidate_id="dormant", rank_prior=1.0, plan_value=1.0, diversity=1.0),
    }

    selected = ParentSelector().select([strong, weak, dormant], limit=2, advisory_features=features)

    assert {candidate.id for candidate in selected} == {"strong", "weak"}
    assert "dormant" not in {candidate.id for candidate in selected}


def test_disabled_parent_selection_default_matches_no_sidecar() -> None:
    candidates = [
        CandidateGenome(id="a", current_fate="Active", multihead_scores={"objective_alignment": 0.6}),
        CandidateGenome(id="b", current_fate="Active", multihead_scores={"objective_alignment": 0.4}),
    ]
    selector = ParentSelector()

    baseline = [candidate.id for candidate in selector.select(candidates, limit=2)]
    disabled = [candidate.id for candidate in selector.select(candidates, limit=2, advisory_features=None)]

    assert disabled == baseline
