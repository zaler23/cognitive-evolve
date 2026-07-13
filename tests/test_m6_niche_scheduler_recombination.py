from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evolution.cross_niche import (
    cross_niche_recombine,
    requires_reseal,
    translate_candidate_to_niche,
)
from cognitive_evolve_runtime.evolution.niche_scheduler import CostAwareUCBScheduler
from cognitive_evolve_runtime.evolution.niches import SpeciesIndex, is_niche_isolated


def _candidate(candidate_id: str, niche: str, *, species_hint: str = "route") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact=f"artifact:{candidate_id}",
        artifact_type="answer",
        concise_claim=f"{niche}:{species_hint}",
        core_mechanism=species_hint,
        novelty_descriptors=[species_hint],
        niche_memberships=[niche],
        metadata={
            "niche_id": niche,
            "species_hint": species_hint,
            "improvement_certificate": {"status": "verified", "verified": True},
            "e_wealth": 12.0,
        },
        verification_result={
            "improvement_certificate": {"status": "verified", "verified": True},
            "e_wealth": 12.0,
        },
        evidence_delta={"verified": ["old-seal"], "added": ["raw:1"]},
        verification_trace=[{"status": "verified", "certificate": "old"}],
    )


def test_species_index_assigns_with_niche_isolation() -> None:
    index = SpeciesIndex()
    alpha = _candidate("A", "alpha", species_hint="same-route")
    beta = _candidate("B", "beta", species_hint="same-route")

    alpha_assignment = index.assign(alpha)
    beta_assignment = index.assign(beta)

    assert alpha_assignment.niche_id == "alpha"
    assert beta_assignment.niche_id == "beta"
    assert alpha_assignment.species_id != beta_assignment.species_id
    assert alpha.metadata["species_id"] == alpha_assignment.species_id
    assert alpha.niche_memberships == ["alpha"]
    assert is_niche_isolated(alpha, "alpha") is True
    assert is_niche_isolated(CandidateGenome(id="leaky", niche_memberships=["alpha", "beta"]), "alpha") is False


def test_cost_aware_ucb_prefers_lower_cost_and_rewards_only_verified_closure() -> None:
    scheduler = CostAwareUCBScheduler(exploration=1.0)
    scheduler.register_niche("expensive", estimated_cost=10.0)
    scheduler.register_niche("cheap", estimated_cost=1.0)

    assert scheduler.select().niche_id == "cheap"
    assert scheduler.record_trial("cheap", closure_certificate={"objective_solved": True}, reward=1.0, cost=1.0) is False
    assert scheduler.arms["cheap"].reward_sum == 0.0
    assert scheduler.arms["cheap"].rejected_rewards == 1

    assert scheduler.record_trial("cheap", closure_certificate={"verified": True, "critical_failures": []}, reward=0.8, cost=1.0) is True
    assert scheduler.arms["cheap"].reward_sum == 0.8
    assert scheduler.arms["cheap"].verified_closures == 1


def test_cross_niche_recombination_clears_verified_certificate_and_e_wealth() -> None:
    alpha = _candidate("A", "alpha")
    beta = _candidate("B", "beta")

    child = cross_niche_recombine(alpha, beta, target_niche_id="hybrid")

    assert child.niche_memberships == ["hybrid"]
    assert child.metadata["requires_reseal"] is True
    assert child.metadata["seal_status"] == "unsealed"
    assert requires_reseal(child) is True
    assert "improvement_certificate" not in child.metadata
    assert "e_wealth" not in child.metadata
    assert "improvement_certificate" not in child.verification_result
    assert "e_wealth" not in child.verification_result
    assert child.evidence_delta.get("verified") is None
    assert child.verification_trace == []


def test_translation_to_new_niche_requires_reseal_and_fresh_species_assignment() -> None:
    alpha = _candidate("A", "alpha", species_hint="portable")

    translated = translate_candidate_to_niche(alpha, "beta")

    assert translated.id != alpha.id
    assert translated.parent_ids == ["A"]
    assert translated.niche_memberships == ["beta"]
    assert translated.metadata["requires_reseal"] is True
    assert translated.metadata["niche_runtime"]["reseal_reason"] == "niche_translation_requires_fresh_verified_closure"
    assert translated.metadata["species_id"].startswith("sp:beta:")
    assert "improvement_certificate" not in translated.metadata
    assert "e_wealth" not in translated.verification_result
