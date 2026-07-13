from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.v23_theory_config import MinimaxBudgetConfig
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.minimax_budget import _allocate_adversarial_budget
from cognitive_evolve_runtime.verification.obligation_runner import run_obligations_for_population
from cognitive_evolve_runtime.verification.regime import compile_grounding_regime
from cognitive_evolve_runtime.verification.types import VerificationResult


def _verified_candidate(cid: str, strength: VerificationStrength) -> CandidateGenome:
    result = VerificationResult(
        passed=True,
        score=1.0,
        strength=strength,
        replayable=True,
        metadata={
            "measured_strength": strength.name,
            "measured_strength_value": int(strength),
            "honesty_measurements": {
                "exogeneity_score": 1.0,
                "variety_score": 1.0,
                "falsification_score": 1.0,
                "replay_score": 1.0,
            },
        },
    )
    return CandidateGenome(id=cid, artifact=cid, concise_claim=cid, core_mechanism=cid, verification_result=result.to_dict())


def test_minimax_budget_stronger_candidate_gets_at_least_weak_budget_and_sum_conserved() -> None:
    weak = CandidateGenome(id="W", artifact="weak")
    strong = _verified_candidate("S", VerificationStrength.FORMAL)
    allocation = _allocate_adversarial_budget([weak, strong], base_budget=2, config=MinimaxBudgetConfig(min_budget_per_candidate=1))

    assert allocation["S"] >= allocation["W"]
    assert sum(allocation.values()) == 4
    assert min(allocation.values()) >= 1


def test_minimax_budget_all_none_is_uniform_and_nonzero() -> None:
    candidates = [CandidateGenome(id=f"C{i}", artifact=str(i)) for i in range(3)]
    allocation = _allocate_adversarial_budget(candidates, base_budget=2, config=MinimaxBudgetConfig(min_budget_per_candidate=1))

    assert set(allocation.values()) == {2}
    assert sum(allocation.values()) == 6


def test_minimax_budget_total_override_is_conserved() -> None:
    weak = _verified_candidate("W", VerificationStrength.ADVERSARIAL)
    strong = _verified_candidate("S", VerificationStrength.EXECUTABLE)
    allocation = _allocate_adversarial_budget([weak, strong], base_budget=2, config=MinimaxBudgetConfig(min_budget_per_candidate=1), total_override=7)

    assert allocation["S"] >= allocation["W"]
    assert sum(allocation.values()) == 7


def test_grounding_regime_override_budget_preempts_obligation_budget() -> None:
    regime = compile_grounding_regime(
        candidate=CandidateGenome(id="C", artifact="x"),
        verifier_fingerprint="vf",
        raw_obligation={"falsification_budget": {"count": 2}, "adversarial_budget": {"count": 3}},
        oracle_kind="toolrunner",
        override_adversarial_budget=9,
    )

    assert regime.adversarial_budget == 9


def test_obligation_cache_key_distinguishes_actual_adversarial_budget() -> None:
    candidate = CandidateGenome(id="C", artifact="safe artifact")
    cache: dict[str, dict] = {}
    run_obligations_for_population([candidate], [{"id": "o", "diagnostic_matcher": "bad", "adversarial_budget": {"count": 1}}], cache=cache, max_checks=1)
    run_obligations_for_population([candidate], [{"id": "o", "diagnostic_matcher": "bad", "adversarial_budget": {"count": 3}}], cache=cache, max_checks=1)

    assert len(cache) == 2
    budgets = {entry["grounding_regime"]["adversarial_budget"] for entry in cache.values()}
    assert budgets == {1, 3}
