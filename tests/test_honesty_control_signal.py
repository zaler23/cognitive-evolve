from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis
from cognitive_evolve_runtime.nexus.honesty_control import compute_honesty_control_signal
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.v23_theory_config import HonestyControlConfig
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult


def _candidate_with_honesty(cid: str, *, exogeneity: float, variety: float, falsification: float, replay: float) -> CandidateGenome:
    result = VerificationResult(
        passed=True,
        score=1.0,
        strength=VerificationStrength.EXECUTABLE,
        replayable=True,
        metadata={
            "measured_strength": VerificationStrength.EXECUTABLE.name,
            "measured_strength_value": int(VerificationStrength.EXECUTABLE),
            "honesty_measurements": {
                "exogeneity_score": exogeneity,
                "variety_score": variety,
                "falsification_score": falsification,
                "replay_score": replay,
            },
        },
    )
    return CandidateGenome(id=cid, artifact=cid, verification_result=result.to_dict())


def test_honesty_control_neutral_when_data_insufficient() -> None:
    signal = compute_honesty_control_signal(candidates=[], config=HonestyControlConfig())

    assert signal.sample_count == 0
    assert signal.pressure["frontier_exploration_pressure"] == 0.0
    assert signal.error_vector["exogeneity"] == 0.0
    assert signal.diagnostics


def test_exogeneity_and_falsification_errors_raise_adversarial_pressure() -> None:
    signal = compute_honesty_control_signal(
        candidates=[_candidate_with_honesty("C", exogeneity=0.0, variety=1.0, falsification=0.0, replay=1.0)],
        config=HonestyControlConfig(),
    )

    assert signal.error_vector["exogeneity"] == 1.0
    assert signal.error_vector["falsification"] == 1.0
    assert signal.pressure["adversarial_budget_pressure"] > 0.0


def test_low_variety_raises_rarity_edge_and_frontier_pressure() -> None:
    signal = compute_honesty_control_signal(
        candidates=[_candidate_with_honesty("C", exogeneity=1.0, variety=0.0, falsification=1.0, replay=1.0)],
        config=HonestyControlConfig(),
    )

    assert signal.pressure["rarity_budget_pressure"] > 0.0
    assert signal.pressure["edge_seed_pressure"] > 0.0
    assert signal.pressure["frontier_exploration_pressure"] > 0.0


def test_policy_updater_converts_honesty_signal_to_bounded_policy_pressure() -> None:
    signal = compute_honesty_control_signal(
        candidates=[_candidate_with_honesty("C", exogeneity=1.0, variety=0.0, falsification=1.0, replay=0.0)],
        config=HonestyControlConfig(),
    )
    diagnosis = SearchDiagnosis(metadata={"honesty_control": signal.to_dict()})
    updated = PolicyUpdater().update(EvolutionPolicy(), diagnosis)

    pressure = updated.metadata["honesty_control"]["pressure"]
    assert 0.0 < pressure["frontier_exploration_pressure"] <= 1.0
    assert 0.0 < pressure["replay_verifier_pressure"] <= 1.0
    assert updated.metadata["honesty_control"]["effect"] == "search_pressure_only_verification_strength_unchanged"
