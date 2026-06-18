from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.verification.cache import check_with_cache, verification_cache_key
from cognitive_evolve_runtime.verification.honesty_core import GroundingRegime, ProbeCase, measure_honesty
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.obligation_runner import run_obligations_for_population
from cognitive_evolve_runtime.verification.probe_executor import execute_probes
from cognitive_evolve_runtime.verification.replay_runner import build_replay_record
from cognitive_evolve_runtime.verification.strength import measured_strength_from_result
from cognitive_evolve_runtime.verification.types import VerificationResult


class _Verifier:
    verifier_id = "executable-verifier"
    fingerprint = "vf-exec"
    plan = {"probe_requirements": [{"kind": "counterfactual", "expected_verdict_flip": False}], "falsification_budget": {"count": 4}}

    def check(self, candidate):
        return VerificationResult(
            passed=True,
            score=1.0,
            strength=VerificationStrength.EXECUTABLE,
            evidence_ref="ev",
            replayable=True,
            metadata={"oracle_kind": "executable"},
        )


def test_known_bad_probe_zeroes_exogeneity() -> None:
    result = VerificationResult(passed=True, replayable=False, metadata={"oracle_kind": "diagnostic_matcher"})
    regime = GroundingRegime(
        regime_id="r",
        probes=[ProbeCase("p", "engine", "engine", False)],
        adversarial_budget=1,
        isolation_enforced=True,
        replay_artifact_hash="a",
        verifier_fingerprint="v",
        oracle_kind="diagnostic_matcher",
    )
    observed = execute_probes(result, regime, raw_obligation={"known_bad_probe": True, "diagnostic_matcher": "x"})
    measurements = measure_honesty(result, regime, observed, {"frozen_artifact_hash": "a", "verifier_fingerprint": "v", "replay_verified": False})
    assert measurements.exogeneity_score == 0.0


def test_text_candidate_capped_at_adversarial_without_oracle() -> None:
    candidate = CandidateGenome(id="T", artifact="plain prose", concise_claim="plain prose")
    records = run_obligations_for_population(
        [candidate],
        [{"id": "obl", "verifier_fingerprint": "vf-text", "must_pass": False, "diagnostic_matcher": "absent", "falsification_budget": {"count": 4}}],
        cache={},
    )
    result = VerificationResult.from_dict(records[0]["verification_result"])
    assert measured_strength_from_result(result) <= VerificationStrength.ADVERSARIAL
    assert result.replayable is False


def test_legacy_cache_without_replay_record_is_rerun_and_marked_diagnostics_only() -> None:
    candidate = CandidateGenome(id="C", artifact="def f(x):\n    return x + 1\n", metadata={"verification_command": "pytest"})
    legacy_key = verification_cache_key(candidate, _Verifier().fingerprint)
    legacy_cache = {legacy_key: {"strength": "FORMAL", "passed": True, "replayable": True}}
    result, _key, cache_hit = check_with_cache(candidate, _Verifier(), legacy_cache)
    assert cache_hit is False
    assert result.metadata.get("honesty_measurements")
    assert measured_strength_from_result(result) >= VerificationStrength.FORMAL
    # The old entry remains diagnostics only; the certified result is newly measured.
    assert legacy_cache[legacy_key]["diagnostics_only"] is True


def test_replay_runner_requires_replayable_oracle() -> None:
    candidate = CandidateGenome(id="R", artifact="text")
    text_result = VerificationResult(passed=True, replayable=True, metadata={"oracle_kind": "diagnostic_matcher"})
    record = build_replay_record(candidate, text_result, verifier_fingerprint="vf", oracle_kind="diagnostic_matcher")
    assert record["replay_verified"] is False
    exec_result = VerificationResult(passed=True, replayable=True, metadata={"oracle_kind": "executable"})
    assert build_replay_record(candidate, exec_result, verifier_fingerprint="vf", oracle_kind="executable")["replay_verified"] is True
