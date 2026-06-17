from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.discovery.illumination import MapElitesIllumination, behavior_descriptor
from cognitive_evolve_runtime.discovery.operators import operator_registry
from cognitive_evolve_runtime.discovery.tension_map import TensionMap
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord
from cognitive_evolve_runtime.verification.grading import GradedOutput, VerifiedResult
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.modalities.executable import ExecutableVerifier
from cognitive_evolve_runtime.verification.modalities.formal import FormalVerifier
from cognitive_evolve_runtime.verification.synthesizer import VerificationSynthesizer


def test_grading_invariant_rejects_low_strength_verified_result() -> None:
    with pytest.raises(AssertionError):
        GradedOutput(mode="verified_result", verification_strength=VerificationStrength.DECOMPOSED, result=VerifiedResult("x", replayable=True), replay_certificate={"x": 1})


def test_synthesizer_selects_executable_for_code_problem_and_reformulates_open_problem() -> None:
    synth = VerificationSynthesizer()
    executable_plan = synth.synthesize("Write a Python function and run pytest")
    assert executable_plan.modality == "executable"
    assert executable_plan.metadata.get("diagnostics_only") is True
    assert executable_plan.strength is VerificationStrength.NONE
    open_plan = synth.synthesize("What is a good theory of this phenomenon?")
    assert open_plan.modality in {"adversarial", "decomposed"}
    assert open_plan.strength is VerificationStrength.NONE


def test_executable_verifier_runs_in_allowlisted_runner() -> None:
    candidate = CandidateGenome(id="C1", artifact="print('ok')\n")
    result = ExecutableVerifier(command=["python", "-c", "print('ok')"]).check(candidate)
    assert result.passed is True
    assert result.replayable is True


def test_formal_verifier_does_not_attempt_z3_cli_when_binding_missing_or_runs_in_process() -> None:
    result = FormalVerifier(formula=True).check(CandidateGenome(id="C1", artifact="x"))
    assert "cli_not_attempted" in result.metadata or result.replayable is True


def test_discovery_operator_registry_returns_distinct_descriptors() -> None:
    candidate = CandidateGenome(id="C1", artifact="x", concise_claim="base claim")
    descriptors = [op.propose(candidate, None, None, k=1)[0]["descriptor"][0] for op in operator_registry().values()]
    assert len(descriptors) == len(set(descriptors))


def test_map_elites_wraps_quality_diversity_and_tension_map_rules_out_region() -> None:
    candidate = CandidateGenome(id="C1", artifact="x", artifact_type="program", multihead_scores={"frontier_score": 0.8})
    illum = MapElitesIllumination()
    added = illum.add(candidate)
    descriptor = behavior_descriptor(candidate)
    assert added["candidate_id"] == "C1"
    assert descriptor
    tensions = TensionMap()
    record = EvidenceRecord(candidate_id="C1", diagnostics=["missing_required_fields: output"])
    tensions.memory.ingest(record, round_index=1)
    assert tensions.open_tensions
    tensions.mark_ruled_out(candidate_id="C1", descriptor=descriptor, evidence_ref="e1")
    assert tensions.is_ruled_out(descriptor) is True
