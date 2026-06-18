"""Verify concurrent verify plumbing: journal safety, env toggle, order preservation."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop import round as round_module
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack
from cognitive_evolve_runtime.tools.verification_stack import VerificationStackResult
from cognitive_evolve_runtime.verification.obligation_runner import run_obligations_for_population


def _candidate(cid: str, artifact: str = "") -> CandidateGenome:
    c = CandidateGenome(id=cid, core_mechanism=artifact, artifact=artifact)
    return c


def _obligation(oid: str, matcher: str = "") -> dict[str, Any]:
    return {"id": oid, "diagnostic_matcher": matcher, "must_pass": False}


# ---------------------------------------------------------------------------
# 1. env toggle: COGEV_VERIFY_CONCURRENCY=1 falls back to serial
# ---------------------------------------------------------------------------

def test_serial_fallback_via_env(monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "1")
    candidates = [_candidate(f"c{i}", f"text {i}") for i in range(4)]
    obligations = [_obligation("o1")]
    records = run_obligations_for_population(candidates, obligations, max_checks=4)
    assert len(records) == 4
    assert all(r["reason"] == "obligation_checked" for r in records)


# ---------------------------------------------------------------------------
# 2. concurrent mode returns same number of records as serial
# ---------------------------------------------------------------------------

def test_concurrent_same_record_count(monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "4")
    candidates = [_candidate(f"c{i}", f"text {i}") for i in range(6)]
    obligations = [_obligation("o1"), _obligation("o2")]
    records = run_obligations_for_population(candidates, obligations, max_checks=12)
    # max_checks limits per-obligation, so each obligation gets up to 12 candidates
    assert len(records) == 12
    assert all(r.get("candidate_id") for r in records)


# ---------------------------------------------------------------------------
# 3. cache is populated and keyed correctly after concurrent run
# ---------------------------------------------------------------------------

def test_cache_populated_concurrent(monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "4")
    candidates = [_candidate(f"c{i}", f"text {i}") for i in range(4)]
    obligations = [_obligation("o1")]
    cache: dict = {}
    run_obligations_for_population(candidates, obligations, cache=cache, max_checks=4)
    # Cache must have one entry per (candidate × obligation) key
    assert len(cache) == 4
    for entry in cache.values():
        assert "measured_result" in entry


def test_obligation_concurrent_preserves_candidate_order(monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "4")
    candidates = [_candidate(f"c{i}", f"text {i}") for i in range(8)]
    records = run_obligations_for_population(candidates, [_obligation("o1")], max_checks=8)
    assert [record["candidate_id"] for record in records] == [candidate.id for candidate in candidates]


def test_verifier_stack_concurrent_candidates_preserve_order(monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "3")
    stack = NexusVerifierStack()
    candidates = [_candidate(f"c{i}", f"text {i}") for i in range(6)]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _verify_candidate(candidate: CandidateGenome, **kwargs: Any) -> VerificationStackResult:
        nonlocal active, max_active
        assert isinstance(kwargs.get("existing_formal_signatures"), set)
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return VerificationStackResult(candidate.id)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(stack, "verify_candidate", _verify_candidate)
    results = stack.verify_population(candidates)
    assert [result.candidate_id for result in results] == [candidate.id for candidate in candidates]
    assert max_active > 1


def test_round_three_verifier_entrypoints_respect_serial_toggle(monkeypatch):
    max_active, results = _run_round_verifier_entrypoint_probe(monkeypatch, concurrency="1")
    assert max_active == 1
    assert results == ["stack", "synthesized", "obligations"]


def test_round_three_verifier_entrypoints_can_overlap(monkeypatch):
    max_active, results = _run_round_verifier_entrypoint_probe(monkeypatch, concurrency="3")
    assert max_active > 1
    assert sorted(results) == ["obligations", "stack", "synthesized"]


# ---------------------------------------------------------------------------
# 4. journal file not corrupted under concurrent writes (each line is valid JSON)
# ---------------------------------------------------------------------------

def test_journal_not_corrupted_concurrent(tmp_path, monkeypatch):
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", "4")
    journal_path = tmp_path / "llm-calls.jsonl"

    # Simulate concurrent appends using the same lock the journal uses
    from cognitive_evolve_runtime.llm.journal import _JOURNAL_LOCK  # noqa: PLC0415

    errors: list[str] = []

    def _append(i: int) -> None:
        line = json.dumps({"i": i, "status": "ok"})
        with _JOURNAL_LOCK:
            with journal_path.open("a") as f:
                f.write(line + "\n")

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = journal_path.read_text().splitlines()
    assert len(lines) == 20
    for line in lines:
        json.loads(line)  # must not raise — each line is valid JSON


def _run_round_verifier_entrypoint_probe(monkeypatch: pytest.MonkeyPatch, *, concurrency: str) -> tuple[int, list[Any]]:
    monkeypatch.setenv("COGEV_VERIFY_CONCURRENCY", concurrency)
    monkeypatch.setattr(round_module, "ingest_latent_feedback", lambda **_: None)
    monkeypatch.setattr(round_module, "ingest_runtime_trial_feedback", lambda **_: None)
    round_runner = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=3))
    population = CandidatePopulation(candidates=[_candidate("c0", "text")])
    archives = ArchiveManager()
    policy = EvolutionPolicy()
    contract = NexusObjectiveContract(original_user_goal="probe", normalized_goal="probe")
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(round_runner.critique_engine, "critique", lambda **_: [])
    monkeypatch.setattr(round_runner.critique_engine, "apply", lambda **_: None)

    def _record(name: str) -> list[str]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return [name]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(round_runner.verifier_stack, "verify_population", lambda *_, **__: _record("stack"))
    monkeypatch.setattr(round_runner, "_run_synthesized_verifier", lambda *_, **__: _record("synthesized"))
    monkeypatch.setattr(round_runner, "_run_verification_obligations", lambda *_, **__: _record("obligations"))
    _, results = round_runner.critique_and_verify(
        current_round=0,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
    )
    return max_active, results
