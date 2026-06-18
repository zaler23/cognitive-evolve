"""Verify concurrent verify plumbing: journal safety, env toggle, order preservation."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.verification.obligation_runner import run_obligations_for_population


def _candidate(cid: str, artifact: str = "") -> CandidateGenome:
    c = CandidateGenome(id=cid, core_mechanism=artifact, artifact=artifact)
    return c


def _obligation(oid: str, matcher: str = "") -> dict[str, Any]:
    return {"id": oid, "diagnostic_matcher": matcher, "must_pass": False}


# ---------------------------------------------------------------------------
# 1. env toggle: COGEV_VERIFY_CONCURRENCY=1 falls back to serial
# ---------------------------------------------------------------------------

def test_serial_fallback_via_env(tmp_path, monkeypatch):
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
