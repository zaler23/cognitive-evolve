from __future__ import annotations

import json

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.llm.fanout import run_ordered_fanout
from cognitive_evolve_runtime.llm.call_ledger import ledger_summary, record_call_state
from cognitive_evolve_runtime.llm.session import LLMSession, current_llm_session, llm_session
from cognitive_evolve_runtime.persistence.checkpoint import build_checkpoint_state
from cognitive_evolve_runtime.verification.executor import VerificationExecutor, VerificationExecutorConfig


def test_completed_unattached_call_explained_by_ledger(tmp_path, monkeypatch) -> None:
    path = tmp_path / "llm-call-ledger.jsonl"
    monkeypatch.setenv("COGEV_LLM_CALL_LEDGER", str(path))
    record_call_state("started", call_id="c22", request_type="nexus_generate_offspring", request_hash="h", round_id="22")
    record_call_state("completed", call_id="c22", request_type="nexus_generate_offspring", request_hash="h", round_id="22")
    summary = ledger_summary(path)
    assert summary["unattached_completed_count"] == 1
    assert summary["status_counts"]["completed"] == 1
    record_call_state("discarded_after_stop", call_id="c22", request_type="nexus_generate_offspring", request_hash="h", round_id="22")
    assert ledger_summary(path)["unattached_completed_count"] == 0


def test_call_ledger_reports_observed_concurrency(tmp_path, monkeypatch) -> None:
    path = tmp_path / "llm-call-ledger.jsonl"
    monkeypatch.setenv("COGEV_LLM_CALL_LEDGER", str(path))
    record_call_state("started", call_id="c1", request_type="nexus_generate_offspring", extra={"event_time": 10.0})
    record_call_state("started", call_id="c2", request_type="nexus_generate_offspring", extra={"event_time": 11.0})
    record_call_state("completed", call_id="c1", request_type="nexus_generate_offspring", extra={"event_time": 12.0})
    record_call_state("completed", call_id="c2", request_type="nexus_generate_offspring", extra={"event_time": 13.0})

    summary = ledger_summary(path)

    assert summary["completed_interval_count"] == 2
    assert summary["max_observed_concurrent_calls"] == 2


def test_session_call_ledger_path_overrides_env_and_propagates_through_fanout(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / "env-ledger.jsonl"
    session_path = tmp_path / "session-ledger.jsonl"
    monkeypatch.setenv("COGEV_LLM_CALL_LEDGER", str(env_path))

    with llm_session(LLMSession(call_ledger_path=str(session_path))):
        def _worker(index: int) -> str:
            assert current_llm_session().call_ledger_path == str(session_path)
            record_call_state("started", call_id=f"call-{index}", request_type="probe", request_hash=f"h-{index}", extra={"event_time": float(index)})
            record_call_state("completed", call_id=f"call-{index}", request_type="probe", request_hash=f"h-{index}", extra={"event_time": float(index) + 0.5, "estimated_cost_usd": 0.01})
            return str(current_llm_session().call_ledger_path)

        paths = run_ordered_fanout([1, 2], _worker, max_workers=2)

    assert paths == [str(session_path), str(session_path)]
    assert session_path.exists()
    assert not env_path.exists()
    summary = ledger_summary(session_path)
    assert summary["status_counts"]["completed"] == 2


def test_checkpoint_namespaces_llm_provider_cost_apart_from_research_extensions() -> None:
    candidate = CandidateGenome(id="C", artifact="x")
    session = LLMSession(events=[{"estimated_cost_usd": 0.125}, {"estimated_cost_usd": 0.375}])

    with llm_session(session):
        checkpoint = build_checkpoint_state(
            round=1,
            max_rounds=2,
            population=CandidatePopulation([candidate]),
            archives=ArchiveManager(),
            cost_ledger={"research_extension": {"estimated_cost_usd": 99.0}},
        )

    assert checkpoint.cost_ledger["research_extension"]["estimated_cost_usd"] == 99.0
    assert checkpoint.cost_ledger["llm_provider"]["estimated_cost_usd"] == 0.5
    assert checkpoint.cost_ledger["llm_provider"]["event_count"] == 2


def test_thin_checkpoint_roundtrip_keeps_last_three_verification_entries(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_CHECKPOINT_PROFILE", "thin")
    candidate = CandidateGenome(id="C", artifact="x", verification_trace=[{"passed": bool(i % 2), "score": i, "metadata": {"cache_key": str(i)}} for i in range(6)])
    checkpoint = build_checkpoint_state(round=1, max_rounds=3, population=CandidatePopulation([candidate]), archives=ArchiveManager(), budget_history=[{"round": i} for i in range(250)])
    restored = CandidatePopulation.from_dict(checkpoint.population)
    assert checkpoint.checkpoint_profile["name"] == "thin"
    assert len(restored.candidates[0].verification_trace) == 3
    assert len(checkpoint.budget_history) == 200


def test_verification_executor_serial_and_threaded_order() -> None:
    items = [1, 2, 3]
    serial = VerificationExecutor(VerificationExecutorConfig(mode="serial", max_workers=1)).map(lambda x: x * 2, items)
    threaded = VerificationExecutor(VerificationExecutorConfig(mode="threaded_local", max_workers=2)).map(lambda x: x * 2, items)
    assert serial == [2, 4, 6]
    assert threaded == [2, 4, 6]


def test_tech_debt_audit_has_before_and_after_sections() -> None:
    text = open("docs/V2_2_1_TECH_DEBT_AUDIT.md", encoding="utf-8").read()
    assert "## Before baseline" in text
    assert "## After validation" in text
    assert "TD-HONESTY-001" in text
