from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.llm.mock_provider import MockProviderResponse
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.session import LLMSession, _LAST_RETRY_HISTORY, llm_round, llm_session, logical_llm_call
from cognitive_evolve_runtime.llm.telemetry import build_round_cost_ledger
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopResult
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRunResult
from cognitive_evolve_runtime.nexus.runtime_services import NexusPersistenceService
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult


_TOTAL_FIELDS = (
    "logical_calls",
    "physical_calls",
    "remote_attempts",
    "cache_hits",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "unpriced_remote_attempts",
    "estimated_cost_usd",
)


def _assert_totals_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    for key in _TOTAL_FIELDS:
        assert actual[key] == pytest.approx(expected[key])


def _assert_hierarchy_conserves(round_record: dict[str, object]) -> None:
    slots = [*round_record["slots"], round_record["unattributed"]]
    slot_sum = {key: sum(slot["totals"][key] for slot in slots) for key in _TOTAL_FIELDS}
    _assert_totals_equal(slot_sum, round_record["totals"])
    for slot in slots:
        candidates = [*slot["candidates"], slot["unattributed"]]
        candidate_sum = {key: sum(candidate["totals"][key] for candidate in candidates) for key in _TOTAL_FIELDS}
        _assert_totals_equal(candidate_sum, slot["totals"])


class _SuccessProvider:
    provider_id = "fake"

    def __init__(self, payload: dict[str, object], *, attempts: int, cost: float, tokens: int) -> None:
        self.payload = payload
        self.attempts = attempts
        self.cost = cost
        self.tokens = tokens
        self.calls = 0

    def complete_json(self, **_: object) -> LLMProviderResult:
        self.calls += 1
        response = MockProviderResponse(self.payload)
        response.usage = {
            "prompt_tokens": self.tokens - 2,
            "completion_tokens": 2,
            "total_tokens": self.tokens,
        }
        return LLMProviderResult(response=response, attempts=self.attempts, estimated_cost_usd=self.cost)


class _FailingProvider:
    provider_id = "fake"

    def complete_json(self, **_: object) -> LLMProviderResult:
        _LAST_RETRY_HISTORY.set(
            [{"attempt": 1, "category": "network_or_transient", "retryable": True, "final": True}]
        )
        raise RuntimeError("synthetic partial slot failure")


def _slot_payload(slot_id: str, *, intervention_ref: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "plans": [{"metadata": {"branch_slots": [{"slot_id": slot_id, "parent_id": "parent"}]}}],
        "requested_candidate_count": 1,
    }
    if intervention_ref is not None:
        payload["intervention_ref"] = intervention_ref
    return payload


def _child(candidate_id: str, slot_id: str) -> dict[str, object]:
    return {
        "offspring": [
            {
                "id": candidate_id,
                "artifact": f"artifact-{candidate_id}",
                "metadata": {"branch_slot_id": slot_id},
            }
        ]
    }


def test_fake_transport_trace_conserves_costs_and_separates_retry_cache_and_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "fake/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "3")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)
    slot_1 = _SuccessProvider(_child("child-1", "slot-1"), attempts=2, cost=0.3, tokens=30)
    slot_2 = _SuccessProvider(_child("child-2", "slot-2"), attempts=1, cost=0.2, tokens=20)
    session = LLMSession(run_id="run-a", response_dir=str(tmp_path))

    with llm_session(session), llm_round(1):
        with logical_llm_call("run-a/round-1/slot-1"):
            llm_json(
                "nexus_generate_offspring",
                _slot_payload("slot-1", intervention_ref="intervention-7"),
                system="Return JSON",
                schema_hint={},
                provider=slot_1,
            )
        with logical_llm_call("run-a/round-1/slot-1"):
            llm_json(
                "nexus_generate_offspring",
                _slot_payload("slot-1", intervention_ref="intervention-7"),
                system="Return JSON",
                schema_hint={},
                provider=slot_1,
            )
        with logical_llm_call("run-a/round-1/slot-2"):
            llm_json(
                "nexus_generate_offspring",
                _slot_payload("slot-2"),
                system="Return JSON",
                schema_hint={},
                provider=slot_2,
            )
        with logical_llm_call("run-a/round-1/slot-3"), pytest.raises(Exception, match="partial slot failure"):
            llm_json(
                "nexus_generate_offspring",
                _slot_payload("slot-3"),
                system="Return JSON",
                schema_hint={},
                provider=_FailingProvider(),
            )

    history = [
        {
            "round": 1,
            "generation_plan": {
                "offspring_ids": ["child-1", "child-2", "child-2"],
                "offspring_harvest": {
                    "slot_errors": ["slot-3: synthetic partial slot failure"],
                    "reservoir_truncated_count": 1,
                },
            },
            "offspring_verification": [
                {"candidate_id": "child-1", "passed": True},
                {"candidate_id": "child-2", "passed": False},
            ],
            "evaluator_qualified_survivors": 1,
        }
    ]
    ledger = build_round_cost_ledger(session.snapshot(), budget_history=history)
    round_record = ledger["rounds"][0]

    assert slot_1.calls == 1
    assert slot_2.calls == 1
    assert round_record["round"] == 1
    assert round_record["totals"] == {
        "logical_calls": 3,
        "physical_calls": 3,
        "remote_attempts": 4,
        "cache_hits": 1,
        "prompt_tokens": 46,
        "completion_tokens": 4,
        "total_tokens": 50,
        "unpriced_remote_attempts": 1,
        "estimated_cost_usd": pytest.approx(0.5),
    }
    assert round_record["observations"] == {
        "physical_calls": 3,
        "unique_valid_children": 2,
        "evaluator_qualified_survivors": 1,
        "truncation_count": 1,
        "partial_failure_blast_radius": 1,
    }
    assert next(
        candidate
        for slot in round_record["slots"]
        for candidate in slot["candidates"]
        if candidate["candidate_id"] == "child-1"
    )["intervention_ref"] == "intervention-7"
    _assert_hierarchy_conserves(round_record)

    replay_session = LLMSession(run_id="run-a", response_dir=str(tmp_path))
    with llm_session(replay_session), llm_round(1), logical_llm_call("run-a/round-1/slot-1"):
        llm_json(
            "nexus_generate_offspring",
            _slot_payload("slot-1", intervention_ref="intervention-7"),
            system="Return JSON",
            schema_hint={},
            provider=slot_1,
        )
    resumed = build_round_cost_ledger(
        replay_session.snapshot(),
        budget_history=history,
        existing_ledger=ledger,
    )
    assert replay_session.total_estimated_cost_usd() == 0.0
    assert slot_1.calls == 1
    assert resumed["rounds"][0]["totals"] == round_record["totals"]


def test_cost_ledger_uses_existing_events_checkpoint_and_resume_idempotency(tmp_path: Path) -> None:
    history = [
        {
            "round": 1,
            "generation_plan": {"offspring_ids": ["child-1"], "offspring_harvest": {}},
            "offspring_verification": [{"candidate_id": "child-1", "passed": True}],
            "evaluator_qualified_survivors": 1,
        }
    ]
    ledger = build_round_cost_ledger(
        [
            {
                "round_id": "1",
                "logical_call_id": "run/round-1/slot-1",
                "physical_call_id": "physical-1",
                "attempts": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "estimated_cost_usd": 0.05,
                "slot_ids": ["slot-1"],
                "candidate_bindings": [{"candidate_id": "child-1", "slot_id": "slot-1"}],
            }
        ],
        budget_history=history,
    )
    result = EvolutionLoopResult(
        population=CandidatePopulation(),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        diagnosis=SearchDiagnosis(),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate output"),
        progress_events=[{"type": "evolution_progress", "round": 1, "max_rounds": 1}],
        budget_history=history,
        current_round=1,
        max_rounds=1,
        stop_reason="max_rounds",
        completion_status="completed",
        cost_ledger=ledger,
    )
    run = NexusRunResult(mode="text", contract={}, policy={}, world={}, evolution=result.to_dict())
    service = NexusPersistenceService(output_dir=tmp_path)

    for _ in range(2):
        service.persist(
            run,
            result,
            contract={},
            world={},
            budget_history=history,
            budget=EvolutionBudget(max_rounds=1, current_round=1),
        )

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))

    assert len([event for event in events if event.get("type") == "round_cost_ledger" and event.get("round") == 1]) == 1
    assert checkpoint["cost_ledger"]["schema_version"] == "round-cost-ledger/v1"
    assert checkpoint["cost_ledger"]["rounds"][0]["observations"]["physical_calls"] == 1
    assert result.budget_history[0]["cost_ledger"]["totals"]["estimated_cost_usd"] == pytest.approx(0.05)
