from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.mock_provider import MockProviderResponse
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.llm.session import (
    LLMSession,
    current_logical_llm_call,
    llm_session,
    logical_llm_call,
)
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _slot_sampling_policy
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def _configure_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)


class _CaptureProvider:
    provider_id = "capture"

    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self.barrier = barrier
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def complete_json(self, **kwargs: Any) -> LLMProviderResult:
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        with self.lock:
            self.calls.append(dict(kwargs))
        return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.0)


def test_call_local_sampling_reaches_provider_journal_telemetry_and_replay_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    provider = _CaptureProvider()
    session = LLMSession(run_id="run-sampling", journal_dir=str(tmp_path))
    base_policy = LLMRequestPolicy(structured_prompt=True)
    profiles = [
        LLMRequestPolicy(temperature=0.9, top_p=0.8, seed=11),
        LLMRequestPolicy(temperature=0.4, top_p=0.7, seed=22),
        LLMRequestPolicy(temperature=0.9, top_p=0.8, seed=11),
    ]

    with llm_session(session):
        for profile in profiles:
            with logical_llm_call("round-1/slot-1", template_version="offspring/v1", request_policy=profile):
                response = llm_json(
                    "sampling_test",
                    {"same": "request"},
                    system="Return JSON",
                    schema_hint={},
                    provider=provider,
                    request_policy=base_policy,
                )
                assert response["ok"] is True

    assert [(call["temperature"], call["top_p"], call["seed"]) for call in provider.calls] == [
        (0.9, 0.8, 11),
        (0.4, 0.7, 22),
    ]
    rows = [json.loads(line) for line in (tmp_path / "llm-calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["temperature"] for row in rows} == {0.4, 0.9}
    assert {row["top_p"] for row in rows} == {0.7, 0.8}
    assert {row["seed"] for row in rows} == {11, 22}
    assert len({row["request_hash"] for row in rows}) == 2
    assert session.snapshot()[-1]["sampling"] == {"temperature": 0.9, "top_p": 0.8, "seed": 11}
    assert session.snapshot()[-1]["cache_replayed"] is True


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            LLMRequestPolicy(temperature=0.2, top_p=0.8, seed=11),
            LLMRequestPolicy(temperature=0.3, top_p=0.8, seed=11),
        ),
        (
            LLMRequestPolicy(temperature=0.2, top_p=0.8, seed=11),
            LLMRequestPolicy(temperature=0.2, top_p=0.9, seed=11),
        ),
        (
            LLMRequestPolicy(temperature=0.2, top_p=0.8, seed=11),
            LLMRequestPolicy(temperature=0.2, top_p=0.8, seed=12),
        ),
    ],
)
def test_each_resolved_sampling_field_changes_replay_signature(
    first: LLMRequestPolicy,
    second: LLMRequestPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    provider = _CaptureProvider()
    with llm_session(LLMSession(run_id="sampling-signature", journal_dir=str(tmp_path))):
        for profile in (first, second):
            with logical_llm_call("round-1/slot-1", request_policy=profile):
                llm_json("sampling_signature", {}, system="Return JSON", schema_hint={}, provider=provider)

    assert len(provider.calls) == 2


def test_rank_replay_signature_tracks_candidate_order_round_and_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    provider = _CaptureProvider()
    session = LLMSession(run_id="rank-run", journal_dir=str(tmp_path))
    calls = [
        ("rank-run/round-1/relative-rank/logical-pass", "rank/v1", ["A", "B"]),
        ("rank-run/round-1/relative-rank/logical-pass", "rank/v1", ["A", "B"]),
        ("rank-run/round-1/relative-rank/logical-pass", "rank/v1", ["B", "A"]),
        ("rank-run/round-2/relative-rank/logical-pass", "rank/v1", ["A", "B"]),
        ("rank-run/round-2/relative-rank/logical-pass", "rank/v2", ["A", "B"]),
    ]
    with llm_session(session):
        for logical_id, template, candidate_ids in calls:
            with logical_llm_call(logical_id, template_version=template):
                llm_json(
                    "nexus_relative_rank",
                    {"candidate_ids": candidate_ids},
                    system="Return JSON",
                    schema_hint={},
                    provider=provider,
                )

    assert len(provider.calls) == 4


class _World:
    kind = "text"


class _Contract:
    objective = "sampling fanout"

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective}


class _TransportCallingOffspringModel:
    def __init__(self, provider: _CaptureProvider) -> None:
        self.provider = provider

    def generate_offspring(
        self,
        *,
        plans: list[MutationPlan],
        parents: list[CandidateGenome],
        world: Any,
        contract: Any,
        policy: EvolutionPolicy,
        provided_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del world, contract, policy, provided_context
        slot_id = str(plans[0].metadata["branch_slots"][0]["slot_id"])
        llm_json(
            "slot_sampling_test",
            {"slot_id": slot_id},
            system="Return JSON",
            schema_hint={},
            provider=self.provider,
            request_policy=LLMRequestPolicy(structured_prompt=True),
        )
        return [
            {
                "id": f"child-{slot_id}",
                "parent_ids": [parents[0].id],
                "artifact": f"artifact-{slot_id}",
                "concise_claim": slot_id,
                "core_mechanism": f"mechanism-{slot_id}",
            }
        ]


def test_slot_sampling_profiles_are_context_local_across_parallel_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "2")
    provider = _CaptureProvider(threading.Barrier(2))
    parents = [
        CandidateGenome(id=f"P{index}", artifact=f"parent-{index}", concise_claim="claim", core_mechanism="mechanism")
        for index in range(2)
    ]
    slots = [
        {
            "slot_id": f"slot-{index}",
            "arm_id": parent.id,
            "parent_id": parent.id,
            "intent": "explore_fresh",
            "variation_index": index,
        }
        for index, parent in enumerate(parents)
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id for parent in parents],
        metadata={"plan_id": "runtime-plan", "plan_source": "runtime_lineage_envelope", "branch_slots": slots},
    )
    policy = EvolutionPolicy(
        metadata={
            "search_phase": "explore",
            "slot_sampling_profiles": {
                "explore": {
                    "explore_fresh": [
                        {"temperature": 0.9, "top_p": 0.8, "seed": 11},
                        {"temperature": 0.4, "top_p": 0.7, "seed": 22},
                    ]
                },
                "exit_sweep": {
                    "explore_fresh": [
                        {"temperature": 0.1, "top_p": 0.4, "seed": 33},
                    ]
                },
            }
        }
    )

    with llm_session(LLMSession(run_id="run-fanout", journal_dir=str(tmp_path))):
        offspring = _generate_offspring(
            model=_TransportCallingOffspringModel(provider),
            mutation_engine=MutationEngine(),
            parents=parents,
            plans=[plan],
            world=_World(),
            contract=_Contract(),
            policy=policy,
            target_size=2,
        )
        assert current_logical_llm_call() is None

    assert {call["seed"]: (call["temperature"], call["top_p"]) for call in provider.calls} == {
        11: (0.9, 0.8),
        22: (0.4, 0.7),
    }
    assert [candidate.metadata["branch_slot_id"] for candidate in offspring] == ["slot-0", "slot-1"]


def test_phase_compiled_sampling_is_journaled_and_changes_replay_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    provider = _CaptureProvider()
    profiles = {
        "explore": {
            "explore_fresh": [{"temperature": 0.9, "top_p": 0.95, "seed": 101}],
        },
        "exit_sweep": {
            "explore_fresh": [{"temperature": 0.1, "top_p": 0.4, "seed": 201}],
        },
    }
    slot = {"slot_id": "slot-phase", "intent": "explore_fresh", "variation_index": 0}

    with llm_session(LLMSession(run_id="phase-sampling", journal_dir=str(tmp_path))):
        for phase in ("explore", "exit_sweep"):
            policy = EvolutionPolicy(metadata={"search_phase": phase, "slot_sampling_profiles": profiles})
            sampling_policy = _slot_sampling_policy(policy, slot)
            assert sampling_policy is not None
            with logical_llm_call("round/slot-phase", request_policy=sampling_policy):
                llm_json("phase_sampling", {"same": "payload"}, system="Return JSON", schema_hint={}, provider=provider)

    assert [(call["temperature"], call["top_p"], call["seed"]) for call in provider.calls] == [
        (0.9, 0.95, 101),
        (0.1, 0.4, 201),
    ]
    rows = [json.loads(line) for line in (tmp_path / "llm-calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["search_phase"] for row in rows} == {"explore", "exit_sweep"}
    assert {row["sampling_profile_id"] for row in rows} == {
        "explore:explore_fresh:0",
        "exit_sweep:explore_fresh:0",
    }
    assert len({row["request_hash"] for row in rows}) == 2
    assert len(list((tmp_path / "llm-responses" / "v1").glob("*.json"))) == 2


def test_grounded_emitter_can_select_an_available_slot_sampling_profile() -> None:
    profiles = {
        "explore": {
            "default": [
                {"temperature": 0.9, "top_p": 0.95, "seed": 101},
                {"temperature": 0.2, "top_p": 0.5, "seed": 202},
            ],
        }
    }
    policy = EvolutionPolicy(metadata={"search_phase": "explore", "slot_sampling_profiles": profiles})
    slot = {
        "slot_id": "slot-replay-profile",
        "intent": "standard_variation",
        "variation_index": 0,
        "directive": {
            "move_replay": {
                "preferred_emitter": {
                    "sampling_profile": "explore:default:1",
                }
            }
        },
    }

    sampling = _slot_sampling_policy(policy, slot)

    assert sampling is not None
    assert sampling.sampling_profile_id == "explore:default:1"
    assert (sampling.temperature, sampling.top_p, sampling.seed) == (0.2, 0.5, 202)


def test_phase_and_profile_identity_are_part_of_replay_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch)
    provider = _CaptureProvider()
    policies = [
        LLMRequestPolicy(temperature=0.4, top_p=0.7, seed=11, search_phase="explore", sampling_profile_id="explore:default:0"),
        LLMRequestPolicy(temperature=0.4, top_p=0.7, seed=11, search_phase="exit_sweep", sampling_profile_id="exit_sweep:default:0"),
        LLMRequestPolicy(temperature=0.4, top_p=0.7, seed=11, search_phase="exit_sweep", sampling_profile_id="exit_sweep:default:1"),
    ]

    with llm_session(LLMSession(run_id="phase-profile-signature", journal_dir=str(tmp_path))):
        for policy in policies:
            with logical_llm_call("round/slot-same", request_policy=policy):
                llm_json("phase_profile_signature", {"same": "payload"}, system="Return JSON", schema_hint={}, provider=provider)

    assert len(provider.calls) == 3
    assert len(list((tmp_path / "llm-responses" / "v1").glob("*.json"))) == 3


def test_relative_rank_binds_run_round_logical_pass_and_template_version() -> None:
    seen: list[tuple[str, str] | None] = []

    class _Rater:
        def rank(self, **_: Any) -> RelativeRankingResult:
            context = current_logical_llm_call()
            seen.append(context[:2] if context is not None else None)
            return RelativeRankingResult(
                best_final_answer_id="C",
                strongest_mechanism_id="C",
                mutation_worthy_ids=["C"],
                multihead_observations={"C": {}},
            )

    candidate = CandidateGenome(id="C", artifact="candidate", concise_claim="candidate", core_mechanism="mechanism")
    stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=1))
    stage.rater = _Rater()
    with llm_session(LLMSession(run_id="rank-run")):
        stage.rank(
            population=CandidatePopulation([candidate]),
            archives=ArchiveManager(),
            policy=EvolutionPolicy(),
            contract=NexusObjectiveContract(original_user_goal="rank", normalized_goal="rank"),
            current_round=2,
        )

    assert seen == [("rank-run/round-2/relative-rank/logical-pass", "nexus-relative-rank/v1")]
