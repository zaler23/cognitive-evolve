from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.llm import transport as transport_module
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.artifact_contract import DynamicArtifactContract
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view, candidate_prompt_view


def _large_candidate(index: int, blob: str) -> CandidateGenome:
    return CandidateGenome(
        id=f"C{index}",
        generation=index % 5,
        artifact=f"artifact-{index}-" + blob,
        concise_claim=f"candidate {index}",
        core_mechanism=f"mechanism {index % 4}",
        failure_lessons=["lesson " + blob for _ in range(3)],
        inherited_genes=["gene " + blob for _ in range(2)],
        verification_trace=[{"status": "failed", "diagnostics": blob, "raw_output_ref": blob}],
        tool_results=[{"tool_id": "t", "status": "ok", "raw_output_ref": blob}],
        mutation_history=["mutate " + blob for _ in range(4)],
        edge_knowledge_seeds=["rare seed"] if index % 3 == 0 else [],
        novelty_descriptors=["novel"],
        niche_memberships=[f"niche-{index % 5}"],
        multihead_scores={
            "objective_alignment": 0.1 + index / 100,
            "answer_likelihood": 0.2 + index / 100,
            "core_mechanism_strength": 0.3,
            "rarity": 0.9 if index % 3 == 0 else 0.1,
            "verifiability": 0.2,
        },
    )


def test_nexus_rank_uses_deterministic_coverage_instead_of_lossy_candidate_compression(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROMPT_MAX_CHARS", "30000")
    blob = "X" * 50_000
    candidates = [_large_candidate(i, blob) for i in range(36)]
    archives = ArchiveManager()
    archives.update(
        [FateAssignment(candidates[0].id, CandidateFate.ELITE)]
        + [FateAssignment(c.id, CandidateFate.DORMANT) for c in candidates[1:]],
        candidates=candidates,
    )
    full_local_state = json.dumps(archives.to_dict(), ensure_ascii=False, default=str)
    assert len(full_local_state) > 500_000

    captured: list[dict] = []

    def caller(request_type: str, payload: dict, schema: dict) -> dict:
        captured.append(payload)
        return {
            "best_final_answer_id": payload["candidates"][0]["id"],
            "strongest_mechanism_id": payload["candidates"][0]["id"],
            "mutation_worthy_ids": [payload["candidates"][0]["id"]],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
            "raw_notes": "bounded",
        }

    adapter = StructuredModelAdapter(caller=caller)
    ranking = adapter.relative_rank(candidates=candidates, contract=NexusObjectiveContract(original_user_goal="g", normalized_goal="g"), policy=EvolutionPolicy(), archives=archives)

    assert captured == []
    assert set(ranking["multihead_observations"]) == {candidate.id for candidate in candidates}
    assert "exact_candidate_exceeded_prompt_cap" in ranking["raw_notes"]
    assert adapter.metadata["last_exact_candidate_batching"]["deterministic_coverage_count"] == len(candidates)


def test_candidate_prompt_view_keeps_full_local_genome_out_of_model_payload() -> None:
    blob = "Y" * 20_000
    candidate = _large_candidate(1, blob)

    full = json.dumps(candidate.to_dict(), ensure_ascii=False, default=str)
    view = json.dumps(candidate_prompt_view(candidate), ensure_ascii=False, default=str)

    assert len(full) > 60_000
    assert len(view) < 20_000
    assert "Y" * 1000 not in view
    assert candidate_prompt_view(candidate)["artifact_sha256"]


def test_build_prompt_view_fits_configured_budget(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROMPT_MAX_CHARS", "12000")
    blob = "Z" * 40_000
    payload = {"population": [_large_candidate(i, blob) for i in range(50)], "archives": {}, "history": [{"round": i, "ranking": {"raw_notes": blob}} for i in range(8)]}

    view = build_prompt_view("nexus_diagnose_search_state", payload)

    assert view.metadata["sent_payload_chars"] <= 12_000
    assert view.metadata["compressed"] is True
    assert view.payload["candidate_population_stats"]["count"] == 50


def test_source_context_preserves_raw_code_in_prompt_view() -> None:
    code = "def target():\n    return 'real source'\n" + "# keep\n" * 900
    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "source_context": {
                "selected_files": ["pkg/mod.py"],
                "budget_policy": "top_1_files_capped_6000chars_from_context_packets",
                "slices": [{"path": "pkg/mod.py", "hash": "abc", "start": 1, "end": 901, "text": code}],
            }
        },
    )

    source_context = view.payload["source_context"]
    assert source_context["budget_policy"] == "top_1_files_capped_6000chars_from_context_packets"
    assert source_context["slices"][0]["path"] == "pkg/mod.py"
    assert source_context["slices"][0]["text"] == code
    assert "type" not in source_context


def test_text_world_model_prompt_preserves_bounded_large_input_packet(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROMPT_MAX_CHARS", "20000")
    canary = "PACKET-END-CANARY-7F3A"
    raw_text = "Causal task input.\n" + "observation " * 450 + canary
    claim = "coolant flow causally changes defect rate"
    constraint = "must compare interventions at flow 8 and 12"
    packet = TextInputPacket(
        raw_text=raw_text,
        extracted_claims=[claim],
        constraints=[constraint],
        available_evidence=[{"kind": "input_evidence", "source": "raw_text", "content": raw_text, "confidence": 1.0}],
    )
    assert len(packet.to_json()) > 4_000
    captured: list[dict] = []

    def caller(_request_type: str, payload: dict, _schema: dict) -> dict:
        captured.append(payload)
        return {"kind": "text", "input_packet_id": packet.packet_id, "goal_summary": raw_text}

    StructuredModelAdapter(caller=caller).build_text_world_model(packet=packet)

    sent = captured[0]
    packet_view = sent["packet"]
    assert len(json.dumps(sent, ensure_ascii=False)) <= 20_000
    assert packet_view["raw_text"] == raw_text
    assert packet_view["extracted_claims"] == [claim]
    assert packet_view["constraints"] == [constraint]
    assert packet_view["available_evidence"][0]["source"] == "raw_text"
    assert canary in packet_view["available_evidence"][0]["content"]
    assert canary in packet_view["raw_text"]
    assert set(packet_view) != {"type", "chars", "sha256"}


def test_objective_contract_prompt_preserves_text_world_semantics() -> None:
    world = TextWorldModel(
        input_packet_id="textpkt-f4",
        goal_summary="Solve the microgrid optimization and emit the required proof DAG.",
        evidence_boundaries={"authoritative_source": ["raw prompt"]},
        likely_task_types=["constrained optimization"],
        constraint_summary=[
            "capacity is 7",
            "clinic must be on",
            "pump implies radio",
            "problem_id must be F4",
        ],
        edge_seed_pool=["capacity->optimality"],
    )
    captured: list[dict] = []

    def caller(_request_type: str, payload: dict, _schema: dict) -> dict:
        captured.append(payload)
        return {"original_user_goal": "solve F4", "normalized_goal": "solve F4"}

    StructuredModelAdapter(caller=caller).build_objective_contract(user_goal="solve F4", world=world)

    sent_world = captured[0]["world"]
    assert sent_world["input_packet_id"] == "textpkt-f4"
    assert sent_world["goal_summary"].startswith("Solve the microgrid")
    assert sent_world["constraint_summary"][-1] == "problem_id must be F4"
    assert sent_world["evidence_boundaries"] == {"authoritative_source": ["raw prompt"]}
    assert "summary" not in sent_world
    assert "constraints" not in sent_world


def test_frozen_spec_and_initial_candidate_survive_seed_and_plan_views(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROMPT_MAX_CHARS", "30000")
    problem_text = (
        "PROBLEM-BEGIN-CANARY\n"
        '{"type":"object","properties":{"answer":{"enum":["SCHEMA-END-CANARY"]}}}\n'
        + ("constraint line\n" * 60)
        + "PROBLEM-END-CANARY"
    )
    frozen_spec = {
        "problem_text": problem_text,
        "spec_sha256": hashlib.sha256(problem_text.encode("utf-8")).hexdigest(),
    }
    incumbent_artifact = {
        "answer": "INCUMBENT-BEGIN-" + ("A" * 1200) + "-INCUMBENT-END-CANARY",
    }
    incumbent = CandidateGenome(
        id="operator-start",
        artifact=incumbent_artifact,
        artifact_type="machine",
        concise_claim="operator-provided start",
        core_mechanism="preserve the supplied work product",
    )
    contract = NexusObjectiveContract(
        original_user_goal=problem_text,
        normalized_goal="solve the frozen problem",
        frozen_spec=frozen_spec,
    )
    source_context = {"frozen_spec": frozen_spec, "initial_candidates": [incumbent]}

    seed_view = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": contract,
            "world": {},
            "policy": EvolutionPolicy(),
            "source_context": source_context,
            "requested_candidate_count": 3,
        },
        max_chars=30_000,
    )

    assert seed_view.payload["requested_candidate_count"] == 3
    assert seed_view.payload["contract"]["frozen_spec"] == frozen_spec
    assert "frozen_spec" not in seed_view.payload["source_context"]
    assert seed_view.payload["source_context"]["initial_candidates"][0]["artifact"] == incumbent_artifact
    assert "PROBLEM-BEGIN-CANARY" in seed_view.payload["contract"]["frozen_spec"]["problem_text"]
    assert "SCHEMA-END-CANARY" in seed_view.payload["contract"]["frozen_spec"]["problem_text"]
    assert seed_view.payload["contract"]["frozen_spec"]["problem_text"].endswith("PROBLEM-END-CANARY")

    plan_view = build_prompt_view(
        "nexus_plan_mutations",
        {
            "parents": [incumbent],
            "policy": EvolutionPolicy(),
            "diagnosis": {},
            "archives": {},
            "source_context": source_context,
        },
        max_chars=30_000,
    )

    assert plan_view.payload["contract"] == {}
    assert plan_view.payload["source_context"]["frozen_spec"] == frozen_spec
    assert plan_view.payload["source_context"]["initial_candidates"][0]["artifact"] == incumbent_artifact


def test_frozen_spec_is_canonical_and_changes_contract_identity() -> None:
    problem_text = "BEGIN\nrequired schema: answer:string\nEND"
    expected_hash = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()
    contract = NexusObjectiveContract(
        original_user_goal=problem_text,
        normalized_goal="solve",
        frozen_spec={"problem_text": problem_text, "spec_sha256": "not-canonical"},
    )
    changed = NexusObjectiveContract(
        original_user_goal=problem_text + "-changed",
        normalized_goal="solve",
        frozen_spec={"problem_text": problem_text + "-changed"},
    )

    assert contract.frozen_spec == {"problem_text": problem_text, "spec_sha256": expected_hash}
    assert contract.contract_hash() != changed.contract_hash()
    assert "frozen_spec" not in NexusObjectiveContract(
        original_user_goal="legacy",
        normalized_goal="legacy",
    ).canonical_payload()


def test_frozen_spec_over_transport_cap_fails_instead_of_truncating(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "1400")
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "5000")
    problem_text = "PROBLEM-BEGIN\n" + ("X" * 5000) + "\nPROBLEM-END"
    contract = NexusObjectiveContract(
        original_user_goal=problem_text,
        normalized_goal="solve",
        frozen_spec={"problem_text": problem_text},
    )

    def should_not_call(_request_type: str, _payload: dict, _schema: dict) -> dict:
        pytest.fail("lossy prompt must fail before invoking the model caller")

    with pytest.raises(LLMResponseError, match="could not fit effective cap"):
        StructuredModelAdapter(caller=should_not_call)._call(
            "nexus_seed_population",
            {"contract": contract, "world": {}, "policy": EvolutionPolicy()},
            {"type": "object"},
        )


def test_direct_parent_over_transport_cap_fails_instead_of_dropping_artifact(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "30000")
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "30000")
    parent = CandidateGenome(
        id="P-over-cap",
        artifact="PARENT-BEGIN\n" + ("X" * 50_000) + "\nPARENT-END",
        artifact_type="permutation_path",
    )
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        metadata={"plan_source": "runtime_lineage_envelope"},
    )

    def should_not_call(_request_type: str, _payload: dict, _schema: dict) -> dict:
        pytest.fail("lossy parent prompt must fail before invoking the model caller")

    with pytest.raises(LLMResponseError, match="could not fit effective cap"):
        StructuredModelAdapter(caller=should_not_call).generate_offspring(
            plans=[plan],
            parents=[parent],
            world={},
            contract=NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
            policy=EvolutionPolicy(),
        )


def test_transport_preserves_complete_prompt_before_provider(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "1400")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class CaptureProvider:
        provider_id = "capture"

        def __init__(self) -> None:
            self.messages = []

        def complete_json(self, **kwargs):
            assert kwargs["_retry_allow_prompt_shrink"] is True
            self.messages = list(kwargs["messages"])
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
            return LLMProviderResult(response=response, attempts=1)

    provider = CaptureProvider()
    assert llm_json("huge_request", {"blob": "Q" * 50_000}, system="Return JSON", schema_hint={}, provider=provider) == {
        "ok": True,
        "provider": "litellm",
        "model": "unit-test-model",
    }
    user_message = provider.messages[1]["content"]
    assert len(user_message) > 1400
    assert json.loads(user_message)["payload"]["blob"] == "Q" * 50_000


def test_nexus_adapter_structurally_fits_to_transport_cap(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "1400")
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "5000")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class CaptureProvider:
        provider_id = "capture"

        def __init__(self) -> None:
            self.messages = []

        def complete_json(self, **kwargs):
            assert kwargs["_retry_allow_prompt_shrink"] is False
            self.messages = list(kwargs["messages"])
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
            return LLMProviderResult(response=response, attempts=1)

    provider = CaptureProvider()
    monkeypatch.setattr(transport_module, "_default_provider_for_status", lambda _status: provider)
    adapter = StructuredModelAdapter.from_configured_llm()

    adapter._call(
        "nexus_diagnose_search_state",
        {f"field_{index}": "Q" * 3000 for index in range(30)},
        {"type": "object"},
    )

    user_message = provider.messages[1]["content"]
    request = json.loads(user_message)
    assert len(user_message) <= 1400
    assert request["payload"].get("_transport_prompt_truncated") is not True
    assert "bounded_request_excerpt" not in request["payload"]
    assert request["payload"]["request_type"] == "nexus_diagnose_search_state"
    assert adapter.metadata["last_prompt_view"]["effective_max_prompt_chars"] == 1400
    assert adapter.metadata["last_prompt_view"]["sent_request_chars"] == len(user_message)
    assert adapter.metadata["last_prompt_view"]["truncated"] is True


def test_f4_shaped_plan_view_keeps_authority_without_duplicate_prompt_scaffolding() -> None:
    problem_text = "F4-BEGIN\n" + ("hard constraint\n" * 120) + "F4-END-CANARY"
    incumbent = CandidateGenome(
        id="operator-start",
        artifact={"dispatch": "INCUMBENT-ARTIFACT-CANARY", "proof": ["capacity", "optimality"]},
        artifact_type="machine",
        concise_claim="operator supplied incumbent",
        core_mechanism="preserve exact dispatch and proof object",
    )
    archive_candidates = {
        f"A{index}": CandidateGenome(
            id=f"A{index}",
            artifact={"dispatch": f"archive-{index}-" + ("X" * 2_000)},
            concise_claim=f"archive candidate {index}",
            core_mechanism=f"mechanism {index}",
            multihead_scores={"answer_likelihood": 1.0 - index / 100},
        )
        for index in range(12)
    }
    archives = SimpleNamespace(
        summary=lambda: {"answer_candidates": 12, "dormant_candidates": 12},
        answer_archive={"operator-start": incumbent, **archive_candidates},
        rarity_archive=SimpleNamespace(candidates=dict(archive_candidates)),
        dormant_archive=SimpleNamespace(candidates=dict(archive_candidates)),
        auxiliary_archive=SimpleNamespace(candidates=dict(archive_candidates)),
        failure_archive=SimpleNamespace(records={}),
    )
    contract = NexusObjectiveContract(
        original_user_goal=problem_text,
        normalized_goal="solve frozen F4",
        frozen_spec={"problem_text": problem_text},
    )
    policy = EvolutionPolicy(
        metadata={
            "minimal_core_ablation": {f"profile_{index}": "CONTROL-SCAFFOLD-" + ("Y" * 4_000) for index in range(6)},
            "algorithm_efficiency": {"direct_evolution": True},
        }
    )

    view = build_prompt_view(
        "nexus_plan_mutations",
        {
            "contract": contract,
            "parents": [incumbent],
            "policy": policy,
            "archives": archives,
            "source_context": {"initial_candidates": [incumbent]},
            "diagnosis": {},
        },
        max_chars=120_000,
    )
    serialized = json.dumps(view.payload, ensure_ascii=False, sort_keys=True)
    archive_ids = [
        item["id"]
        for key in ("answer_elites", "rarity_elites", "dormant_hints", "auxiliary_hints")
        for item in view.payload["archives"][key]
    ]

    assert len(serialized) < 40_000
    assert "F4-END-CANARY" in serialized
    assert "INCUMBENT-ARTIFACT-CANARY" in serialized
    assert "minimal_core_ablation" not in serialized
    assert "CONTROL-SCAFFOLD" not in serialized
    assert "operator-start" not in archive_ids
    assert len(archive_ids) == len(set(archive_ids))


def test_direct_offspring_view_keeps_exact_parent_and_evaluator_vector() -> None:
    artifact = "PARENT-BEGIN\n" + ("A" * 2_400) + "\nPARENT-MIDDLE-CANARY\n" + ("B" * 2_400) + "\nPARENT-END"
    parent = CandidateGenome(
        id="P",
        artifact=artifact,
        artifact_type="permutation_path",
        concise_claim="current incumbent",
        core_mechanism="improve the measured path",
        metadata={
            "evaluator": {
                "status": "failed",
                "passed": False,
                "metrics": {"length": 873, "coverage": 720, "distinct": 719},
                "diagnostics": ["one duplicate path remains"],
                "details": {
                    "per_instance": [{"instance_id": "feedback-1", "score": 0.8}],
                    "failures": [{"kind": "oracle_process", "detail": "oracle unavailable"}],
                    "_cache": {"cache_hit": True, "cache_path": "/private/e2-cache/item.json"},
                },
            },
            "evidence_records": [
                {
                    "candidate_id": "P",
                    "diagnostics": [
                        'repair diagnostic: {"failure":"oracle unavailable","candidate_sha256":"abc","evaluated_at":"now"}'
                    ],
                    "hints": [
                        'repair diagnostic: {"failure":"oracle unavailable","cache_hit":true,"oracle_invocations":1}'
                    ],
                    "metadata": {"metrics": {"length": 873}, "cache_path": "/private/e2-cache/item.json"},
                }
            ],
            "evidence_state": {
                "search_score": 0.82,
                "repair_value": 0.9,
                "final_blocked": True,
                "target_challenge_ids": ["duplicate-path"],
            },
        },
        verification_trace=[
            {
                "status": "passed",
                "metadata": {"cache_key": "runtime-cache"},
                "replay_record": {"verification_cache_key": "verification:abc"},
            },
            {
                "tool_id": "external_evaluator",
                "status": "failed",
                "diagnostics": ["one duplicate path remains"],
                "raw_output_ref": "external_evaluator:failed",
            },
        ],
    )

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "parents": [parent],
            "plans": [{"operator": "ModelDirected", "parent_ids": ["P"]}],
            "contract": NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
            "policy": EvolutionPolicy(),
            "source_context": {"problem_spec": "improve", "initial_candidates": [parent]},
        },
        max_chars=30_000,
    )

    sent_parent = view.payload["parents"][0]
    assert sent_parent["artifact"] == artifact
    assert "PARENT-MIDDLE-CANARY" in json.dumps(view.payload, ensure_ascii=False)
    assert sent_parent["external_evaluator"]["metrics"] == {"length": 873, "coverage": 720, "distinct": 719}
    assert sent_parent["external_evaluator"]["diagnostics"] == ["one duplicate path remains"]
    assert sent_parent["external_evaluator"]["details"] == {
        "per_instance": [{"instance_id": "feedback-1", "score": 0.8}],
        "failures": [{"kind": "oracle_process", "detail": "oracle unavailable"}],
    }
    assert sent_parent["external_evaluator"]["evidence_state"]["target_challenge_ids"] == ["duplicate-path"]
    assert not {"evaluator", "evidence_records", "evidence_state"} & sent_parent["metadata"].keys()
    assert all(item.get("tool_id") != "external_evaluator" for item in sent_parent["verification_trace"])
    assert "initial_candidates" not in view.payload["source_context"]
    assert "supersede earlier packet/world absence claims" in view.payload["prompt_contract"]["current_state_precedence"]
    prompt_text = json.dumps(view.payload, ensure_ascii=False, sort_keys=True)
    assert "oracle unavailable" in prompt_text
    assert prompt_text.count("one duplicate path remains") == 1
    for volatile_key in (
        "_cache",
        "cache_hit",
        "cache_key",
        "cache_path",
        "candidate_sha256",
        "evaluated_at",
        "oracle_invocations",
        "verification_cache_key",
    ):
        assert volatile_key not in prompt_text


def test_direct_offspring_view_cleans_evaluator_telemetry_from_runtime_plan() -> None:
    failure = {
        "_cache": {
            "cache_hit": False,
            "cache_path": "/private/e2-cache/item.json",
            "candidate_sha256": "abc",
            "evaluated_at": "now",
            "oracle_invocations": 1,
            "cache_key": "runtime-cache",
            "verification_cache_key": "verification:abc",
        },
        "failures": [{"kind": "evaluator_error", "detail": "candidate contains no source"}],
        "mean_score": None,
    }
    plan = {
        "operator": "ModelDirected",
        "parent_ids": ["P"],
        "instruction": "repair the evaluator-visible artifact",
        "metadata": {
            "plan_source": "runtime_lineage_envelope",
            "completion_mode": "complete_task_artifact_only",
            "branch_slots": [
                {
                    "slot_id": "slot-p",
                    "directive": {
                        "search_pressure": {
                            "success_criteria": [{"summary": json.dumps(failure)}],
                        }
                    },
                }
            ],
        },
    }

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "plans": [plan],
            "mutation_instruction": f"repair this failure: {json.dumps(failure)}; keep the measured failure",
        },
        max_chars=30_000,
    )

    prompt_text = json.dumps(view.payload, ensure_ascii=False, sort_keys=True)
    assert "candidate contains no source" in prompt_text
    assert "mean_score" in prompt_text
    assert "keep the measured failure" in prompt_text
    for volatile_key in (
        "_cache",
        "cache_hit",
        "cache_key",
        "cache_path",
        "candidate_sha256",
        "evaluated_at",
        "oracle_invocations",
        "verification_cache_key",
    ):
        assert volatile_key not in prompt_text


def test_evaluator_led_runtime_lineage_offspring_requires_complete_task_artifacts_only() -> None:
    payload = {
        "plans": [
            MutationPlan(
                operator="ModelDirected",
                parent_ids=["P"],
                metadata={
                    "plan_source": "runtime_lineage_envelope",
                    "completion_mode": "complete_task_artifact_only",
                },
            )
        ],
        "contract": NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
        "policy": EvolutionPolicy(),
    }

    direct = build_prompt_view("nexus_generate_offspring", payload).payload["artifact_generation_contract"]
    assert direct["completion_mode"] == "complete_task_artifact_only"
    assert "complete evaluator-visible task artifact" in direct["non_negotiable_runtime_invariant"]
    assert "unified_diff" in direct["project_patch_output_rule"]
    assert "invalid" in direct["project_patch_output_rule"]
    assert "executable repair step" not in direct["non_negotiable_runtime_invariant"]
    assert "repair obligation" not in direct["when_incomplete"]

    payload["plans"] = [
        MutationPlan(
            operator="ModelDirected",
            parent_ids=["P"],
            metadata={
                "plan_source": "runtime_lineage_envelope",
                "completion_mode": "concrete_progress_allowed",
            },
        )
    ]
    exploratory = build_prompt_view("nexus_generate_offspring", payload).payload["artifact_generation_contract"]
    assert "completion_mode" not in exploratory
    assert "executable repair step" in exploratory["non_negotiable_runtime_invariant"]
    assert "repair obligation" in exploratory["when_incomplete"]


def test_schema_bound_seed_requires_complete_task_artifact() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="return a machine artifact",
        normalized_goal="return a machine artifact",
        dynamic_artifact_contract={
            "required_work_product": {
                "artifact_type": "machine",
                "required_fields": ["answer"],
            }
        },
    )

    strict = build_prompt_view(
        "nexus_seed_population",
        {"contract": contract, "policy": EvolutionPolicy(), "world": {}},
    ).payload["artifact_generation_contract"]
    assert strict["completion_mode"] == "complete_task_artifact_only"
    assert "when present" in strict["model_defined_required_work_product"]
    assert "when parents are supplied" in strict["model_defined_minimum_delta"]
    assert "empty result array" in strict["when_incomplete"]
    assert "preserves any already accepted incumbent" in strict["when_incomplete"]
    assert "Return no offspring" not in strict["when_incomplete"]

    shape_bound = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": {
                "dynamic_artifact_contract": {
                    "required_work_product": {},
                    "allowed_artifact_shapes": [{"required_fields": ["answer"]}],
                }
            },
            "policy": EvolutionPolicy(),
            "world": {},
        },
    ).payload["artifact_generation_contract"]
    assert shape_bound["completion_mode"] == "complete_task_artifact_only"

    adapter_bound = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": NexusObjectiveContract(
                original_user_goal="return an adapter-bound artifact",
                normalized_goal="return an adapter-bound artifact",
                dynamic_artifact_contract=DynamicArtifactContract(
                    objective="return an adapter-bound artifact",
                    adapter_requirements={"required_fields": ["answer"]},
                ).to_dict(),
            ),
            "policy": EvolutionPolicy(),
            "world": {},
        },
    ).payload["artifact_generation_contract"]
    assert adapter_bound["completion_mode"] == "complete_task_artifact_only"

    legacy = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": NexusObjectiveContract(original_user_goal="explore", normalized_goal="explore"),
            "policy": EvolutionPolicy(),
            "world": {},
        },
    ).payload["artifact_generation_contract"]
    assert "completion_mode" not in legacy
    assert "repair obligation" in legacy["when_incomplete"]
