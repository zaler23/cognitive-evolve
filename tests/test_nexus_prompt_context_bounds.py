from __future__ import annotations

import json
from types import SimpleNamespace

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.transport import llm_json
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


def test_nexus_prompt_view_compresses_candidates_and_archives(monkeypatch) -> None:
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
    adapter.relative_rank(candidates=candidates, contract=NexusObjectiveContract(original_user_goal="g", normalized_goal="g"), policy=EvolutionPolicy(), archives=archives)

    sent = json.dumps(captured[0], ensure_ascii=False, default=str)
    assert len(sent) <= 30_000
    assert "X" * 1000 not in sent
    assert captured[0]["prompt_contract"]["state_is_compressed"] is True
    assert captured[0]["archives"]["summary"]["dormant_candidates"] >= 1
    assert adapter.metadata["last_prompt_view"]["raw_payload_chars"] > adapter.metadata["last_prompt_view"]["sent_payload_chars"]


def test_candidate_prompt_view_keeps_full_local_genome_out_of_model_payload() -> None:
    blob = "Y" * 20_000
    candidate = _large_candidate(1, blob)

    full = json.dumps(candidate.to_dict(), ensure_ascii=False, default=str)
    view = json.dumps(candidate_prompt_view(candidate), ensure_ascii=False, default=str)

    assert len(full) > 60_000
    assert len(view) < 8_000
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


def test_transport_applies_prompt_bound_before_provider(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "1400")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class CaptureProvider:
        provider_id = "capture"

        def __init__(self) -> None:
            self.messages = []

        def complete_json(self, **kwargs):
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
    assert len(user_message) <= 1400
    assert json.loads(user_message)["payload"]["_transport_prompt_truncated"] is True
