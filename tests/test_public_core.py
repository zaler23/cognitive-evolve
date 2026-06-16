from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.llm import llm_public_status, llm_status
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map


def test_fixture_provider_status_is_public_test_path(monkeypatch) -> None:
    fixture = Path(__file__).parent / "fixtures" / "llm_fixture.json"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(fixture))
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)
    status = llm_status()
    assert status["provider"] == "fixture"
    assert status["configured"] is True
    assert status["test_provider_only"] is True


def test_generic_litellm_provider_status_has_no_host_specific_driver(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "provider/model-id")
    monkeypatch.setenv("COGEV_LLM_API_BASE", "https://your-provider.example/v1")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "replace-with-your-upstream-model-api-key")
    status = llm_status()
    assert status["provider"] == "litellm"
    assert status["model"] == "provider/model-id"
    assert status["api_key_placeholder"] is True
    assert "model_key" not in status
    assert "model_enum" not in status

    public = llm_public_status(status)
    assert public["credential_placeholder"] is True
    assert "api_key_placeholder" not in public
    assert "api_key_configured" not in public


def test_parent_selector_preserves_underexplored_rare_candidate() -> None:
    common = CandidateGenome(id="common", artifact="standard", concise_claim="standard", core_mechanism="known", multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.8})
    rare = CandidateGenome(id="rare", artifact="edge", concise_claim="edge", core_mechanism="edge", edge_knowledge_seeds=["obscure analogy"], multihead_scores={"answer_likelihood": 0.4, "objective_alignment": 0.5, "rarity": 1.0, "novelty": 0.9})
    selected = ParentSelector().select([common, rare], limit=2)
    assert {candidate.id for candidate in selected} == {"common", "rare"}


def test_search_space_map_remains_available_for_context_analysis() -> None:
    search_map = build_search_space_map({"task_type": "proof_resolution", "real_objective": "Study a frontier theorem carefully."}, 4)
    assert search_map["candidate_target_count"] >= 1
    assert search_map["route_family"]
