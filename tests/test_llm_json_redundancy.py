from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.json_tools import extract_json_from_text
from cognitive_evolve_runtime.llm.session import _LAST_RETRY_HISTORY
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.controller import EvolutionLoopController
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
import pytest


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_llm_json_uses_single_five_attempt_default_budget_for_parse_repair(monkeypatch) -> None:
    class Provider:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls < 5:
                return LLMProviderResult(response=_response(""))
            return LLMProviderResult(response=_response('{"ok": true}'))

    provider = Provider()
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.delenv("COGEV_LLM_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("COGEV_LLM_JSON_RETRY_ATTEMPTS", raising=False)

    response = llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=provider)

    assert response["ok"] is True
    assert provider.calls == 5
    assert [item["attempt"] for item in _LAST_RETRY_HISTORY.get()] == [1, 2, 3, 4]


def test_llm_json_exception_after_parse_retry_does_not_self_extend_history(monkeypatch) -> None:
    class Provider:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return LLMProviderResult(response=_response(""))
            raise TimeoutError("provider timeout")

    provider = Provider()
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "2")

    with pytest.raises(LLMResponseError, match="provider timeout"):
        llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 2
    assert [item["attempt"] for item in _LAST_RETRY_HISTORY.get()] == [1]


def test_extract_json_from_text_accepts_embedded_fenced_json() -> None:
    text = "Model note before JSON.\n```json\n{\"ok\": true, \"mode\": \"embedded\"}\n```\ntrailing note"

    assert extract_json_from_text(text) == {"ok": True, "mode": "embedded"}


def test_extract_json_from_text_accepts_prose_wrapped_object() -> None:
    text = "Here is the result: {\"ok\": true, \"value\": 3} and nothing else."

    assert extract_json_from_text(text) == {"ok": True, "value": 3}


def test_extract_json_from_text_rejects_refusal_with_hint() -> None:
    with pytest.raises(LLMResponseError) as exc:
        extract_json_from_text("I cannot fulfill this request.")

    assert "parse_hint=refusal_or_empty" in str(exc.value)


def test_extract_json_from_text_still_rejects_non_object_json() -> None:
    with pytest.raises(LLMResponseError, match="must be a JSON object"):
        extract_json_from_text("```json\n[1, 2, 3]\n```")


def test_invalid_provider_json_secret_is_redacted_before_checkpoint_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "unit-test-provider-token-123456789"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", secret)
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("COGEV_LLM_JSON_RETRY_ATTEMPTS", "1")

    class Provider:
        provider_id = "test"

        def complete_json(self, **kwargs):  # noqa: ANN001
            return LLMProviderResult(response=_response(f"not-json {secret}"), attempts=1)

    with pytest.raises(LLMResponseError) as exc_info:
        llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=Provider())

    controller = EvolutionLoopController(
        population=CandidatePopulation([]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="test", normalized_goal="test"),
        world={"kind": "text"},
        budget=EvolutionBudget(max_rounds=1),
    )
    controller._checkpoint_interruption(
        0,
        exc_info.value,
        stop_reason="model_error_checkpointed",
        stagnation_type="ProviderError",
        actions=["resume_from_checkpoint"],
    )
    store = LiveNexusStore(
        tmp_path,
        mode="text",
        contract=controller.contract,
        world=controller.world,
        max_rounds=1,
    )
    store({
        "phase": "error_checkpoint",
        "round": 0,
        "population": controller.population,
        "archives": controller.archives,
        "policy": controller.policy,
        "diagnosis": controller.diagnosis,
        "progress_event": {"type": "evolution_progress", "round": 0, "max_rounds": 1},
        "budget_history": controller.budget.history,
        "error": controller.error,
    })

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    assert secret not in str(exc_info.value)
    assert secret not in persisted
