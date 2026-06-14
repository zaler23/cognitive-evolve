from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.api.jobs import _job_public
from cognitive_evolve_runtime.api.security import _matches_service_api_key
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.core.redaction import redact
from cognitive_evolve_runtime.durable.checkpoint_store import CheckpointStore as DurableCheckpointStore
from cognitive_evolve_runtime.durable.event_log import append_jsonl
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.http_provider import DirectHTTPProviderError
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.retry import completion_with_retry, provider_error_category, retry_sleep_seconds
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.nexus._serde import stable_hash, stable_json
from cognitive_evolve_runtime.persistence.event_store import EventStore


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_event_store_append_fsyncs_and_redacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fsynced: list[int] = []
    monkeypatch.setenv("COGEV_LLM_API_KEY", "sk-secret-runtime-key")
    monkeypatch.setattr("cognitive_evolve_runtime.persistence.event_store.os.fsync", lambda fd: fsynced.append(fd))

    store = EventStore(tmp_path / "events.jsonl")
    payload = store.append({"type": "provider_error", "api_key": "sk-secret-runtime-key", "message": "Bearer sk-secret-runtime-key failed"})

    assert fsynced
    assert payload["api_key"] == "[REDACTED]"
    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "sk-secret-runtime-key" not in raw
    assert "[REDACTED]" in raw


def test_redact_secret_shaped_key_handles_nested_payloads_without_crashing() -> None:
    payload = {
        "type": "latent_metadata",
        "token_metadata": {"nested": {"still": "private"}},
        "authorization": ["Bearer sk-nested-runtime-key", {"inner": "value"}],
        "session_id": None,
        "api_key": "",
        "safe": {"nested": "kept"},
    }

    redacted = redact(payload)

    assert redacted["token_metadata"] == "[REDACTED]"
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["session_id"] is None
    assert redacted["api_key"] == ""
    assert redacted["safe"] == {"nested": "kept"}


def test_redact_handles_sets_cycles_and_nested_secret_values() -> None:
    payload: dict[str, object] = {"meta": {"values": {"sk-secret-runtime-key", "safe"}}}
    payload["self"] = payload

    redacted = redact(payload)

    assert redacted["self"] == "[CIRCULAR]"
    values = redacted["meta"]["values"]
    assert "[REDACTED]" in values
    assert "sk-secret-runtime-key" not in json.dumps(redacted, ensure_ascii=False)


def test_stable_json_orders_sets_and_marks_cycles() -> None:
    payload: dict[str, object] = {"items": {"b", "a", "c"}}
    payload["self"] = payload

    encoded = stable_json(payload)

    assert '"items":["a","b","c"]' in encoded
    assert '"self":"[CIRCULAR]"' in encoded
    assert stable_hash({"items": {"c", "b", "a"}}) == stable_hash({"items": {"a", "b", "c"}})


def test_event_logs_redact_set_values_and_survive_cycles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_LLM_API_KEY", "sk-secret-runtime-key")
    payload: dict[str, object] = {"type": "provider_error", "values": {"sk-secret-runtime-key"}}
    payload["self"] = payload

    store = EventStore(tmp_path / "events.jsonl")
    stored = store.append(payload)
    append_jsonl(tmp_path / "durable.jsonl", payload)

    assert stored["self"] == "[CIRCULAR]"
    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8") + (tmp_path / "durable.jsonl").read_text(encoding="utf-8")
    assert "sk-secret-runtime-key" not in raw
    assert "[REDACTED]" in raw


def test_durable_checkpoint_store_handles_sets_and_cycles(tmp_path: Path) -> None:
    payload: dict[str, object] = {"values": {"b", "a"}}
    payload["self"] = payload

    store = DurableCheckpointStore(tmp_path / "durable")
    status = store.write_input("round", "step", "name", payload)

    assert status.input_hash
    persisted = store.read_input("round", "step", "name")
    assert persisted == {"self": "[CIRCULAR]", "values": ["a", "b"]}


def test_retry_defaults_to_five_attempts_with_1_2_4_8_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COGEV_LLM_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("COGEV_LLM_RETRY_BASE_SLEEP", raising=False)
    monkeypatch.delenv("COGEV_LLM_RETRY_JITTER", raising=False)
    sleeps: list[float] = []
    calls = {"count": 0}

    def always_fails(**_: object) -> None:
        calls["count"] += 1
        raise RuntimeError("temporary network failure")

    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(RuntimeError):
        completion_with_retry(always_fails, messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    assert calls["count"] == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_llm_json_parse_repair_does_not_multiply_transport_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class Provider:
        provider_id = "unit"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **kwargs):  # noqa: ANN001
            self.calls += 1
            assert kwargs["_retry_max_attempts"] == max(1, 5 - (self.calls - 1))
            return LLMProviderResult(response=_response("not json"), attempts=1)

    provider = Provider()
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit/model")
    monkeypatch.delenv("COGEV_LLM_RETRY_ATTEMPTS", raising=False)
    monkeypatch.setenv("COGEV_LLM_JSON_RETRY_ATTEMPTS", "99")

    with pytest.raises(LLMResponseError):
        llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 5


def test_budget_and_quota_errors_are_not_ordinary_transient_retries() -> None:
    class QuotaError(RuntimeError):
        status_code = 429

    assert provider_error_category(QuotaError("RESOURCE_EXHAUSTED quota exceeded")) == "quota_exhausted"
    assert provider_error_category(LLMResponseError("LLM cost budget already exhausted before next call")) == "budget_exhausted"
    assert provider_error_category(DirectHTTPProviderError("EMPTY_ASSISTANT_CONTENT from direct_http provider", status_code=502)) == "empty_assistant_content"
    assert provider_error_category(DirectHTTPProviderError("TRUNCATED assistant content; finish_reason=length")) == "truncated_response"
    assert retry_sleep_seconds(RuntimeError("temporary network failure"), 4) == 8.0


def test_empty_assistant_retry_mutates_request_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("COGEV_LLM_RETRY_JITTER", "0")
    monkeypatch.setenv("COGEV_LLM_RETRY_MAX_TOKENS", "32768")
    seen_tokens: list[int] = []

    def flaky(**kwargs):  # noqa: ANN001
        seen_tokens.append(int(kwargs.get("max_tokens") or 0))
        if len(seen_tokens) == 1:
            raise DirectHTTPProviderError("EMPTY_ASSISTANT_CONTENT from direct_http provider", status_code=502)
        return SimpleNamespace(ok=True)

    result, attempts = completion_with_retry(flaky, messages=[{"role": "user", "content": "x" * 5000}], max_tokens=4096, _retry_max_attempts=2)

    assert result.ok is True
    assert attempts == 2
    assert seen_tokens == [4096, 8192]


def test_create_app_refuses_public_bind_without_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cognitive_evolve_runtime.api.openai_compat import create_app

    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "false")
    monkeypatch.delenv("COGEV_ALLOW_INSECURE_BIND", raising=False)

    with pytest.raises(RuntimeError, match="Refusing to serve"):
        create_app()


def test_service_api_key_matching_is_constant_time_membership() -> None:
    assert _matches_service_api_key("secret-b", ("secret-a", "secret-b")) is True
    assert _matches_service_api_key("secret-c", ("secret-a", "secret-b")) is False


def test_job_public_downgrades_durable_resume_claim(tmp_path: Path) -> None:
    job = {"id": "job-test", "status": "completed", "artifact_root": str(tmp_path), "task_dir": str(tmp_path)}

    public = _job_public(job)

    assert public["durable_resume_plan"]["status"] == "snapshot_only"
    assert public["durable_resume_plan"]["api_resume_supported"] is False
    assert public["resume_available"] is False


def test_archive_elite_precedes_auxiliary_and_tombstone_is_removed() -> None:
    candidate = CandidateGenome(
        id="winner",
        current_fate=CandidateFate.ACTIVE,
        artifact="answer",
        multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.7, "auxiliary_value": 0.99},
    )
    archives = ArchiveManager()

    assignments = archives.assign_by_policy([candidate])
    archives.update(assignments, candidates=[candidate])

    assert assignments[0].fate == CandidateFate.ELITE.value
    assert "winner" in archives.answer_archive
    assert "winner" not in archives.auxiliary_archive.candidates

    candidate.mark_fate(CandidateFate.FAILED.value)
    archives.update([candidate])
    assert "winner" in archives.terminal_tombstones
    candidate.mark_fate(CandidateFate.ACTIVE.value)
    archives.update([candidate])
    assert "winner" not in archives.terminal_tombstones


def test_final_synthesis_prompt_uses_compact_evidence_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "9000")
    blob = "X" * 20_000
    candidates = [
        CandidateGenome(
            id=f"C{i}",
            artifact=blob,
            concise_claim=f"claim {i}",
            core_mechanism=f"mechanism {i}",
            current_fate=CandidateFate.DORMANT,
            multihead_scores={"objective_alignment": 0.5, "answer_likelihood": 0.4},
        )
        for i in range(30)
    ]
    archives = ArchiveManager()
    archives.update(candidates)

    view = build_prompt_view("nexus_synthesize_result", {"population": candidates, "archives": archives, "contract": {"normalized_goal": "g"}})
    encoded = json.dumps(view.payload, ensure_ascii=False, default=str)

    assert view.metadata["sent_payload_chars"] <= 9000
    assert view.payload["synthesis_requirements"]["return_non_empty_json"] is True
    assert "X" * 1000 not in encoded


def test_specs_directory_declares_reference_status() -> None:
    text = Path(".cogev/specs/README.md").read_text(encoding="utf-8")

    assert "reference/design specifications" in text
    assert "Do not assume every YAML/JSON file" in text
