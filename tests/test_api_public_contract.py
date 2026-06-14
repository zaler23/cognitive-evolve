from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from cognitive_evolve_runtime.api.jobs import JobQueue, _get_job, _job_public, _set_job, _task_dir_for_request
from cognitive_evolve_runtime.api.payloads import _completion_payload, _nexus_verification_passed
from cognitive_evolve_runtime.api.server import status as api_status, status_cli
from cognitive_evolve_runtime.api.streaming import _heartbeat_seconds, _stream_engine_chunks, _stream_heartbeat_chunks


def _fake_nexus(rounds: int = 1) -> dict[str, Any]:
    return {
        "mode": "text",
        "evolution": {"progress_events": [{"round": rounds}]},
        "verification_summaries": [],
    }


def _solved_nexus(*, objective_solved: bool, critical_failures: list[str] | None = None) -> dict[str, Any]:
    return {
        "mode": "text",
        "evolution": {
            "completion_status": "solved",
            "progress_events": [{"round": 1}],
            "synthesis": {
                "completion_status": "solved",
                "objective_solved": objective_solved,
                "closure_certificate": {
                    "objective_solved": objective_solved,
                    "critical_failures": list(critical_failures or []),
                },
            },
        },
        "verification_summaries": [{"passed": True}],
    }


def test_api_verification_passed_is_bound_to_objective_closure() -> None:
    assert _nexus_verification_passed(_solved_nexus(objective_solved=True)) is True
    assert _nexus_verification_passed(_solved_nexus(objective_solved=False)) is False
    assert _nexus_verification_passed(_solved_nexus(objective_solved=True, critical_failures=["missing_verified_improvement_certificate"])) is False
    assert _nexus_verification_passed(_fake_nexus(1)) is False


def test_completion_payload_exposes_objective_solved_separately() -> None:
    payload = _completion_payload(
        request_id="req",
        model="cognitive-evolve-one-shot",
        prompt="task",
        answer="answer",
        nexus_data=_solved_nexus(objective_solved=True),
    )

    assert payload["cognitive_evolve"]["verification_passed"] is True
    assert payload["cognitive_evolve"]["objective_solved"] is True


def test_openai_compatible_app_health_models_and_completion(tmp_path: Path) -> None:
    # FastAPI/Pydantic are intentionally exercised in a subprocess so the main
    # pytest process can keep the rest of the Nexus test suite memory-light.
    script = textwrap.dedent(
        r"""
        import json
        from pathlib import Path
        from fastapi.testclient import TestClient
        from cognitive_evolve_runtime.api import openai_compat

        def fake_run_engine(prompt, **kwargs):
            assert "Audit API" in prompt
            return "API answer", {
                "mode": "text",
                "evolution": {"progress_events": [{"round": 2}]},
                "verification_summaries": [],
            }

        openai_compat._run_engine = fake_run_engine
        with TestClient(openai_compat.create_app()) as client:
            health = client.get("/health")
            models = client.get("/v1/models")
            payload = {
                "model": "cognitive-evolve-one-shot",
                "messages": [{"role": "user", "content": "Audit API"}],
            }
            completion = client.post("/v1/chat/completions", json=payload)
            bad_model = client.post("/v1/chat/completions", json={**payload, "model": "missing"})
            bad_prompt = client.post("/v1/chat/completions", json={"model": "cognitive-evolve-one-shot", "messages": []})

        print(json.dumps({
            "health_status": health.status_code,
            "health_provider": health.json()["llm"]["provider"],
            "models_status": models.status_code,
            "models_object": models.json()["object"],
            "models_semantics": models.json()["data"][0]["cognitive_evolve"]["streaming_semantics"],
            "completion_status": completion.status_code,
            "completion_answer": completion.json()["choices"][0]["message"]["content"],
            "completion_rounds": completion.json()["cognitive_evolve"]["actual_rounds"],
            "completion_semantics": completion.json()["cognitive_evolve"]["completion_semantics"],
            "bad_model_status": bad_model.status_code,
            "bad_prompt_status": bad_prompt.status_code,
        }))
        """
    )
    env = os.environ.copy()
    env.update({
        "COGEV_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "COGEV_API_TASK_ROOT": str(tmp_path / "api-runs"),
        "COGEV_SERVER_REQUIRE_AUTH": "false",
        "COGEV_LLM_PROVIDER": "fixture",
        "COGEV_LLM_FIXTURE": str(Path(__file__).parent / "fixtures" / "llm_fixture.json"),
        "COGEV_HERMETIC_TEST": "1",
    })
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(Path(__file__).parents[1]), env=env, text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["health_status"] == 200
    assert data["health_provider"] == "fixture"
    assert data["models_status"] == 200
    assert data["models_object"] == "list"
    assert "not token-by-token" in data["models_semantics"]
    assert data["completion_status"] == 200
    assert data["completion_answer"] == "API answer"
    assert data["completion_rounds"] == 2
    assert "multi-round" in data["completion_semantics"]
    assert data["bad_model_status"] == 404
    assert data["bad_prompt_status"] == 400


def test_api_request_size_and_rate_guards(tmp_path: Path) -> None:
    script = textwrap.dedent(
        r"""
        import json
        from fastapi.testclient import TestClient
        from cognitive_evolve_runtime.api import openai_compat

        with TestClient(openai_compat.create_app()) as client:
            too_large = client.post(
                "/v1/chat/completions",
                json={"model": "cognitive-evolve-one-shot", "messages": [{"role": "user", "content": "x" * 2000}]},
            )
            first = client.get("/v1/models")
            second = client.get("/v1/models")
        print(json.dumps({"too_large": too_large.status_code, "first": first.status_code, "second": second.status_code}))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "COGEV_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "COGEV_API_TASK_ROOT": str(tmp_path / "api-runs"),
            "COGEV_SERVER_REQUIRE_AUTH": "false",
            "COGEV_LLM_PROVIDER": "fixture",
            "COGEV_LLM_FIXTURE": str(Path(__file__).parent / "fixtures" / "llm_fixture.json"),
            "COGEV_HERMETIC_TEST": "1",
            "COGEV_API_MAX_REQUEST_BYTES": "128",
            "COGEV_API_RATE_LIMIT_PER_MINUTE": "2",
        }
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(Path(__file__).parents[1]), env=env, text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data == {"too_large": 413, "first": 200, "second": 429}


def test_jobs_registry_public_rehydration_and_status_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(tmp_path / "api-runs"))
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_API_KEY", "secret-123456")
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "true")

    task_dir = _task_dir_for_request(tmp_path / "api-runs", "job-test")
    job = _set_job(
        "job-test",
        status="completed",
        model="cognitive-evolve-one-shot",
        task_dir=str(task_dir),
        artifact_root=str(task_dir),
        answer="done",
        nexus_data=_fake_nexus(3),
        cancellation_requested=False,
    )
    public = _job_public(job)
    assert public["answer"] == "done"
    assert public["cognitive_evolve"]["actual_rounds"] == 3
    assert _get_job("job-test")["status"] == "completed"
    assert _get_job("../bad") is None

    assert api_status()["auth_required"] is True
    assert api_status()["api_key_precedence"].startswith("COGEV_SERVER_API_KEY")
    assert status_cli() == 0
    assert "CognitiveEvolve" in capsys.readouterr().out


def test_terminal_jobs_are_pruned_without_deleting_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cognitive_evolve_runtime.api import jobs

    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(tmp_path / "api-runs"))
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_API_JOB_TTL_SECONDS", "60")
    monkeypatch.setenv("COGEV_API_MAX_TRACKED_JOBS", "1000")
    job_id = "job-prune-test"
    task_dir = _task_dir_for_request(tmp_path / "api-runs", job_id)
    _set_job(job_id, status="completed", task_dir=str(task_dir), artifact_root=str(task_dir), answer="done")
    with jobs._JOBS_LOCK:
        jobs._JOBS[job_id]["updated"] = 1
        removed = jobs._prune_jobs_locked(now=10_000)

    assert removed >= 1
    assert job_id not in jobs._JOBS
    assert (task_dir / "job-status.json").exists()


def test_jobqueue_is_lazy_stateless_facade_over_existing_registry() -> None:
    sys.modules.pop("cognitive_evolve_runtime.api.openai_compat", None)

    from cognitive_evolve_runtime.api import JobQueue as PackageJobQueue

    assert PackageJobQueue is JobQueue
    assert "cognitive_evolve_runtime.api.openai_compat" not in sys.modules

    queue = JobQueue()
    job_id = "jobqueue-facade-test"
    pushed = queue.push({"id": job_id, "status": "queued", "model": "fixture"})

    assert pushed["id"] == job_id
    assert _get_job(job_id)["status"] == "queued"
    _set_job(job_id, status="completed", answer="done")
    assert queue.get(job_id)["answer"] == "done"
    assert any(job["id"] == job_id for job in queue.snapshot())


def test_jobqueue_rejects_missing_id_and_handles_concurrent_pushes() -> None:
    queue = JobQueue()
    with pytest.raises(ValueError, match="job id"):
        queue.push({"status": "queued"})

    prefix = "jobqueue-thread-"

    def push(index: int) -> None:
        queue.push({"id": f"{prefix}{index}", "status": "queued", "index": index})

    threads = [threading.Thread(target=push, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = {job["id"]: job for job in queue.snapshot() if str(job.get("id", "")).startswith(prefix)}
    assert len(snapshot) == 8
    assert {job["index"] for job in snapshot.values()} == set(range(8))


def test_api_status_warns_when_auth_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "false")
    monkeypatch.setenv("COGEV_SERVER_HOST", "0.0.0.0")

    status = api_status()

    assert status["auth_required"] is False
    assert status["auth_warning"] == "authentication_disabled_on_non_loopback_host"


def test_stream_engine_chunks_final_and_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(tmp_path / "api-runs"))
    monkeypatch.setenv("COGEV_STREAM_HEARTBEAT_SECONDS", "1")

    class FakeResult:
        final_answer = "streamed answer"

        def to_dict(self) -> dict[str, Any]:
            return _fake_nexus(4)

    class FakeEngine:
        def run(self, prompt: str, context: dict[str, Any], progress_callback=None) -> FakeResult:
            if progress_callback:
                progress_callback({"stage": "fake", "status": "running"})
            return FakeResult()

    monkeypatch.setattr("cognitive_evolve_runtime.api.streaming.EngineOrchestrator", FakeEngine)
    chunks = list(_stream_engine_chunks("hello", request_id="chatcmpl-test", model="cognitive-evolve-one-shot", raw_request={"messages": []}))
    decoded = b"".join(chunks).decode("utf-8")
    assert "streamed answer" in decoded
    assert "cogev progress" in decoded
    assert "[DONE]" in decoded

    class FailingEngine:
        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr("cognitive_evolve_runtime.api.streaming.EngineOrchestrator", FailingEngine)
    decoded_error = b"".join(_stream_engine_chunks("hello", request_id="chatcmpl-error", model="cognitive-evolve-one-shot", raw_request={})).decode("utf-8")
    assert "pipeline failed" in decoded_error


def test_stream_engine_chunks_can_close_on_configured_lifetime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(tmp_path / "api-runs"))
    monkeypatch.setenv("COGEV_STREAM_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("COGEV_STREAM_MAX_SECONDS", "0.02")

    class SlowEngine:
        def run(self, prompt: str, context: dict[str, Any], progress_callback=None, cancellation_callback=None) -> Any:
            del prompt, context, progress_callback
            for _ in range(100):
                if cancellation_callback and cancellation_callback():
                    raise InterruptedError("cancelled")
                time.sleep(0.01)
            raise RuntimeError("timeout test did not cancel")

    monkeypatch.setattr("cognitive_evolve_runtime.api.streaming.EngineOrchestrator", SlowEngine)
    decoded = b"".join(_stream_engine_chunks("hello", request_id="chatcmpl-timeout", model="cognitive-evolve-one-shot", raw_request={})).decode("utf-8")

    assert "COGEV_STREAM_MAX_SECONDS" in decoded
    assert "[DONE]" in decoded


def test_stream_heartbeat_chunks_are_openai_stream_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COGEV_STREAM_HEARTBEAT_SECONDS", raising=False)
    assert _heartbeat_seconds() == 5.0
    monkeypatch.setenv("COGEV_STREAM_HEARTBEAT_SECONDS", "not-a-number")
    assert _heartbeat_seconds() == 5.0
    monkeypatch.setenv("COGEV_STREAM_HEARTBEAT_SECONDS", "0.2")
    assert _heartbeat_seconds() == 1.0

    decoded = b"".join(_stream_heartbeat_chunks(request_id="chatcmpl-heartbeat", model="cognitive-evolve-one-shot", created=123)).decode("utf-8")
    assert ": cogev heartbeat: wait" in decoded
    assert '"stage": "heartbeat"' in decoded
    assert '"think_display": "wait"' in decoded
    assert '"content": ""' in decoded
    assert '"reasoning_content": "wait\\n"' in decoded
    assert '"reasoning": "wait\\n"' in decoded
    assert '"thinking": "wait\\n"' in decoded
    assert "chain_of_thought_exposed" in decoded
