from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognitive_evolve_runtime.evidence.planner as ep
import cognitive_evolve_runtime.nexus.evaluation as evaluation


class _Adapter:
    def __init__(self, result: dict | Exception) -> None:
        self.result = result

    def execute(self, source_type: str, *, assessment: dict, context: dict) -> dict:
        if isinstance(self.result, Exception):
            raise self.result
        return dict(self.result)


def test_evidence_planner_adapters_executor_and_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    planner = ep.EvidencePlanner()
    plan = planner.plan(
        {
            "surface_request": "Refactor the code and cite current docs",
            "real_objective": "audit architecture",
            "task_type": "architecture_refactor_or_migration",
            "weak_signals": {"tool_or_code_reference": True, "current_or_research_dependency": True},
            "complexity_assessment": {"external_evidence_dependency": 0.9},
        },
        {"tool_verification": "required"},
    )
    assert plan["required"] is True
    assert {"local_files", "test_runner", "repo_reader", "conflict_audit", "primary_or_current_external_sources"}.issubset(set(plan["required_source_types"]))
    assert planner.plan({"task_type": "general"}, {})["status"] == "not_required_for_current_answer"

    safe_payload = ep._safe_evidence_payload(
        "repo_reader",
        assessment={"real_objective": "objective", "surface_request": "surface", "task_type": "code"},
        context={"api_request_id": "r1", "secret": "drop", "workspace_root": str(tmp_path)},
    )
    assert safe_payload["query"] == "objective"
    assert "secret" not in safe_payload["context"]
    assert ep._iterative_json_depth({"a": [{"b": [1]}]}) >= 4
    assert ep._iterative_json_depth({"a": {"b": {"c": 1}}}, max_depth=2) > 2

    with pytest.raises(ValueError):
        ep.MCPStdioEvidenceAdapter("")
    mcp = ep.MCPStdioEvidenceAdapter([
        "python3",
        "-c",
        "import sys,json; json.load(sys.stdin); print('{\"claims_supported\":[\"ok\"]}')",
    ])
    mcp_result = mcp.execute("mcp_context", assessment={"real_objective": "x"}, context={})
    assert mcp_result["adapter"] == "mcp_stdio"
    failing_mcp = ep.MCPStdioEvidenceAdapter(["python3", "-c", "import sys; sys.exit(2)"])
    with pytest.raises(RuntimeError):
        failing_mcp.execute("mcp_context", assessment={}, context={})

    with pytest.raises(ValueError):
        ep.HTTPJsonEvidenceAdapter("not-a-url")
    with pytest.raises(ValueError):
        ep.HTTPJsonEvidenceAdapter("http://localhost:8765")

    monkeypatch.setattr(ep.HTTPJsonEvidenceAdapter, "_resolve_endpoint_addresses", lambda self, host, port: {"93.184.216.34"})

    class _Response:
        def __init__(self, body: bytes) -> None:
            self.body = body
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def read(self, amount: int) -> bytes:
            return self.body

    def fake_urlopen(req, timeout: float):
        assert req.headers["Content-type"] == "application/json"
        return _Response(b'{"claims_supported":["source checked"]}')

    monkeypatch.setattr(ep.urlrequest, "urlopen", fake_urlopen)
    http = ep.HTTPJsonEvidenceAdapter("https://example.com/evidence", bearer_token="tok", max_response_bytes=100)
    http_result = http.execute("primary_or_current_external_sources", assessment={"surface_request": "x"}, context={})
    assert http_result["adapter"] == "http_json"
    assert http._is_private_or_local_address("127.0.0.1") is True
    assert http._is_private_or_local_address("bad-address") is True

    monkeypatch.setattr(ep.urlrequest, "urlopen", lambda *a, **k: _Response(b'x' * 20))
    tiny = ep.HTTPJsonEvidenceAdapter("https://example.com/evidence", max_response_bytes=8)
    with pytest.raises(RuntimeError):
        tiny.execute("primary_or_current_external_sources", assessment={}, context={})

    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.setenv("COGEV_REPO_MANIFEST_MAX_ENTRIES", "bad-int")
    executor = ep.EvidenceExecutor(workspace_root=tmp_path, adapters={
        "mcp_context": _Adapter({"summary": "mcp ok"}),
        "primary_or_current_external_sources": _Adapter(RuntimeError("down")),
    })
    not_required = executor.execute({"required": False}, {}, {})
    assert not_required["status"] == "not_required"
    executed = executor.execute(
        {"required": True, "required_source_types": ["repo_reader", "test_runner", "mcp_context", "primary_or_current_external_sources"]},
        {"real_objective": "objective"},
        {"workspace_root": str(tmp_path)},
    )
    statuses = {item["source_type"]: item["status"] for item in executed["results"]}
    assert statuses["repo_reader"] in {"ok", "partial"}
    assert statuses["test_runner"] == "skipped_disabled"
    assert statuses["mcp_context"] == "ok"
    assert statuses["primary_or_current_external_sources"] == "adapter_failed"

    request_result = executor.request({"source_type": "mcp_context", "candidate_id": "c1", "query": "q"}, {"real_objective": "objective"})
    assert request_result["source"] == "evidence_service_request"
    assert request_result["request"]["candidate_id"] == "c1"

    monkeypatch.setenv("COGEV_ENABLE_TEST_RUNNER", "1")
    monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1 passed\n"))
    assert executor._maybe_run_tests(tmp_path)["status"] == "ok"
    monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert executor._maybe_run_tests(tmp_path)["status"] == "failed_to_execute"

    assert ep.EvidenceStore().write_plan(plan) == {}
    task_dir = tmp_path / "task"
    artifacts = ep.EvidenceStore(task_dir).write_plan(plan, claims=[{"claim": "x"}], execution=executed)
    assert set(artifacts) == {"sources", "claim_evidence_ledger", "tool_calls"}
    assert (task_dir / artifacts["sources"]).exists()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_native_eval_check_branches(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    assert evaluation.eval_check(task, {"id": "native_eval_output"})["passed"] is True

    (task / "result.md").write_text("alpha beta gamma", encoding="utf-8")
    _write_json(task / "report.json", {"status": "ok", "score": 3, "items": ["a", "b"]})
    generic = evaluation.eval_check(task, {
        "id": "generic",
        "required_files": ["result.md"],
        "text_file": "result.md",
        "contains_any": ["delta", "alpha"],
        "contains_all": ["beta"],
        "contains_all_groups": [["alpha", "gamma"]],
        "forbidden_any": ["forbidden"],
        "json_file": "report.json",
        "json_required_fields": ["status"],
        "json_equals": {"status": "ok"},
        "json_number_at_least": {"score": 2},
        "json_array_contains": {"items": ["a"]},
        "json_array_min_length": {"items": 2},
    })
    assert generic["passed"] is True
    failing = evaluation.eval_check(task, {
        "id": "generic-fail",
        "required_files": ["missing.md"],
        "text_file": "result.md",
        "contains_all": ["absent"],
        "forbidden_any": ["alpha"],
        "json_file": "missing.json",
    })
    assert failing["passed"] is False
    assert "missing:missing.md" in failing["errors"]
    assert "missing_json:missing.json" in failing["json_errors"]


def test_native_optimizer_success_and_failure_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert evaluation.native_optimize_run(str(missing)) == 1

    task = tmp_path / "task"
    task.mkdir()
    source = tmp_path / "prompt.txt"
    source.write_text("base prompt", encoding="utf-8")
    assert evaluation.native_optimize_run(str(task), source=str(source)) == 0
    report = json.loads((task / "evaluations" / "prompt-optimization-report.json").read_text(encoding="utf-8"))
    assert report["runtime_architecture"] == "nexus"
    assert report["variant_count"] >= 1

    empty = tmp_path / "empty"
    empty.mkdir()
    assert evaluation.native_optimize_run(str(empty)) == 1
