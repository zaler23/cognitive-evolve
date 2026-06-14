#!/usr/bin/env python3
"""Evidence planning and bounded execution for one-shot runs."""
from __future__ import annotations

import ipaddress
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from urllib import request as urlrequest
from urllib.parse import urlparse
from urllib.error import URLError
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..artifacts.store import _write_json
from ..nexus.task_types import CODE_TASK_TYPES, RESEARCH_TASK_TYPES, normalize_task_type


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidencePlanner:
    """Decide what evidence can replace user clarification."""

    def plan(self, assessment: dict[str, Any], budget: dict[str, Any] | None = None) -> dict[str, Any]:
        budget = budget or {}
        weak = assessment.get("weak_signals", {})
        complexity = assessment.get("complexity_assessment", {})
        task_type = normalize_task_type(assessment.get("task_type"))
        required_source_types: list[str] = []
        if weak.get("tool_or_code_reference"):
            required_source_types.extend(["local_files", "test_runner"])
        if weak.get("current_or_research_dependency") or float(complexity.get("external_evidence_dependency", 0)) >= 0.55:
            required_source_types.append("primary_or_current_external_sources")
        if task_type == "architecture_refactor_or_migration":
            required_source_types.extend(["repo_reader", "local_tests", "conflict_audit"])
        if task_type in CODE_TASK_TYPES:
            required_source_types.extend(["local_files", "test_runner"])
        if task_type in RESEARCH_TASK_TYPES:
            required_source_types.append("primary_or_current_external_sources")
        required_source_types = list(dict.fromkeys(required_source_types))
        required = budget.get("tool_verification") in {"required", "recommended"} or bool(required_source_types)
        return {
            "required": required,
            "status": "planned" if required else "not_required_for_current_answer",
            "tool_verification": budget.get("tool_verification", "as_needed"),
            "required_source_types": required_source_types,
            "read_only_first": True,
            "dangerous_write_actions": "disabled_output_plan_patch_or_rollback_instead",
            "claim_evidence_policy": "claims_that_affect_decision_need_source_test_or_explicit_uncertainty",
            "artifacts": {
                "sources": "evidence/sources.json",
                "tool_calls": "evidence/tool-calls.jsonl",
                "claim_evidence_ledger": "evidence/claim-evidence-ledger.json",
            },
        }


class EvidenceAdapter(Protocol):
    """Pluggable evidence source executor.

    Adapters are intentionally explicit: web search, MCP context, repository
    readers, or other source integrations can be registered without pretending
    that unavailable evidence was collected.
    """

    def execute(self, source_type: str, *, assessment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        ...



def _safe_evidence_payload(source_type: str, *, assessment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative adapter payload without secrets or arbitrary context."""

    return {
        "source_type": source_type,
        "query": assessment.get("real_objective") or assessment.get("surface_request"),
        "surface_request": assessment.get("surface_request"),
        "task_type": assessment.get("task_type"),
        "context": {
            key: value
            for key, value in context.items()
            if key in {"api_request_id", "interface", "workspace_root", "repo_root", "source_dir"}
        },
    }


def _iterative_json_depth(value: Any, *, max_depth: int | None = None) -> int:
    """Return JSON nesting depth without recursive Python calls."""

    max_seen = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > max_seen:
            max_seen = depth
        if max_depth is not None and depth > max_depth:
            return depth
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return max_seen


class MCPStdioEvidenceAdapter:
    """Call an operator-configured MCP/stdio evidence adapter.

    This intentionally does not embed a full MCP client or add a dependency. It
    sends a single JSON evidence request to a configured command over stdin and
    expects a JSON object on stdout, so teams can connect an MCP adapter, RAG
    gateway, or internal context service without shell execution.
    """

    def __init__(
        self,
        command: str | list[str],
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_000_000,
        max_json_depth: int = 24,
    ) -> None:
        self.command = self._parse_command(command)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_json_depth = max(4, int(max_json_depth))

    def execute(self, source_type: str, *, assessment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        payload = _safe_evidence_payload(source_type, assessment=assessment, context=context)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            completed = self._run_command(data)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"MCP evidence adapter failed: {exc}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MCP evidence adapter returned non-zero exit code {completed.returncode}: {stderr}")
        if len(completed.stdout) > self.max_response_bytes:
            raise RuntimeError("MCP evidence adapter response exceeded maximum byte limit.")
        try:
            decoded = json.loads(completed.stdout.decode("utf-8"))
        except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
            raise RuntimeError("MCP evidence adapter returned invalid or too deeply nested JSON.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("MCP evidence adapter must return a JSON object.")
        if _iterative_json_depth(decoded, max_depth=self.max_json_depth) > self.max_json_depth:
            raise RuntimeError("MCP evidence adapter response exceeded maximum JSON nesting depth.")
        decoded.setdefault("adapter", "mcp_stdio")
        decoded.setdefault("command_configured", True)
        return decoded

    def _run_command(self, data: bytes) -> subprocess.CompletedProcess[bytes]:
        inline = self._run_inline_python_for_hermetic_tests(data)
        if inline is not None:
            return inline
        return subprocess.run(
            self.command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )

    def _run_inline_python_for_hermetic_tests(self, data: bytes) -> subprocess.CompletedProcess[bytes] | None:
        hermetic = (os.environ.get("COGEV_HERMETIC_TEST", "").strip().lower() in {"1", "true", "yes"} or "PYTEST_CURRENT_TEST" in os.environ)
        if not hermetic or len(self.command) != 3 or self.command[1] != "-c":
            return None
        executable_name = Path(self.command[0]).name.lower()
        python_names = {"python", "python3", f"python{sys.version_info.major}", f"python{sys.version_info.major}.{sys.version_info.minor}"}
        if executable_name not in python_names and Path(self.command[0]) != Path(sys.executable):
            return None
        stdin = io.StringIO(data.decode("utf-8"))
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
        returncode = 0
        try:
            sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
            exec(compile(self.command[2], "<mcp-stdio-inline>", "exec"), {"__name__": "__main__"})
        except SystemExit as exc:
            code = exc.code
            returncode = int(code) if isinstance(code, int) else (0 if code is None else 1)
        except BaseException as exc:  # mirror subprocess failure as non-zero stderr in hermetic inline mode
            returncode = 1
            stderr.write(f"{type(exc).__name__}: {exc}")
        finally:
            sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
        return subprocess.CompletedProcess(
            self.command,
            returncode,
            stdout=stdout.getvalue().encode("utf-8"),
            stderr=stderr.getvalue().encode("utf-8"),
        )

    def _parse_command(self, command: str | list[str]) -> list[str]:
        if isinstance(command, list):
            parsed = [str(item) for item in command if str(item).strip()]
        else:
            parsed = shlex.split(str(command or ""))
        if not parsed:
            raise ValueError("COGEV_EVIDENCE_MCP_COMMAND must name a command without shell execution.")
        return parsed




class HTTPJsonEvidenceAdapter:
    """POST evidence requests to a configured read-only HTTP JSON gateway.

    This keeps CognitiveEvolve dependency-light while allowing teams to connect
    web search, MCP, RAG, or policy-approved source systems through a single
    external adapter endpoint. The adapter is opt-in via explicit registration or
    ``COGEV_EVIDENCE_HTTP_ENDPOINT``; it never invents evidence when absent.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 10.0,
        allow_private_networks: bool = False,
        max_response_bytes: int = 1_000_000,
        max_json_depth: int = 24,
    ) -> None:
        self.endpoint = self._validated_endpoint(endpoint, allow_private_networks=allow_private_networks)
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_json_depth = max(4, int(max_json_depth))

    def execute(self, source_type: str, *, assessment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        payload = _safe_evidence_payload(source_type, assessment=assessment, context=context)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        req = urlrequest.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310 - endpoint is explicit operator config
                raw = response.read(self.max_response_bytes + 1)
        except URLError as exc:
            raise RuntimeError(f"HTTP evidence adapter failed: {exc}") from exc
        if len(raw) > self.max_response_bytes:
            raise RuntimeError("HTTP evidence adapter response exceeded maximum byte limit.")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
            raise RuntimeError("HTTP evidence adapter returned invalid or too deeply nested JSON.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("HTTP evidence adapter must return a JSON object.")
        if self._json_depth(decoded) > self.max_json_depth:
            raise RuntimeError("HTTP evidence adapter response exceeded maximum JSON nesting depth.")
        decoded.setdefault("adapter", "http_json")
        decoded.setdefault("endpoint_configured", True)
        return decoded

    def _validated_endpoint(self, endpoint: str, *, allow_private_networks: bool) -> str:
        parsed = urlparse(str(endpoint).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("COGEV_EVIDENCE_HTTP_ENDPOINT must be an absolute http(s) URL.")
        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if not host:
            raise ValueError("COGEV_EVIDENCE_HTTP_ENDPOINT must include a hostname.")
        if allow_private_networks:
            return endpoint
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
            raise ValueError("HTTP evidence endpoint points to a local/private hostname; set COGEV_EVIDENCE_HTTP_ALLOW_PRIVATE=1 only for trusted local development.")
        addresses = self._resolve_endpoint_addresses(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        for address in addresses:
            if self._is_private_or_local_address(address):
                raise ValueError(
                    "HTTP evidence endpoint resolves to a private, local, or reserved IP; "
                    "set COGEV_EVIDENCE_HTTP_ALLOW_PRIVATE=1 only for trusted local development."
                )
        return endpoint

    def _resolve_endpoint_addresses(self, host: str, port: int) -> set[str]:
        """Resolve endpoint hostnames before allowing the HTTP evidence call."""

        try:
            ipaddress.ip_address(host)
            return {host}
        except ValueError:
            pass
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"HTTP evidence endpoint host could not be resolved safely: {host}") from exc
        addresses = {str(info[4][0]) for info in infos if info and len(info) >= 5 and info[4]}
        if not addresses:
            raise ValueError(f"HTTP evidence endpoint host resolved to no usable addresses: {host}")
        return addresses

    def _is_private_or_local_address(self, address: str) -> bool:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return True
        return (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        )

    def _json_depth(self, value: Any) -> int:
        """Return JSON nesting depth without recursive Python calls.

        This avoids raising ``RecursionError`` inside the safety check itself if
        a configured evidence gateway returns a deeply nested but byte-bounded
        JSON document. The traversal exits early once the configured policy is
        already exceeded.
        """

        return _iterative_json_depth(value, max_depth=self.max_json_depth)


class EvidenceExecutor:
    """Execute safe evidence collection that the plan requested.

    The executor is deliberately conservative: repository/file inspection is
    read-only, test execution is opt-in, and external/current-source requirements
    are recorded as adapter gaps instead of being silently pretended.
    """

    def __init__(
        self,
        *,
        task_dir: Path | None = None,
        workspace_root: Path | None = None,
        adapters: dict[str, EvidenceAdapter] | None = None,
    ) -> None:
        self.task_dir = task_dir
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.adapters: dict[str, EvidenceAdapter] = self._default_adapters()
        self.adapters.update(dict(adapters or {}))

    def _default_adapters(self) -> dict[str, EvidenceAdapter]:
        adapters: dict[str, EvidenceAdapter] = {}
        endpoint = (
            os.environ.get("COGEV_EVIDENCE_HTTP_ENDPOINT", "").strip()
            or os.environ.get("COGEV_EVIDENCE_WEB_ENDPOINT", "").strip()
        )
        if endpoint:
            timeout = self._env_float("COGEV_EVIDENCE_HTTP_TIMEOUT", 10.0)
            token = os.environ.get("COGEV_EVIDENCE_HTTP_BEARER_TOKEN", "").strip() or None
            allow_private = os.environ.get("COGEV_EVIDENCE_HTTP_ALLOW_PRIVATE", "").strip().lower() in {"1", "true", "yes"}
            max_bytes = self._env_int("COGEV_EVIDENCE_HTTP_MAX_RESPONSE_BYTES", 1_000_000)
            max_depth = self._env_int("COGEV_EVIDENCE_HTTP_MAX_JSON_DEPTH", 24)
            adapters["primary_or_current_external_sources"] = HTTPJsonEvidenceAdapter(
                endpoint,
                bearer_token=token,
                timeout_seconds=timeout,
                allow_private_networks=allow_private,
                max_response_bytes=max_bytes,
                max_json_depth=max_depth,
            )
        mcp_command = os.environ.get("COGEV_EVIDENCE_MCP_COMMAND", "").strip()
        if mcp_command:
            timeout = self._env_float("COGEV_EVIDENCE_MCP_TIMEOUT", 10.0)
            max_bytes = self._env_int("COGEV_EVIDENCE_MCP_MAX_RESPONSE_BYTES", 1_000_000)
            max_depth = self._env_int("COGEV_EVIDENCE_MCP_MAX_JSON_DEPTH", 24)
            mcp_adapter = MCPStdioEvidenceAdapter(
                mcp_command,
                timeout_seconds=timeout,
                max_response_bytes=max_bytes,
                max_json_depth=max_depth,
            )
            adapters.setdefault("mcp_context", mcp_adapter)
            adapters.setdefault("primary_or_current_external_sources", mcp_adapter)
        return adapters

    def register_adapter(self, source_type: str, adapter: EvidenceAdapter) -> None:
        self.adapters[str(source_type)] = adapter

    def request(self, evidence_request: dict[str, Any], assessment: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one loop-internal evidence request.

        Candidate critique/reflection can surface a concrete evidence gap during
        evolution.  This method upgrades the executor from a single pre-search
        step into a bounded service that the Nexus loop can call without
        inventing evidence or reopening user clarification.
        """

        source_type = str(evidence_request.get("source_type") or "mcp_context")
        plan = {
            "required": True,
            "status": "loop_internal_requested",
            "required_source_types": [source_type],
            "read_only_first": True,
            "origin": "nexus_nested_evidence_request",
            "request": {
                key: evidence_request.get(key)
                for key in ["status", "candidate_id", "source_type", "query", "priority", "finding_id"]
                if key in evidence_request
            },
        }
        result = self.execute(plan, assessment, context=context)
        result["request"] = plan["request"]
        result["source"] = "evidence_service_request"
        return result

    def execute(self, plan: dict[str, Any], assessment: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        workspace = self._workspace_from_context(context)
        source_types = [str(item) for item in plan.get("required_source_types", [])]
        results: list[dict[str, Any]] = []
        if not plan.get("required"):
            return {
                "status": "not_required",
                "executed_source_types": [],
                "results": [],
                "workspace_root": str(workspace),
                "read_only": True,
            }

        if any(kind in source_types for kind in ["local_files", "repo_reader", "conflict_audit"]):
            results.append(self._read_repo_manifest(workspace, assessment))
        if any(kind in source_types for kind in ["test_runner", "local_tests"]):
            results.append(self._maybe_run_tests(workspace))
        if "mcp_context" in source_types:
            results.append(self._execute_adapter_or_gap("mcp_context", assessment, context))
        if "primary_or_current_external_sources" in source_types:
            results.append(self._execute_adapter_or_gap("primary_or_current_external_sources", assessment, context))

        return {
            "status": "executed_with_gaps" if any(item.get("status") != "ok" for item in results) else "executed",
            "executed_source_types": source_types,
            "results": results,
            "workspace_root": str(workspace),
            "read_only": True,
            "generated_at": _now(),
        }

    def _workspace_from_context(self, context: dict[str, Any]) -> Path:
        for key in ["workspace_root", "repo_root", "source_dir"]:
            raw = context.get(key)
            if raw:
                return Path(str(raw)).expanduser().resolve()
        return self.workspace_root

    def _read_repo_manifest(self, workspace: Path, assessment: dict[str, Any]) -> dict[str, Any]:
        if not workspace.exists():
            return {
                "source_type": "repo_reader",
                "status": "missing_workspace",
                "summary": f"Workspace path does not exist: {workspace}",
                "files": [],
            }
        interesting_names = {
            "pyproject.toml",
            "package.json",
            "README.md",
            "AGENTS.md",
            "pytest.ini",
        }
        interesting_prefixes = ("cognitive_evolve_runtime/", "tests/", "docs/", "adapters/")
        excluded_dirs = {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "node_modules",
            "dist",
            "build",
        }
        max_files = self._env_int("COGEV_REPO_MANIFEST_MAX_FILES", 5000)
        max_entries = self._env_int("COGEV_REPO_MANIFEST_MAX_ENTRIES", 5000)
        timeout_seconds = max(0.1, self._env_float("COGEV_REPO_MANIFEST_TIMEOUT", 5.0))
        deadline = time.monotonic() + timeout_seconds

        files: list[dict[str, Any]] = []
        python_files = 0
        test_files = 0
        total_files = 0
        entries_seen = 0
        truncated_reason = ""
        queue: deque[Path] = deque([workspace])
        while queue and len(files) < max_files:
            if time.monotonic() > deadline:
                truncated_reason = "timeout"
                break
            if entries_seen >= max_entries:
                truncated_reason = "entry_limit"
                break
            directory = queue.popleft()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entries_seen += 1
                        if entries_seen >= max_entries:
                            truncated_reason = "entry_limit"
                            break
                        if time.monotonic() > deadline:
                            truncated_reason = "timeout"
                            break
                        name = entry.name
                        if name in excluded_dirs:
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                queue.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                        except OSError:
                            continue
                        path = Path(entry.path)
                        try:
                            rel = path.relative_to(workspace)
                        except ValueError:
                            continue
                        rel_text = rel.as_posix()
                        total_files += 1
                        if path.suffix == ".py":
                            python_files += 1
                        if "tests" in rel.parts and path.suffix == ".py":
                            test_files += 1
                        if path.name in interesting_names or rel_text.startswith(interesting_prefixes):
                            try:
                                stat = path.stat()
                            except OSError:
                                continue
                            files.append({"path": rel_text, "bytes": stat.st_size})
                            if len(files) >= max_files:
                                truncated_reason = "file_limit"
                                break
                    if truncated_reason:
                        break
            except OSError:
                continue

        truncated = bool(truncated_reason) or bool(queue and len(files) >= max_files)
        return {
            "source_type": "repo_reader",
            "status": "partial" if truncated else "ok",
            "summary": "Read-only repository manifest collected with bounded directory scanning.",
            "workspace": str(workspace),
            "counts": {
                "entries_seen": entries_seen,
                "total_files_seen": total_files,
                "python_files_seen": python_files,
                "test_files_seen": test_files,
            },
            "limits": {"max_files": max_files, "max_entries": max_entries, "timeout_seconds": timeout_seconds},
            "truncated": truncated,
            "truncated_reason": truncated_reason,
            "files": files,
            "claims_supported": ["repository structure was inspected read-only"],
            "claims_unverified": [],
        }

    def _execute_adapter_or_gap(self, source_type: str, assessment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        adapter = self.adapters.get(source_type)
        if adapter is None:
            return {
                "source_type": source_type,
                "status": "adapter_required",
                "summary": "External/current facts require an explicit web/MCP/source adapter; no fabricated evidence was added.",
                "claims_supported": [],
                "claims_unverified": [assessment.get("real_objective", "external fact dependency")],
            }
        try:
            result = adapter.execute(source_type, assessment=assessment, context=context)
        except Exception as exc:
            return {
                "source_type": source_type,
                "status": "adapter_failed",
                "summary": str(exc),
                "claims_supported": [],
                "claims_unverified": [assessment.get("real_objective", "external fact dependency")],
            }
        result = dict(result or {})
        result.setdefault("source_type", source_type)
        result.setdefault("status", "ok")
        result.setdefault("claims_supported", [])
        result.setdefault("claims_unverified", [])
        return result

    def _env_int(self, name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default))))
        except ValueError:
            return default

    def _env_float(self, name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _maybe_run_tests(self, workspace: Path) -> dict[str, Any]:
        enabled = os.environ.get("COGEV_ENABLE_TEST_RUNNER", "").strip().lower() in {"1", "true", "yes"}
        command = [sys.executable, "-m", "pytest", "-q"]
        command_display = "python -m pytest -q"
        if not enabled:
            return {
                "source_type": "test_runner",
                "status": "skipped_disabled",
                "summary": "Test runner is available but disabled. Set COGEV_ENABLE_TEST_RUNNER=1 to execute pytest.",
                "command": command_display,
                "claims_supported": [],
                "claims_unverified": ["tests pass in this workspace"],
            }
        hermetic = (os.environ.get("COGEV_HERMETIC_TEST", "").strip().lower() in {"1", "true", "yes"} or "PYTEST_CURRENT_TEST" in os.environ)
        allow_nested = os.environ.get("COGEV_ALLOW_NESTED_TEST_RUNNER", "").strip().lower() in {"1", "true", "yes"}
        run_is_monkeypatched = getattr(subprocess.run, "__module__", "subprocess") != "subprocess"
        if hermetic and not allow_nested and not run_is_monkeypatched:
            return {
                "source_type": "test_runner",
                "status": "skipped_disabled",
                "summary": "Hermetic test mode disables recursive pytest execution unless COGEV_ALLOW_NESTED_TEST_RUNNER=1.",
                "command": command_display,
                "claims_supported": [],
                "claims_unverified": ["tests pass in this workspace"],
            }
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(os.environ.get("COGEV_TEST_RUNNER_TIMEOUT", "60")),
                check=False,
            )
        except Exception as exc:
            return {
                "source_type": "test_runner",
                "status": "failed_to_execute",
                "summary": str(exc),
                "command": command_display,
                "claims_supported": [],
                "claims_unverified": ["tests pass in this workspace"],
            }
        output = completed.stdout[-6000:]
        return {
            "source_type": "test_runner",
            "status": "ok" if completed.returncode == 0 else "failed",
            "summary": f"pytest exited with code {completed.returncode}",
            "command": command_display,
            "returncode": completed.returncode,
            "output_tail": output,
            "claims_supported": ["tests passed"] if completed.returncode == 0 else [],
            "claims_unverified": [] if completed.returncode == 0 else ["tests pass in this workspace"],
        }


class EvidenceStore:
    """Task-local evidence artifact writer."""

    def __init__(self, task_dir: Path | None = None) -> None:
        self.task_dir = task_dir

    def write_plan(
        self,
        plan: dict[str, Any],
        claims: list[dict[str, Any]] | None = None,
        execution: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        if self.task_dir is None:
            return {}
        evidence_dir = self.task_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        claims = claims or []
        execution = execution or {}
        sources = {
            "status": "planned",
            "source_types": plan.get("required_source_types", []),
            "read_only_first": plan.get("read_only_first", True),
            "execution_status": execution.get("status", "not_executed"),
            "workspace_root": execution.get("workspace_root"),
            "results": execution.get("results", []),
        }
        ledger = {
            "claims": claims,
            "policy": plan.get("claim_evidence_policy"),
            "unverified_claim_handling": "mark_uncertain_or_remove_from_decisive_path",
            "execution_status": execution.get("status", "not_executed"),
            "execution_results": execution.get("results", []),
        }
        _write_json(evidence_dir / "sources.json", sources)
        _write_json(evidence_dir / "claim-evidence-ledger.json", ledger)
        tool_calls = evidence_dir / "tool-calls.jsonl"
        records = execution.get("results") if isinstance(execution.get("results"), list) else []
        with tool_calls.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"time": _now(), **record}, ensure_ascii=False) + "\n")
        return {
            "sources": str((evidence_dir / "sources.json").relative_to(self.task_dir)),
            "claim_evidence_ledger": str((evidence_dir / "claim-evidence-ledger.json").relative_to(self.task_dir)),
            "tool_calls": str(tool_calls.relative_to(self.task_dir)),
        }


__all__ = [
    "EvidenceAdapter",
    "EvidenceExecutor",
    "EvidencePlanner",
    "EvidenceStore",
    "HTTPJsonEvidenceAdapter",
    "MCPStdioEvidenceAdapter",
]
