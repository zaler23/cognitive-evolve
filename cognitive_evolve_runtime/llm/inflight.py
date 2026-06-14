#!/usr/bin/env python3
"""Nexus LLM provider inflight-call registry."""
from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class InflightCall:
    call_id: str
    provider: str
    request_type: str
    started_at_monotonic: float
    pid: int | None = None
    status: str = "starting"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["age_seconds"] = round(time.monotonic() - self.started_at_monotonic, 3)
        return data


class ProviderInflightRegistry:
    """Track active provider calls and kill subprocesses on timeout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, InflightCall] = {}

    def start(self, *, provider: str, request_type: str, metadata: dict[str, Any] | None = None) -> str:
        call_id = f"{provider}:{request_type}:{uuid.uuid4().hex[:10]}"
        call = InflightCall(
            call_id=call_id,
            provider=provider,
            request_type=request_type,
            started_at_monotonic=time.monotonic(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._calls[call_id] = call
        return call_id

    def set_pid(self, call_id: str, pid: int | None) -> None:
        with self._lock:
            call = self._calls.get(call_id)
            if call:
                call.pid = pid
                call.status = "running"

    def finish(self, call_id: str, *, status: str = "finished", metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            call = self._calls.get(call_id)
            if call:
                call.status = status
                if metadata:
                    call.metadata.update(metadata)
                self._calls.pop(call_id, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_count": len(self._calls),
                "active": [call.to_dict() for call in self._calls.values()],
            }

    def run_subprocess(
        self,
        argv: list[str],
        *,
        provider: str,
        request_type: str,
        timeout: float,
        text: bool = True,
        capture_output: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        call_id = self.start(provider=provider, request_type=request_type, metadata=metadata)
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(argv, text=text, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE if capture_output else None)  # noqa: S603 - argv is constructed by runtime, no shell.
            self.set_pid(call_id, proc.pid)
            stdout, stderr = proc.communicate(timeout=timeout)
            completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
            self.finish(call_id, status="finished", metadata={"returncode": proc.returncode})
            return completed
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                proc.kill()
                stdout, stderr = proc.communicate()
                exc.output = stdout
                exc.stderr = stderr
            self.finish(call_id, status="killed_timeout", metadata={"timeout": timeout})
            raise
        except Exception:
            self.finish(call_id, status="error")
            raise


_REGISTRY = ProviderInflightRegistry()


def provider_inflight_registry() -> ProviderInflightRegistry:
    return _REGISTRY


def provider_inflight_status() -> dict[str, Any]:
    return _REGISTRY.snapshot()


__all__ = [
    "InflightCall",
    "ProviderInflightRegistry",
    "provider_inflight_registry",
    "provider_inflight_status",
]
