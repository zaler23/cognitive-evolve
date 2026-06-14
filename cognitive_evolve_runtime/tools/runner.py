"""Offline local command runner."""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Any

from .feedback import ToolFeedback
from .verifier_environment import VerifierEnvironment

DEFAULT_ALLOWED_EXECUTABLES = {"python", "python3", "pytest", "ruff", "mypy", "npm", Path(sys.executable).name}
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_FILE_SIZE_LIMIT_MB = 256
DEFAULT_OPEN_FILE_LIMIT = 256


class ToolRunner:
    def __init__(self, *, timeout_seconds: float = 60.0, hermetic: bool = True, allowed_executables: set[str] | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.hermetic = hermetic
        self.allowed_executables = set(allowed_executables or DEFAULT_ALLOWED_EXECUTABLES)

    def run(self, command: str | list[str], *, cwd: str | Path, env: dict[str, str] | None = None, timeout_seconds: float | None = None) -> ToolFeedback:
        start = time.monotonic()
        args = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
        allowed, reason = self._command_allowed(args)
        if not allowed:
            return ToolFeedback(
                tool_id=" ".join(args),
                status="blocked",
                diagnostics=[reason],
                failed_fragments=["command_not_allowlisted"],
                cost={"seconds": round(time.monotonic() - start, 4)},
                confidence=1.0,
                raw_output_ref=reason,
            )
        verifier_env = VerifierEnvironment.for_path(cwd, extra_env=env, hermetic=self.hermetic)
        try:
            proc = subprocess.run(
                args,
                cwd=verifier_env.cwd,
                env=verifier_env.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds or self.timeout_seconds,
                check=False,
                start_new_session=True,
                preexec_fn=_resource_limiter(timeout_seconds or self.timeout_seconds),
            )
            status = "passed" if proc.returncode == 0 else "failed"
            diagnostics = [line for line in (proc.stderr or proc.stdout).splitlines() if line.strip()][:50]
            return ToolFeedback(
                tool_id=" ".join(args),
                status=status,
                diagnostics=diagnostics,
                verified_fragments=[] if proc.returncode else ["command_exit_0"],
                failed_fragments=[] if proc.returncode == 0 else ["nonzero_exit"],
                cost={"seconds": round(time.monotonic() - start, 4), "returncode": proc.returncode},
                confidence=1.0 if proc.returncode == 0 else 0.8,
                raw_output_ref=(proc.stdout + proc.stderr)[-4000:],
            )
        except subprocess.TimeoutExpired as exc:
            return ToolFeedback(
                tool_id=" ".join(args),
                status="timeout",
                diagnostics=[f"timeout after {timeout_seconds or self.timeout_seconds} seconds"],
                failed_fragments=["timeout"],
                cost={"seconds": round(time.monotonic() - start, 4)},
                confidence=0.5,
                raw_output_ref=str(exc),
            )
        except OSError as exc:
            return ToolFeedback(
                tool_id=" ".join(args),
                status="error",
                diagnostics=[str(exc)],
                failed_fragments=["runner_error"],
                cost={"seconds": round(time.monotonic() - start, 4)},
                confidence=0.0,
                raw_output_ref=str(exc),
            )

    def _command_allowed(self, args: list[str]) -> tuple[bool, str]:
        if not args:
            return False, "empty verifier command"
        executable = Path(args[0])
        name = executable.name
        if str(executable) == sys.executable or name in self.allowed_executables:
            return True, ""
        return False, f"verifier command not allowlisted: {args[0]}"


def _resource_limiter(timeout_seconds: float):
    """Return a POSIX preexec_fn that bounds verifier subprocess damage."""

    try:
        import resource
    except Exception:
        return None

    def _limit() -> None:
        cpu_seconds = max(1, int(float(timeout_seconds or 1)) + 1)
        memory_mb = _positive_int(os.environ.get("COGEV_VERIFIER_MEMORY_MB"), DEFAULT_MEMORY_LIMIT_MB)
        file_mb = _positive_int(os.environ.get("COGEV_VERIFIER_FILE_SIZE_MB"), DEFAULT_FILE_SIZE_LIMIT_MB)
        open_files = _positive_int(os.environ.get("COGEV_VERIFIER_OPEN_FILES"), DEFAULT_OPEN_FILE_LIMIT)
        _safe_setrlimit(resource, resource.RLIMIT_CPU, cpu_seconds, cpu_seconds)
        if hasattr(resource, "RLIMIT_AS"):
            _safe_setrlimit(resource, resource.RLIMIT_AS, memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
        if hasattr(resource, "RLIMIT_FSIZE"):
            _safe_setrlimit(resource, resource.RLIMIT_FSIZE, file_mb * 1024 * 1024, file_mb * 1024 * 1024)
        if hasattr(resource, "RLIMIT_NOFILE"):
            _safe_setrlimit(resource, resource.RLIMIT_NOFILE, open_files, open_files)
        if hasattr(resource, "RLIMIT_NPROC"):
            _safe_setrlimit(resource, resource.RLIMIT_NPROC, 64, 64)

    return _limit


def _safe_setrlimit(resource_module: Any, limit_name: int, soft: int, hard: int) -> None:
    try:
        current_soft, current_hard = resource_module.getrlimit(limit_name)
        if current_hard not in (-1, resource_module.RLIM_INFINITY):
            hard = min(hard, int(current_hard))
        if current_soft not in (-1, resource_module.RLIM_INFINITY):
            soft = min(soft, int(current_soft))
        resource_module.setrlimit(limit_name, (soft, hard))
    except Exception:
        return


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


__all__ = ["DEFAULT_ALLOWED_EXECUTABLES", "ToolRunner"]
