"""Small local verifier adapters exposed through one tool suite."""
from __future__ import annotations

import compileall
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .feedback import ToolFeedback
from .runner import ToolRunner


@dataclass
class ToolCommandSpec:
    tool_id: str
    command: list[str]
    required_executable: str | None = None
    optional: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return self.required_executable is None or shutil.which(self.required_executable) is not None


class LocalToolSuite:
    """Run supported offline verifiers without hard-coding task semantics."""

    def __init__(self, *, runner: ToolRunner | None = None) -> None:
        self.runner = runner or ToolRunner()

    def default_specs_for_project(self, root: str | Path, *, include_tests: bool = False) -> list[ToolCommandSpec]:
        root_path = Path(root)
        specs: list[ToolCommandSpec] = []
        if any(path.suffix == ".py" for path in root_path.rglob("*.py")):
            specs.append(ToolCommandSpec("compileall", [sys.executable, "-m", "compileall", "-q", "."], required_executable=sys.executable, optional=False))
        if include_tests and ((root_path / "tests").exists() or (root_path / "pyproject.toml").exists() or (root_path / "pytest.ini").exists()):
            specs.append(ToolCommandSpec("pytest", [sys.executable, "-m", "pytest", "-q"], required_executable=sys.executable, optional=True))
        if (root_path / "pyproject.toml").exists() or (root_path / "ruff.toml").exists():
            specs.append(ToolCommandSpec("ruff", ["ruff", "check", "."], required_executable="ruff", optional=True))
            specs.append(ToolCommandSpec("mypy", ["mypy", "."], required_executable="mypy", optional=True))
        if (root_path / "package.json").exists():
            specs.append(ToolCommandSpec("npm_test", ["npm", "test"], required_executable="npm", optional=True))
        return specs

    def run_specs(self, specs: list[ToolCommandSpec], *, cwd: str | Path, timeout_seconds: float | None = None) -> list[ToolFeedback]:
        feedback: list[ToolFeedback] = []
        for spec in specs:
            if spec.tool_id == "compileall":
                feedback.append(_run_compileall_in_process(cwd))
                continue
            if not spec.available():
                if not spec.optional:
                    feedback.append(
                        ToolFeedback(
                            tool_id=spec.tool_id,
                            status="unavailable",
                            diagnostics=[f"required executable not found: {spec.required_executable}"],
                            failed_fragments=["missing_executable"],
                            confidence=0.0,
                        )
                    )
                continue
            result = self.runner.run(spec.command, cwd=cwd, timeout_seconds=timeout_seconds)
            result.tool_id = spec.tool_id
            feedback.append(result)
        return feedback


def _run_compileall_in_process(cwd: str | Path) -> ToolFeedback:
    start = time.monotonic()
    root = Path(cwd)
    ok = compileall.compile_dir(str(root), quiet=1, force=False, legacy=False)
    return ToolFeedback(
        tool_id="compileall",
        status="passed" if ok else "failed",
        diagnostics=[] if ok else ["compileall reported at least one syntax/import compilation failure"],
        verified_fragments=["compileall_passed"] if ok else [],
        failed_fragments=[] if ok else ["compileall_failed"],
        cost={"seconds": round(time.monotonic() - start, 4)},
        confidence=1.0 if ok else 0.9,
        raw_output_ref=f"compileall.compile_dir({root}) -> {ok}",
    )


__all__ = ["ToolCommandSpec", "LocalToolSuite"]
