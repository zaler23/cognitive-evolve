"""`cogev attack` entrypoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


def run_attack(*, problem_path: str | None, budget: int | None, out_dir: str | None, resume_dir: str | None = None, offline: bool = False) -> int:
    out = Path(out_dir or resume_dir or "cogev-attack-out").resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        runtime = NexusRuntime(model=None, output_dir=out) if offline else NexusRuntime.with_configured_llm(output_dir=out)
    except Exception as exc:
        print(f"Failed to configure LLM runtime: {exc}", file=sys.stderr)
        return 2
    try:
        if resume_dir:
            result = runtime.resume_from_checkpoint(max_rounds=budget)
        else:
            if not problem_path:
                print("Missing problem.yaml", file=sys.stderr)
                return 2
            problem = _load_problem(Path(problem_path))
            prompt = _problem_prompt(problem)
            adaptive = problem.get("adaptive") if isinstance(problem.get("adaptive"), dict) else None
            result = runtime.run_text(prompt, user_goal=str(problem.get("goal") or problem.get("objective") or prompt[:200]), max_rounds=max(1, int(budget or problem.get("budget") or 1)), adaptive_config=adaptive)
    except Exception as exc:
        print(f"cogev attack failed: {exc}", file=sys.stderr)
        return 1
    data = result.to_dict()
    (out / "graded-output.json").write_text(json.dumps(data.get("evolution", {}).get("graded_output") or data.get("evolution", {}).get("synthesis", {}).get("closure_certificate", {}).get("graded_output") or {}, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "attack-result.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completion_status": data.get("evolution", {}).get("completion_status"), "graded_output": str(out / "graded-output.json")}, ensure_ascii=False, sort_keys=True))
    return 0


def _load_problem(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return {"problem": data}
    return {}


def _problem_prompt(problem: dict[str, Any]) -> str:
    for key in ("problem", "prompt", "objective", "goal"):
        value = problem.get(key)
        if value:
            return str(value)
    return json.dumps(problem, ensure_ascii=False, sort_keys=True)


__all__ = ["run_attack"]
