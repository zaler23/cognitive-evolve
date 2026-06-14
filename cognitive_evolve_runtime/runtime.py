"""Thin CLI-facing Nexus runtime commands.

This module is intentionally not a second runtime.  It prepares task artifacts,
resolves Nexus semantic/budget hints, invokes ``NexusRuntime``, and writes the
canonical runtime state.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .artifacts.store import _append_trace, _format_list, _load_json, _now, _task_dir, _write_json
from .artifacts.task_files import _task_seed_prompt
from .llm.env import LLMConfigurationError, LLMResponseError, require_llm_config
from .nexus.evaluation import write_runtime_validation_report
from .nexus.budget_factory import evolution_budget_from_round_budget
from .nexus.difficulty_estimator import (
    runtime_entry_difficulty as _runtime_entry_difficulty,
    runtime_profile_from_difficulty as _runtime_profile_from_difficulty,
    runtime_round_budget as _runtime_round_budget,
)
from .nexus.model_adapter import StructuredModelAdapter
from .nexus.runtime import NexusRuntime
from .nexus.semantics import DEFAULT_CAPABILITIES, classify, ensure_enhanced_task_contract, required_capabilities
from .nexus.state import nexus_runtime_state
from .nexus.state_contract import RUNTIME_PATH, normalize_runtime_state



def _canonical_capabilities(capability_ids: list[str]) -> list[str]:
    selected = list(dict.fromkeys(capability_ids))
    for capability_id in reversed(DEFAULT_CAPABILITIES):
        if capability_id not in selected:
            selected.insert(0, capability_id)
    return list(dict.fromkeys(selected))


def _resolve_runtime_model(*, offline: bool) -> tuple[object | None, int]:
    if offline:
        return None, 0
    try:
        require_llm_config()
        return StructuredModelAdapter.from_configured_llm(), 0
    except LLMConfigurationError as exc:
        print(f"LLM configuration required for runtime run: {exc}", file=sys.stderr)
        print("Use --offline for an explicit deterministic local run.", file=sys.stderr)
        return None, 2
    except LLMResponseError as exc:
        print(f"LLM runtime initialization failed: {exc}", file=sys.stderr)
        return None, 2


def runtime_run(path: str | None, prompt: str | None, activate_all: bool = False, rounds: int | None = None, *, offline: bool = False) -> int:
    task_dir = _task_dir(path)
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}", file=sys.stderr)
        return 1

    seed_prompt = _task_seed_prompt(task_dir, prompt)
    runtime_model, model_status = _resolve_runtime_model(offline=offline)
    if model_status:
        return model_status
    route = _classify_runtime_seed(seed_prompt, runtime_model)
    ensure_enhanced_task_contract(
        task_dir,
        seed_prompt,
        print_summary=True,
        force=bool(prompt and prompt.strip()),
        model=runtime_model,
    )

    selected = _canonical_capabilities(DEFAULT_CAPABILITIES if activate_all else required_capabilities(seed_prompt, route))
    route_semantic = getattr(route, "semantic", {}) if isinstance(getattr(route, "semantic", {}), dict) else {}
    difficulty_assessment = _runtime_entry_difficulty(route)
    route_profile = _runtime_profile_from_difficulty(getattr(route, "profile", "balanced"), difficulty_assessment)
    round_budget = _runtime_round_budget(
        route_profile=route_profile,
        route_semantic=route_semantic,
        rounds=rounds,
        difficulty_assessment=difficulty_assessment,
    )
    evolution_budget = evolution_budget_from_round_budget(round_budget)

    output_dir = task_dir / "nexus-runtime"
    result = NexusRuntime(model=runtime_model, output_dir=output_dir).run_text(
        seed_prompt,
        user_goal=seed_prompt,
        budget=evolution_budget,
        runtime_metadata={
            "semantic_route": route.to_dict() if hasattr(route, "to_dict") else dict(getattr(route, "__dict__", {})),
            "entry_difficulty_assessment": difficulty_assessment,
            "round_budget": round_budget.to_dict(),
            "route_incomplete_diagnostic_clamped": False,
            "runtime_entry": "runtime_run",
            "offline": offline,
            "model_backed": runtime_model is not None,
        },
    )
    run_status = _runtime_state_status(result.evolution)
    state = nexus_runtime_state(
        task_dir=task_dir,
        prompt=seed_prompt,
        run_data=result.to_dict(),
        selected_capabilities=selected,
        status=run_status,
    )
    state["last_run_at"] = _now()
    state["route"] = route.to_dict() if hasattr(route, "to_dict") else dict(getattr(route, "__dict__", {}))
    state = normalize_runtime_state(state)
    _write_json(task_dir / "runtime-state.json", state)
    validation_report = write_runtime_validation_report(task_dir)
    _append_trace(
        task_dir,
        "nexus_runtime_run",
        {
            "status": "completed",
            "runtime_path": RUNTIME_PATH,
            "active_capabilities": selected,
            "validation_status": validation_report.get("status"),
            "rounds": evolution_budget.round_limit,
            "completion_status": result.evolution.get("completion_status"),
        },
    )
    print(f"Runtime state written: {task_dir / 'runtime-state.json'}")
    print(f"Active capabilities: {_format_list(selected)}")
    print("Runtime path: nexus")
    return 0 if validation_report.get("status") == "pass" else 1


def _classify_runtime_seed(seed_prompt: str, runtime_model: object | None) -> object:
    if runtime_model is not None:
        try:
            return classify(seed_prompt, model=runtime_model)
        except TypeError as exc:
            if "model" not in str(exc):
                raise
    return classify(seed_prompt)


def _runtime_state_status(evolution: dict[str, object]) -> str:
    if evolution.get("interrupted"):
        return "interrupted"
    completion = str(evolution.get("completion_status") or "").strip()
    if completion in {"best_current_route", "needs_continuation", "route_incomplete"}:
        return completion
    return "completed"


def runtime_status(path: str | None) -> int:
    task_dir = _task_dir(path)
    state_path = task_dir / "runtime-state.json"
    if not state_path.exists():
        print(f"Runtime state not found: {state_path}", file=sys.stderr)
        return 1
    state = _load_json(state_path)
    print(f"task: {state.get('task')}")
    print(f"status: {state.get('status')}")
    print(f"version: {state.get('version')}")
    print(f"interaction_mode: {state.get('interaction_mode')}")
    print(f"runtime_path: {state.get('runtime_path')}")
    print(f"last_run_at: {state.get('last_run_at')}")
    print(f"activation_mode: {state.get('activation_mode', 'nexus')}")
    print(f"active_capabilities: {_format_list(state.get('active_capabilities', []))}")
    if "nexus_evolution" in state:
        print(f"nexus_evolution.actual_rounds: {state.get('nexus_evolution', {}).get('actual_rounds')}")
    print("nodes:")
    for node in state.get("nodes", []):
        print(f"- {node.get('id')}: {node.get('status')} ({node.get('capability')})")
    return 0


__all__ = ["runtime_run", "runtime_status"]
