"""Thin CLI-facing Nexus runtime commands.

This module is intentionally not a second runtime.  It prepares task artifacts,
resolves Nexus semantic/budget hints, invokes ``NexusRuntime``, and writes the
canonical runtime state.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from .artifacts.store import _append_trace, _format_list, _load_json, _now, _task_dir, _write_json
from .artifacts.task_files import _task_seed_prompt
from .configuration import parse_simple_yaml
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

    raw_seed_prompt = _task_seed_prompt(task_dir, prompt)
    adaptive_config = _task_adaptive_config(task_dir)
    seed_prompt = _seed_prompt_with_artifact_policy(raw_seed_prompt, adaptive_config)
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
        adaptive_config=adaptive_config,
        runtime_metadata={
            "semantic_route": route.to_dict() if hasattr(route, "to_dict") else dict(getattr(route, "__dict__", {})),
            "entry_difficulty_assessment": difficulty_assessment,
            "round_budget": round_budget.to_dict(),
            "route_incomplete_diagnostic_clamped": False,
            "runtime_entry": "runtime_run",
            "offline": offline,
            "model_backed": runtime_model is not None,
            "artifact_policy_hint_enabled": raw_seed_prompt != seed_prompt,
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


def _task_adaptive_config(task_dir: Path) -> dict[str, object]:
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        return {}
    try:
        data = parse_simple_yaml(task_yaml.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to parse %s; falling back to empty adaptive config", task_yaml, exc_info=True
        )
        return {}
    adaptive = data.get("adaptive")
    evaluator = data.get("evaluator")
    evidence = data.get("evidence")
    config: dict[str, object] = {}
    if isinstance(adaptive, dict):
        config.update(adaptive)
        nested_evaluator = config.get("evaluator")
        if isinstance(nested_evaluator, dict):
            evaluator_config = dict(nested_evaluator)
            evaluator_config.setdefault("cwd", str(task_dir))
            config["evaluator"] = evaluator_config
            config.setdefault("enabled", True)
    if isinstance(evaluator, dict):
        evaluator_config = dict(evaluator)
        evaluator_config.setdefault("cwd", str(task_dir))
        if isinstance(evaluator_config.get("evidence"), dict) and not isinstance(config.get("evidence"), dict):
            config["evidence"] = dict(evaluator_config["evidence"])
        existing = config.get("evaluator")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(evaluator_config)
            evaluator_config = merged
        config["evaluator"] = evaluator_config
        config.setdefault("enabled", True)
    if isinstance(evidence, dict):
        existing_evidence = config.get("evidence")
        merged_evidence = dict(existing_evidence) if isinstance(existing_evidence, dict) else {}
        merged_evidence.update(evidence)
        config["evidence"] = merged_evidence
        config.setdefault("enabled", True)
    return config


def _seed_prompt_with_artifact_policy(seed_prompt: str, adaptive_config: dict[str, object]) -> str:
    hint = _artifact_policy_prompt_hint(adaptive_config)
    if not hint or "## CognitiveEvolve machine artifact contract" in seed_prompt:
        return seed_prompt
    return seed_prompt.rstrip() + "\n\n" + hint


def _artifact_policy_prompt_hint(adaptive_config: dict[str, object]) -> str:
    evidence = _as_dict(adaptive_config.get("evidence"))
    if not evidence:
        return ""
    machine_required = _truthy(evidence.get("machine_artifact_required") or evidence.get("machine_readable_required"))
    artifact_type = str(evidence.get("artifact_type") or "").strip()
    required_fields = _string_list(evidence.get("required_fields"))
    if not (machine_required or artifact_type or required_fields):
        return ""
    field_aliases = _as_dict(evidence.get("field_aliases"))
    type_aliases = _as_dict(evidence.get("artifact_type_aliases"))
    metadata = _as_dict(evidence.get("metadata"))
    allowed_terms = _string_list(metadata.get("domain_vocabulary") or metadata.get("allowed_domain_terms") or evidence.get("domain_vocabulary") or evidence.get("allowed_domain_terms"))
    forbidden_terms = _string_list(metadata.get("forbidden_semantic_terms") or evidence.get("forbidden_semantic_terms"))
    lines = [
        "## CognitiveEvolve machine artifact contract",
        "",
        "The final candidate artifact must satisfy this machine-artifact boundary. Treat this section as a runtime contract hint, not as content to copy into the artifact.",
    ]
    if artifact_type:
        lines.append(f"- Emit artifact_type exactly: `{artifact_type}`.")
    if required_fields:
        lines.append("- Required top-level artifact fields: `" + "`, `".join(required_fields) + "`.")
    if type_aliases:
        lines.append("- Do not use artifact_type aliases: `" + "`, `".join(str(key) for key in type_aliases.keys()) + "`.")
    if field_aliases:
        lines.append("- Do not use field aliases: `" + "`, `".join(str(key) for key in field_aliases.keys()) + "`.")
    if machine_required:
        lines.append("- Output a machine-readable artifact object, not prose that wraps or describes the object.")
    if allowed_terms:
        lines.append("- Prefer domain vocabulary: `" + "`, `".join(allowed_terms[:24]) + "`.")
    if forbidden_terms:
        lines.append("- Avoid forbidden semantic terms in the artifact: `" + "`, `".join(forbidden_terms[:24]) + "`.")
    lines.append("- Do not emit internal runtime contracts, route summaries, checkpoint metadata, or evaluator implementation details as the artifact itself.")
    return "\n".join(lines)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "required"}


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
    if completion in {"best" + "_current" + "_route", "route" + "_incomplete", "completed", "solved"}:
        return "completed"
    if completion in {"needs_continuation"}:
        return completion
    if completion in {"interrupted_checkpointed", "paused_quota"}:
        return "interrupted"
    if completion in {"failed", "failed_verification"}:
        return "failed"
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
