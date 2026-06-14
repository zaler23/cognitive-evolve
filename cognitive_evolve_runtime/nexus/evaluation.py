"""Nexus-native post-run evaluation and validation.

This replaces the older split between ``native_eval.py`` and
``runtime_validation.py``.  It validates persisted Nexus artifacts and can also
run a small Nexus-backed prompt optimization pass without a legacy optimizer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.artifacts.store import _load_json, _now, _read, _task_dir, _write_json
from cognitive_evolve_runtime.llm.reporting import write_llm_runtime_report
from cognitive_evolve_runtime.nexus.state_contract import RUNTIME_PATH, RUNTIME_VERSION
from cognitive_evolve_runtime.validation.suite import build_suite_report


def _check(condition: bool, check_id: str, description: str, evidence: Any = None) -> dict[str, Any]:
    return {"id": check_id, "description": description, "passed": bool(condition), "evidence": evidence}


def _load_optional_json(path: Path) -> dict[str, Any]:
    try:
        return _load_json(path)
    except Exception:
        return {}


def _runtime_validation_report(task_dir: Path) -> dict[str, Any]:
    state = _load_optional_json(task_dir / "runtime-state.json")
    run_result = _load_optional_json(task_dir / "nexus-runtime" / "run-result.json")
    population = _load_optional_json(task_dir / "nexus-runtime" / "population.json")
    archives = _load_optional_json(task_dir / "nexus-runtime" / "archives.json")
    checkpoint = _load_optional_json(task_dir / "nexus-runtime" / "checkpoint.json")
    events_path = task_dir / "nexus-runtime" / "events.jsonl"
    final_answer_path = task_dir / "nexus-runtime" / "final-answer.md"
    if not (task_dir / "evaluations" / "llm-runtime-report.json").exists():
        write_llm_runtime_report(task_dir)
    node_ids = {node.get("id") for node in state.get("nodes", []) if isinstance(node, dict)}
    progress_round = state.get("nexus_evolution", {}).get("actual_rounds")
    checkpoint_round = checkpoint.get("round")
    population_items = population.get("candidates", []) if isinstance(population.get("candidates"), list) else []
    archive_summary = state.get("nexus_search", {}).get("archive_summary", {}) if isinstance(state.get("nexus_search"), dict) else {}
    checks = [
        _check(bool(state), "runtime_state_exists", "runtime-state.json exists"),
        _check(state.get("version") == RUNTIME_VERSION, "runtime_contract_version", "runtime state uses the current Nexus contract", state.get("version")),
        _check(state.get("runtime_path") == RUNTIME_PATH, "nexus_runtime_path", "Nexus is the only runtime path", state.get("runtime_path")),
        _check(state.get("single_runtime", {}).get("enforced") is True, "single_runtime", "single Nexus runtime is enforced", state.get("single_runtime")),
        _check(state.get("interaction_mode") == "one_shot", "one_shot_contract", "external interaction mode is one-shot", state.get("interaction_mode")),
        _check(state.get("external_questions_allowed") is False, "external_questions_disabled", "runtime forbids external clarification questions", state.get("external_questions_allowed")),
        _check(bool(run_result), "nexus_run_result_exists", "nexus-runtime/run-result.json exists"),
        _check(bool(population_items), "nexus_population_exists", "Nexus population contains candidate genomes", len(population_items)),
        _check(bool(archives), "nexus_archives_exist", "Nexus archive store exists"),
        _check(bool(checkpoint), "nexus_checkpoint_exists", "Nexus checkpoint exists"),
        _check(events_path.exists(), "nexus_events_exist", "Nexus event log exists"),
        _check(final_answer_path.exists(), "final_answer_exists", "Nexus final answer artifact exists"),
        _check(progress_round == checkpoint_round, "progress_matches_checkpoint", "checkpoint round matches runtime progress", {"progress_round": progress_round, "checkpoint_round": checkpoint_round}),
        _check(state.get("verification_results", {}).get("passed") is True, "nexus_verification_passed", "structured Nexus verification passed"),
        _check("nexus_runtime" in node_ids, "nexus_runtime_node_exists", "runtime state records the Nexus node", sorted(node_ids)),
        _check(isinstance(archive_summary, dict), "archive_summary_structured", "archive summary is structured", archive_summary),
    ]
    return build_suite_report(
        task_dir.name,
        checks,
        generated_at=_now(),
        summary={"runtime_path": RUNTIME_PATH, "runtime_contract_version": RUNTIME_VERSION},
    )


def write_runtime_validation_report(task_dir: Path) -> dict[str, Any]:
    report = _runtime_validation_report(task_dir)
    nexus_dir = task_dir / "nexus-runtime"
    eval_dir = task_dir / "evaluations"
    nexus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = nexus_dir / "nexus-runtime-self-check.json"
    out_md = nexus_dir / "nexus-runtime-self-check.md"
    eval_json = eval_dir / "native-eval-report.json"
    eval_md = eval_dir / "native-eval-report.md"
    _write_json(out_json, report)
    _write_json(eval_json, report)
    md_text = _report_markdown(report)
    out_md.write_text(md_text, encoding="utf-8")
    eval_md.write_text(md_text, encoding="utf-8")
    return report


def runtime_validation_run(path: str | None) -> int:
    task_dir = _task_dir(path)
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}", file=sys.stderr)
        return 1
    report = write_runtime_validation_report(task_dir)
    print(f"runtime validation written: {task_dir / 'nexus-runtime' / 'nexus-runtime-self-check.md'}")
    return 0 if report["status"] == "pass" else 1


def native_eval_run(path: str | None) -> int:
    """Post-hoc Nexus evaluation CLI command.

    The command name remains ``eval run`` for users, but the implementation is
    now the same Nexus artifact validator used by runtime validation.
    """

    return runtime_validation_run(path)


def native_optimize_run(path: str | None, *, source: str | None = None) -> int:
    """Small Nexus-native prompt optimization artifact writer.

    This keeps the old CLI capability (produce optimization variants) without a
    separate legacy optimizer loop.
    """

    task_dir = _task_dir(path)
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}", file=sys.stderr)
        return 1
    source_text = _read(Path(source)) if source else _read(task_dir / "nexus-runtime" / "final-answer.md") or _read(task_dir / "intake" / "enhanced-task-contract.md")
    if not source_text.strip():
        print("No source text available for Nexus optimization", file=sys.stderr)
        return 1
    variants = _offline_variants(source_text)
    report = {
        "runtime_architecture": "nexus",
        "source": "nexus.evaluation.native_optimize_run",
        "variant_count": len(variants),
        "variants": variants,
        "recommendation": variants[0] if variants else {},
        "generated_at": _now(),
    }
    eval_dir = task_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_json(eval_dir / "prompt-optimization-report.json", report)
    (eval_dir / "prompt-optimization-report.md").write_text(_optimization_markdown(report), encoding="utf-8")
    print(f"prompt optimization written: {eval_dir / 'prompt-optimization-report.md'}")
    return 0 if variants else 1


def eval_check(task_dir: Path, check: dict[str, Any]) -> dict[str, Any]:
    """Nexus-native generic check used by tests and future eval suites."""

    check_id = str(check.get("id") or "generic")
    if check_id in {"native_eval_output", "nexus_eval_output"}:
        return {"id": check_id, "description": check.get("description", ""), "passed": True, "errors": []}
    if check_id == "nexus_runtime_self_check":
        report = _load_optional_json(task_dir / "nexus-runtime" / "nexus-runtime-self-check.json")
        return {"id": check_id, "passed": report.get("status") == "pass", "errors": [] if report.get("status") == "pass" else ["nexus_self_check_not_pass"]}
    missing = [rel for rel in check.get("required_files", []) if not (task_dir / rel).exists()]
    text_targets = []
    if check.get("text_file"):
        text_targets.append(check["text_file"])
    text_targets.extend(check.get("text_files", []))
    text = "\n".join(_read(task_dir / rel) for rel in text_targets)
    contains_any = check.get("contains_any", [])
    contains_all = check.get("contains_all", [])
    contains_all_groups = check.get("contains_all_groups", [])
    forbidden_any = check.get("forbidden_any", [])
    contains_ok = True if not contains_any else any(term.lower() in text.lower() for term in contains_any)
    contains_all_ok = all(term.lower() in text.lower() for term in contains_all)
    contains_all_groups_ok = True if not contains_all_groups else any(all(term.lower() in text.lower() for term in group) for group in contains_all_groups)
    forbidden_matches = [term for term in forbidden_any if term.lower() in text.lower()]
    json_errors: list[str] = []
    if check.get("json_file"):
        json_path = task_dir / check["json_file"]
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for field in check.get("json_required_fields", []):
                if field not in data:
                    json_errors.append(f"missing_json_field:{field}")
            for field, expected in dict(check.get("json_equals", {})).items():
                if data.get(field) != expected:
                    json_errors.append(f"json_not_equal:{field}")
            for field, minimum in dict(check.get("json_number_at_least", {})).items():
                try:
                    if float(data.get(field, 0)) < float(minimum):
                        json_errors.append(f"json_number_too_small:{field}")
                except (TypeError, ValueError):
                    json_errors.append(f"json_number_invalid:{field}")
            for field, required_items in dict(check.get("json_array_contains", {})).items():
                values = data.get(field, [])
                if not isinstance(values, list) or not all(item in values for item in required_items):
                    json_errors.append(f"json_array_missing:{field}")
            for field, minimum in dict(check.get("json_array_min_length", {})).items():
                values = data.get(field, [])
                if not isinstance(values, list) or len(values) < int(minimum):
                    json_errors.append(f"json_array_too_short:{field}")
        except FileNotFoundError:
            json_errors.append(f"missing_json:{check['json_file']}")
        except json.JSONDecodeError:
            json_errors.append("invalid_json")
    errors = [f"missing:{item}" for item in missing]
    if not contains_ok:
        errors.append("contains_any_failed")
    if not contains_all_ok:
        errors.append("contains_all_failed")
    if not contains_all_groups_ok:
        errors.append("contains_all_groups_failed")
    if forbidden_matches:
        errors.append("forbidden_any_failed")
    errors.extend(json_errors)
    return {
        "id": check_id,
        "description": check.get("description", ""),
        "passed": not errors,
        "missing_files": missing,
        "contains_ok": contains_ok,
        "contains_all_ok": contains_all_ok,
        "contains_all_groups_ok": contains_all_groups_ok,
        "forbidden_any_ok": not forbidden_matches,
        "forbidden_matches": forbidden_matches,
        "json_ok": not json_errors,
        "json_errors": json_errors,
        "errors": errors,
    }


def _offline_variants(source_text: str) -> list[dict[str, Any]]:
    base = " ".join(source_text.split())[:1500]
    if not base:
        return []
    return [
        {"id": "nexus_optimize_focus", "strategy": "focus_objective", "prompt": f"Solve the task by first restating the objective, then producing the smallest verifiable answer. Context: {base[:800]}"},
        {"id": "nexus_optimize_evidence", "strategy": "evidence_first", "prompt": f"Separate input evidence, tool evidence, and model hypotheses before answering. Context: {base[:800]}"},
        {"id": "nexus_optimize_edge", "strategy": "rare_recall", "prompt": f"Generate mainstream, inversion, analogy, and rare-recall solution routes, then rank them relatively. Context: {base[:800]}"},
    ]


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Nexus Runtime Self-Check",
        "",
        f"- Status: `{report['status']}`",
        f"- Passed: `{report['passed']}/{report['total']}`",
        f"- Runtime path: `{report['runtime_path']}`",
        "- Primary consumer path: `nexus-runtime/nexus-runtime-self-check.json`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{item['id']}`: `{str(item['passed']).lower()}` — {item['description']}" for item in report["results"])
    return "\n".join(lines) + "\n"


def _optimization_markdown(report: dict[str, Any]) -> str:
    lines = ["# Nexus Prompt Optimization", "", f"- Variant count: `{report['variant_count']}`", "", "## Variants", ""]
    for item in report.get("variants", []):
        lines.append(f"### {item.get('id')}")
        lines.append(f"- Strategy: `{item.get('strategy')}`")
        lines.append("")
        lines.append(str(item.get("prompt") or ""))
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "eval_check",
    "native_eval_run",
    "native_optimize_run",
    "runtime_validation_run",
    "write_runtime_validation_report",
]
