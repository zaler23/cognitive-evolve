#!/usr/bin/env python3
"""CLI argument parsing for the Nexus-only CognitiveEvolve runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api.config import load_service_env
from .api.server import serve as api_serve, status_cli as api_status_cli
from .config_templates import available_env_profiles, env_template_info, render_env_template, write_env_template
from .artifacts.task_files import check_task, list_tasks, new_task, _slug_from_prompt
from .core.redaction import redact_text
from .cli.attack import run_attack
from .doctor import doctor
from .llm.env import LLMConfigurationError, LLMResponseError, require_llm_config
from .llm import llm_json, llm_public_status, llm_status_cli
from .nexus.evaluation import native_eval_run, native_optimize_run
from .nexus.model_adapter import StructuredModelAdapter
from .nexus.semantics import (
    DEFAULT_CAPABILITIES,
    NexusRoute,
    build_routed_prompt,
    classify,
    enhance_request,
    ensure_enhanced_task_contract,
    route_prompt,
    select_capability_ids,
)
from .runtime import runtime_run, runtime_status



def config_init(profile: str = "local", output: str = ".env", *, force: bool = False, print_template: bool = False) -> int:
    """Create or print a safe generic deployment env template."""

    try:
        info = env_template_info(profile)
        rendered = render_env_template(info.profile)
        if print_template:
            print(rendered, end="" if rendered.endswith("\n") else "\n")
            return 0
        path = write_env_template(output, profile=info.profile, force=force)
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Env template written: {path}")
    print(f"profile: {info.profile}")
    print(f"description: {info.description}")
    print("boundary: generic model provider only; no local private application relay configuration")
    return 0

def list_capabilities() -> int:
    for capability in DEFAULT_CAPABILITIES:
        print(capability)
    return 0


def list_ports() -> int:
    print("nexus_runtime")
    print("openai_compatible_api")
    print("local_tool_runner")
    return 0


def show_capability(capability_id: str) -> int:
    known = set(DEFAULT_CAPABILITIES)
    if capability_id not in known:
        print(f"Unknown capability: {capability_id}", file=sys.stderr)
        return 1
    print(f"{capability_id}: Nexus capability hint")
    return 0


def select_capabilities(prompt: str) -> int:
    for capability in select_capability_ids(prompt, model=_optional_classifier_model()):
        print(capability)
    return 0


def _optional_classifier_model(*, offline: bool = False) -> object | None:
    if offline:
        return None
    try:
        require_llm_config()
        return StructuredModelAdapter.from_configured_llm()
    except (LLMConfigurationError, LLMResponseError):
        return None


def _classify_prompt(prompt: str, model: object | None = None) -> NexusRoute:
    if model is not None:
        try:
            return classify(prompt, model=model)
        except TypeError as exc:
            if "model" not in str(exc):
                raise
    return classify(prompt)


def _apply_cognitive_intake(task_dir: Path, prompt: str, print_summary: bool = True) -> dict:
    return ensure_enhanced_task_contract(task_dir, prompt, print_summary=print_summary, force=True, model=_optional_classifier_model())


def llm_smoke() -> int:
    try:
        response = llm_json(
            "llm_smoke",
            {"prompt": "Return JSON with ok=true for a CognitiveEvolve LLM smoke check."},
            system="Return only JSON for a smoke check.",
            schema_hint={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )
    except Exception as exc:
        print(redact_text(f"llm.smoke failed: {exc}"), file=sys.stderr)
        return 1
    status = llm_public_status()
    print(json.dumps({"event": "llm.smoke", "ok": bool(response.get("ok", True)), "provider": status.get("provider"), "model": status.get("model")}, ensure_ascii=False))
    return 0


def run_standalone(prompt: str, dry_run: bool = False, *, offline: bool = False) -> int:
    classifier_model = _optional_classifier_model(offline=offline)
    route = _classify_prompt(prompt, classifier_model)
    task_dir = new_task("nexus", _slug_from_prompt(prompt))
    ensure_enhanced_task_contract(task_dir, prompt, print_summary=True, force=True, model=classifier_model)
    routed = build_routed_prompt(prompt, route, task_dir=task_dir)
    print("Selected Nexus route:")
    print(f"level: {route.level}")
    print(f"profile: {route.profile}")
    print(f"search: {str(route.search).lower()}")
    print(f"checkmodel_required: {str(route.checkmodel).lower()}")
    print(f"artifacts_required: {str(route.artifacts).lower()}")
    print(f"reason: {route.reason}")
    print("\nStandalone task:")
    print(task_dir)
    if dry_run:
        print("\nDry run: not invoking runtime.")
        print("\nRouted prompt preview:")
        print(routed)
        return 0
    return runtime_run(str(task_dir), prompt, activate_all=True, offline=offline)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("--type", default="general")
    p_new.add_argument("--slug", required=True)
    p_new.add_argument("--no-enhance", action="store_true", help="create only the raw skeleton; mainly for template development")

    sub.add_parser("list")

    p_config = sub.add_parser("config")
    config_sub = p_config.add_subparsers(dest="config_cmd", required=True)
    p_config_init = config_sub.add_parser("init")
    p_config_init.add_argument("--profile", choices=list(available_env_profiles()), default="local")
    p_config_init.add_argument("--output", default=".env")
    p_config_init.add_argument("--force", action="store_true")
    p_config_init.add_argument("--print", action="store_true", dest="print_template", help="print the selected template instead of writing a file")

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--scope", choices=["core", "runtime", "task", "all"], default="all")

    p_check = sub.add_parser("check")
    p_check.add_argument("path", nargs="?")

    p_enhance = sub.add_parser("enhance")
    p_enhance.add_argument("--path")
    p_enhance.add_argument("--json", action="store_true", dest="print_json")
    p_enhance.add_argument("prompt", nargs=argparse.REMAINDER)

    p_cap = sub.add_parser("capability")
    cap_sub = p_cap.add_subparsers(dest="capability_cmd", required=True)
    cap_sub.add_parser("list")
    cap_sub.add_parser("ports")
    p_cap_show = cap_sub.add_parser("show")
    p_cap_show.add_argument("capability_id")
    p_cap_select = cap_sub.add_parser("select")
    p_cap_select.add_argument("prompt", nargs=argparse.REMAINDER)

    p_runtime = sub.add_parser("runtime")
    runtime_sub = p_runtime.add_subparsers(dest="runtime_cmd", required=True)
    p_runtime_run = runtime_sub.add_parser("run")
    p_runtime_run.add_argument("path", nargs="?")
    p_runtime_run.add_argument("--prompt", default="")
    p_runtime_run.add_argument("--all", action="store_true", dest="activate_all")
    p_runtime_run.add_argument("--rounds", type=int, default=None, help="optional Nexus evolution round cap")
    p_runtime_run.add_argument("--offline", action="store_true", help="explicit deterministic local run without a configured LLM")
    p_runtime_status = runtime_sub.add_parser("status")
    p_runtime_status.add_argument("path", nargs="?")

    p_eval = sub.add_parser("eval")
    eval_sub = p_eval.add_subparsers(dest="eval_cmd", required=True)
    p_eval_run = eval_sub.add_parser("run")
    p_eval_run.add_argument("path", nargs="?")

    p_optimize = sub.add_parser("optimize")
    optimize_sub = p_optimize.add_subparsers(dest="optimize_cmd", required=True)
    p_optimize_run = optimize_sub.add_parser("run")
    p_optimize_run.add_argument("path", nargs="?")
    p_optimize_run.add_argument("--source")

    p_route = sub.add_parser("route")
    p_route.add_argument("prompt", nargs=argparse.REMAINDER)

    p_llm = sub.add_parser("llm")
    llm_sub = p_llm.add_subparsers(dest="llm_cmd", required=True)
    llm_sub.add_parser("status")
    llm_sub.add_parser("smoke")

    p_attack = sub.add_parser("attack")
    p_attack.add_argument("problem", nargs="?", help="problem.yaml for a new attack campaign")
    p_attack.add_argument("--budget", type=int, default=1, help="round budget or resume target rounds")
    p_attack.add_argument("--out", dest="out_dir", default=None, help="output directory for artifacts")
    p_attack.add_argument("--resume", dest="resume_dir", default=None, help="resume from an existing attack output directory")
    p_attack.add_argument("--offline", action="store_true", help="run without configured LLM for local smoke tests")

    p_api = sub.add_parser("api")
    api_sub = p_api.add_subparsers(dest="api_cmd", required=True)
    api_sub.add_parser("serve")
    api_sub.add_parser("status")

    p_run = sub.add_parser("run")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--offline", action="store_true", help="explicit deterministic local run without a configured LLM")
    p_run.add_argument("prompt", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.cmd == "new":
        p = new_task(args.type, args.slug)
        if not args.no_enhance:
            ensure_enhanced_task_contract(p, args.slug, print_summary=True, force=True, model=_optional_classifier_model())
        print(p)
        return 0
    if args.cmd == "list":
        list_tasks()
        return 0
    if args.cmd == "config":
        if args.config_cmd == "init":
            return config_init(args.profile, args.output, force=args.force, print_template=args.print_template)
    if args.cmd == "doctor":
        return doctor(args.scope)
    if args.cmd == "check":
        return check_task(args.path)
    if args.cmd == "enhance":
        prompt = " ".join(args.prompt).strip()
        return enhance_request(prompt, path=args.path, print_json=args.print_json, model=_optional_classifier_model())
    if args.cmd == "capability":
        if args.capability_cmd == "list":
            return list_capabilities()
        if args.capability_cmd == "ports":
            return list_ports()
        if args.capability_cmd == "show":
            return show_capability(args.capability_id)
        if args.capability_cmd == "select":
            prompt = " ".join(args.prompt).strip()
            if not prompt:
                print("Missing prompt", file=sys.stderr)
                return 2
            return select_capabilities(prompt)
    if args.cmd == "runtime":
        if args.runtime_cmd == "run":
            return runtime_run(args.path, args.prompt, activate_all=args.activate_all, rounds=args.rounds, offline=args.offline)
        if args.runtime_cmd == "status":
            return runtime_status(args.path)
    if args.cmd == "eval":
        if args.eval_cmd == "run":
            return native_eval_run(args.path)
    if args.cmd == "optimize":
        if args.optimize_cmd == "run":
            return native_optimize_run(args.path, source=args.source)
    if args.cmd == "route":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            print("Missing prompt", file=sys.stderr)
            return 2
        route_prompt(prompt, model=_optional_classifier_model())
        return 0
    if args.cmd == "llm":
        if args.llm_cmd == "status":
            load_service_env()
            return llm_status_cli()
        if args.llm_cmd == "smoke":
            load_service_env()
            return llm_smoke()
    if args.cmd == "attack":
        return run_attack(problem_path=args.problem, budget=args.budget, out_dir=args.out_dir, resume_dir=args.resume_dir, offline=args.offline)
    if args.cmd == "api":
        if args.api_cmd == "serve":
            return api_serve()
        if args.api_cmd == "status":
            return api_status_cli()
    if args.cmd == "run":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            print("Missing prompt", file=sys.stderr)
            return 2
        return run_standalone(prompt, dry_run=args.dry_run, offline=args.offline)
    return 1


# Backward-compatible symbol names inside this file only; no legacy module remains.


__all__ = [
    "build_routed_prompt",
    "classify",
    "list_capabilities",
    "list_ports",
    "llm_smoke",
    "main",
    "native_eval_run",
    "native_optimize_run",
    "route_prompt",
    "run_standalone",
    "runtime_run",
    "runtime_status",
    "select_capabilities",
    "show_capability",
]
