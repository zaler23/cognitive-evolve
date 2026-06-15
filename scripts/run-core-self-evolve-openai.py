#!/usr/bin/env python3
"""Launch a bounded OpenAI-compatible CognitiveEvolve core self-evolution run.

This helper is intentionally explicit: it writes the same top-level status/result
files used by the monitoring automations, while NexusRuntime owns the canonical
``nexus-runtime/*`` checkpoint/population/events artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cognitive_evolve_runtime.configuration import load_layered_config
from cognitive_evolve_runtime.core.redaction import redact, redact_text
from cognitive_evolve_runtime.llm.env import llm_public_status, llm_status
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime

DEFAULT_GOAL = """Run a fresh model-backed bootstrap/self-evolution exploration of the full CognitiveEvolve core through the configured OpenAI-compatible upstream.

Primary objective: find implementable improvements to the core self-evolution engine and produce patch-sized update candidates. Treat the whole core as input: architecture boundaries, model/runtime protocols, stage-adaptive eligibility, Incubating/Dormant repair, Active pool preservation, relative_rater robustness, provider/model error recovery, checkpoint/event consistency, live store, archive memory, obligation/evidence ledger, verification, synthesis, package/release safety, and upstream model execution.

Priority validation targets for this run:
- formal_artifact schemas using type=assertion_set or type=verification_witness must pass structural proof gates when executable assertions/checks are present;
- source_binding / patch target paths that do not exist in the current project snapshot must be rejected quickly with source_binding_missing_path or patch_target_missing;
- generic artifact_type=code_patch candidates with artifact-level unified diffs, including guarded artifact.content unified diffs, must enter the project patch sandbox and either apply or fail with concrete patch diagnostics;
- if docs-only / NEXUS_SEED_NOTE or no_parents_available collapse appears, convert the ranked repair material into concrete runtime/test/schema mutations rather than stopping without a repair attempt;
- patch candidates must use exact project-relative paths from the context packet or allowed_patch_scope, not invented root-level basenames such as nexus_*_hardening.py; when emitting unified diffs, prefer artifact.patch or artifact.unified_diff containing diff --git a/<path> b/<path> with real context from the provided raw slices;
- avoid proof_object_structurally_weak false negatives, hallucinated source files, malformed patches, and patch_no_effect loops.

Prefer low-risk runtime/test/schema patches backed by formal_artifacts, obligation_delta, evidence_refs, source_bindings, and local verification results. Do not treat narrative architecture notes as final progress unless they are converted into executable or testable project changes."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def default_run_dir(label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label).strip("-") or "core-self-evolve"
    return Path.home() / ".cognitive-evolve" / ".cogev" / "api-runs" / f"self-evolve-core-openai-{safe_label}-{stamp}"


def load_runtime_environment(project_dir: Path, env_file: Path | None) -> None:
    candidate = env_file or (Path.home() / ".cognitive-evolve" / ".env")
    if candidate.exists():
        load_dotenv(candidate, override=False)
    load_layered_config(override=False)
    # Public release default: use a caller-configured OpenAI-compatible upstream.
    # Do not inject host-app or provider-specific defaults here.
    os.environ.setdefault("COGEV_LLM_PROVIDER", "litellm")
    os.environ["COGEV_LLM_TEMPERATURE"] = os.environ.get("COGEV_CORE_SELF_EVOLVE_TEMPERATURE", "0.7")
    os.environ.setdefault("COGEV_LLM_TIMEOUT", "900")
    os.environ.setdefault("COGEV_LLM_RETRY_ATTEMPTS", "5")
    os.environ.setdefault("COGEV_LLM_RETRY_BASE_SLEEP", "1")
    os.environ.setdefault("COGEV_LLM_RETRY_MAX_SLEEP", "8")
    os.environ.setdefault("COGEV_LLM_RETRY_JITTER", "0")
    os.environ.setdefault("COGEV_LLM_MAX_TOKENS", "8192")
    os.environ.setdefault("COGEV_LLM_JSON_RETRY_ATTEMPTS", "5")
    os.environ.setdefault("COGEV_MAX_PROMPT_CHARS", os.environ.get("COGEV_LLM_MAX_PROMPT_CHARS", ""))
    os.environ["COGEV_PROJECT_DIR"] = str(project_dir)


def status_payload(*, status: str, run_dir: Path, project_dir: Path, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "run_dir": str(run_dir),
        "project_dir": str(project_dir),
        "nexus_runtime_dir": str(run_dir / "nexus-runtime"),
        "updated_at": utc_now(),
        "label": args.label,
        "resume": bool(args.resume),
        "budget": {
            "max_rounds": args.max_rounds,
            "branch_factor": args.branch_factor,
            "initial_candidate_count": args.initial_candidates,
            "min_rounds_before_stop": args.min_rounds_before_stop,
            "round_safety_limit": args.round_safety_limit or args.max_rounds,
            "include_tests": args.include_tests,
        },
        "llm_status": _safe_llm_status(),
        "llm_temperature": os.environ.get("COGEV_LLM_TEMPERATURE", ""),
        "preflight": preflight_status(args),
    }
    if extra:
        payload.update(extra)
    return payload


def _safe_llm_status() -> dict[str, Any]:
    return llm_public_status(llm_status())


def parse_iso_time(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def upstream_health(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310 - local operator-supplied health URL
            body = response.read(2048).decode("utf-8", errors="replace")
            return {"ok": 200 <= int(response.status) < 300, "status": int(response.status), "body": body[:500]}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def preflight_status(args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    status: dict[str, Any] = {"ok": True, "checked_at": now.isoformat(), "blockers": []}
    if args.not_before:
        try:
            target = parse_iso_time(args.not_before)
        except ValueError as exc:
            status["ok"] = False
            status["blockers"].append(f"invalid_not_before:{exc}")
            target = None
        if target is not None:
            status["not_before"] = target.isoformat()
            if now < target:
                status["ok"] = False
                status["blockers"].append(f"waiting_until:{target.isoformat()}")
    if args.require_upstream_health:
        if not args.upstream_health_url:
            status["ok"] = False
            status["blockers"].append("upstream_health_url_missing")
        else:
            health = upstream_health(args.upstream_health_url)
            status["upstream_health"] = health
            if not health.get("ok"):
                status["ok"] = False
                status["blockers"].append("upstream_health_unavailable")
    return status


def build_budget(args: argparse.Namespace) -> EvolutionBudget:
    return EvolutionBudget(
        max_rounds=max(1, int(args.max_rounds)),
        branch_factor=max(0, int(args.branch_factor)),
        initial_candidate_count=max(0, int(args.initial_candidates)),
        stop_policy="llm_after_minimum",
        min_rounds_before_stop=max(1, int(args.min_rounds_before_stop)),
        adaptive=True,
        round_safety_limit=max(1, int(args.round_safety_limit or args.max_rounds)),
        completion_requires_stop_signal=True,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CognitiveEvolve core self-evolution through an OpenAI-compatible upstream")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--label", default="core-bootstrap")
    parser.add_argument("--goal-file", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--resume", action="store_true", help="resume from run-dir/nexus-runtime/checkpoint.json instead of starting a fresh project run")
    parser.add_argument("--max-rounds", type=int, default=48)
    parser.add_argument("--round-safety-limit", type=int, default=48)
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--initial-candidates", type=int, default=16)
    parser.add_argument("--min-rounds-before-stop", type=int, default=8)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--not-before", default="", help="ISO timestamp; exit without model calls before this time")
    parser.add_argument("--require-upstream-health", action="store_true", help="require an operator-supplied upstream health endpoint before model calls")
    parser.add_argument("--upstream-health-url", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir(args.label)
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else None
    load_runtime_environment(project_dir, env_file)
    os.environ["COGEV_RUN_ID"] = run_dir.name
    goal = read_text_if_exists(Path(args.goal_file).expanduser()) if args.goal_file else DEFAULT_GOAL
    status_path = run_dir / "self-evolve-status.json"
    result_path = run_dir / "self-evolve-result.json"
    error_path = run_dir / "self-evolve-error.txt"
    preflight = preflight_status(args)
    if args.dry_run:
        print(json.dumps(redact(status_payload(status="dry_run", run_dir=run_dir, project_dir=project_dir, args=args, extra={"goal_chars": len(goal), "preflight": preflight})), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not preflight.get("ok"):
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(status_path, status_payload(status="waiting_preflight", run_dir=run_dir, project_dir=project_dir, args=args, extra={"preflight": preflight, "finished_at": utc_now()}))
        print(json.dumps(redact({"status": "waiting_preflight", "run_dir": str(run_dir), "preflight": preflight}), ensure_ascii=False, sort_keys=True))
        return 75
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, status_payload(status="running", run_dir=run_dir, project_dir=project_dir, args=args, extra={"started_at": utc_now()}))
    try:
        model = StructuredModelAdapter.from_configured_llm()
        runtime = NexusRuntime(model=model, output_dir=run_dir / "nexus-runtime")
        if args.resume:
            result = runtime.resume_from_checkpoint(max_rounds=args.max_rounds)
        else:
            result = runtime.run_project(
                project_dir,
                user_goal=goal,
                budget=build_budget(args),
                include_tests=bool(args.include_tests),
            )
        data = result.to_dict()
        write_json(result_path, data)
        completion_status = str(data.get("evolution", {}).get("completion_status") or data.get("evolution", {}).get("stop_reason") or "completed")
        write_json(
            status_path,
            status_payload(
                status="completed",
                run_dir=run_dir,
                project_dir=project_dir,
                args=args,
                extra={
                    "completion_status": completion_status,
                    "finished_at": utc_now(),
                    "artifacts": data.get("artifacts", {}),
                },
            ),
        )
        if error_path.exists():
            error_path.unlink()
        print(json.dumps(redact({"status": "completed", "completion_status": completion_status, "run_dir": str(run_dir)}), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        error_path.write_text(redact_text("".join(traceback.format_exception(exc))), encoding="utf-8")
        write_json(
            status_path,
            status_payload(
                status="failed",
                run_dir=run_dir,
                project_dir=project_dir,
                args=args,
                extra={"error": f"{exc.__class__.__name__}: {exc}", "finished_at": utc_now()},
            ),
        )
        print(redact_text(f"failed: {exc.__class__.__name__}: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
