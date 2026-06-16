from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.redaction import redact
from .inflight import provider_inflight_status
from .budget import budget_usd
from .env import llm_public_status
from .governor import llm_governor_status
from .session import current_llm_session
from .utils import cli_logger, now_iso, write_json


def _event_signature(event: dict[str, Any]) -> str:
    return json.dumps(
        {"time": event.get("time"), "request_type": event.get("request_type"), "provider": event.get("provider"), "model": event.get("model"), "confidence": event.get("confidence")},
        sort_keys=True,
        ensure_ascii=False,
    )


def _preserve_existing_report(task_dir: Path, existing: dict[str, Any]) -> None:
    (task_dir / "evaluations" / "llm-runtime-report.md").write_text(
        "# LLM Runtime Report\n\n"
        f"- Status: `{existing.get('status')}`\n"
        f"- Provider: `{existing.get('provider')}`\n"
        f"- Model: `{existing.get('model')}`\n"
        f"- Test provider only: `{str(existing.get('test_provider_only')).lower()}`\n"
        "- No no-LLM fallback: `true`\n"
        f"- LLM event count: `{existing.get('event_count')}`\n"
        f"- Request types: `{', '.join(existing.get('request_types', []))}`\n"
        f"- Total tokens: `{(existing.get('usage') or {}).get('total_tokens', 0)}`\n"
        f"- Estimated cost USD: `{existing.get('estimated_cost_usd', 0)}`\n",
        encoding="utf-8",
    )


def _stage_usage(existing_events: list[Any], new_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for event in existing_events + new_events:
        if not isinstance(event, dict):
            continue
        stage = str(event.get("stage") or "unscoped")
        usage_item = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        bucket = buckets.setdefault(stage, {"event_count": 0, "estimated_cost_usd": 0.0, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "request_types": []})
        bucket["event_count"] += 1
        bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + float(event.get("estimated_cost_usd") or 0.0), 6)
        bucket["usage"]["prompt_tokens"] += int(usage_item.get("prompt_tokens") or 0)
        bucket["usage"]["completion_tokens"] += int(usage_item.get("completion_tokens") or 0)
        bucket["usage"]["total_tokens"] += int(usage_item.get("total_tokens") or 0)
        request_type = str(event.get("request_type") or "")
        if request_type and request_type not in bucket["request_types"]:
            bucket["request_types"].append(request_type)
    return buckets


def write_llm_runtime_report(task_dir: Path) -> None:
    status = llm_public_status()
    existing_path = task_dir / "evaluations" / "llm-runtime-report.json"
    existing: dict[str, Any] = {}
    if existing_path.exists():
        try:
            loaded = json.loads(existing_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    events = current_llm_session().snapshot()
    if not events and int(existing.get("event_count") or 0) > 0:
        _preserve_existing_report(task_dir, existing)
        return

    existing_events = existing.get("events") if isinstance(existing.get("events"), list) else []
    existing_signatures = {_event_signature(event) for event in existing_events if isinstance(event, dict)}
    new_events = [event for event in events if _event_signature(event) not in existing_signatures]
    merged_events = (existing_events + new_events[-100:])[-100:]
    existing_request_types = {str(item) for item in existing.get("request_types", [])}
    current_request_types = {str(event.get("request_type")) for event in new_events}
    existing_usage = existing.get("usage") if isinstance(existing.get("usage"), dict) else {}
    current_usage = {
        "prompt_tokens": sum(int((event.get("usage") or {}).get("prompt_tokens") or 0) for event in new_events),
        "completion_tokens": sum(int((event.get("usage") or {}).get("completion_tokens") or 0) for event in new_events),
        "total_tokens": sum(int((event.get("usage") or {}).get("total_tokens") or 0) for event in new_events),
    }
    usage = {
        "prompt_tokens": int(existing_usage.get("prompt_tokens") or 0) + current_usage["prompt_tokens"],
        "completion_tokens": int(existing_usage.get("completion_tokens") or 0) + current_usage["completion_tokens"],
        "total_tokens": int(existing_usage.get("total_tokens") or 0) + current_usage["total_tokens"],
    }
    report = {
        "name": "llm-runtime-report",
        "version": "1.0",
        "task": task_dir.name,
        "generated_at": now_iso(),
        "status": "pass" if status.get("configured") else "missing_llm_configuration",
        "provider": status.get("provider"),
        "model": status.get("model"),
        "test_provider_only": status.get("test_provider_only", False),
        "api_base": status.get("api_base", ""),
        "credential_configured": status.get("credential_configured", False),
        "credential_placeholder": status.get("credential_placeholder", False),
        "no_llm_fallback": True,
        "event_count": int(existing.get("event_count") or 0) + len(new_events),
        "request_types": sorted(existing_request_types | current_request_types),
        "usage": usage,
        "estimated_cost_usd": round(float(existing.get("estimated_cost_usd") or 0.0) + sum(float(event.get("estimated_cost_usd") or 0.0) for event in new_events), 6),
        "budget_usd": budget_usd(),
        "governor_config": llm_governor_status(),
        "inflight_registry": provider_inflight_status(),
        "stage_usage": _stage_usage(existing_events, new_events),
        "events": merged_events,
        "merge_policy": "preserve_existing_task_events_and_append_current_process_events",
    }
    write_json(task_dir / "evaluations" / "llm-runtime-report.json", report)
    (task_dir / "evaluations" / "llm-runtime-report.md").write_text(
        "# LLM Runtime Report\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Provider: `{report['provider']}`\n"
        f"- Model: `{report['model']}`\n"
        f"- Test provider only: `{str(report['test_provider_only']).lower()}`\n"
        f"- API base override: `{report.get('api_base', '')}`\n"
        f"- Credential configured: `{str(report.get('credential_configured', False)).lower()}`\n"
        f"- Credential placeholder: `{str(report.get('credential_placeholder', False)).lower()}`\n"
        "- No no-LLM fallback: `true`\n"
        f"- LLM event count: `{report['event_count']}`\n"
        f"- Request types: `{', '.join(report['request_types'])}`\n"
        f"- Total tokens: `{report['usage']['total_tokens']}`\n"
        f"- Estimated cost USD: `{report['estimated_cost_usd']}`\n"
        f"- Governor max concurrent: `{report['governor_config'].get('max_concurrent')}`\n"
        f"- Governor RPM: `{report['governor_config'].get('rpm')}`\n"
        f"- Governor TPM: `{report['governor_config'].get('tpm')}`\n",
        encoding="utf-8",
    )


def log_cli_json(event: str, payload: dict[str, Any]) -> None:
    cli_logger().info(json.dumps(redact({"event": event, **payload}), ensure_ascii=False))


def llm_status_cli() -> int:
    log_cli_json("llm.status", llm_public_status())
    return 0
