#!/usr/bin/env python3
"""Smoke test the CognitiveEvolve OpenAI-compatible API.

Usage:
  python scripts/cogev_api_smoke.py

Env:
  COGEV_SERVER_PUBLIC_BASE_URL=http://127.0.0.1:8765/v1
  COGEV_SERVER_API_KEY=...
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_evolve_runtime.api.config import get_service_config, load_service_env  # noqa: E402


def _request(method: str, url: str, key: str, data: dict | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    load_service_env()
    config = get_service_config()
    base_url = config.public_base_url.rstrip("/")
    key = config.api_keys[0] if config.api_keys else "ce-local-dev-key-change-me"
    model = config.default_model
    try:
        models = _request("GET", f"{base_url}/models", key)
        print(json.dumps({"models": [item["id"] for item in models.get("data", [])]}, ensure_ascii=False, indent=2))
        result = _request(
            "POST",
            f"{base_url}/chat/completions",
            key,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Use CognitiveEvolve's one-shot flow to output a pre-v1.0 release checklist."}
                ],
            },
        )
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"API smoke failed: {exc}", file=sys.stderr)
        return 1
    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("\n--- assistant preview ---")
    print(answer[:1200])
    return 0 if answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
