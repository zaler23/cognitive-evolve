from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .env import LLMConfigurationError, LLMResponseError


def load_fixture_response(request_type: str, payload: dict[str, Any], fixture_path: str) -> dict[str, Any]:
    try:
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LLMConfigurationError(f"Invalid LLM fixture {fixture_path}: {exc}") from exc
    responses = fixture.get("responses", fixture)
    if request_type == "classify_route":
        prompt = str(payload.get("prompt", "")).lower()
        for case in responses.get("classify_route_cases", []):
            contains = [str(term).lower() for term in case.get("contains", [])]
            if contains and all(term in prompt for term in contains):
                response = case.get("response", {})
                if isinstance(response, dict):
                    return response
    response = responses.get(request_type)
    if response is None:
        response = responses.get("default", {}).get(request_type)
    if not isinstance(response, dict):
        raise LLMResponseError(f"Fixture has no response for request_type={request_type}")
    return response
