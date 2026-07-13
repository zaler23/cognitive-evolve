#!/usr/bin/env python3
"""Deterministic mini cache-trace evaluator for EvoCachePolicy-style tests.

This fixture intentionally keeps the simulator small.  It validates machine
artifact shape and applies a bounded, replayable trace to separate obviously
usable policies from schema-only candidates.  It is not an industrial cache
benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("admission", "eviction", "parameters", "update_or_state_update")
TRACE = [
    ("A", 3), ("B", 3), ("A", 3), ("C", 8), ("A", 3), ("B", 3),
    ("D", 3), ("A", 3), ("E", 8), ("A", 3), ("B", 3), ("D", 3),
]
CAPACITY_BYTES = 9


def main(path: str) -> None:
    candidate = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact = candidate.get("artifact")
    diagnostics: list[str] = []
    if candidate.get("artifact_type") != "cache_policy":
        diagnostics.append("candidate artifact_type must be cache_policy")
    if not isinstance(artifact, dict):
        diagnostics.append("candidate artifact must be a JSON object")
        emit(False, diagnostics, schema_cleanliness=0.0)
        return
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        diagnostics.append("missing required cache policy sections: " + ", ".join(missing))
    schema_cleanliness = 1.0 if not diagnostics else max(0.0, 1.0 - 0.2 * len(diagnostics))
    if diagnostics:
        emit(False, diagnostics, schema_cleanliness=schema_cleanliness)
        return
    hit_rate, byte_hit_rate = simulate(artifact)
    score = round(0.5 * hit_rate + 0.3 * byte_hit_rate + 0.2 * schema_cleanliness, 4)
    passed = score >= 0.35 and hit_rate > 0.0
    emit(
        passed,
        [] if passed else ["trace score below threshold"],
        schema_cleanliness=schema_cleanliness,
        score=score,
        hit_rate=hit_rate,
        byte_hit_rate=byte_hit_rate,
    )


def simulate(policy: dict[str, Any]) -> tuple[float, float]:
    params = policy.get("parameters") if isinstance(policy.get("parameters"), dict) else {}
    admission_text = json.dumps(policy.get("admission", {}), sort_keys=True).lower()
    eviction_text = json.dumps(policy.get("eviction", {}), sort_keys=True).lower()
    update_text = json.dumps(policy.get("update_or_state_update", {}), sort_keys=True).lower()
    size_threshold = _float(params.get("size_threshold_ratio"), _float(params.get("size_scaling_factor"), 2.5))
    freq_weight = _float(params.get("frequency_multiplier"), 128.0)
    recency_weight = _float(params.get("base_recency_weight"), 4.0)
    cache: dict[str, dict[str, float]] = {}
    tick = 0
    hits = 0
    byte_hits = 0
    total_bytes = 0
    for key, size in TRACE:
        tick += 1
        total_bytes += size
        if key in cache:
            hits += 1
            byte_hits += size
            cache[key]["frequency"] = min(65535, cache[key]["frequency"] + 1)
            cache[key]["last_tick"] = tick
            continue
        avg_size = sum(item["size"] for item in cache.values()) / max(1, len(cache)) if cache else size
        if not admit(size=size, avg_size=avg_size, admission_text=admission_text, threshold=size_threshold):
            continue
        cache[key] = {"size": float(size), "frequency": 1.0, "last_tick": float(tick)}
        while sum(item["size"] for item in cache.values()) > CAPACITY_BYTES and cache:
            victim = min(cache, key=lambda item_key: eviction_score(cache[item_key], tick=tick, eviction_text=eviction_text, update_text=update_text, freq_weight=freq_weight, recency_weight=recency_weight))
            cache.pop(victim, None)
    return round(hits / len(TRACE), 4), round(byte_hits / max(1, total_bytes), 4)


def admit(*, size: int, avg_size: float, admission_text: str, threshold: float) -> bool:
    if "deny" in admission_text and "size" in admission_text:
        return size <= max(avg_size * threshold, avg_size + 1)
    return True


def eviction_score(entry: dict[str, float], *, tick: int, eviction_text: str, update_text: str, freq_weight: float, recency_weight: float) -> float:
    age = max(0.0, tick - entry.get("last_tick", 0.0))
    score = 0.0
    if "freq" in eviction_text or "frequency" in eviction_text or "freq" in update_text:
        score += entry.get("frequency", 0.0) * freq_weight
    if "last" in eviction_text or "recency" in eviction_text or "tick" in eviction_text:
        score -= age * recency_weight
    if score == 0.0:
        score = -age
    return score


def emit(passed: bool, diagnostics: list[str], **metrics: Any) -> None:
    metrics.setdefault("correctness", bool(passed))
    metrics.setdefault("score", 0.0 if not passed else 1.0)
    metrics.setdefault("challenge_pass_rate", 1.0 if passed else 0.0)
    print(json.dumps({"passed": passed, "metrics": metrics, "diagnostics": diagnostics}, ensure_ascii=False, sort_keys=True))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: cache_trace_evaluator.py candidate.json")
    main(sys.argv[1])
