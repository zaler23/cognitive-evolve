"""Verifier-on-frozen-artifact cache."""
from __future__ import annotations

import threading
from typing import Any

from cognitive_evolve_runtime.core.serialization import stable_hash
from .honesty_core import measure_verification_result
from .probe_executor import execute_probes
from .regime import compile_grounding_regime
from .replay_runner import build_replay_record
from .types import VerificationResult, SynthesizedVerifier

_CACHE_LOCK = threading.RLock()


def candidate_artifact_hash(candidate: Any) -> str:
    artifact = getattr(candidate, "artifact", candidate)
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    verifier_relevant = {}
    if isinstance(metadata, dict):
        verifier_relevant = {k: metadata.get(k) for k in ("verification_command", "z3_formula", "required_claims", "empirical_score") if k in metadata}
    return "artifact-" + stable_hash({"artifact": artifact, "artifact_type": getattr(candidate, "artifact_type", ""), "verifier_metadata": verifier_relevant})[:24]


def verification_cache_key(candidate: Any, verifier_fingerprint: str) -> str:
    return "verification:" + stable_hash({"artifact_sha": candidate_artifact_hash(candidate), "verifier_fingerprint": verifier_fingerprint})


def check_with_cache(candidate: Any, verifier: SynthesizedVerifier, cache: dict[str, dict[str, Any]]) -> tuple[VerificationResult, str, bool]:
    with _CACHE_LOCK:
        return _check_with_cache_locked(candidate, verifier, cache)


def _check_with_cache_locked(candidate: Any, verifier: SynthesizedVerifier, cache: dict[str, dict[str, Any]]) -> tuple[VerificationResult, str, bool]:
    fingerprint = str(getattr(verifier, "fingerprint", "") or getattr(verifier, "verifier_id", ""))
    key = verification_cache_key(candidate, fingerprint)
    legacy_entry_seen = False
    if key in cache and isinstance(cache[key], dict):
        entry = cache[key]
        if isinstance(entry.get("measured_result"), dict):
            result = VerificationResult.from_dict(entry["measured_result"])
            metadata = dict(result.metadata)
            metadata["cache_hit"] = True
            metadata.setdefault("cache_key", key)
            metadata.setdefault("artifact_sha256", candidate_artifact_hash(candidate))
            metadata.setdefault("verifier_fingerprint", fingerprint)
            result = VerificationResult(result.passed, result.score, result.strength, result.evidence_ref, result.replayable, list(result.diagnostics), metadata)
            return result, key, True
        # Legacy entries did not carry honesty measurements.  Keep them for
        # diagnostics, but rerun instead of certifying from stale strength.
        entry["legacy_cache"] = True
        entry["diagnostics_only"] = True
        legacy_entry_seen = True

    raw_result = verifier.check(candidate)
    artifact_sha = candidate_artifact_hash(candidate)
    oracle_kind = str(raw_result.metadata.get("oracle_kind") or getattr(verifier, "verifier_id", "").replace("-verifier", "") or "")
    metadata = dict(raw_result.metadata)
    metadata.update({
        "cache_hit": False,
        "cache_key": key,
        "artifact_sha256": artifact_sha,
        "verifier_fingerprint": fingerprint,
        "replay_scope": "verifier_on_frozen_artifact",
        "diagnostics_only": bool(metadata.get("diagnostics_only", True)),
    })
    raw_result = VerificationResult(raw_result.passed, raw_result.score, raw_result.strength, raw_result.evidence_ref, raw_result.replayable, list(raw_result.diagnostics), metadata)
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint=fingerprint,
        artifact_hash=artifact_sha,
        plan=getattr(verifier, "plan", None) if isinstance(getattr(verifier, "plan", None), dict) else None,
        oracle_kind=oracle_kind,
    )
    actual_probe_verdicts = execute_probes(raw_result, regime, candidate=candidate)
    replay_record = build_replay_record(
        candidate,
        raw_result,
        verifier_fingerprint=fingerprint,
        cache_key=key,
        oracle_kind=oracle_kind,
    )
    measured = measure_verification_result(
        raw_result,
        regime,
        actual_probe_verdicts=actual_probe_verdicts,
        replay_record=replay_record,
    )
    result = measured.to_verification_result()
    cache[key] = {
        "raw_result": raw_result.to_dict(),
        "measured_result": result.to_dict(),
        "honesty_measurements": result.metadata.get("honesty_measurements"),
        "artifact_sha256": artifact_sha,
        "verifier_fingerprint": fingerprint,
        "replay_record": replay_record,
        "actual_probe_verdicts": actual_probe_verdicts,
        "grounding_regime": regime.to_dict(),
        "legacy_cache": legacy_entry_seen,
        "diagnostics_only": legacy_entry_seen,
    }
    return result, key, False


__all__ = ["candidate_artifact_hash", "check_with_cache", "verification_cache_key"]
