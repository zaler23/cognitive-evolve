"""Verifier-on-frozen-artifact cache."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash
from .types import VerificationResult, SynthesizedVerifier


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
    fingerprint = str(getattr(verifier, "fingerprint", "") or getattr(verifier, "verifier_id", ""))
    key = verification_cache_key(candidate, fingerprint)
    if key in cache and isinstance(cache[key], dict):
        result = VerificationResult.from_dict(cache[key].get("result") if isinstance(cache[key].get("result"), dict) else cache[key])
        metadata = dict(result.metadata)
        metadata["cache_hit"] = True
        metadata.setdefault("cache_key", key)
        metadata.setdefault("artifact_sha256", candidate_artifact_hash(candidate))
        metadata.setdefault("verifier_fingerprint", fingerprint)
        result = VerificationResult(result.passed, result.score, result.strength, result.evidence_ref, result.replayable, list(result.diagnostics), metadata)
        return result, key, True
    result = verifier.check(candidate)
    metadata = dict(result.metadata)
    metadata.update({
        "cache_hit": False,
        "cache_key": key,
        "artifact_sha256": candidate_artifact_hash(candidate),
        "verifier_fingerprint": fingerprint,
        "replay_scope": "verifier_on_frozen_artifact",
    })
    result = VerificationResult(result.passed, result.score, result.strength, result.evidence_ref, result.replayable, list(result.diagnostics), metadata)
    cache[key] = {"result": result.to_dict(), "artifact_sha256": metadata["artifact_sha256"], "verifier_fingerprint": fingerprint}
    return result, key, False


__all__ = ["candidate_artifact_hash", "check_with_cache", "verification_cache_key"]
