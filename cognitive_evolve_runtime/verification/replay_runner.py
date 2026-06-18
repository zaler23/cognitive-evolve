"""Replay evidence builder for verifier-on-frozen-artifact certification."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash

from .types import VerificationResult

_REPLAYABLE_ORACLES = {"formal", "executable", "toolrunner", "tool_runner", "empirical"}


def build_replay_record(
    candidate: Any,
    raw_result: VerificationResult,
    *,
    verifier_fingerprint: str,
    cache_key: str = "",
    oracle_kind: str = "",
) -> dict[str, Any]:
    """Build a replay record scoped to a frozen artifact and verifier.

    Cache hits are evidence that a result was previously computed, not replay by
    themselves.  ``replay_verified`` is true only when the raw verifier is
    replayable and its oracle kind is one the runtime can independently rerun or
    re-evaluate on a frozen artifact.
    """

    artifact_sha = _candidate_artifact_hash(candidate)
    kind = str(oracle_kind or raw_result.metadata.get("oracle_kind") or "").lower()
    replay_verified = bool(raw_result.replayable and (kind in _REPLAYABLE_ORACLES or raw_result.metadata.get("replay_verified") is True))
    return {
        "frozen_artifact_hash": artifact_sha,
        "artifact_sha256": artifact_sha,
        "verifier_fingerprint": str(verifier_fingerprint or raw_result.metadata.get("verifier_fingerprint") or raw_result.metadata.get("fingerprint") or ""),
        "verification_cache_key": str(cache_key or ""),
        "oracle_kind": kind,
        "replay_verified": replay_verified,
        "replay_scope": "verifier_on_frozen_artifact" if replay_verified else "not_replay_certified",
    }


def _candidate_artifact_hash(candidate: Any) -> str:
    artifact = getattr(candidate, "artifact", candidate)
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    verifier_relevant = {}
    if isinstance(metadata, dict):
        verifier_relevant = {k: metadata.get(k) for k in ("verification_command", "z3_formula", "required_claims", "empirical_score") if k in metadata}
    return "artifact-" + stable_hash({"artifact": artifact, "artifact_type": getattr(candidate, "artifact_type", ""), "verifier_metadata": verifier_relevant})[:24]


__all__ = ["build_replay_record"]
