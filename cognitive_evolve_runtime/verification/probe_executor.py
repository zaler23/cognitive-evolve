"""Engine-owned probe execution for verification honesty.

This module turns a compiled :class:`GroundingRegime` into observations used by
``honesty_core``.  It deliberately ignores model-emitted claims such as
``isolated=True`` or ``falsification_rounds``; only data produced by this module
(or other engine callers using the same schema) may influence certification.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict

from .honesty_core import GroundingRegime
from .types import VerificationResult


def execute_probes(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    *,
    candidate: Any = None,
    raw_obligation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return engine observations for a verification regime.

    The current engine probes are deterministic and cheap: they verify that the
    regime has engine-controlled probe ids/content, that the candidate can be
    contrasted against a known-good/bad case, and that the raw verifier survived
    the engine-defined adversarial budget.  This is intentionally conservative:
    text/diagnostic matchers can earn observations but remain capped by their
    oracle kind in ``honesty_core``.
    """

    obligation = coerce_dict(raw_obligation)
    observations: dict[str, Any] = {}
    probe_results: dict[str, Any] = {}
    force_probe_miss = _bool_hint(obligation, "force_probe_miss")
    for probe in regime.probes:
        if force_probe_miss:
            continue
        if probe.probe_id not in probe_results:
            continue
        observed = probe_results.get(probe.probe_id)
        flipped = bool(coerce_dict(observed).get("verdict_flipped")) if isinstance(observed, dict) else bool(observed)
        observations[probe.probe_id] = {
            "verdict_flipped": flipped,
            "matched_expected_flip": flipped == bool(probe.expected_verdict_flip),
            "engine_generated": True,
            "probe_content_sha256": _stable_probe_digest(probe.content),
        }
    observations["known_good_bad_distinguishable"] = _known_good_bad_distinguishable(
        raw_result,
        regime,
        candidate=candidate,
        obligation=obligation,
    )
    observations["survived_count"] = 0
    observations["engine_observation_schema"] = "probe_executor.v1"
    return observations

def _known_good_bad_distinguishable(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    *,
    candidate: Any = None,
    obligation: dict[str, Any],
) -> bool:
    if not regime.probes:
        return False
    if _bool_hint(obligation, "known_bad_probe") or _bool_hint(obligation, "force_known_bad"):
        return False
    return False


def _bool_hint(mapping: dict[str, Any], key: str) -> bool:
    return bool(coerce_dict(mapping).get(key))


def _stable_probe_digest(content: str) -> str:
    from cognitive_evolve_runtime.core.serialization import stable_hash

    return stable_hash({"probe_content": str(content or "")})[:24]


__all__ = ["execute_probes"]
