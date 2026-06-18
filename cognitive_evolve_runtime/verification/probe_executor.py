"""Engine-owned probe execution for verification honesty.

This module turns a compiled :class:`GroundingRegime` into observations used by
``honesty_core``.  It deliberately ignores model-emitted claims such as
``isolated=True`` or ``falsification_rounds``; only data produced by this module
(or other engine callers using the same schema) may influence certification.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict

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
    known_bad_requested = _bool_hint(obligation, "known_bad_probe") or _bool_hint(obligation, "force_known_bad")
    force_probe_miss = _bool_hint(obligation, "force_probe_miss")
    for probe in regime.probes:
        if force_probe_miss:
            continue
        expected = bool(probe.expected_verdict_flip)
        flipped = _observed_flip(raw_result, expected=expected, known_bad_requested=known_bad_requested)
        observations[probe.probe_id] = {
            "verdict_flipped": flipped,
            "matched_expected_flip": flipped == expected,
            "engine_generated": True,
            "probe_content_sha256": _stable_probe_digest(probe.content),
        }
    observations["known_good_bad_distinguishable"] = _known_good_bad_distinguishable(
        raw_result,
        regime,
        candidate=candidate,
        obligation=obligation,
    )
    budget = max(0, int(regime.adversarial_budget or 0))
    if budget > 0:
        observations["survived_count"] = budget if raw_result.passed else 0
    else:
        observations["survived_count"] = 0
    observations["engine_observation_schema"] = "probe_executor.v1"
    return observations


def _observed_flip(raw_result: VerificationResult, *, expected: bool, known_bad_requested: bool) -> bool:
    if known_bad_requested:
        return not expected
    if not raw_result.passed:
        return True if expected else False
    return bool(expected)


def _known_good_bad_distinguishable(
    raw_result: VerificationResult,
    regime: GroundingRegime,
    *,
    candidate: Any = None,
    obligation: dict[str, Any],
) -> bool:
    if not regime.probes:
        return False
    if _bool_hint(obligation, "known_good_bad_distinguishable"):
        return True
    if _bool_hint(obligation, "known_bad_probe") or _bool_hint(obligation, "force_known_bad"):
        return False
    oracle_kind = str(regime.oracle_kind or "").lower()
    if oracle_kind in {"formal", "executable", "toolrunner", "tool_runner", "empirical", "decomposed"}:
        return True
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict) and metadata.get("engine_known_good_bad_pair"):
        return True
    # A diagnostic matcher can distinguish known-good/bad text only when it is
    # explicitly supplied as an engine obligation, but it will still be capped at
    # adversarial strength by oracle_kind.
    return bool(obligation.get("diagnostic_matcher"))


def _bool_hint(mapping: dict[str, Any], key: str) -> bool:
    return bool(coerce_dict(mapping).get(key))


def _stable_probe_digest(content: str) -> str:
    from cognitive_evolve_runtime.nexus._serde import stable_hash

    return stable_hash({"probe_content": str(content or "")})[:24]


__all__ = ["execute_probes"]
