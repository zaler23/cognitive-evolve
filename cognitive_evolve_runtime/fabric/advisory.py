"""Advisory-signal guards for the Exploration Fabric.

Fabric outputs may influence scheduling, retention, and prompt views, but they
must never grant verification authority.  These helpers keep that boundary
centralized for new v3 code and tests.
"""
from __future__ import annotations

from typing import Any

FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "verification_result",
        "objective_solved",
        "passed",
        "replayable",
        "verification_strength",
        "verification_strength_value",
        "graded_output",
        "verified_result",
        "formal",
        "executable",
    }
)


def authority_key_violations(value: Any, *, path: str = "") -> list[str]:
    """Return dotted paths where advisory payloads try to carry authority keys."""

    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_s = str(key)
            current = f"{path}.{key_s}" if path else key_s
            if key_s in FORBIDDEN_AUTHORITY_KEYS:
                violations.append(current)
            violations.extend(authority_key_violations(item, path=current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            current = f"{path}[{index}]" if path else f"[{index}]"
            violations.extend(authority_key_violations(item, path=current))
    return violations


def assert_advisory_payload(value: Any) -> None:
    """Reject model/scheduler advisory payloads that carry authority fields."""

    violations = authority_key_violations(value)
    if violations:
        raise ValueError("advisory payload contains verification-authority keys: " + ", ".join(violations[:8]))


def advisory_dict(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy marked advisory after checking authority-field leakage."""

    out = dict(data or {})
    if out.get("advisory") is False:
        out.setdefault("diagnostics", [])
        diagnostics = out["diagnostics"] if isinstance(out["diagnostics"], list) else []
        diagnostics.append("advisory_false_coerced_to_true")
        out["diagnostics"] = diagnostics
    out["advisory"] = True
    assert_advisory_payload({k: v for k, v in out.items() if k != "diagnostics"})
    return out


__all__ = ["FORBIDDEN_AUTHORITY_KEYS", "advisory_dict", "assert_advisory_payload", "authority_key_violations"]
