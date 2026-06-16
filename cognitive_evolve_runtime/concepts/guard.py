"""Runtime and CI guards for concept authority contracts.

This module deliberately avoids importing ``ResearchSignal`` at runtime.  It
uses duck typing so ``nexus.adaptive.research.signal`` remains the pure data and
merge layer instead of importing back into ``concepts``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .contract import CHANNEL_AUTHORITY, PROPOSAL_CHANNELS, ConceptContract
from .effects import effect_to_dict

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers.
    from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal


class ConceptAuthorityError(RuntimeError):
    """Raised by strict guard when a concept writes outside its contract."""


@dataclass(frozen=True)
class GuardViolation:
    concept_id: str
    channel: str
    reason: str
    authority: int = 0
    max_authority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "channel": self.channel,
            "reason": self.reason,
            "authority": int(self.authority),
            "max_authority": int(self.max_authority),
        }


@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    violations: list[GuardViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "violations": [item.to_dict() for item in self.violations]}


def enforce_strict(signal: Any, contract: ConceptContract) -> Any:
    """Fail-closed CI/test guard."""

    result = validate_signal(signal, contract)
    if not result.accepted:
        details = "; ".join(f"{v.channel}:{v.reason}" for v in result.violations)
        raise ConceptAuthorityError(f"concept authority violation for {contract.concept_id}: {details}")
    return signal


def filter_live_signal(signal: Any, contract: ConceptContract, trace: Any | None = None) -> GuardResult:
    """Fail-safe live guard: validate and return whether to keep or drop.

    The caller owns construction of an empty replacement signal to avoid a
    runtime import from this guard module back into ``ResearchSignal``.
    """

    result = validate_signal(signal, contract)
    if result.accepted:
        return result
    if trace is not None:
        payload = {
            "event": "concept_guard_violation",
            "concept_id": contract.concept_id,
            "violations": [item.to_dict() for item in result.violations],
        }
        if hasattr(trace, "record_violation"):
            trace.record_violation(payload)
        elif hasattr(trace, "record"):
            try:
                trace.record(round_index=_round_index(signal), concept_id=contract.concept_id, consumed_refs=[], produced_effects=payload, cost={}, decision_changed=False, replay_hash="guard-violation")
            except TypeError:
                trace.record(payload)
    return result


def validate_signal(signal: Any, contract: ConceptContract) -> GuardResult:
    actual = non_empty_channels(signal)
    violations: list[GuardViolation] = []
    for channel in sorted(actual):
        if channel not in contract.produces:
            violations.append(
                GuardViolation(
                    concept_id=contract.concept_id,
                    channel=channel,
                    reason="channel_not_declared_in_contract",
                    authority=int(CHANNEL_AUTHORITY.get(channel, 0)),
                    max_authority=int(contract.max_authority),
                )
            )
            continue
        if channel in PROPOSAL_CHANNELS:
            violations.extend(_proposal_violations(signal, contract, channel))
            continue
        authority = int(CHANNEL_AUTHORITY.get(channel, 0))
        if authority > int(contract.max_authority):
            violations.append(
                GuardViolation(
                    concept_id=contract.concept_id,
                    channel=channel,
                    reason="channel_authority_exceeds_contract",
                    authority=authority,
                    max_authority=int(contract.max_authority),
                )
            )
    return GuardResult(accepted=not violations, violations=violations)


def non_empty_channels(signal: Any) -> set[str]:
    out: set[str] = set()
    for channel in CHANNEL_AUTHORITY:
        value = getattr(signal, channel, None)
        if _is_non_empty(value):
            out.add(channel)
    return out


def signal_channel_counts(signal: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for channel in CHANNEL_AUTHORITY:
        value = getattr(signal, channel, None)
        if not _is_non_empty(value):
            continue
        if isinstance(value, dict):
            counts[channel] = len(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            counts[channel] = len(value)
        else:
            counts[channel] = 1
    return counts


def _proposal_violations(signal: Any, contract: ConceptContract, channel: str) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    proposals = getattr(signal, channel, []) or []
    if not isinstance(proposals, list):
        proposals = [proposals]
    for proposal in proposals:
        data = effect_to_dict(proposal)
        if data.get("requires_approval") is not True:
            violations.append(
                GuardViolation(
                    concept_id=contract.concept_id,
                    channel=channel,
                    reason="contract_delta_proposal_requires_approval_must_be_true",
                    authority=-1,
                    max_authority=int(contract.max_authority),
                )
            )
        if not str(data.get("objective_hash_before") or "") or not str(data.get("objective_hash_after") or ""):
            violations.append(
                GuardViolation(
                    concept_id=contract.concept_id,
                    channel=channel,
                    reason="contract_delta_proposal_requires_objective_hash_fork",
                    authority=-1,
                    max_authority=int(contract.max_authority),
                )
            )
    return violations


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return bool(value)
    return bool(value)


def _round_index(signal: Any) -> int:
    try:
        return int(getattr(signal, "round_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ConceptAuthorityError",
    "GuardResult",
    "GuardViolation",
    "enforce_strict",
    "filter_live_signal",
    "non_empty_channels",
    "signal_channel_counts",
    "validate_signal",
]
