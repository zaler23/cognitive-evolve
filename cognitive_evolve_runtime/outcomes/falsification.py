"""Deterministic M6 closure falsification gates.

The closure gauntlet is deliberately fail-closed: a solve claim survives only
when every required challenge has trusted, replayable evidence and no challenge
finds a counterexample.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash


FALSIFICATION_VERSION = "m6-closure-falsification/v1"
FALSIFICATION_BUNDLE_PREFIX = "fgb:"


@dataclass(frozen=True)
class FalsificationCase:
    case_id: str
    assertion_ref: str = ""
    challenge: str = ""
    expected_failure_mode: str = ""
    severity: str = "critical"
    required: bool = True
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id or ""))
        object.__setattr__(self, "assertion_ref", str(self.assertion_ref or ""))
        object.__setattr__(self, "challenge", str(self.challenge or ""))
        object.__setattr__(self, "expected_failure_mode", str(self.expected_failure_mode or ""))
        object.__setattr__(self, "severity", str(self.severity or "critical"))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def case_hash(self) -> str:
        return "fgc:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class FalsificationOutcome:
    case_id: str
    survived: bool = False
    evidence_ref: str = ""
    trusted: bool = True
    status: str = ""
    counterexample_ref: str = ""
    notes: str = ""
    passed: bool | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if not status and self.passed is not None:
            status = "survived" if bool(self.passed) else "falsified"
        if not status:
            status = "survived" if bool(self.survived) else "falsified"
        if status in {"pass", "passed", "ok"}:
            status = "survived"
        if status in {"fail", "failed", "counterexample"}:
            status = "falsified"

        survived = status == "survived"
        object.__setattr__(self, "case_id", str(self.case_id or ""))
        object.__setattr__(self, "survived", survived)
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or ""))
        object.__setattr__(self, "trusted", bool(self.trusted))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "counterexample_ref", str(self.counterexample_ref or ""))
        object.__setattr__(self, "notes", str(self.notes or ""))
        object.__setattr__(self, "passed", survived)
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    @property
    def falsified(self) -> bool:
        return self.status == "falsified" or bool(self.counterexample_ref)

    @property
    def inconclusive(self) -> bool:
        return self.status in {"inconclusive", "unknown", "missing"}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["falsified"] = self.falsified
        data["inconclusive"] = self.inconclusive
        return data

    def outcome_hash(self) -> str:
        return "fgo:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class FalsificationAudit:
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    required_case_ids: tuple[str, ...] = ()
    survived_case_ids: tuple[str, ...] = ()
    failed_case_ids: tuple[str, ...] = ()
    missing_case_ids: tuple[str, ...] = ()
    expected_bundle_hash: str = ""
    actual_bundle_hash: str = ""
    version: str = FALSIFICATION_VERSION

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "required_case_ids": list(self.required_case_ids),
            "survived_case_ids": list(self.survived_case_ids),
            "failed_case_ids": list(self.failed_case_ids),
            "missing_case_ids": list(self.missing_case_ids),
            "expected_bundle_hash": self.expected_bundle_hash,
            "actual_bundle_hash": self.actual_bundle_hash,
        }


@dataclass(frozen=True)
class FalsificationGauntlet:
    scope: str
    candidate_id: str
    problem_model_snapshot_hash: str
    cases: tuple[FalsificationCase, ...] = ()
    outcomes: tuple[FalsificationOutcome, ...] = ()
    required_case_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    falsification_bundle_hash: str = ""
    version: str = FALSIFICATION_VERSION

    def __post_init__(self) -> None:
        cases = tuple(item for item in (_case_from_any(raw) for raw in _as_sequence(self.cases)) if item is not None)
        outcomes = tuple(item for item in (_outcome_from_any(raw) for raw in _as_sequence(self.outcomes)) if item is not None)
        required = _str_tuple(self.required_case_ids) or tuple(item.case_id for item in cases if item.required and item.case_id)
        evidence_refs = _str_tuple(self.evidence_refs) or tuple(
            dict.fromkeys(item.evidence_ref for item in outcomes if item.evidence_ref)
        )
        object.__setattr__(self, "scope", str(self.scope or ""))
        object.__setattr__(self, "candidate_id", str(self.candidate_id or ""))
        object.__setattr__(self, "problem_model_snapshot_hash", str(self.problem_model_snapshot_hash or ""))
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "required_case_ids", required)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "version", str(self.version or FALSIFICATION_VERSION))
        if not self.falsification_bundle_hash:
            object.__setattr__(self, "falsification_bundle_hash", self.bundle_hash())

    @property
    def passed(self) -> bool:
        return audit_falsification_gauntlet(self).passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "problem_model_snapshot_hash": self.problem_model_snapshot_hash,
            "cases": [item.to_dict() for item in self.cases],
            "case_hashes": [item.case_hash() for item in self.cases],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "outcome_hashes": [item.outcome_hash() for item in self.outcomes],
            "required_case_ids": list(self.required_case_ids),
            "evidence_refs": list(self.evidence_refs),
            "falsification_bundle_hash": self.falsification_bundle_hash,
        }

    def stable_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("falsification_bundle_hash", None)
        return payload

    def bundle_hash(self) -> str:
        return FALSIFICATION_BUNDLE_PREFIX + stable_hash(self.stable_payload())

    def audit(self) -> FalsificationAudit:
        return audit_falsification_gauntlet(self)


def build_falsification_gauntlet(
    *,
    scope: str,
    candidate_id: str,
    problem_model_snapshot_hash: str,
    cases: list[FalsificationCase | dict[str, Any]] | tuple[FalsificationCase | dict[str, Any], ...],
    outcomes: list[FalsificationOutcome | dict[str, Any]] | tuple[FalsificationOutcome | dict[str, Any], ...],
    required_case_ids: tuple[str, ...] = (),
) -> FalsificationGauntlet:
    return FalsificationGauntlet(
        scope=scope,
        candidate_id=candidate_id,
        problem_model_snapshot_hash=problem_model_snapshot_hash,
        cases=tuple(cases),
        outcomes=tuple(outcomes),
        required_case_ids=required_case_ids,
    )


def audit_falsification_gauntlet(bundle: FalsificationGauntlet | dict[str, Any] | None) -> FalsificationAudit:
    raw_data = coerce_dict(bundle) if isinstance(bundle, dict) else {}
    parsed = _gauntlet_from_any(bundle)
    if parsed is None:
        return FalsificationAudit(False, ("malformed_falsification_bundle",))

    reasons: list[str] = []
    failed_cases: list[str] = []
    survived_cases: list[str] = []

    if not parsed.scope:
        reasons.append("missing_scope")
    if not parsed.candidate_id:
        reasons.append("missing_candidate_id")
    if not parsed.problem_model_snapshot_hash:
        reasons.append("missing_problem_model_snapshot_hash")
    if not parsed.cases:
        reasons.append("missing_falsification_cases")
    if not parsed.required_case_ids:
        reasons.append("missing_required_falsification_cases")

    raw_expected_hash = str(raw_data.get("falsification_bundle_hash") or raw_data.get("bundle_hash") or "")
    expected_hash = raw_expected_hash if isinstance(bundle, dict) else str(parsed.falsification_bundle_hash or "")
    actual_hash = parsed.bundle_hash()
    if not expected_hash:
        reasons.append("missing_falsification_bundle_hash")
    elif expected_hash != actual_hash:
        reasons.append("falsification_bundle_hash_mismatch")

    case_ids = [item.case_id for item in parsed.cases if item.case_id]
    if len(case_ids) != len(set(case_ids)):
        reasons.append("duplicate_falsification_case")
    case_id_set = set(case_ids)

    outcomes_by_case: dict[str, list[FalsificationOutcome]] = {}
    for outcome in parsed.outcomes:
        outcomes_by_case.setdefault(outcome.case_id, []).append(outcome)
        if outcome.case_id not in case_id_set:
            reasons.append("unknown_falsification_outcome")
            failed_cases.append(outcome.case_id)

    missing_cases = [case_id for case_id in parsed.required_case_ids if case_id not in outcomes_by_case]
    if missing_cases:
        reasons.append("missing_required_falsification_outcome")

    required_set = set(parsed.required_case_ids)
    for case_id, results in outcomes_by_case.items():
        if len(results) > 1:
            reasons.append("duplicate_falsification_outcome")
            failed_cases.append(case_id)
            continue
        outcome = results[0]
        is_required = case_id in required_set
        outcome_reasons = _outcome_failure_reasons(outcome, required=is_required)
        if outcome_reasons:
            reasons.extend(outcome_reasons)
            failed_cases.append(case_id)
        elif is_required:
            survived_cases.append(case_id)

    return FalsificationAudit(
        passed=not reasons,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        required_case_ids=parsed.required_case_ids,
        survived_case_ids=tuple(dict.fromkeys(survived_cases)),
        failed_case_ids=tuple(dict.fromkeys(case for case in failed_cases if case)),
        missing_case_ids=tuple(dict.fromkeys(missing_cases)),
        expected_bundle_hash=expected_hash,
        actual_bundle_hash=actual_hash,
    )


def verify_falsification_gauntlet(bundle: FalsificationGauntlet | dict[str, Any] | None) -> bool:
    return audit_falsification_gauntlet(bundle).passed


def falsification_bundle_hash(bundle: FalsificationGauntlet | dict[str, Any] | None) -> str:
    parsed = _gauntlet_from_any(bundle)
    return parsed.bundle_hash() if parsed is not None else ""


def _outcome_failure_reasons(outcome: FalsificationOutcome, *, required: bool) -> tuple[str, ...]:
    reasons: list[str] = []
    if not outcome.trusted:
        reasons.append("untrusted_falsification_outcome")
    if required and not outcome.evidence_ref:
        reasons.append("missing_replayable_falsification_evidence")
    if outcome.inconclusive:
        reasons.append("inconclusive_falsification_outcome")
    if outcome.falsified:
        reasons.append("falsification_counterexample_found")
    if required and not outcome.survived:
        reasons.append("required_falsification_not_survived")
    return tuple(dict.fromkeys(reasons))


def _gauntlet_from_any(raw: FalsificationGauntlet | dict[str, Any] | None) -> FalsificationGauntlet | None:
    if isinstance(raw, FalsificationGauntlet):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    try:
        return FalsificationGauntlet(
            scope=str(data.get("scope") or ""),
            candidate_id=str(data.get("candidate_id") or ""),
            problem_model_snapshot_hash=str(data.get("problem_model_snapshot_hash") or ""),
            cases=tuple(_as_sequence(data.get("cases"))),
            outcomes=tuple(_as_sequence(data.get("outcomes"))),
            required_case_ids=_str_tuple(data.get("required_case_ids")),
            evidence_refs=_str_tuple(data.get("evidence_refs")),
            falsification_bundle_hash=str(data.get("falsification_bundle_hash") or data.get("bundle_hash") or ""),
            version=str(data.get("version") or FALSIFICATION_VERSION),
        )
    except (TypeError, ValueError):
        return None


def _case_from_any(raw: FalsificationCase | dict[str, Any] | None) -> FalsificationCase | None:
    if isinstance(raw, FalsificationCase):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    return FalsificationCase(
        case_id=str(data.get("case_id") or data.get("probe_id") or data.get("id") or ""),
        assertion_ref=str(data.get("assertion_ref") or data.get("target_digest") or ""),
        challenge=str(data.get("challenge") or data.get("probe") or ""),
        expected_failure_mode=str(data.get("expected_failure_mode") or ""),
        severity=str(data.get("severity") or "critical"),
        required=bool(data.get("required", True)),
        metadata=coerce_dict(data.get("metadata")),
    )


def _outcome_from_any(raw: FalsificationOutcome | dict[str, Any] | None) -> FalsificationOutcome | None:
    if isinstance(raw, FalsificationOutcome):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    passed = data.get("passed")
    return FalsificationOutcome(
        case_id=str(data.get("case_id") or data.get("probe_id") or data.get("id") or ""),
        survived=bool(data.get("survived", data.get("passed", False))),
        evidence_ref=str(data.get("evidence_ref") or ""),
        trusted=bool(data.get("trusted", True)),
        status=str(data.get("status") or ""),
        counterexample_ref=str(data.get("counterexample_ref") or ""),
        notes=str(data.get("notes") or ""),
        passed=None if passed is None else bool(passed),
        metadata=coerce_dict(data.get("metadata")),
    )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value, key=str))
    return (value,)


def _str_tuple(value: Any) -> tuple[str, ...]:
    items = _as_sequence(value)
    return tuple(dict.fromkeys(str(item) for item in items if str(item)))


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return coerce_dict(value.to_dict())
        except (TypeError, ValueError):
            return {}
    if is_dataclass(value):
        return coerce_dict(asdict(value))
    return {}


FalsificationProbe = FalsificationCase
FalsificationResult = FalsificationOutcome
FalsificationBundle = FalsificationGauntlet
build_falsification_bundle = build_falsification_gauntlet
audit_falsification_bundle = audit_falsification_gauntlet
verify_falsification_bundle = verify_falsification_gauntlet


__all__ = [
    "FALSIFICATION_VERSION",
    "FalsificationAudit",
    "FalsificationBundle",
    "FalsificationCase",
    "FalsificationGauntlet",
    "FalsificationOutcome",
    "FalsificationProbe",
    "FalsificationResult",
    "audit_falsification_bundle",
    "audit_falsification_gauntlet",
    "build_falsification_bundle",
    "build_falsification_gauntlet",
    "falsification_bundle_hash",
    "verify_falsification_bundle",
    "verify_falsification_gauntlet",
]
