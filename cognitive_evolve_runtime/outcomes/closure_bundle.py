"""Structural replay bundles for M6 closure gates.

The bundle binds every replay-critical digest into one content-addressed object.
Verification recomputes the structural hash, so changing any bound digest without
the original signing context fails closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash


STRUCTURAL_REPLAY_BUNDLE_VERSION = "m6-structural-replay-bundle/v1"
STRUCTURAL_REPLAY_BUNDLE_PREFIX = "srb:"

REQUIRED_DIGEST_FIELDS = (
    "e_process_digest",
    "calibration_digest",
    "falsification_digest",
    "problem_model_digest",
    "latent_digest",
    "pareto_digest",
    "compaction_digest",
)


@dataclass(frozen=True)
class ClosureReplayDigests:
    e_process_digest: str
    calibration_digest: str
    falsification_digest: str
    problem_model_digest: str
    latent_digest: str
    pareto_digest: str
    compaction_digest: str

    def __post_init__(self) -> None:
        for field in REQUIRED_DIGEST_FIELDS:
            object.__setattr__(self, field, str(getattr(self, field) or ""))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in REQUIRED_DIGEST_FIELDS}

    def missing_fields(self) -> tuple[str, ...]:
        return tuple(field for field in REQUIRED_DIGEST_FIELDS if not getattr(self, field))

    def binding_hash(self) -> str:
        return "crd:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class StructuralReplayVerification:
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    missing_digest_fields: tuple[str, ...] = ()
    mismatched_digest_fields: tuple[str, ...] = ()
    expected_bundle_hash: str = ""
    actual_bundle_hash: str = ""
    digest_binding_hash: str = ""
    version: str = STRUCTURAL_REPLAY_BUNDLE_VERSION

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "missing_digest_fields": list(self.missing_digest_fields),
            "mismatched_digest_fields": list(self.mismatched_digest_fields),
            "expected_bundle_hash": self.expected_bundle_hash,
            "actual_bundle_hash": self.actual_bundle_hash,
            "digest_binding_hash": self.digest_binding_hash,
        }


@dataclass(frozen=True)
class StructuralReplayBundle:
    scope: str
    candidate_id: str
    digests: ClosureReplayDigests | dict[str, Any]
    baseline_id: str = ""
    replay_steps: tuple[dict[str, Any], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    structural_replay_bundle_hash: str = ""
    version: str = STRUCTURAL_REPLAY_BUNDLE_VERSION

    def __post_init__(self) -> None:
        digests = _digests_from_any(self.digests)
        steps = tuple(coerce_dict(_as_dict(step) or step) for step in _as_sequence(self.replay_steps))
        object.__setattr__(self, "scope", str(self.scope or ""))
        object.__setattr__(self, "candidate_id", str(self.candidate_id or ""))
        object.__setattr__(self, "baseline_id", str(self.baseline_id or ""))
        object.__setattr__(self, "digests", digests)
        object.__setattr__(self, "replay_steps", steps)
        object.__setattr__(self, "evidence_refs", _str_tuple(self.evidence_refs))
        object.__setattr__(self, "version", str(self.version or STRUCTURAL_REPLAY_BUNDLE_VERSION))
        if not self.structural_replay_bundle_hash:
            object.__setattr__(self, "structural_replay_bundle_hash", self.bundle_hash())

    @property
    def digest_binding_hash(self) -> str:
        return self.digests.binding_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "digests": self.digests.to_dict(),
            "digest_binding_hash": self.digest_binding_hash,
            "replay_steps": [dict(step) for step in self.replay_steps],
            "evidence_refs": list(self.evidence_refs),
            "structural_replay_bundle_hash": self.structural_replay_bundle_hash,
        }

    def stable_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("structural_replay_bundle_hash", None)
        return payload

    def bundle_hash(self) -> str:
        return STRUCTURAL_REPLAY_BUNDLE_PREFIX + stable_hash(self.stable_payload())

    def verify(self, expected_digests: ClosureReplayDigests | dict[str, Any] | None = None) -> StructuralReplayVerification:
        return audit_structural_replay_bundle(self, expected_digests=expected_digests)


def bind_closure_gate_digests(
    *,
    scope: str,
    candidate_id: str,
    e_process_digest: str,
    calibration_digest: str,
    falsification_digest: str,
    problem_model_digest: str,
    latent_digest: str,
    pareto_digest: str,
    compaction_digest: str,
    baseline_id: str = "",
    replay_steps: tuple[dict[str, Any], ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> StructuralReplayBundle:
    return StructuralReplayBundle(
        scope=scope,
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        digests=ClosureReplayDigests(
            e_process_digest=e_process_digest,
            calibration_digest=calibration_digest,
            falsification_digest=falsification_digest,
            problem_model_digest=problem_model_digest,
            latent_digest=latent_digest,
            pareto_digest=pareto_digest,
            compaction_digest=compaction_digest,
        ),
        replay_steps=replay_steps,
        evidence_refs=evidence_refs,
    )


def build_structural_replay_bundle(
    *,
    scope: str,
    candidate_id: str,
    digests: ClosureReplayDigests | dict[str, Any],
    baseline_id: str = "",
    replay_steps: tuple[dict[str, Any], ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> StructuralReplayBundle:
    return StructuralReplayBundle(
        scope=scope,
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        digests=digests,
        replay_steps=replay_steps,
        evidence_refs=evidence_refs,
    )


def audit_structural_replay_bundle(
    bundle: StructuralReplayBundle | dict[str, Any] | None,
    *,
    expected_digests: ClosureReplayDigests | dict[str, Any] | None = None,
) -> StructuralReplayVerification:
    raw_data = coerce_dict(bundle) if isinstance(bundle, dict) else {}
    parsed = _bundle_from_any(bundle)
    if parsed is None:
        return StructuralReplayVerification(False, ("malformed_structural_replay_bundle",))

    reasons: list[str] = []
    if not parsed.scope:
        reasons.append("missing_scope")
    if not parsed.candidate_id:
        reasons.append("missing_candidate_id")

    missing = parsed.digests.missing_fields()
    if missing:
        reasons.append("missing_required_closure_digest")

    raw_expected_hash = str(raw_data.get("structural_replay_bundle_hash") or raw_data.get("bundle_hash") or "")
    expected_hash = raw_expected_hash if isinstance(bundle, dict) else str(parsed.structural_replay_bundle_hash or "")
    actual_hash = parsed.bundle_hash()
    if not expected_hash:
        reasons.append("missing_structural_replay_bundle_hash")
    elif expected_hash != actual_hash:
        reasons.append("structural_replay_bundle_hash_mismatch")

    mismatched: tuple[str, ...] = ()
    expected = _digests_from_any(expected_digests) if expected_digests is not None else None
    if expected is not None:
        mismatched = tuple(
            field
            for field in REQUIRED_DIGEST_FIELDS
            if getattr(parsed.digests, field) != getattr(expected, field)
        )
        if mismatched:
            reasons.append("closure_digest_mismatch")

    return StructuralReplayVerification(
        passed=not reasons,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        missing_digest_fields=missing,
        mismatched_digest_fields=mismatched,
        expected_bundle_hash=expected_hash,
        actual_bundle_hash=actual_hash,
        digest_binding_hash=parsed.digest_binding_hash,
    )


def verify_structural_replay_bundle(
    bundle: StructuralReplayBundle | dict[str, Any] | None,
    *,
    expected_digests: ClosureReplayDigests | dict[str, Any] | None = None,
) -> bool:
    return audit_structural_replay_bundle(bundle, expected_digests=expected_digests).passed


def structural_replay_bundle_hash(bundle: StructuralReplayBundle | dict[str, Any] | None) -> str:
    parsed = _bundle_from_any(bundle)
    return parsed.bundle_hash() if parsed is not None else ""


def _bundle_from_any(raw: StructuralReplayBundle | dict[str, Any] | None) -> StructuralReplayBundle | None:
    if isinstance(raw, StructuralReplayBundle):
        return raw
    data = coerce_dict(raw)
    if not data:
        return None
    try:
        digests = data.get("digests") or data
        return StructuralReplayBundle(
            scope=str(data.get("scope") or ""),
            candidate_id=str(data.get("candidate_id") or ""),
            baseline_id=str(data.get("baseline_id") or ""),
            digests=_digests_from_any(digests),
            replay_steps=tuple(_as_sequence(data.get("replay_steps"))),
            evidence_refs=_str_tuple(data.get("evidence_refs")),
            structural_replay_bundle_hash=str(data.get("structural_replay_bundle_hash") or data.get("bundle_hash") or ""),
            version=str(data.get("version") or STRUCTURAL_REPLAY_BUNDLE_VERSION),
        )
    except (TypeError, ValueError):
        return None


def _digests_from_any(raw: ClosureReplayDigests | dict[str, Any] | None) -> ClosureReplayDigests:
    if isinstance(raw, ClosureReplayDigests):
        return raw
    data = coerce_dict(raw)
    return ClosureReplayDigests(
        e_process_digest=_digest_value(data, "e_process_digest", "e_process", "e_process_state_hash", "eprocess_digest"),
        calibration_digest=_digest_value(data, "calibration_digest", "calibration", "calibration_snapshot_hash"),
        falsification_digest=_digest_value(data, "falsification_digest", "falsification", "falsification_bundle_hash"),
        problem_model_digest=_digest_value(data, "problem_model_digest", "problem_model", "problem_model_snapshot_hash"),
        latent_digest=_digest_value(data, "latent_digest", "latent", "latent_replay_digest", "latent_snapshot_hash"),
        pareto_digest=_digest_value(data, "pareto_digest", "pareto", "pareto_frontier_hash"),
        compaction_digest=_digest_value(data, "compaction_digest", "compaction", "compaction_hash"),
    )


def _digest_value(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return ""


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
    return tuple(dict.fromkeys(str(item) for item in _as_sequence(value) if str(item)))


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


ClosureGateDigests = ClosureReplayDigests
ClosureGateBundle = StructuralReplayBundle
M6ClosureBundle = StructuralReplayBundle
build_closure_bundle = build_structural_replay_bundle
audit_closure_bundle = audit_structural_replay_bundle
verify_closure_bundle = verify_structural_replay_bundle


__all__ = [
    "REQUIRED_DIGEST_FIELDS",
    "STRUCTURAL_REPLAY_BUNDLE_VERSION",
    "ClosureGateBundle",
    "ClosureGateDigests",
    "ClosureReplayDigests",
    "M6ClosureBundle",
    "StructuralReplayBundle",
    "StructuralReplayVerification",
    "audit_closure_bundle",
    "audit_structural_replay_bundle",
    "bind_closure_gate_digests",
    "build_closure_bundle",
    "build_structural_replay_bundle",
    "structural_replay_bundle_hash",
    "verify_closure_bundle",
    "verify_structural_replay_bundle",
]
