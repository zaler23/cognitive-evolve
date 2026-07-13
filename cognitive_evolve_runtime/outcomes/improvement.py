"""Outcome-improvement primitives for verifiable evolution.

M4 can propose and preserve artifacts.  This module adds the M5 boundary: a
candidate is not "better" because an evaluator liked it; it is better only when
a scoped, replayable comparison against a pinned baseline verifies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash, utc_now

ImprovementStatus = Literal["verified", "rejected", "inconclusive", "quarantined", "revoked", "stale"]
MetricDirection = Literal["maximize", "minimize"]
ComparisonRule = Literal["weighted_lcb", "pareto_lcb"]


@dataclass(frozen=True)
class OutcomeMetric:
    """One objective dimension in a frozen outcome contract."""

    id: str
    weight: float = 1.0
    direction: MetricDirection = "maximize"
    protected_regression_tolerance: float | None = None
    hard: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeContract:
    """Content-addressed definition of what "better" means for one scope."""

    objective: str
    metrics: tuple[OutcomeMetric, ...]
    scope: str = "declared_problem_scope"
    comparison_rule: ComparisonRule = "weighted_lcb"
    min_effect: float = 0.0
    confidence: float = 0.95
    hard_constraints: tuple[str, ...] = ()
    required_same_basis: tuple[str, ...] = ("manifest_hash", "environment_hash", "evaluator_hash")
    require_independent_verifier: bool = True
    version: str = "outcome-contract/v1"

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("outcome contract requires at least one metric")
        for metric in self.metrics:
            if metric.direction not in {"maximize", "minimize"}:
                raise ValueError(f"unknown metric direction: {metric.direction}")
        if self.comparison_rule not in {"weighted_lcb", "pareto_lcb"}:
            raise ValueError(f"unknown comparison rule: {self.comparison_rule}")
        if not 0 < float(self.confidence) < 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "objective": self.objective,
            "scope": self.scope,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "comparison_rule": self.comparison_rule,
            "min_effect": float(self.min_effect),
            "confidence": float(self.confidence),
            "hard_constraints": list(self.hard_constraints),
            "required_same_basis": list(self.required_same_basis),
            "require_independent_verifier": bool(self.require_independent_verifier),
        }

    def contract_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class TrialObservation:
    """Replayable observation of one artifact under a pinned contract basis."""

    artifact_id: str
    contract_hash: str
    manifest_hash: str
    environment_hash: str
    evaluator_hash: str
    scores: dict[str, float]
    uncertainty_radius: dict[str, float] = field(default_factory=dict)
    constraints_passed: bool = True
    hard_constraint_failures: tuple[str, ...] = ()
    raw_observation_ref: str = ""
    resource_usage: dict[str, float] = field(default_factory=dict)
    proposer_ref: str = ""
    verifier_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    seed: str = ""
    source_type: str = ""
    provenance_ref: str = ""
    verifier_run_id: str = ""
    raw_observation_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", _coerce_float_dict(self.scores))
        object.__setattr__(self, "uncertainty_radius", _coerce_nonnegative_float_dict(self.uncertainty_radius))
        object.__setattr__(self, "resource_usage", _coerce_nonnegative_float_dict(self.resource_usage))
        object.__setattr__(self, "hard_constraint_failures", tuple(str(item) for item in self.hard_constraint_failures if str(item)))
        object.__setattr__(self, "evidence_refs", tuple(str(item) for item in self.evidence_refs if str(item)))
        object.__setattr__(self, "source_type", str(self.source_type or ""))
        object.__setattr__(self, "provenance_ref", str(self.provenance_ref or ""))
        object.__setattr__(self, "verifier_run_id", str(self.verifier_run_id or ""))
        object.__setattr__(self, "raw_observation_hash", str(self.raw_observation_hash or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "contract_hash": self.contract_hash,
            "manifest_hash": self.manifest_hash,
            "environment_hash": self.environment_hash,
            "evaluator_hash": self.evaluator_hash,
            "scores": dict(self.scores),
            "uncertainty_radius": dict(self.uncertainty_radius),
            "constraints_passed": bool(self.constraints_passed),
            "hard_constraint_failures": list(self.hard_constraint_failures),
            "raw_observation_ref": self.raw_observation_ref,
            "resource_usage": dict(self.resource_usage),
            "proposer_ref": self.proposer_ref,
            "verifier_ref": self.verifier_ref,
            "evidence_refs": list(self.evidence_refs),
            "seed": self.seed,
            "source_type": self.source_type,
            "provenance_ref": self.provenance_ref,
            "verifier_run_id": self.verifier_run_id,
            "raw_observation_hash": self.raw_observation_hash,
        }

    def observation_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class ImprovementCertificate:
    """Proof-carrying scoped improvement claim."""

    contract_hash: str
    baseline_id: str
    challenger_id: str
    baseline_observation_hash: str
    challenger_observation_hash: str
    evidence_hash: str
    status: ImprovementStatus
    aggregate_delta: float
    aggregate_lcb: float
    metric_deltas: dict[str, float]
    metric_lcbs: dict[str, float]
    checks: tuple[dict[str, Any], ...]
    critical_failures: tuple[str, ...] = ()
    issued_at_utc: str = field(default_factory=utc_now)
    version: str = "improvement-certificate/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "issued_at_utc": self.issued_at_utc,
            "contract_hash": self.contract_hash,
            "baseline_id": self.baseline_id,
            "challenger_id": self.challenger_id,
            "baseline_observation_hash": self.baseline_observation_hash,
            "challenger_observation_hash": self.challenger_observation_hash,
            "evidence_hash": self.evidence_hash,
            "status": self.status,
            "aggregate_delta": float(self.aggregate_delta),
            "aggregate_lcb": float(self.aggregate_lcb),
            "metric_deltas": dict(self.metric_deltas),
            "metric_lcbs": dict(self.metric_lcbs),
            "checks": [dict(check) for check in self.checks],
            "critical_failures": list(self.critical_failures),
        }

    def certificate_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("issued_at_utc", None)
        return stable_hash(payload)

    @property
    def verified(self) -> bool:
        return self.status == "verified" and not self.critical_failures


@dataclass(frozen=True)
class ImprovementEdge:
    """Lineage edge that may advance only when its certificate verifies."""

    parent_artifact_id: str
    child_artifact_id: str
    certificate_hash: str
    status: ImprovementStatus
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_outcomes(contract: OutcomeContract, baseline: TrialObservation, challenger: TrialObservation) -> ImprovementCertificate:
    """Compare challenger against baseline and mint a fail-closed certificate."""

    contract_hash = contract.contract_hash()
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    _record_check(checks, failures, "contract_hash_matches", baseline.contract_hash == contract_hash and challenger.contract_hash == contract_hash, {
        "expected": contract_hash,
        "baseline": baseline.contract_hash,
        "challenger": challenger.contract_hash,
    })
    basis_values = {
        "manifest_hash": (baseline.manifest_hash, challenger.manifest_hash),
        "environment_hash": (baseline.environment_hash, challenger.environment_hash),
        "evaluator_hash": (baseline.evaluator_hash, challenger.evaluator_hash),
    }
    for basis_name in contract.required_same_basis:
        baseline_value, challenger_value = basis_values.get(str(basis_name), ("", ""))
        _record_check(checks, failures, f"basis_present:{basis_name}", bool(baseline_value and challenger_value), {
            "baseline": bool(baseline_value),
            "challenger": bool(challenger_value),
        })
    _record_check(checks, failures, "same_manifest", baseline.manifest_hash == challenger.manifest_hash and bool(baseline.manifest_hash), {
        "baseline": baseline.manifest_hash,
        "challenger": challenger.manifest_hash,
    })
    _record_check(checks, failures, "same_environment", baseline.environment_hash == challenger.environment_hash and bool(baseline.environment_hash), {
        "baseline": baseline.environment_hash,
        "challenger": challenger.environment_hash,
    })
    _record_check(checks, failures, "same_evaluator", baseline.evaluator_hash == challenger.evaluator_hash and bool(baseline.evaluator_hash), {
        "baseline": baseline.evaluator_hash,
        "challenger": challenger.evaluator_hash,
    })
    _record_check(checks, failures, "baseline_constraints_passed", baseline.constraints_passed, list(baseline.hard_constraint_failures))
    _record_check(checks, failures, "challenger_constraints_passed", challenger.constraints_passed, list(challenger.hard_constraint_failures))
    _record_check(checks, failures, "raw_evidence_present", bool(baseline.raw_observation_ref and challenger.raw_observation_ref), {
        "baseline": bool(baseline.raw_observation_ref),
        "challenger": bool(challenger.raw_observation_ref),
    })
    _record_check(checks, failures, "raw_evidence_replayable", _raw_evidence_ref_replayable(baseline) and _raw_evidence_ref_replayable(challenger), {
        "baseline_ref": baseline.raw_observation_ref,
        "challenger_ref": challenger.raw_observation_ref,
        "baseline_hash": bool(baseline.raw_observation_hash),
        "challenger_hash": bool(challenger.raw_observation_hash),
    })
    independent = bool(challenger.verifier_ref and challenger.proposer_ref and challenger.verifier_ref != challenger.proposer_ref)
    _record_check(checks, failures, "independent_verifier", independent or not contract.require_independent_verifier, {
        "required": contract.require_independent_verifier,
        "proposer_ref": challenger.proposer_ref,
        "verifier_ref": challenger.verifier_ref,
    })

    metric_deltas: dict[str, float] = {}
    metric_lcbs: dict[str, float] = {}
    aggregate_delta = 0.0
    aggregate_lcb = 0.0
    pareto_failure = False
    protected_failure = False
    for metric in contract.metrics:
        if metric.id not in baseline.scores or metric.id not in challenger.scores:
            _record_check(checks, failures, f"metric_present:{metric.id}", False, "missing baseline or challenger score")
            continue
        raw_delta = challenger.scores[metric.id] - baseline.scores[metric.id]
        if metric.direction == "minimize":
            raw_delta = -raw_delta
        radius = baseline.uncertainty_radius.get(metric.id, 0.0) + challenger.uncertainty_radius.get(metric.id, 0.0)
        lcb = raw_delta - radius
        metric_deltas[metric.id] = raw_delta
        metric_lcbs[metric.id] = lcb
        aggregate_delta += metric.weight * raw_delta
        aggregate_lcb += metric.weight * lcb
        tolerance = metric.protected_regression_tolerance if metric.protected_regression_tolerance is not None else (0.0 if metric.hard else None)
        if tolerance is not None and lcb < -float(tolerance):
            protected_failure = True
            _record_check(checks, failures, f"protected_metric_non_regression:{metric.id}", False, {
                "lcb": lcb,
                "tolerance": tolerance,
            })
        if contract.comparison_rule == "pareto_lcb" and lcb < 0:
            pareto_failure = True

    if contract.comparison_rule == "pareto_lcb":
        any_practical_gain = any(lcb > contract.min_effect for lcb in metric_lcbs.values())
        dominance_passed = bool(metric_lcbs and not pareto_failure and any_practical_gain)
    else:
        dominance_passed = aggregate_lcb > float(contract.min_effect)
    _record_check(checks, failures, "dominance_lcb_threshold", dominance_passed, {
        "comparison_rule": contract.comparison_rule,
        "aggregate_lcb": aggregate_lcb,
        "min_effect": contract.min_effect,
        "metric_lcbs": metric_lcbs,
    })
    _record_check(checks, failures, "protected_metrics_ok", not protected_failure, metric_lcbs)

    evidence_payload = {
        "contract_hash": contract_hash,
        "baseline_observation_hash": baseline.observation_hash(),
        "challenger_observation_hash": challenger.observation_hash(),
        "metric_deltas": metric_deltas,
        "metric_lcbs": metric_lcbs,
        "aggregate_delta": aggregate_delta,
        "aggregate_lcb": aggregate_lcb,
        "checks": checks,
    }
    status: ImprovementStatus = "verified" if not failures else "rejected"
    return ImprovementCertificate(
        contract_hash=contract_hash,
        baseline_id=baseline.artifact_id,
        challenger_id=challenger.artifact_id,
        baseline_observation_hash=baseline.observation_hash(),
        challenger_observation_hash=challenger.observation_hash(),
        evidence_hash=stable_hash(evidence_payload),
        status=status,
        aggregate_delta=aggregate_delta,
        aggregate_lcb=aggregate_lcb,
        metric_deltas=metric_deltas,
        metric_lcbs=metric_lcbs,
        checks=tuple(checks),
        critical_failures=tuple(dict.fromkeys(failures)),
    )


def _raw_evidence_ref_replayable(observation: TrialObservation) -> bool:
    """Return whether the observation points to replayable raw evidence."""

    ref = str(observation.raw_observation_ref or "").strip()
    if not ref:
        return False
    if observation.raw_observation_hash:
        return True
    if observation.evidence_refs:
        return True
    trusted_prefixes = ("raw:", "verifier:", "trace:", "sha256:", "file:")
    if ref.startswith(trusted_prefixes):
        return True
    path = Path(ref)
    return path.is_absolute() and path.exists()


def improvement_edge(parent_artifact_id: str, child_artifact_id: str, certificate: ImprovementCertificate) -> ImprovementEdge:
    """Create a lineage edge from a certificate without upgrading failed claims."""

    return ImprovementEdge(
        parent_artifact_id=parent_artifact_id,
        child_artifact_id=child_artifact_id,
        certificate_hash=certificate.certificate_hash(),
        status="verified" if certificate.verified else certificate.status,
        reason_codes=tuple(certificate.critical_failures),
    )


def verify_certificate(
    certificate: ImprovementCertificate | dict[str, Any],
    *,
    contract: OutcomeContract,
    baseline: TrialObservation,
    challenger: TrialObservation,
) -> ImprovementCertificate:
    """Recompute a certificate from dependencies and reject digest/status drift."""

    original = certificate if isinstance(certificate, ImprovementCertificate) else certificate_from_dict(certificate)
    recomputed = compare_outcomes(contract, baseline, challenger)
    failures = list(recomputed.critical_failures)
    if original.contract_hash != recomputed.contract_hash:
        failures.append("certificate_contract_hash_drift")
    if original.baseline_observation_hash != recomputed.baseline_observation_hash:
        failures.append("baseline_observation_hash_drift")
    if original.challenger_observation_hash != recomputed.challenger_observation_hash:
        failures.append("challenger_observation_hash_drift")
    if original.evidence_hash != recomputed.evidence_hash:
        failures.append("evidence_hash_drift")
    if original.status != recomputed.status:
        failures.append("certificate_status_drift")
    if not failures:
        return recomputed
    checks = list(recomputed.checks)
    checks.append({"check": "certificate_recomputes", "passed": False, "detail": sorted(set(failures))})
    return ImprovementCertificate(
        contract_hash=recomputed.contract_hash,
        baseline_id=recomputed.baseline_id,
        challenger_id=recomputed.challenger_id,
        baseline_observation_hash=recomputed.baseline_observation_hash,
        challenger_observation_hash=recomputed.challenger_observation_hash,
        evidence_hash=recomputed.evidence_hash,
        status="rejected",
        aggregate_delta=recomputed.aggregate_delta,
        aggregate_lcb=recomputed.aggregate_lcb,
        metric_deltas=recomputed.metric_deltas,
        metric_lcbs=recomputed.metric_lcbs,
        checks=tuple(checks),
        critical_failures=tuple(dict.fromkeys(failures)),
    )


def certificate_from_dict(data: dict[str, Any]) -> ImprovementCertificate:
    return ImprovementCertificate(
        contract_hash=str(data.get("contract_hash") or ""),
        baseline_id=str(data.get("baseline_id") or ""),
        challenger_id=str(data.get("challenger_id") or ""),
        baseline_observation_hash=str(data.get("baseline_observation_hash") or ""),
        challenger_observation_hash=str(data.get("challenger_observation_hash") or ""),
        evidence_hash=str(data.get("evidence_hash") or ""),
        status=_status(data.get("status")),
        aggregate_delta=float(data.get("aggregate_delta") or 0.0),
        aggregate_lcb=float(data.get("aggregate_lcb") or 0.0),
        metric_deltas=_coerce_float_dict(data.get("metric_deltas")),
        metric_lcbs=_coerce_float_dict(data.get("metric_lcbs")),
        checks=tuple(dict(item) for item in data.get("checks", []) if isinstance(item, dict)),
        critical_failures=tuple(str(item) for item in data.get("critical_failures", []) if str(item)),
        issued_at_utc=str(data.get("issued_at_utc") or utc_now()),
        version=str(data.get("version") or "improvement-certificate/v1"),
    )


def _record_check(checks: list[dict[str, Any]], failures: list[str], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    if not passed:
        failures.append(name)


def _status(value: Any) -> ImprovementStatus:
    raw = str(value or "rejected")
    if raw in {"verified", "rejected", "inconclusive", "quarantined", "revoked", "stale"}:
        return raw  # type: ignore[return-value]
    return "rejected"


def _coerce_float_dict(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in coerce_dict(value).items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _coerce_nonnegative_float_dict(value: Any) -> dict[str, float]:
    return {key: max(0.0, val) for key, val in _coerce_float_dict(value).items()}


__all__ = [
    "ImprovementCertificate",
    "ImprovementEdge",
    "OutcomeContract",
    "OutcomeMetric",
    "TrialObservation",
    "certificate_from_dict",
    "compare_outcomes",
    "improvement_edge",
    "verify_certificate",
]
