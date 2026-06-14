"""Adapters from Nexus runtime evidence into M5.1 preference evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.outcomes.improvement import ImprovementCertificate, TrialObservation, certificate_from_dict
from cognitive_evolve_runtime.outcomes.latent import LatentProblemState, PreferenceEvidence


ADAPTER_VERSION = "latent-evidence-feedback/v1"
TRUSTED_CERTIFICATE_SOURCES = frozenset({"runtime_verifier", "verifier_result", "tool_verifier", "m5_verifier", "verified_trial"})


@dataclass(frozen=True)
class EvidenceQuarantineRecord:
    source_type: str
    provenance_ref: str
    reason: str
    raw_hash: str
    adapter_version: str = ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAdapterOutput:
    evidence: tuple[PreferenceEvidence, ...] = ()
    quarantined: tuple[EvidenceQuarantineRecord, ...] = ()
    adapter_version: str = ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "evidence": [item.to_dict() for item in self.evidence],
            "quarantined": [item.to_dict() for item in self.quarantined],
        }


def adapt_latent_feedback(
    *,
    state: LatentProblemState,
    critiques: list[Any] | None = None,
    verifier_results: list[Any] | None = None,
    archive_observations: list[Any] | None = None,
    certificates: list[Any] | None = None,
    trial_observations: list[Any] | None = None,
) -> EvidenceAdapterOutput:
    evidence: list[PreferenceEvidence] = []
    quarantined: list[EvidenceQuarantineRecord] = []
    for source, adapter in [
        (critiques or [], adapt_critique_result),
        (verifier_results or [], adapt_verifier_result),
        (archive_observations or [], adapt_archive_observation),
        (certificates or [], adapt_improvement_certificate),
        (trial_observations or [], adapt_trial_observation),
    ]:
        for raw in source:
            output = adapter(raw, state)
            evidence.extend(output.evidence)
            quarantined.extend(output.quarantined)
    return EvidenceAdapterOutput(evidence=tuple(evidence), quarantined=tuple(quarantined))


def adapt_critique_result(raw: Any, state: LatentProblemState) -> EvidenceAdapterOutput:
    data = _to_mapping(raw)
    provenance = _provenance(data, "critique")
    if not data:
        return _quarantined("critique", provenance, "malformed_critique_result", raw)
    text = " ".join(
        str(item)
        for key in ("strengths", "flaws", "missing_evidence", "proposed_mutations", "reusable_genes")
        for item in _list(data.get(key))
    )
    if not text:
        return _quarantined("critique", provenance, "empty_critique_payload", raw)
    support = 0.18 if _list(data.get("strengths")) or _list(data.get("reusable_genes")) else 0.0
    contradiction = 0.16 if _list(data.get("flaws")) or _list(data.get("missing_evidence")) else 0.0
    severity = _bounded_float(data.get("severity"), default=0.0)
    contradiction = min(0.35, contradiction + 0.12 * severity)
    targets = _target_intents(state, text, explicit_intent_id=data.get("intent_id"), fallback="all")
    return _evidence_output(
        state=state,
        raw=raw,
        targets=targets,
        source_type="critique",
        provenance_ref=provenance,
        support=support,
        contradiction=contradiction,
        weight=0.25,
        confidence=0.50,
        calibration="weak_model_or_rule_critique_not_strong_evidence",
    )


def adapt_verifier_result(raw: Any, state: LatentProblemState) -> EvidenceAdapterOutput:
    data = _to_mapping(raw)
    provenance = _provenance(data, "verifier")
    if not data:
        return _quarantined("verifier", provenance, "malformed_verifier_result", raw)
    passed = bool(data.get("passed", data.get("status") in {"ok", "passed", "verified"}))
    diagnostics = " ".join(str(item) for item in _list(data.get("diagnostics")) + _list(data.get("failure_lessons")))
    score_hint = _score_hint(data)
    if passed:
        support = max(0.22, min(0.55, score_hint or 0.35))
        contradiction = 0.0
    else:
        support = 0.0
        contradiction = max(0.28, min(0.65, 1.0 - (score_hint or 0.35)))
    targets = _target_intents(state, diagnostics or provenance, explicit_intent_id=data.get("intent_id"), fallback="all")
    return _evidence_output(
        state=state,
        raw=raw,
        targets=targets,
        source_type="verifier",
        provenance_ref=provenance,
        support=support,
        contradiction=contradiction,
        weight=0.70,
        confidence=0.75,
        calibration="runtime_verifier_evidence",
    )


def adapt_archive_observation(raw: Any, state: LatentProblemState) -> EvidenceAdapterOutput:
    data = _to_mapping(raw)
    provenance = _provenance(data, "archive")
    if not data:
        return _quarantined("archive", provenance, "malformed_archive_observation", raw)
    fate = str(data.get("fate") or data.get("archive_fate") or data.get("status") or "").lower()
    support = 0.0
    contradiction = 0.0
    if fate in {"elite", "active", "answerarchive", "answer_archive"}:
        support = 0.14
    elif fate in {"failed", "culled", "rejected"}:
        contradiction = 0.18
    elif fate in {"dormant", "incubating"}:
        contradiction = 0.08
    else:
        return _quarantined("archive", provenance, "unsupported_archive_observation", raw)
    text = " ".join(str(data.get(key) or "") for key in ("intent_id", "candidate_id", "fate", "reason", "failure_signature"))
    targets = _target_intents(state, text, explicit_intent_id=data.get("intent_id"), fallback="all")
    return _evidence_output(
        state=state,
        raw=raw,
        targets=targets,
        source_type="archive",
        provenance_ref=provenance,
        support=support,
        contradiction=contradiction,
        weight=0.20,
        confidence=0.50,
        calibration="archive_frequency_is_not_desirability",
    )


def adapt_improvement_certificate(raw: Any, state: LatentProblemState) -> EvidenceAdapterOutput:
    data = _to_mapping(raw)
    certificate = _certificate_from_any(raw)
    provenance = _provenance(data, "certificate")
    if certificate is None:
        return _quarantined("verified_improvement_certificate", provenance, "malformed_improvement_certificate", raw)
    explicit = data.get("intent_id") or coerce_dict(data.get("metadata")).get("intent_id")
    targets = _target_intents(state, str(explicit or certificate.challenger_id or certificate.contract_hash), explicit_intent_id=explicit, fallback="top")
    inferred = not explicit and len(state.intents) > 1
    verified = bool(certificate.verified)
    if verified and not _trusted_certificate_provenance(data):
        return _quarantined("verified_improvement_certificate", provenance, "untrusted_verified_certificate_provenance", raw)
    return _evidence_output(
        state=state,
        raw=certificate.to_dict() | {"intent_id": str(explicit or ""), "source_type": str(data.get("source_type") or ""), "provenance_ref": str(data.get("provenance_ref") or "")},
        targets=targets,
        source_type="verified_improvement_certificate" if verified else "improvement_certificate",
        provenance_ref=provenance or certificate.certificate_hash(),
        support=(0.62 if inferred else 0.90) if verified else 0.0,
        contradiction=0.0 if verified else 0.65,
        weight=1.0,
        confidence=0.72 if inferred else 1.0,
        calibration="verified_m5_certificate" if verified else "rejected_m5_certificate",
    )


def _trusted_certificate_provenance(data: dict[str, Any]) -> bool:
    source = str(data.get("source_type") or "").strip()
    provenance = str(data.get("provenance_ref") or data.get("verifier_run_id") or "").strip()
    container = str(data.get("trial_pair_container_source") or "").strip()
    if source not in TRUSTED_CERTIFICATE_SOURCES or not provenance:
        return False
    if container and not container.startswith("candidate.verification_result"):
        return False
    return True


def adapt_trial_observation(raw: Any, state: LatentProblemState) -> EvidenceAdapterOutput:
    data = _to_mapping(raw)
    provenance = _provenance(data, "trial")
    if not data:
        return _quarantined("trial_observation", provenance, "malformed_trial_observation", raw)
    constraints_passed = bool(data.get("constraints_passed", True))
    scores = coerce_dict(data.get("scores"))
    score_mean = sum(_bounded_float(value, default=0.0) for value in scores.values()) / max(1, len(scores))
    support = min(0.18, 0.10 + 0.08 * score_mean) if constraints_passed else 0.0
    contradiction = 0.0 if constraints_passed else 0.28
    text = " ".join([str(data.get("artifact_id") or ""), " ".join(str(key) for key in scores)])
    targets = _target_intents(state, text, explicit_intent_id=data.get("intent_id"), fallback="all")
    return _evidence_output(
        state=state,
        raw=raw,
        targets=targets,
        source_type="trial_observation",
        provenance_ref=provenance,
        support=support,
        contradiction=contradiction,
        weight=0.20,
        confidence=0.50,
        calibration="weak_without_verified_improvement_certificate",
    )


def _evidence_output(
    *,
    state: LatentProblemState,
    raw: Any,
    targets: list[str],
    source_type: str,
    provenance_ref: str,
    support: float,
    contradiction: float,
    weight: float,
    confidence: float,
    calibration: str,
) -> EvidenceAdapterOutput:
    if not targets:
        return _quarantined(source_type, provenance_ref, "no_matching_latent_intent", raw)
    if support <= 0.0 and contradiction <= 0.0:
        return _quarantined(source_type, provenance_ref, "zero_strength_evidence", raw)
    raw_hash = stable_hash(_to_mapping(raw) or str(raw))
    raw_mapping = _to_mapping(raw)
    evidence_round = raw_mapping.get("round") or raw_mapping.get("round_index")
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "raw_hash": raw_hash,
        "intent_count": len(state.intents),
    }
    if evidence_round not in (None, ""):
        metadata["evidence_round"] = evidence_round
    if "age_rounds" in raw_mapping:
        metadata["age_rounds"] = raw_mapping.get("age_rounds")
    if "stale_decay" in raw_mapping:
        metadata["stale_decay"] = raw_mapping.get("stale_decay")
    evidence = tuple(
        PreferenceEvidence(
            intent_id=intent_id,
            support=support,
            contradiction=contradiction,
            weight=weight,
            evidence_ref=f"{source_type}:{provenance_ref or raw_hash}:{intent_id}",
            source_type=source_type,
            provenance_ref=provenance_ref,
            confidence=confidence,
            calibration=calibration,
            metadata=dict(metadata),
        )
        for intent_id in targets
    )
    return EvidenceAdapterOutput(evidence=evidence)


def _target_intents(state: LatentProblemState, text: str, *, explicit_intent_id: Any = None, fallback: str) -> list[str]:
    known = {intent.id for intent in state.intents}
    explicit = str(explicit_intent_id or "").strip()
    if explicit in known:
        return [explicit]
    lowered = str(text or "").lower()
    matched: list[str] = []
    for intent in state.intents:
        tokens = [intent.id, *intent.utility_dimensions, *intent.hard_constraints, intent.statement]
        if any(str(token).lower() and str(token).lower() in lowered for token in tokens):
            matched.append(intent.id)
    if matched:
        return list(dict.fromkeys(matched))
    if fallback == "top":
        return [state.top_intent().id]
    if fallback == "all":
        return [intent.id for intent in state.intents]
    return []


def _certificate_from_any(raw: Any) -> ImprovementCertificate | None:
    if isinstance(raw, ImprovementCertificate):
        return raw
    data = _to_mapping(raw)
    if not data:
        return None
    try:
        return certificate_from_dict(data)
    except Exception:
        return None


def _to_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, TrialObservation):
        return raw.to_dict()
    if isinstance(raw, ImprovementCertificate):
        return raw.to_dict()
    if hasattr(raw, "to_dict"):
        try:
            data = raw.to_dict()
        except Exception:
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def _quarantined(source_type: str, provenance_ref: str, reason: str, raw: Any) -> EvidenceAdapterOutput:
    return EvidenceAdapterOutput(
        quarantined=(
            EvidenceQuarantineRecord(
                source_type=source_type,
                provenance_ref=provenance_ref,
                reason=reason,
                raw_hash=stable_hash(_to_mapping(raw) or str(raw)),
            ),
        )
    )


def _provenance(data: dict[str, Any], prefix: str) -> str:
    candidate_id = str(data.get("candidate_id") or data.get("challenger_id") or data.get("artifact_id") or "").strip()
    round_index = str(data.get("round") or data.get("round_index") or "").strip()
    status = str(data.get("status") or data.get("fate") or "").strip()
    parts = [prefix, candidate_id, round_index, status]
    value = ":".join(part for part in parts if part)
    return value or f"{prefix}:{stable_hash(data)[:12]}"


def _score_hint(data: dict[str, Any]) -> float:
    for key in ("score", "confidence"):
        if key in data:
            return _bounded_float(data.get(key), default=0.0)
    for nested_key in ("proof_progress", "evidence_obligation", "final_gate", "artifact_contract"):
        nested = coerce_dict(data.get(nested_key))
        if "score" in nested:
            return _bounded_float(nested.get("score"), default=0.0)
    return 0.0


def _list(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set):
        return [item for item in value if item is not None]
    if value in (None, ""):
        return []
    return [value]


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return max(0.0, min(1.0, parsed))


__all__ = [
    "ADAPTER_VERSION",
    "EvidenceAdapterOutput",
    "EvidenceQuarantineRecord",
    "adapt_archive_observation",
    "adapt_critique_result",
    "adapt_improvement_certificate",
    "adapt_latent_feedback",
    "adapt_trial_observation",
    "adapt_verifier_result",
]
