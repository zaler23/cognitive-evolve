"""Stage classification helpers for candidate maturation."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.obligations import candidate_has_obligation_or_evidence_delta

from .constants import EARLY_STAGE, FINAL_STAGE, LATE_STAGE, MIDDLE_STAGE, STAGE_ORDER
from .metrics import parse_metric_value

def stage_for_round(current_round: int, round_limit: int, policy_config: dict[str, Any] | None = None) -> str:
    """Return the coarse global run stage.

    The thresholds intentionally describe search pressure, not completion.
    Adaptive runs may hit the safety limit without being solved; final synthesis
    still uses the strict verifier regardless of this label.
    """

    limit = max(1, int(round_limit or 1))
    current = max(0, int(current_round or 0))
    fraction = current / limit
    thresholds = _stage_thresholds(policy_config)
    if thresholds:
        if fraction < thresholds["early_until"]:
            return EARLY_STAGE
        if fraction < thresholds["middle_until"]:
            return MIDDLE_STAGE
        if fraction < thresholds["late_until"]:
            return LATE_STAGE
        return FINAL_STAGE
    return _signal_adaptive_round_stage(current=current, limit=limit)

def stage_for_candidate_age(candidate_age: int, round_limit: int, policy_config: dict[str, Any] | None = None) -> str:
    """Return the candidate-local maturation stage.

    This prevents a fresh late-run candidate from being killed immediately just
    because the global run is old.  The final answer gate remains strict and is
    not affected by this local leniency.
    """

    age = max(0, int(candidate_age or 0))
    limit = max(1, int(round_limit or 1))
    windows = _candidate_age_windows(policy_config, round_limit=limit)
    if windows:
        early_window = windows["early_until_age"]
        middle_window = windows["middle_until_age"]
        late_window = windows["late_until_age"]
        if age <= early_window:
            return EARLY_STAGE
        if age <= middle_window:
            return MIDDLE_STAGE
        if age <= late_window:
            return LATE_STAGE
        return FINAL_STAGE
    return EARLY_STAGE if age <= 0 else MIDDLE_STAGE

def stage_for_candidate(
    candidate: CandidateGenome,
    *,
    current_round: int = 0,
    round_limit: int = 0,
    policy_config: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, int, int]:
    current = max(0, int(current_round or 0))
    limit = max(1, int(round_limit or 1))
    created = candidate_created_in_round(candidate)
    age = max(0, current - created)
    global_stage = stage_for_round(current, limit, policy_config=policy_config)
    age_stage = stage_for_candidate_age(age, limit, policy_config=policy_config)
    claim_stage = candidate_claim_maturity_stage(candidate)
    # Early and middle phases are deliberately permissive: a candidate that
    # uses final-answer language is not treated as final-grade until late/final
    # pressure.  The final synthesis gate remains strict, but promising partial
    # routes are allowed to incubate and repair instead of being killed early.
    claim_can_raise_early = bool(coerce_dict(policy_config).get("claim_maturity_can_raise_stage_before_late"))
    if global_stage in {EARLY_STAGE, MIDDLE_STAGE} and not claim_can_raise_early:
        stage = global_stage
    else:
        stage = max((global_stage, claim_stage), key=lambda item: STAGE_ORDER[item])
    return stage, global_stage, age_stage, claim_stage, created, age

def candidate_claim_maturity_stage(candidate: CandidateGenome) -> str:
    """Infer whether the candidate's own claim raises its eligibility stage."""

    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    explicit = str(metadata.get("claim_maturity_stage") or metadata.get("evidence_stage") or "").strip().lower()
    if explicit in STAGE_ORDER:
        return explicit
    if metadata.get("objective_solved") or metadata.get("final_claim") or metadata.get("claims_final_answer"):
        return FINAL_STAGE
    text = " ".join(
        str(part or "")
        for part in (
            candidate.concise_claim,
            candidate.core_mechanism,
            candidate.artifact if isinstance(candidate.artifact, str) else "",
        )
    ).lower()
    final_tokens = (
        "final answer",
        "complete proof",
        "proved the theorem",
        "the theorem is proved",
        "objective solved",
        "closed proof",
        "完整证明",
        "最终答案",
        "已经证明",
    )
    if any(_final_token_present(text, token) for token in final_tokens):
        return FINAL_STAGE
    if candidate.formal_artifacts or candidate.evidence_refs or candidate.source_bindings or _has_evidence_progress(candidate):
        return MIDDLE_STAGE
    return EARLY_STAGE

def candidate_created_in_round(candidate: CandidateGenome) -> int:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in ("created_in_round", "reactivated_in_round", "model_seed_batch"):
        value = metadata.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return 0

def candidate_diagnostics(candidate: CandidateGenome) -> set[str]:
    result = getattr(candidate, "verification_result", {}) or {}
    diagnostics: set[str] = set()
    if isinstance(result, dict):
        diagnostics.update(str(item) for item in result.get("diagnostics", []) if item)
        for section in ("proof_progress", "evidence_obligation"):
            payload = result.get(section)
            if isinstance(payload, dict):
                diagnostics.update(str(item) for item in payload.get("diagnostics", []) if item)
    if candidate.missing_parts:
        diagnostics.add("missing_parts")
    return diagnostics

def _has_evidence_progress(candidate: CandidateGenome) -> bool:
    if candidate_has_obligation_or_evidence_delta(candidate):
        return True
    for attr in ("evidence_delta", "obligation_delta"):
        payload = coerce_dict(getattr(candidate, attr, {}))
        if any(bool(value) for value in payload.values()):
            return True
    return False

def _final_token_present(text: str, token: str) -> bool:
    if token == "complete proof" and ("incomplete proof" in text or "not complete proof" in text):
        return False
    if token == "完整证明" and ("缺少完整证明" in text or "没有完整证明" in text):
        return False
    return token in text

def _stage_thresholds(policy_config: dict[str, Any] | None) -> dict[str, float]:
    policy = coerce_dict(policy_config)
    raw = coerce_dict(policy.get("stage_fractions") or policy.get("stage_thresholds"))
    if not raw:
        return {}
    early = _fraction(raw.get("early_until"), default=-1.0)
    middle = _fraction(raw.get("middle_until"), default=-1.0)
    late = _fraction(raw.get("late_until"), default=-1.0)
    if not (0.0 < early < middle < late < 1.0):
        return {}
    return {"early_until": early, "middle_until": middle, "late_until": late}

def _signal_adaptive_round_stage(*, current: int, limit: int) -> str:
    """Fallback phase with no baked-in early/middle/late percentages.

    Model-backed policy may define exact thresholds.  Without that, the runtime
    only treats the safety checkpoint itself as final pressure; all prior rounds
    stay exploratory enough for repairable candidates to keep developing.
    """

    if current <= 0:
        return EARLY_STAGE
    if limit > 0 and current >= limit:
        return FINAL_STAGE
    return MIDDLE_STAGE

def _candidate_age_windows(policy_config: dict[str, Any] | None, *, round_limit: int) -> dict[str, int]:
    policy = coerce_dict(policy_config)
    raw = coerce_dict(policy.get("candidate_age_windows"))
    if raw:
        early = _positive_int(raw.get("early_until_age"))
        middle = _positive_int(raw.get("middle_until_age"))
        late = _positive_int(raw.get("late_until_age"))
        if early and middle and late and early < middle < late:
            return {"early_until_age": early, "middle_until_age": middle, "late_until_age": late}
    fractions = coerce_dict(policy.get("candidate_age_fractions"))
    if not fractions:
        return {}
    early_fraction = _fraction(fractions.get("early_until"), default=-1.0)
    middle_fraction = _fraction(fractions.get("middle_until"), default=-1.0)
    late_fraction = _fraction(fractions.get("late_until"), default=-1.0)
    if not (0.0 < early_fraction < middle_fraction < late_fraction <= 1.0):
        return {}
    limit = max(1, int(round_limit or 1))
    return {
        "early_until_age": max(1, int(round(limit * early_fraction))),
        "middle_until_age": max(1, int(round(limit * middle_fraction))),
        "late_until_age": max(1, int(round(limit * late_fraction))),
    }

def _fraction(value: Any, *, default: float) -> float:
    parsed = parse_metric_value(value)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))

def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0

__all__ = [
    "stage_for_round", "stage_for_candidate_age", "stage_for_candidate",
    "candidate_claim_maturity_stage", "candidate_created_in_round", "candidate_diagnostics",
]
