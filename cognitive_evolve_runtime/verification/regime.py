"""Compile model/extension verification hints into engine-owned regimes."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash

from .honesty_core import GroundingRegime, ProbeCase


def compile_grounding_regime(
    *,
    candidate: Any = None,
    verifier_fingerprint: str,
    artifact_hash: str = "",
    raw_obligation: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    oracle_kind: str = "",
    override_adversarial_budget: int | None = None,
) -> GroundingRegime:
    """Turn semantic verifier hints into a regime controlled by the engine.

    Raw obligations may describe intent, but the probe content, budget,
    isolation flag, artifact hash, and fingerprint are supplied here.  This
    prevents model output from self-reporting the inputs later used for
    certification strength.
    """

    obligation = coerce_dict(raw_obligation)
    plan_data = coerce_dict(plan)
    candidate_id = str(getattr(candidate, "id", "") or "")
    artifact_hash = str(artifact_hash or "")
    fingerprint = str(verifier_fingerprint or obligation.get("verifier_fingerprint") or "")
    kind = str(oracle_kind or obligation.get("oracle_kind") or plan_data.get("modality") or "").strip().lower()
    probes = _compile_probes(candidate_id=candidate_id, obligation=obligation, plan=plan_data)
    budget = _engine_falsification_budget(obligation=obligation, plan=plan_data, oracle_kind=kind) if override_adversarial_budget is None else max(0, int(override_adversarial_budget or 0))
    regime_id = "regime-" + stable_hash(
        {
            "candidate_id": candidate_id,
            "artifact_hash": artifact_hash,
            "verifier_fingerprint": fingerprint,
            "obligation_id": obligation.get("id"),
            "oracle_kind": kind,
            "adversarial_budget": budget,
        }
    )[:16]
    return GroundingRegime(
        regime_id=regime_id,
        probes=probes,
        adversarial_budget=budget,
        isolation_enforced=bool(probes and budget > 0),
        replay_artifact_hash=artifact_hash,
        verifier_fingerprint=fingerprint,
        oracle_kind=kind,
    )


def _compile_probes(*, candidate_id: str, obligation: dict[str, Any], plan: dict[str, Any]) -> list[ProbeCase]:
    hints: list[dict[str, Any]] = []
    for source in (obligation.get("exogeneity_probe"), obligation.get("variety_probe"), plan.get("probe_requirements")):
        if isinstance(source, dict):
            hints.append(source)
        elif isinstance(source, list):
            hints.extend(item for item in source if isinstance(item, dict))
    if not hints and not obligation and not plan:
        return []
    if not hints:
        hints = [{"kind": "default_counterfactual", "expected_verdict_flip": False}]
    probes: list[ProbeCase] = []
    for index, hint in enumerate(hints[:4]):
        semantic_label = str(hint.get("kind") or hint.get("label") or "verification_probe")
        # The content is deliberately engine-authored and stable; raw hint text is
        # only treated as a semantic label, not as certification evidence.
        content = "engine_probe:" + stable_hash({"candidate_id": candidate_id, "index": index, "label": semantic_label})[:24]
        probes.append(
            ProbeCase(
                probe_id="probe-" + stable_hash({"candidate_id": candidate_id, "index": index, "label": semantic_label})[:12],
                content=content,
                provenance="engine",
                expected_verdict_flip=bool(hint.get("expected_verdict_flip", False)),
            )
        )
    return probes


def _engine_falsification_budget(*, obligation: dict[str, Any], plan: dict[str, Any], oracle_kind: str) -> int:
    raw_budget = obligation.get("falsification_budget") or obligation.get("adversarial_budget")
    count: Any = None
    if isinstance(raw_budget, dict):
        count = raw_budget.get("count") or raw_budget.get("budget")
    if count is None and isinstance(plan.get("falsification_budget"), dict):
        count = plan["falsification_budget"].get("count") or plan["falsification_budget"].get("budget")
    if count is None and isinstance(plan.get("adversarial_budget"), dict):
        count = plan["adversarial_budget"].get("count") or plan["adversarial_budget"].get("budget")
    if count is None and not isinstance(plan.get("adversarial_budget"), dict):
        count = plan.get("adversarial_budget")
    try:
        value = int(count)
    except (TypeError, ValueError):
        value = 1 if oracle_kind in {"formal", "executable", "toolrunner", "empirical", "decomposed"} else 0
    return max(0, min(value, 32))


__all__ = ["compile_grounding_regime"]
