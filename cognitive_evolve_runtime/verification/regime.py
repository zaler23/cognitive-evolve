"""Compile model/extension verification hints into engine-owned regimes."""
from __future__ import annotations

import json
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash

from .honesty_core import GroundingRegime, ProbeCase


_ARTIFACT_ASSERTION_TEMPLATE = "artifact_assertion/v1"
_TYPED_ARTIFACT_RELATION_TEMPLATE = "artifact_json_relation/v2"
_ARTIFACT_ASSERTION_OPERATORS = {
    "exists",
    "not_exists",
    "equal",
    "not_equal",
    "contains",
    "not_contains",
    "lt",
    "lte",
    "gt",
    "gte",
    "between",
}
_UNSUPPORTED_MODEL_EXECUTION_FIELDS = {
    "args",
    "command",
    "cwd",
    "executable",
    "expression",
    "file",
    "file_path",
    "module",
    "python",
    "python_expression",
    "shell",
}


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
    probes = _compile_probes(candidate=candidate, candidate_id=candidate_id, obligation=obligation, plan=plan_data)
    budget = _engine_falsification_budget(obligation=obligation, plan=plan_data, oracle_kind=kind) if override_adversarial_budget is None else max(0, int(override_adversarial_budget or 0))
    probe_signature = "probe-" + stable_hash(
        {
            "artifact_hash": artifact_hash,
            "adversarial_budget": budget,
            "probes": [probe.to_dict() for probe in probes],
            "template_version": _ARTIFACT_ASSERTION_TEMPLATE,
            "verifier_fingerprint": fingerprint,
        }
    )[:24]
    regime_id = "regime-" + stable_hash(
        {
            "candidate_id": candidate_id,
            "artifact_hash": artifact_hash,
            "verifier_fingerprint": fingerprint,
            "obligation_id": obligation.get("id"),
            "oracle_kind": kind,
            "adversarial_budget": budget,
            "probe_signature": probe_signature,
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
        probe_signature=probe_signature,
    )


def _compile_probes(*, candidate: Any, candidate_id: str, obligation: dict[str, Any], plan: dict[str, Any]) -> list[ProbeCase]:
    parameterized: list[ProbeCase] = []
    sources = [obligation]
    sources.extend(coerce_dict(item) for item in getattr(candidate, "proof_obligations", []) if isinstance(item, dict))
    for source_index, source in enumerate(sources):
        cases = source.get("probe_cases")
        if not isinstance(cases, list):
            continue
        obligation_id = str(source.get("id") or obligation.get("id") or f"obligation-{source_index}")
        for case_index, raw_case in enumerate(cases):
            if not isinstance(raw_case, dict):
                continue
            parameterized.append(
                _compile_artifact_assertion_case(
                    candidate_id=candidate_id,
                    obligation_id=obligation_id,
                    case_index=case_index,
                    raw_case=raw_case,
                )
            )

    hints: list[dict[str, Any]] = []
    for source in (obligation.get("exogeneity_probe"), obligation.get("variety_probe"), plan.get("probe_requirements")):
        if isinstance(source, dict):
            hints.append(source)
        elif isinstance(source, list):
            hints.extend(item for item in source if isinstance(item, dict))
    if not hints and not obligation and not plan and not parameterized:
        return []
    if not hints and not parameterized:
        hints = [{"kind": "default_counterfactual", "expected_verdict_flip": False}]
    probes: list[ProbeCase] = list(parameterized)
    for index, hint in enumerate(hints):
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
    return list({probe.probe_id: probe for probe in probes}.values())


def _compile_artifact_assertion_case(
    *,
    candidate_id: str,
    obligation_id: str,
    case_index: int,
    raw_case: dict[str, Any],
) -> ProbeCase:
    declared_template_id = str(raw_case.get("probe_template_id") or raw_case.get("template") or "")
    if declared_template_id == _TYPED_ARTIFACT_RELATION_TEMPLATE:
        template_id, assertion_id, pointer, operator, expected, reason = _compile_typed_relation_fields(raw_case)
    else:
        template_id = declared_template_id
        assertion_id = str(raw_case.get("assertion_id") or "")
        pointer = str(raw_case.get("path") or "")
        operator = str(raw_case.get("operator") or "")
        expected = raw_case.get("expected")
        forbidden = sorted(key for key in raw_case if str(key) in _UNSUPPORTED_MODEL_EXECUTION_FIELDS)
        reason = ""
        if template_id != _ARTIFACT_ASSERTION_TEMPLATE:
            reason = "unsupported_template"
        elif forbidden:
            reason = "unsupported_fields:" + ",".join(forbidden)
        elif not assertion_id:
            reason = "assertion_id_required"
        elif pointer and not pointer.startswith("/"):
            reason = "invalid_json_pointer"
        elif operator not in _ARTIFACT_ASSERTION_OPERATORS:
            reason = "unsupported_operator"
    parameters = {
        "assertion_id": assertion_id,
        "path": pointer,
        "operator": operator,
        "expected": expected,
    }
    if reason:
        parameters["unsupported_reason"] = reason
    identity = {
        "candidate_id": candidate_id,
        "obligation_id": obligation_id,
        "case_index": case_index,
        "template_id": template_id,
        "parameters": parameters,
    }
    return ProbeCase(
        probe_id="probe-" + stable_hash(identity)[:12],
        content="engine_template:" + template_id + ":" + stable_hash(identity)[:24],
        provenance="engine_template_model_parameters",
        expected_verdict_flip=False,
        template_id=template_id,
        parameters=parameters,
    )


def _compile_typed_relation_fields(raw_case: dict[str, Any]) -> tuple[str, str, str, str, Any, str]:
    template_id = _TYPED_ARTIFACT_RELATION_TEMPLATE
    unsupported = sorted(str(key) for key in raw_case if key not in {"probe_template_id", "args", "expected_relation"})
    if unsupported:
        return template_id, "", "", "", None, "unsupported_fields:" + ",".join(unsupported)
    args = raw_case.get("args")
    relation = raw_case.get("expected_relation")
    if not isinstance(args, dict) or set(args) != {"path"} or not isinstance(args.get("path"), str):
        return template_id, "", "", "", None, "invalid_typed_args"
    if not isinstance(relation, dict):
        return template_id, "", "", "", None, "expected_relation_must_be_object"
    operator = str(relation.get("operator") or "")
    required_relation_fields = {"operator"} if operator in {"exists", "not_exists"} else {"operator", "value"}
    if set(relation) != required_relation_fields:
        return template_id, "", "", operator, None, "invalid_expected_relation_fields"
    pointer = args["path"]
    expected = relation.get("value")
    reason = ""
    if pointer and not pointer.startswith("/"):
        reason = "invalid_json_pointer"
    elif len(pointer) > 512:
        reason = "json_pointer_exceeds_limit"
    elif operator not in _ARTIFACT_ASSERTION_OPERATORS:
        reason = "unsupported_operator"
    elif not _typed_relation_value(operator, expected):
        reason = "invalid_expected_relation_value"
    assertion_id = "relation-" + stable_hash({"args": args, "expected_relation": relation})[:12]
    return template_id, assertion_id, pointer, operator, expected, reason


def _typed_relation_value(operator: str, expected: Any) -> bool:
    if operator in {"exists", "not_exists"}:
        return True
    try:
        json.dumps(expected, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    if operator in {"lt", "lte", "gt", "gte"}:
        return not isinstance(expected, bool) and isinstance(expected, (int, float))
    if operator == "between":
        return (
            isinstance(expected, list)
            and len(expected) == 2
            and all(not isinstance(value, bool) and isinstance(value, (int, float)) for value in expected)
        )
    return True


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
