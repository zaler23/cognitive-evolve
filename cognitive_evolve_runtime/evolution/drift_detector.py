"""Semantic drift detection; Chinese terms support multilingual drift signals."""
from __future__ import annotations

from typing import Any

JARGON_TERMS = {
    "selmer", "poitou", "tate", "langlands", "motivic", "derived", "tensor", "spectral", "category", "高级", "术语"
}
META_TERMS = {"router", "classifier", "classification", "logging", "dashboard", "metadata", "summary", "summarization", "路由", "分类", "日志", "总结"}
MECHANISM_TERMS = {"construction", "test", "verifier", "evidence", "lemma", "mechanism", "implementation", "example", "trace", "构造", "测试", "验证", "证据", "机制"}


class DriftDetector:
    def detect(self, candidates: list[Any], *, contract: dict[str, Any] | None = None, search_descriptor: dict[str, Any] | None = None, lineage_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        signals: list[dict[str, Any]] = []
        for raw_candidate in candidates:
            candidate = _candidate_mapping(raw_candidate)
            text = _candidate_text(candidate)
            lowered = text.lower()
            has_jargon = any(term in lowered for term in JARGON_TERMS)
            has_meta = any(term in lowered for term in META_TERMS)
            has_mechanism = _has_positive_mechanism(lowered) or bool(candidate.get("mechanism_trace") or candidate.get("core_mechanism"))
            gate_score = _gate_score(candidate)
            if has_jargon and not has_mechanism and gate_score <= 0:
                signals.append({"candidate_id": candidate.get("id"), "signal": "terminology_attractor", "severity": "high"})
            if has_meta and gate_score <= 0 and not _objective_is_meta(contract):
                signals.append({"candidate_id": candidate.get("id"), "signal": "meta_substitution", "severity": "medium"})
        lineage_signal = _lineage_lock_in(lineage_history or [])
        if lineage_signal:
            signals.append(lineage_signal)
        status = "semantic_drift_detected" if any(item.get("severity") == "high" for item in signals) else "ok"
        return {
            "status": status,
            "signals": signals,
            "restart_decision": "branch_restart" if status == "semantic_drift_detected" else "continue_search",
            "search_descriptor_version": (search_descriptor or {}).get("version"),
        }


def _has_positive_mechanism(lowered: str) -> bool:
    for term in MECHANISM_TERMS:
        if term not in lowered:
            continue
        negated = any(pattern in lowered for pattern in [f"no {term}", f"without {term}", f"lacks {term}", f"缺少{term}", f"无{term}"])
        if not negated:
            return True
    return False


def _candidate_mapping(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    if hasattr(candidate, "to_dict"):
        data = candidate.to_dict()
        return data if isinstance(data, dict) else {"artifact": str(candidate)}
    if hasattr(candidate, "__dict__"):
        return dict(vars(candidate))
    return {"artifact": str(candidate)}


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ["title", "summary", "strategy", "actions", "design_lines", "claims", "validation", "risks", "artifact", "concise_claim", "core_mechanism"]:
        value = candidate.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _gate_score(candidate: dict[str, Any]) -> float:
    gate = candidate.get("hard_gate_satisfaction")
    if isinstance(gate, (int, float)):
        return float(gate)
    gate_data = candidate.get("contract_gate") if isinstance(candidate.get("contract_gate"), dict) else {}
    try:
        return float(gate_data.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _objective_is_meta(contract: dict[str, Any] | None) -> bool:
    text = str((contract or {}).get("objective") or "").lower()
    return any(term in text for term in META_TERMS)


def _lineage_lock_in(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(history) < 3:
        return None
    recent = history[-3:]
    ids = [str(item.get("lineage_id") or item.get("source_candidate") or item.get("candidate_id") or "") for item in recent]
    if ids and len(set(ids)) == 1 and not any(item.get("hard_gate_progress") or item.get("evidence_progress") or item.get("verifier_progress") for item in recent):
        return {"candidate_id": ids[-1], "signal": "lineage_without_gate_progress", "severity": "high"}
    return None


__all__ = ["DriftDetector"]
