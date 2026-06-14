"""Text input packet and text world model for Nexus runtime."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_str_list, stable_hash, utc_now

_SENTENCE_SPLIT = re.compile(r"(?<=[。.!?？])\s+|\n+")


@dataclass
class TextInputPacket:
    raw_text: str
    extracted_claims: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    task_type_hypotheses: list[str] = field(default_factory=list)
    available_evidence: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_zones: list[str] = field(default_factory=list)
    possible_edge_knowledge_seeds: list[str] = field(default_factory=list)
    packet_id: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.packet_id:
            self.packet_id = "textpkt-" + stable_hash({"raw_text": self.raw_text})[:12]
        self.extracted_claims = coerce_str_list(self.extracted_claims)
        self.constraints = coerce_str_list(self.constraints)
        self.task_type_hypotheses = coerce_str_list(self.task_type_hypotheses)
        self.uncertainty_zones = coerce_str_list(self.uncertainty_zones)
        self.possible_edge_knowledge_seeds = coerce_str_list(self.possible_edge_knowledge_seeds)

    @classmethod
    def from_text(cls, text: str) -> "TextInputPacket":
        raw = str(text or "")
        segments = [seg.strip() for seg in _SENTENCE_SPLIT.split(raw) if seg.strip()]
        constraints = [seg for seg in segments if _looks_like_constraint(seg)]
        claims = [seg for seg in segments if seg not in constraints][:20]
        hypotheses = _task_type_hypotheses(raw)
        uncertainties = [seg for seg in segments if any(marker in seg.lower() for marker in ["uncertain", "unknown", "maybe", "不确定", "可能", "风险"])]
        edge = _edge_seed_hints(raw)
        evidence = [{"kind": "input_evidence", "source": "raw_text", "content": raw, "confidence": 1.0}]
        return cls(
            raw_text=raw,
            extracted_claims=claims,
            constraints=constraints,
            task_type_hypotheses=hypotheses,
            available_evidence=evidence,
            uncertainty_zones=uncertainties,
            possible_edge_knowledge_seeds=edge,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextInputPacket":
        return cls(
            raw_text=str(data.get("raw_text") or ""),
            extracted_claims=coerce_str_list(data.get("extracted_claims")),
            constraints=coerce_str_list(data.get("constraints")),
            task_type_hypotheses=coerce_str_list(data.get("task_type_hypotheses")),
            available_evidence=[dict(item) for item in data.get("available_evidence", []) if isinstance(item, dict)],
            uncertainty_zones=coerce_str_list(data.get("uncertainty_zones")),
            possible_edge_knowledge_seeds=coerce_str_list(data.get("possible_edge_knowledge_seeds")),
            packet_id=str(data.get("packet_id") or ""),
            created_at=str(data.get("created_at") or utc_now()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "TextInputPacket":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("text packet JSON must decode to an object")
        return cls.from_dict(data)


@dataclass
class TextWorldModel:
    kind: str = "text"
    input_packet_id: str = ""
    goal_summary: str = ""
    evidence_boundaries: dict[str, list[str]] = field(default_factory=dict)
    likely_task_types: list[str] = field(default_factory=list)
    constraint_summary: list[str] = field(default_factory=list)
    uncertainty_zones: list[str] = field(default_factory=list)
    edge_seed_pool: list[str] = field(default_factory=list)

    @classmethod
    def from_packet(cls, packet: TextInputPacket) -> "TextWorldModel":
        return cls(
            input_packet_id=packet.packet_id,
            goal_summary=(packet.extracted_claims[0] if packet.extracted_claims else packet.raw_text[:240]),
            evidence_boundaries={
                "input_evidence": [item.get("source", "raw_text") for item in packet.available_evidence],
                "tool_evidence": [],
                "model_hypothesis": packet.possible_edge_knowledge_seeds,
            },
            likely_task_types=list(packet.task_type_hypotheses),
            constraint_summary=list(packet.constraints),
            uncertainty_zones=list(packet.uncertainty_zones),
            edge_seed_pool=list(packet.possible_edge_knowledge_seeds),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextWorldModel":
        return cls(
            kind=str(data.get("kind") or "text"),
            input_packet_id=str(data.get("input_packet_id") or ""),
            goal_summary=str(data.get("goal_summary") or ""),
            evidence_boundaries={str(k): _coerce_boundary_values(v) for k, v in dict(data.get("evidence_boundaries") or {}).items()},
            likely_task_types=coerce_str_list(data.get("likely_task_types")),
            constraint_summary=coerce_str_list(data.get("constraint_summary")),
            uncertainty_zones=coerce_str_list(data.get("uncertainty_zones")),
            edge_seed_pool=coerce_str_list(data.get("edge_seed_pool")),
        )


class TextInputProcessor:
    def build_world_model(self, text: str) -> tuple[TextInputPacket, TextWorldModel]:
        packet = TextInputPacket.from_text(text)
        return packet, TextWorldModel.from_packet(packet)


def _looks_like_constraint(segment: str) -> bool:
    lowered = segment.lower()
    return any(token in lowered for token in ["must", "should", "require", "do not", "不要", "必须", "要求", "不能", "禁止"])


def _coerce_boundary_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    return [str(value)]


def _task_type_hypotheses(text: str) -> list[str]:
    lowered = text.lower()
    hypotheses: list[str] = []
    if any(token in lowered for token in ["code", "repo", "patch", "pytest", "project", "仓库", "项目", "测试"]):
        hypotheses.append("project_or_code")
    if any(token in lowered for token in ["prove", "math", "theorem", "证明", "数学"]):
        hypotheses.append("math_or_formal_reasoning")
    if any(token in lowered for token in ["research", "paper", "报告", "研究"]):
        hypotheses.append("research_or_report")
    if not hypotheses:
        hypotheses.append("general_text_task")
    return hypotheses


def _edge_seed_hints(text: str) -> list[str]:
    lowered = text.lower()
    seeds: list[str] = []
    if any(token in lowered for token in ["rare", "edge", "obscure", "低频", "边缘", "冷门"]):
        seeds.append("explicit_edge_or_obscure_route")
    if any(token in lowered for token in ["invert", "反例", "对偶", "逆"]):
        seeds.append("inversion_or_dual_route")
    if any(token in lowered for token in ["analogy", "类比", "迁移"]):
        seeds.append("analogy_transfer_route")
    return seeds


__all__ = ["TextInputPacket", "TextWorldModel", "TextInputProcessor"]
