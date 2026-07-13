"""Normalize input/tool/model evidence into explicit provenance records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, utc_now

INPUT_EVIDENCE = "Input Evidence"
TOOL_EVIDENCE = "Tool Evidence"
MODEL_HYPOTHESIS = "Model Hypothesis"


@dataclass
class EvidenceRecord:
    kind: str
    content: Any
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceNormalizer:
    def normalize(self, value: Any, *, kind: str, source: str = "", confidence: float = 0.0) -> EvidenceRecord:
        if kind not in {INPUT_EVIDENCE, TOOL_EVIDENCE, MODEL_HYPOTHESIS, "input_evidence", "tool_evidence", "model_hypothesis"}:
            raise ValueError(f"unsupported evidence kind: {kind}")
        return EvidenceRecord(kind=kind, content=value, source=source, confidence=float(confidence), metadata=coerce_dict(getattr(value, "metadata", {})))


__all__ = ["INPUT_EVIDENCE", "TOOL_EVIDENCE", "MODEL_HYPOTHESIS", "EvidenceRecord", "EvidenceNormalizer"]
