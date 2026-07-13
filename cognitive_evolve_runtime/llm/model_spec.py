"""Safe per-call model routing specifications.

``LLMModelSpec`` intentionally carries only public routing coordinates. Secrets
such as API keys remain in the existing operator configuration and are never
stored in this object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.llm.env import normalize_reasoning_effort


@dataclass(frozen=True)
class LLMModelSpec:
    profile_id: str | None = None
    provider: str | None = None
    model: str | None = None
    api_base: str | None = None
    fixture: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_reasoning_effort(self.reasoning_effort, source="LLMModelSpec.reasoning_effort"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "LLMModelSpec" | None) -> "LLMModelSpec | None":
        if isinstance(data, LLMModelSpec):
            return data
        payload = coerce_dict(data)
        if not payload:
            return None
        return cls(
            profile_id=_clean(payload.get("profile_id") or payload.get("model_profile_id") or payload.get("id")),
            provider=_clean(payload.get("provider")),
            model=_clean(payload.get("model")),
            api_base=_clean(payload.get("api_base") or payload.get("base_url")),
            fixture=_clean(payload.get("fixture")),
            reasoning_effort=payload.get("reasoning_effort"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value}

    def public_summary(self) -> dict[str, Any]:
        data = self.to_dict()
        if data.get("api_base"):
            data["api_base_configured"] = True
            data.pop("api_base", None)
        if data.get("fixture"):
            data["fixture"] = "configured"
        return data

    @property
    def spec_hash(self) -> str:
        return "llm-model-spec-" + stable_hash(self.public_summary())[:16]

    def apply_to_status(self, status: dict[str, Any]) -> dict[str, Any]:
        out = dict(status or {})
        if self.provider:
            out["provider"] = self.provider.strip().lower()
        if self.profile_id:
            out["model_profile_id"] = self.profile_id.strip()
        if self.model:
            out["model"] = self.model.strip()
            out["configured"] = True
        if self.fixture:
            out["fixture"] = self.fixture.strip()
            out["provider"] = "fixture"
            out["model"] = "fixture"
            out["configured"] = True
            out["requires_real_llm"] = False
            out["test_provider_only"] = True
        if self.api_base:
            out["api_base"] = self.api_base.strip()
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
            out["reasoning_effort_configured"] = True
        return out


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None

__all__ = ["LLMModelSpec"]
