"""Safe per-call model routing specifications.

``LLMModelSpec`` intentionally carries only public routing coordinates. Secrets
such as API keys remain in the existing operator configuration and are never
stored in this object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash


@dataclass(frozen=True)
class LLMModelSpec:
    profile_id: str | None = None
    provider: str | None = None
    model: str | None = None
    api_base: str | None = None
    fixture: str | None = None

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value}

    def public_summary(self) -> dict[str, Any]:
        data = self.to_dict()
        if data.get("api_base"):
            data["api_base"] = _redact_url(data["api_base"])
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
        return out


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _redact_url(value: str) -> str:
    text = str(value or "")
    if "@" in text:
        scheme, _, rest = text.partition("://")
        host = rest.rsplit("@", 1)[-1]
        return f"{scheme}://[redacted]@{host}" if scheme else "[redacted]@" + host
    return text


__all__ = ["LLMModelSpec"]
