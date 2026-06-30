"""Per-call LLM identity for profile-safe accounting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LLMCallIdentity:
    profile_id: str
    provider: str
    model: str
    route_role: str

    @property
    def breaker_key(self) -> str:
        return self.profile_id or f"{self.provider}:{self.model}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def identity_from_status(status: dict[str, Any], *, request_type: str) -> LLMCallIdentity:
    provider = str(status.get("provider") or "").strip() or "unknown"
    model = str(status.get("model") or status.get("fixture") or "").strip() or "unknown"
    profile_id = str(status.get("model_profile_id") or status.get("profile_id") or "").strip()
    if not profile_id:
        profile_id = f"{provider}:{model}"
    return LLMCallIdentity(profile_id=profile_id, provider=provider, model=model, route_role=str(request_type or "default"))


__all__ = ["LLMCallIdentity", "identity_from_status"]
