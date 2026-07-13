"""Domain adapter protocol defaults for progressive evaluation."""
from __future__ import annotations

from typing import Any, Protocol


class ProgressiveDomainAdapter(Protocol):
    domain_id: str

    def artifact_similarity(self, left: Any, right: Any) -> float: ...

    def project_artifact_for_user(self, artifact: Any, *, evidence: dict[str, Any] | None = None) -> str: ...


class GenericDomainAdapter:
    domain_id = "general"

    def artifact_similarity(self, left: Any, right: Any) -> float:
        return 1.0 if left == right else 0.0

    def project_artifact_for_user(self, artifact: Any, *, evidence: dict[str, Any] | None = None) -> str:
        return str(artifact or "")


__all__ = ["GenericDomainAdapter", "ProgressiveDomainAdapter"]
