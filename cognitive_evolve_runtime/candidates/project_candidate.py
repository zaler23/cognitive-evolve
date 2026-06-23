"""Project-level candidate genomes and patch application records."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, coerce_str_list, utc_now


@dataclass
class PatchOperation:
    """A small, serializable patch primitive used by the offline sandbox.

    Supported operations are intentionally simple for deterministic tests:
    ``write`` creates/replaces a file with ``content``; ``append`` appends
    content; ``replace`` replaces ``old_text`` with ``new_text``; and ``delete``
    removes a file.
    """

    path: str
    operation: str = "write"
    content: str = ""
    old_text: str = ""
    new_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchOperation":
        return cls(
            path=str(data.get("path") or ""),
            operation=str(data.get("operation") or "write"),
            content=str(data.get("content") or ""),
            old_text=str(data.get("old_text") or ""),
            new_text=str(data.get("new_text") or ""),
            metadata=coerce_dict(data.get("metadata")),
        )


@dataclass
class PatchApplicationResult:
    status: str
    diagnostics: list[str] = field(default_factory=list)
    applied_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    pre_hash: str = ""
    post_hash: str = ""
    sandbox_path: str = ""
    raw_output_ref: str = ""
    created_at: str = field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        return self.status == "applied"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchApplicationResult":
        return cls(
            status=str(data.get("status") or "failed"),
            diagnostics=coerce_str_list(data.get("diagnostics")),
            applied_files=coerce_str_list(data.get("applied_files")),
            failed_files=coerce_str_list(data.get("failed_files")),
            pre_hash=str(data.get("pre_hash") or ""),
            post_hash=str(data.get("post_hash") or ""),
            sandbox_path=str(data.get("sandbox_path") or ""),
            raw_output_ref=str(data.get("raw_output_ref") or ""),
            created_at=str(data.get("created_at") or utc_now()),
        )


@dataclass
class ProjectCandidateGenome(CandidateGenome):
    artifact_type: str = "project_patch"
    patch_set: list[PatchOperation] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    touched_symbols: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    affected_tests: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    patch_application_result: dict[str, Any] = field(default_factory=dict)
    commands_run: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] = field(default_factory=dict)
    mutation_operator: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.artifact_type = self.artifact_type or "project_patch"
        self.patch_set = [item if isinstance(item, PatchOperation) else PatchOperation.from_dict(item) for item in self.patch_set if isinstance(item, (PatchOperation, dict))]
        if not self.touched_files:
            self.touched_files = [op.path for op in self.patch_set if op.path]
        else:
            self.touched_files = coerce_str_list(self.touched_files)
        self.touched_symbols = coerce_str_list(self.touched_symbols)
        self.expected_effects = coerce_str_list(self.expected_effects)
        self.affected_tests = coerce_str_list(self.affected_tests)
        self.risk_notes = coerce_str_list(self.risk_notes)
        self.patch_application_result = coerce_dict(self.patch_application_result)
        self.commands_run = [dict(item) for item in self.commands_run if isinstance(item, dict)]
        self.verification_result = coerce_dict(self.verification_result)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["patch_set"] = [op.to_dict() for op in self.patch_set]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectCandidateGenome":
        base = CandidateGenome.from_dict(data)
        if not data.get("artifact_type"):
            base.artifact_type = "project_patch"
        base_data = base.to_dict()
        verification_result = coerce_dict(data.get("verification_result")) or coerce_dict(base_data.pop("verification_result", {}))
        base_data.pop("verification_result", None)
        return cls(
            **base_data,
            patch_set=[PatchOperation.from_dict(item) for item in data.get("patch_set", []) if isinstance(item, dict)],
            touched_files=coerce_str_list(data.get("touched_files")),
            touched_symbols=coerce_str_list(data.get("touched_symbols")),
            expected_effects=coerce_str_list(data.get("expected_effects") or data.get("expected_effect")),
            affected_tests=coerce_str_list(data.get("affected_tests")),
            risk_notes=coerce_str_list(data.get("risk_notes")),
            patch_application_result=coerce_dict(data.get("patch_application_result")),
            commands_run=[dict(item) for item in data.get("commands_run", []) if isinstance(item, dict)],
            verification_result=verification_result,
            mutation_operator=str(data.get("mutation_operator") or ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "ProjectCandidateGenome":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("project candidate JSON must decode to an object")
        return cls.from_dict(data)


__all__ = ["PatchOperation", "PatchApplicationResult", "ProjectCandidateGenome"]
