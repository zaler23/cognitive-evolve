#!/usr/bin/env python3
"""Source and claim ledger for evidence-bound tasks."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifacts.store import _write_json

URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
ABS_PATH_RE = re.compile(r"(?<![\w.-])/(?:Users|Volumes|tmp|var|private|opt|usr|etc)/[^\s\"'<>]+")
CLAIM_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?。！？])\s+)")
DECISIVE_TERMS = {
    "proof",
    "prove",
    "proven",
    "theorem",
    "counterexample",
    "disprove",
    "disproved",
    "solved",
    "complete",
    "refute",
    "result",
    "证明",
    "定理",
    "反例",
    "推翻",
    "解决",
    "完整",
}
UNCERTAINTY_TERMS = {
    "uncertain",
    "unchecked",
    "unverified",
    "speculative",
    "partial",
    "gap",
    "needs expert",
    "未验证",
    "推测",
    "不确定",
    "缺口",
    "部分",
    "需要专家",
}
NEGATION_TERMS = {
    "not",
    "no",
    "cannot",
    "can't",
    "impossible",
    "false",
    "failed",
    "does not",
    "did not",
    "未",
    "没有",
    "不能",
    "无法",
    "不可能",
    "失败",
}
AFFIRMATIVE_SOURCE_TERMS = {
    "released",
    "announced",
    "introduced",
    "implemented",
    "published",
    "available",
    "exists",
    "result",
    "we show",
    "we prove",
    "证明",
    "发布",
    "推出",
    "存在",
}

MODEL_HYPOTHESIS = "model_hypothesis"
EXTERNAL_EVIDENCE = "external_evidence"
COMPUTED_EVIDENCE = "computed_evidence"
VERIFIED_CLAIM = "verified_claim"
UNSUPPORTED_CLAIM = "unsupported_claim"
CONTRADICTED_CLAIM = "contradicted_claim"
NON_EVIDENCE_SOURCE_TYPES = {MODEL_HYPOTHESIS, "llm_hypothesis", "semantic_controller", "model_memory"}
EVIDENCE_SOURCE_TYPES = {EXTERNAL_EVIDENCE, COMPUTED_EVIDENCE, "url", "local_path", "repo_reader", "test_runner", "local_tests", "primary_or_current_external_sources"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceRef:
    source_id: str
    locator: str
    relevance: str = ""


@dataclass
class SourceRecord:
    id: str
    source_type: str
    locator: str
    sha256: str | None = None
    exists: bool = True
    authority: str = "bound_source"
    text_digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimRecord:
    id: str
    text: str
    candidate_id: str | None = None
    status: str = "unchecked"
    support: list[EvidenceRef] = field(default_factory=list)
    risk: str = "medium"
    reason: str = ""
    decisive: bool = False
    claim_kind: str = UNSUPPORTED_CLAIM


@dataclass
class EvidenceLedger:
    id: str = "evidence-ledger:v1"
    created_at: str = field(default_factory=_now)
    sources: list[SourceRecord] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    policy: str = "bound_sources_override_llm_preference_for_decisive_claims"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "EvidenceLedger" | None) -> "EvidenceLedger":
        if isinstance(data, EvidenceLedger):
            return data
        if not isinstance(data, dict) or not data:
            return cls()
        return cls(
            id=str(data.get("id") or "evidence-ledger:v1"),
            created_at=str(data.get("created_at") or _now()),
            sources=[
                SourceRecord(
                    id=str(item.get("id")),
                    source_type=str(item.get("source_type") or "unknown"),
                    locator=str(item.get("locator") or ""),
                    sha256=item.get("sha256"),
                    exists=bool(item.get("exists", True)),
                    authority=str(item.get("authority") or "bound_source"),
                    text_digest=str(item.get("text_digest") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in data.get("sources", [])
                if isinstance(item, dict) and item.get("id")
            ],
            claims=[
                ClaimRecord(
                    id=str(item.get("id")),
                    text=str(item.get("text") or ""),
                    candidate_id=item.get("candidate_id"),
                    status=str(item.get("status") or "unchecked"),
                    support=[
                        EvidenceRef(
                            source_id=str(ref.get("source_id") or ""),
                            locator=str(ref.get("locator") or ""),
                            relevance=str(ref.get("relevance") or ""),
                        )
                        for ref in item.get("support", [])
                        if isinstance(ref, dict)
                    ],
                    risk=str(item.get("risk") or "medium"),
                    reason=str(item.get("reason") or ""),
                    decisive=bool(item.get("decisive", False)),
                    claim_kind=str(item.get("claim_kind") or item.get("evidence_kind") or UNSUPPORTED_CLAIM),
                )
                for item in data.get("claims", [])
                if isinstance(item, dict) and item.get("id")
            ],
            policy=str(data.get("policy") or "bound_sources_override_llm_preference_for_decisive_claims"),
        )

    @classmethod
    def from_prompt(
        cls,
        *,
        prompt: str,
        context: dict[str, Any] | None = None,
        task_dir: Path | None = None,
        evidence_execution: dict[str, Any] | None = None,
    ) -> "EvidenceLedger":
        context = context or {}
        ledger = cls()
        seen: set[str] = set()
        for url in URL_RE.findall(prompt):
            clean = url.rstrip(".,;")
            if clean in seen:
                continue
            seen.add(clean)
            ledger.sources.append(
                SourceRecord(
                    id=f"url:{len(ledger.sources) + 1}",
                    source_type="url",
                    locator=clean,
                    authority="prompt_bound_public_source",
                    text_digest="URL was explicitly supplied in the prompt; content must be fetched or attached before decisive claims rely on it.",
                    metadata={"fetch_required": True},
                )
            )
        for raw_path in list(ABS_PATH_RE.findall(prompt)) + [str(item) for item in context.get("source_paths", []) if str(item).startswith("/")]:
            path_text = raw_path.rstrip(".,;:)")
            if path_text in seen:
                continue
            seen.add(path_text)
            ledger.sources.append(_source_from_path(Path(path_text), source_id=f"path:{len(ledger.sources) + 1}"))
        if evidence_execution:
            ledger.sources.extend(_sources_from_evidence_execution(evidence_execution, start_index=len(ledger.sources) + 1))
        if task_dir is not None:
            ledger.write_artifacts(task_dir)
        return ledger

    def extract_claims(self, text: str, *, candidate_id: str | None = None, limit: int = 80) -> list[ClaimRecord]:
        return extract_claims(text, candidate_id=candidate_id, limit=limit)

    def assess_claims(self, claims: list[ClaimRecord]) -> list[ClaimRecord]:
        assessed: list[ClaimRecord] = []
        for index, claim in enumerate(claims, start=1):
            claim.id = claim.id or f"claim:{index}"
            claim.decisive = claim.decisive or _is_decisive(claim.text)
            contradiction = self._contradiction_source(claim.text)
            if contradiction:
                claim.status = "contradicted"
                claim.risk = "high"
                claim.reason = "claim conflicts with bound public/source evidence"
                claim.support = [EvidenceRef(source_id=contradiction.id, locator=contradiction.locator, relevance="contradiction")]
                claim.claim_kind = CONTRADICTED_CLAIM
            else:
                support = self._support_sources(claim.text)
                if support:
                    explicit_support = [ref for ref in support if ref.relevance.startswith(("explicit_tool_support", "semantic_support", "nli_support", "llm_support"))]
                    if explicit_support:
                        claim.status = "supported"
                        claim.risk = "low"
                        claim.reason = "explicit semantic/tool evidence supports the claim"
                        claim.support = explicit_support
                        claim.claim_kind = VERIFIED_CLAIM
                    elif claim.decisive:
                        claim.status = "needs_semantic_review"
                        claim.risk = "high"
                        claim.reason = "lexical overlap is only a support candidate for decisive claims"
                        claim.support = support
                    else:
                        claim.status = "supported_candidate"
                        claim.risk = "medium"
                        claim.reason = "lexical overlap is a support candidate; semantic review still required before final support"
                        claim.support = support
                elif _is_uncertainty_labeled(claim.text):
                    claim.status = "labeled_uncertain"
                    claim.risk = "low" if claim.decisive else "medium"
                    claim.reason = "claim is explicitly labeled as uncertain/partial"
                    claim.claim_kind = UNSUPPORTED_CLAIM
                elif claim.decisive:
                    claim.status = "unsupported_decisive"
                    claim.risk = "high"
                    claim.reason = "decisive claim lacks bound support"
                    claim.claim_kind = UNSUPPORTED_CLAIM
                else:
                    claim.status = "unchecked"
                    claim.risk = "medium"
                    claim.reason = "no bound source support found"
                    claim.claim_kind = UNSUPPORTED_CLAIM
            assessed.append(claim)
        self.claims = assessed
        return assessed

    def add_model_hypothesis(self, text: str, *, candidate_id: str | None = None) -> ClaimRecord:
        claim = ClaimRecord(
            id=f"model_hypothesis:{len(self.claims) + 1}",
            text=str(text),
            candidate_id=candidate_id,
            status=MODEL_HYPOTHESIS,
            risk="medium",
            reason="model hypothesis can seed search but is not evidence",
            decisive=_is_decisive(text),
            claim_kind=MODEL_HYPOTHESIS,
        )
        self.claims.append(claim)
        return claim

    def add_external_evidence(self, *, locator: str, text_digest: str = "", claims_supported: list[str] | None = None, source_id: str | None = None) -> SourceRecord:
        source = SourceRecord(
            id=source_id or f"external:{len(self.sources) + 1}",
            source_type=EXTERNAL_EVIDENCE,
            locator=str(locator),
            authority="external_bound_source",
            text_digest=str(text_digest or "external evidence registered")[:8000],
            metadata={"claims_supported": list(claims_supported or [])},
        )
        self.sources.append(source)
        return source

    def add_computed_evidence(self, *, locator: str, text_digest: str = "", claims_supported: list[str] | None = None, source_id: str | None = None) -> SourceRecord:
        source = SourceRecord(
            id=source_id or f"computed:{len(self.sources) + 1}",
            source_type=COMPUTED_EVIDENCE,
            locator=str(locator),
            authority="computed_runtime_artifact",
            text_digest=str(text_digest or "computed evidence registered")[:8000],
            metadata={"claims_supported": list(claims_supported or [])},
        )
        self.sources.append(source)
        return source

    def evidence_score(self) -> float:
        if not self.claims:
            return 0.0
        supported = 0
        considered = 0
        evidence_source_ids = {source.id for source in self.sources if source.source_type in EVIDENCE_SOURCE_TYPES and source.source_type not in NON_EVIDENCE_SOURCE_TYPES}
        for claim in self.claims:
            if claim.claim_kind == MODEL_HYPOTHESIS or claim.status == MODEL_HYPOTHESIS:
                continue
            considered += 1
            if claim.status == "supported" and any(ref.source_id in evidence_source_ids for ref in claim.support):
                supported += 1
        return round(supported / max(1, considered), 4)

    def evidence_requirement_status(self, *, required: bool, adapter_available: bool) -> dict[str, Any]:
        if required and not adapter_available:
            return {
                "status": "evidence_blocked",
                "reason": "adapter_required",
                "model_hypothesis_counts_as_evidence": False,
            }
        return {
            "status": "evidence_ready" if required else "not_required",
            "evidence_score": self.evidence_score(),
            "model_hypothesis_counts_as_evidence": False,
        }

    def contradiction_count(self) -> int:
        return sum(1 for claim in self.claims if claim.status == "contradicted")

    def unsupported_decisive_count(self) -> int:
        return sum(1 for claim in self.claims if claim.status == "unsupported_decisive")

    def source_count(self) -> int:
        return sum(1 for source in self.sources if source.exists)

    def write_artifacts(self, task_dir: Path) -> dict[str, str]:
        evidence_dir = task_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / "evidence-ledger-v1.json"
        _write_json(path, self.to_dict())
        return {"evidence_ledger_v1": str(path.relative_to(task_dir))}

    def _support_sources(self, claim_text: str) -> list[EvidenceRef]:
        claim_tokens = _tokens(claim_text)
        if not claim_tokens:
            return []
        refs: list[EvidenceRef] = []
        for source in self.sources:
            if source.source_type in NON_EVIDENCE_SOURCE_TYPES:
                continue
            if _explicitly_supports_claim(source, claim_text):
                refs.append(EvidenceRef(source_id=source.id, locator=source.locator, relevance="explicit_tool_support"))
                continue
            digest_tokens = _tokens(source.text_digest)
            overlap = claim_tokens & digest_tokens
            if len(overlap) >= min(4, max(2, len(claim_tokens) // 5)):
                refs.append(EvidenceRef(source_id=source.id, locator=source.locator, relevance=f"lexical_candidate_overlap:{len(overlap)}"))
        return refs[:5]

    def _contradiction_source(self, claim_text: str) -> SourceRecord | None:
        # Lexical contradiction detection has been removed.
        # Fall back to LLM semantic verifier for accurate contradiction checking.
        return None


def _source_from_path(path: Path, *, source_id: str) -> SourceRecord:
    expanded = path.expanduser()
    exists = expanded.exists()
    sha: str | None = None
    digest = ""
    metadata: dict[str, Any] = {}
    if exists and expanded.is_file():
        sha = _sha256_file(expanded)
        with expanded.open("rb") as handle:
            data = handle.read(64_000)
        try:
            digest = data.decode("utf-8")
        except UnicodeDecodeError:
            digest = f"binary file; {len(data)} bytes sampled"
        metadata = {"bytes": expanded.stat().st_size, "name": expanded.name}
    elif exists:
        digest = "path exists but is not a regular file"
        metadata = {"path_kind": "directory_or_special"}
    else:
        digest = "path was prompt-bound but does not exist in this runtime"
    return SourceRecord(
        id=source_id,
        source_type="local_path",
        locator=str(expanded),
        sha256=sha,
        exists=exists,
        authority="prompt_bound_local_source",
        text_digest=digest[:8000],
        metadata=metadata,
    )


def _sources_from_evidence_execution(evidence_execution: dict[str, Any], *, start_index: int) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    results = evidence_execution.get("results") if isinstance(evidence_execution.get("results"), list) else []
    for offset, result in enumerate(results, start=start_index):
        if not isinstance(result, dict):
            continue
        digest = json.dumps(
            {
                "summary": result.get("summary"),
                "status": result.get("status"),
                "claims_supported": result.get("claims_supported", []),
                "claims_unverified": result.get("claims_unverified", []),
                "source_type": result.get("source_type"),
            },
            ensure_ascii=False,
        )
        records.append(
            SourceRecord(
                id=f"evidence_exec:{offset}",
                source_type=str(result.get("source_type") or "evidence_execution"),
                locator=str(result.get("workspace") or result.get("source_type") or "evidence_execution"),
                exists=True,
                authority="runtime_evidence_execution",
                text_digest=digest[:8000],
                metadata={
                    "status": result.get("status"),
                    "adapter": result.get("adapter"),
                    "claims_supported": result.get("claims_supported", []),
                    "evidence_kind": _evidence_kind_for_source_type(str(result.get("source_type") or "")),
                },
            )
        )
    return records


def _evidence_kind_for_source_type(source_type: str) -> str:
    lowered = source_type.lower()
    if any(term in lowered for term in ["test", "local", "repo", "computed", "proof", "smt"]):
        return COMPUTED_EVIDENCE
    if any(term in lowered for term in ["external", "primary", "current", "url", "paper", "source"]):
        return EXTERNAL_EVIDENCE
    if lowered in NON_EVIDENCE_SOURCE_TYPES:
        return MODEL_HYPOTHESIS
    return UNSUPPORTED_CLAIM


def _explicitly_supports_claim(source: SourceRecord, claim_text: str) -> bool:
    supported = source.metadata.get("claims_supported") if isinstance(source.metadata, dict) else None
    if not isinstance(supported, list) or not supported:
        return False
    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return False
    for item in supported:
        supported_tokens = _tokens(str(item))
        if not supported_tokens:
            continue
        overlap = claim_tokens & supported_tokens
        if len(overlap) >= min(4, max(2, len(claim_tokens) // 4)):
            return True
    return False


def extract_claims(text: str, *, candidate_id: str | None = None, limit: int = 80) -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    for piece in CLAIM_SPLIT_RE.split(str(text or "")):
        cleaned = piece.strip().lstrip("-*•0123456789.）) ").strip()
        if len(cleaned) < 18:
            continue
        if cleaned.startswith("#"):
            continue
        decisive = _is_decisive(cleaned)
        if decisive or len(cleaned) <= 360:
            claims.append(
                ClaimRecord(
                    id=f"claim:{len(claims) + 1}",
                    text=cleaned[:700],
                    candidate_id=candidate_id,
                    decisive=decisive,
                )
            )
        if len(claims) >= limit:
            break
    return claims


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{4,}|[\u4e00-\u9fff]{2,}", str(text).lower())
        if token not in {"that", "this", "with", "from", "there", "their", "which", "should", "would", "could"}
    }


def _is_decisive(text: str) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in DECISIVE_TERMS)


def _is_uncertainty_labeled(text: str) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in UNCERTAINTY_TERMS)


__all__ = [
    "MODEL_HYPOTHESIS",
    "EXTERNAL_EVIDENCE",
    "COMPUTED_EVIDENCE",
    "VERIFIED_CLAIM",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "EvidenceRef",
    "SourceRecord",
    "ClaimRecord",
    "EvidenceLedger",
    "extract_claims",
]
