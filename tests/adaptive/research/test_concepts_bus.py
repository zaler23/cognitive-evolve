from __future__ import annotations

import pytest

from cognitive_evolve_runtime.concepts.contract import AuthorityLevel, CONTRACTS, ConceptContract
from cognitive_evolve_runtime.concepts.effects import (
    ArchiveDirective,
    BudgetDirective,
    CandidateTransform,
    ContextTransform,
    ContractDeltaProposal,
    VerificationObligation,
)
from cognitive_evolve_runtime.concepts.guard import ConceptAuthorityError, enforce_strict, filter_live_signal
from cognitive_evolve_runtime.concepts.trace import TraceLedger
from cognitive_evolve_runtime.nexus.adaptive.research.registry import _KNOWN_EXTENSION_IDS
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal, merge_research_signals


def test_effect_dataclasses_roundtrip_and_dirty_input_is_empty() -> None:
    effects = [
        VerificationObligation("v", "fp", True, 1, True, "test"),
        ArchiveDirective("add_cell", ("a", 1), {"x": 1}),
        BudgetDirective("C1", 0.7, "roi", 0.8),
        ContextTransform(["spec"], ["old"], "view"),
        CandidateTransform("C1", "compress", {"x": 1}, 0.02),
        ContractDeltaProposal("d", "change", "reason", "h1", "h2", True),
    ]
    for effect in effects:
        restored = type(effect).from_dict(effect.to_dict())
        assert restored.to_dict() == effect.to_dict()
        assert type(effect).from_dict("dirty").to_dict() == type(effect).empty().to_dict()


def test_contracts_cover_known_extensions_and_authority_levels() -> None:
    assert _KNOWN_EXTENSION_IDS <= set(CONTRACTS)
    assert CONTRACTS["contract_refinement"].max_authority is AuthorityLevel.OBSERVE
    assert "contract_delta_proposals" in CONTRACTS["contract_refinement"].produces
    assert CONTRACTS["immune_necropsy"].max_authority is AuthorityLevel.VERIFY


def test_strict_guard_rejects_authority_and_undeclared_channel() -> None:
    contract = ConceptContract("fake", frozenset(), frozenset({"metrics", "warnings"}), AuthorityLevel.ADVISE, False, "a", "f")
    with pytest.raises(ConceptAuthorityError):
        enforce_strict(ResearchSignal(source="fake", final_gate_directives=[{"kind": "block"}]), contract)
    contract2 = ConceptContract("fake", frozenset(), frozenset({"metrics", "warnings"}), AuthorityLevel.GATE, False, "a", "f")
    with pytest.raises(ConceptAuthorityError):
        enforce_strict(ResearchSignal(source="fake", selection_advisory={"C1": {"risk": 0.5}}), contract2)


def test_live_guard_drops_but_records_violation() -> None:
    contract = ConceptContract("fake", frozenset(), frozenset({"metrics", "warnings"}), AuthorityLevel.OBSERVE, False, "a", "f")
    trace = TraceLedger()
    result = filter_live_signal(ResearchSignal(source="fake", final_gate_directives=[{"kind": "block"}]), contract, trace)
    assert result.accepted is False
    assert trace.entries


def test_signal_v2_merge_dedupes_new_effect_channels_without_guard_imports() -> None:
    first = ResearchSignal(source="a", round_index=1, verification_obligations=[{"id": "v1"}], candidate_transforms=[{"candidate_id": "C1", "kind": "compress"}])
    second = ResearchSignal(source="b", round_index=1, verification_obligations=[{"id": "v1"}], candidate_transforms=[{"candidate_id": "C1", "kind": "compress"}])
    merged = merge_research_signals([second, first])
    assert len(merged.verification_obligations) == 1
    assert len(merged.candidate_transforms) == 1
