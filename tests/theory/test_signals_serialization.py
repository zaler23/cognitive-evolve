from __future__ import annotations

import json

import pytest

from cognitive_evolve_runtime.theory import AdvisoryRankingFeatures, TheorySignal, forbidden_key_paths, validate_theory_signal_json_safe


def test_theory_signal_json_safe_and_frozen_meta() -> None:
    signal = TheorySignal(
        source="mdl",
        kind="rank_prior",
        target_type="candidate",
        cycle_id="r1",
        target_id="C1",
        value=0.25,
        confidence=0.5,
        interval=(0.0, 0.5),
        provenance=("unit",),
        meta={"nested": {"items": ["a", 1]}},
    )

    validate_theory_signal_json_safe(signal)
    payload = signal.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["advisory_only"] is True
    with pytest.raises(TypeError):
        signal.meta["new"] = "blocked"  # type: ignore[index]


def test_theory_signal_rejects_non_finite_values_and_bad_interval() -> None:
    with pytest.raises(ValueError, match="finite"):
        TheorySignal(source="mdl", kind="rank_prior", target_type="candidate", cycle_id="r1", target_id="C1", value=float("nan"))
    with pytest.raises(ValueError, match="confidence"):
        TheorySignal(source="mdl", kind="rank_prior", target_type="candidate", cycle_id="r1", target_id="C1", value=0.1, confidence=1.5)
    with pytest.raises(ValueError, match="interval"):
        TheorySignal(source="mdl", kind="rank_prior", target_type="candidate", cycle_id="r1", target_id="C1", value=0.1, interval=(2.0, 1.0))


def test_forbidden_structured_keys_are_rejected_recursively() -> None:
    assert forbidden_key_paths({"GateResult": {"safe": 1}, "nested": [{"certificate-id": "x"}]}) == (
        "GateResult",
        "nested[0].certificate-id",
    )
    with pytest.raises(ValueError, match="forbidden"):
        TheorySignal(
            source="observer",
            kind="diagnostic",
            target_type="outcome",
            cycle_id="r1",
            target_id="O1",
            value=0.1,
            meta={"promotion decision": "no"},
        )


def test_advisory_ranking_features_reject_non_finite_candidate_values() -> None:
    with pytest.raises(ValueError):
        AdvisoryRankingFeatures(candidate_id="C1", rank_prior=float("inf"))
