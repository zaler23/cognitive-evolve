from __future__ import annotations

from cognitive_evolve_runtime.outcomes import (
    EVIDENCE_ADDED,
    EVIDENCE_DEDUPLICATED,
    EVIDENCE_REJECTED,
    EVIDENCE_RETRACTED,
    EVIDENCE_SUPERSEDED,
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    PreferenceEvidence,
    bounded_update_intent_posteriors,
    materialize_posterior_snapshot,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="prefer clarity", posterior=0.5, uncertainty=0.5),
            IntentHypothesis(id="impact", statement="prefer impact", posterior=0.5, uncertainty=0.5),
        )
    )


def _evidence(intent_id: str, *, ref: str, support: float = 0.8, contradiction: float = 0.0, source_type: str = "verifier") -> PreferenceEvidence:
    return PreferenceEvidence(
        intent_id=intent_id,
        support=support,
        contradiction=contradiction,
        weight=1.0,
        evidence_ref=ref,
        source_type=source_type,
        provenance_ref=ref,
        confidence=1.0,
    )


def test_latent_ledger_append_only_monotonic_offsets_and_dedup() -> None:
    ledger = LatentLedger()

    first = ledger.add_evidence(_evidence("clarity", ref="verifier:C1"), idempotency_key="k1")
    duplicate = ledger.add_evidence(_evidence("clarity", ref="verifier:C1"), idempotency_key="k1")

    assert first.event_type == EVIDENCE_ADDED
    assert duplicate.event_type == EVIDENCE_DEDUPLICATED
    assert [event.sequence for event in ledger.events] == [1, 2]
    replay = ledger.replay()
    assert len(replay.active_evidence) == 1
    assert replay.active_idempotency_keys == ("k1",)


def test_replay_reconstructs_same_posterior_snapshot() -> None:
    state = _state()
    ledger = LatentLedger()
    ledger.add_evidence(_evidence("impact", ref="verifier:impact"), idempotency_key="impact")

    snapshot = materialize_posterior_snapshot(state, ledger)
    replayed = LatentLedger.from_dict(ledger.to_dict())
    snapshot2 = materialize_posterior_snapshot(state, replayed)

    assert snapshot.snapshot_hash() == snapshot2.snapshot_hash()
    assert snapshot.state.top_intent().id == "impact"
    assert snapshot.ledger_cursor == 1


def test_retraction_rebuilds_expected_posterior() -> None:
    state = _state()
    ledger = LatentLedger()
    added = ledger.add_evidence(_evidence("clarity", ref="verifier:clarity"), idempotency_key="clarity")
    shifted = materialize_posterior_snapshot(state, ledger)

    retract = ledger.retract_evidence(added.evidence_id, reason="bad_source")
    rebuilt = materialize_posterior_snapshot(state, ledger)

    assert retract.event_type == EVIDENCE_RETRACTED
    assert shifted.state.top_intent().id == "clarity"
    assert abs(rebuilt.state.intents[0].posterior - 0.5) < 1e-9
    assert abs(rebuilt.state.intents[1].posterior - 0.5) < 1e-9
    assert ledger.replay().active_evidence == ()


def test_supersession_replaces_prior_evidence() -> None:
    state = _state()
    ledger = LatentLedger()
    added = ledger.add_evidence(_evidence("clarity", ref="verifier:clarity"), idempotency_key="clarity")
    supersede = ledger.supersede_evidence(
        added.evidence_id,
        _evidence("impact", ref="verifier:impact"),
        idempotency_key="impact",
    )
    replay = ledger.replay()
    snapshot = materialize_posterior_snapshot(state, ledger)

    assert supersede.event_type == EVIDENCE_SUPERSEDED
    assert len(replay.active_evidence) == 1
    assert replay.active_evidence[0].intent_id == "impact"
    assert snapshot.state.top_intent().id == "impact"


def test_malformed_evidence_quarantined_not_ingested() -> None:
    ledger = LatentLedger()

    event = ledger.add_evidence({"support": 1.0})

    assert event.event_type == EVIDENCE_REJECTED
    assert ledger.replay().active_evidence == ()


def test_single_source_influence_cap_and_conflict_preserves_uncertainty() -> None:
    state = _state()
    repeated = [
        _evidence("clarity", ref=f"critique:{index}", support=1.0, source_type="critique")
        for index in range(10)
    ]
    capped, trace = bounded_update_intent_posteriors(state, repeated)

    clarity = next(intent for intent in capped.intents if intent.id == "clarity")
    assert clarity.posterior < 0.60
    assert trace["capped_source_updates"]["clarity"]["critique"] == 0.25

    conflicted, _ = bounded_update_intent_posteriors(
        state,
        [
            _evidence("clarity", ref="verifier:positive", support=0.8, source_type="verifier"),
            _evidence("clarity", ref="verifier:negative", support=0.0, contradiction=0.8, source_type="verifier"),
        ],
    )
    clarity_conflicted = next(intent for intent in conflicted.intents if intent.id == "clarity")
    assert clarity_conflicted.uncertainty >= 0.5
