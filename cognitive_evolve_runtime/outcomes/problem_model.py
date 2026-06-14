"""M6-alpha problem-model evolution primitives.

M5.1 updates a posterior over latent objectives.  M6-alpha lifts the search one
level higher: the runtime can propose, replay, and validate changes to the
*problem model* that defines which objectives, constraints, observables,
mechanisms, and subproblems are even in the search space.

The module is deliberately deterministic and ledger-friendly.  It does not let
structural novelty prove improvement; promoted structures still need future or
held-out evidence plus the M5/M5.2 verifier/certificate gates before they can
support closure.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash, utc_now
from cognitive_evolve_runtime.outcomes.latent import LatentProblemState, PreferenceEvidence


PROBLEM_MODEL_EVOLUTION_VERSION = "problem-model-evolution/v1"
PROBLEM_MODEL_LEDGER_VERSION = "problem-model-ledger/v1"

MODEL_ADDED = "problem_model_added"
MODEL_DEDUPLICATED = "problem_model_deduplicated"
MODEL_REJECTED = "problem_model_rejected"
MODEL_RETIRED = "problem_model_retired"
MODEL_SUPERSEDED = "problem_model_superseded"
MODEL_VALIDATED = "problem_model_validated"
MODEL_PROMOTED = "problem_model_promoted"
MODEL_DECISION_BOUND = "problem_model_decision_bound"

PROBLEM_MODEL_EVENT_TYPES = {
    MODEL_ADDED,
    MODEL_DEDUPLICATED,
    MODEL_REJECTED,
    MODEL_RETIRED,
    MODEL_SUPERSEDED,
    MODEL_VALIDATED,
    MODEL_PROMOTED,
    MODEL_DECISION_BOUND,
}


@dataclass(frozen=True)
class ProblemObjective:
    id: str
    statement: str
    utility_dimensions: tuple[str, ...] = ("quality",)
    hard_constraints: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    posterior_hint: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id or "").strip())
        object.__setattr__(self, "statement", str(self.statement or self.id).strip())
        object.__setattr__(self, "utility_dimensions", _str_tuple(self.utility_dimensions) or ("quality",))
        object.__setattr__(self, "hard_constraints", _str_tuple(self.hard_constraints))
        object.__setattr__(self, "evidence_refs", _str_tuple(self.evidence_refs))
        object.__setattr__(self, "posterior_hint", _bounded_float(self.posterior_hint, default=0.0))
        if not self.id:
            raise ValueError("problem objective requires an id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "ProblemObjective" | None:
        data = coerce_dict(raw)
        objective_id = str(data.get("id") or data.get("name") or "").strip()
        if not objective_id:
            return None
        return cls(
            id=objective_id,
            statement=str(data.get("statement") or data.get("description") or objective_id),
            utility_dimensions=tuple(_str_list(data.get("utility_dimensions"))),
            hard_constraints=tuple(_str_list(data.get("hard_constraints"))),
            evidence_refs=tuple(_str_list(data.get("evidence_refs"))),
            posterior_hint=_bounded_float(data.get("posterior_hint"), default=0.0),
        )


@dataclass(frozen=True)
class ProblemModelHypothesis:
    """A typed hypothesis about the structure of the problem itself."""

    id: str
    statement: str
    objectives: tuple[ProblemObjective, ...]
    constraints: tuple[str, ...] = ()
    observable_mappings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    causal_mechanisms: tuple[str, ...] = ()
    subproblems: tuple[str, ...] = ()
    unknown_regions: tuple[str, ...] = ()
    unknown_mass: float = 0.1
    parent_model_hashes: tuple[str, ...] = ()
    proposal_operator: str = "initial"
    evidence_basis_hashes: tuple[str, ...] = ()
    validation_rules: tuple[str, ...] = ()
    falsification_conditions: tuple[str, ...] = ()
    niche_id: str = "default"
    complexity_score: float = 0.0
    provenance_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PROBLEM_MODEL_EVOLUTION_VERSION

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("problem model requires at least one objective")
        object.__setattr__(self, "id", str(self.id or "problem-model").strip())
        object.__setattr__(self, "statement", str(self.statement or self.id).strip())
        objectives = tuple(_dedupe_objectives(self.objectives))
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "constraints", _str_tuple(self.constraints))
        object.__setattr__(self, "observable_mappings", _coerce_mapping_tuple(self.observable_mappings))
        object.__setattr__(self, "causal_mechanisms", _str_tuple(self.causal_mechanisms))
        object.__setattr__(self, "subproblems", _str_tuple(self.subproblems))
        object.__setattr__(self, "unknown_regions", _str_tuple(self.unknown_regions))
        object.__setattr__(self, "unknown_mass", max(0.0, min(1.0, float(self.unknown_mass))))
        object.__setattr__(self, "parent_model_hashes", _str_tuple(self.parent_model_hashes))
        object.__setattr__(self, "proposal_operator", str(self.proposal_operator or "unspecified"))
        object.__setattr__(self, "evidence_basis_hashes", _str_tuple(self.evidence_basis_hashes))
        object.__setattr__(self, "validation_rules", _str_tuple(self.validation_rules))
        object.__setattr__(self, "falsification_conditions", _str_tuple(self.falsification_conditions))
        object.__setattr__(self, "niche_id", str(self.niche_id or "default"))
        if self.complexity_score <= 0:
            object.__setattr__(self, "complexity_score", compute_problem_model_complexity(self))
        else:
            object.__setattr__(self, "complexity_score", max(0.0, float(self.complexity_score)))
        object.__setattr__(self, "provenance_ref", str(self.provenance_ref or ""))
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    def stable_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "statement": self.statement,
            "objectives": [objective.to_dict() for objective in self.objectives],
            "constraints": list(self.constraints),
            "observable_mappings": {key: list(value) for key, value in sorted(self.observable_mappings.items())},
            "causal_mechanisms": list(self.causal_mechanisms),
            "subproblems": list(self.subproblems),
            "unknown_regions": list(self.unknown_regions),
            "unknown_mass": self.unknown_mass,
            "parent_model_hashes": list(self.parent_model_hashes),
            "proposal_operator": self.proposal_operator,
            "evidence_basis_hashes": list(self.evidence_basis_hashes),
            "validation_rules": list(self.validation_rules),
            "falsification_conditions": list(self.falsification_conditions),
            "niche_id": self.niche_id,
            "complexity_score": self.complexity_score,
            "provenance_ref": self.provenance_ref,
        }

    def model_hash(self) -> str:
        return "pm:" + stable_hash(self.stable_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "statement": self.statement,
            "objectives": [objective.to_dict() for objective in self.objectives],
            "constraints": list(self.constraints),
            "observable_mappings": {key: list(value) for key, value in sorted(self.observable_mappings.items())},
            "causal_mechanisms": list(self.causal_mechanisms),
            "subproblems": list(self.subproblems),
            "unknown_regions": list(self.unknown_regions),
            "unknown_mass": self.unknown_mass,
            "parent_model_hashes": list(self.parent_model_hashes),
            "proposal_operator": self.proposal_operator,
            "evidence_basis_hashes": list(self.evidence_basis_hashes),
            "validation_rules": list(self.validation_rules),
            "falsification_conditions": list(self.falsification_conditions),
            "niche_id": self.niche_id,
            "complexity_score": self.complexity_score,
            "provenance_ref": self.provenance_ref,
            "metadata": dict(self.metadata),
            "model_hash": self.model_hash(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ProblemModelHypothesis" | None:
        if isinstance(raw, ProblemModelHypothesis):
            return raw
        data = coerce_dict(raw)
        objectives = tuple(
            objective
            for objective in (ProblemObjective.from_dict(item) for item in data.get("objectives", []))
            if objective is not None
        )
        if not objectives:
            return None
        return cls(
            id=str(data.get("id") or "problem-model"),
            statement=str(data.get("statement") or data.get("id") or "problem model"),
            objectives=objectives,
            constraints=tuple(_str_list(data.get("constraints"))),
            observable_mappings=_coerce_mapping_tuple(data.get("observable_mappings")),
            causal_mechanisms=tuple(_str_list(data.get("causal_mechanisms"))),
            subproblems=tuple(_str_list(data.get("subproblems"))),
            unknown_regions=tuple(_str_list(data.get("unknown_regions"))),
            unknown_mass=_bounded_float(data.get("unknown_mass"), default=0.1),
            parent_model_hashes=tuple(_str_list(data.get("parent_model_hashes"))),
            proposal_operator=str(data.get("proposal_operator") or "initial"),
            evidence_basis_hashes=tuple(_str_list(data.get("evidence_basis_hashes"))),
            validation_rules=tuple(_str_list(data.get("validation_rules"))),
            falsification_conditions=tuple(_str_list(data.get("falsification_conditions"))),
            niche_id=str(data.get("niche_id") or "default"),
            complexity_score=_bounded_float(data.get("complexity_score"), default=0.0, upper=1_000_000.0),
            provenance_ref=str(data.get("provenance_ref") or ""),
            metadata=coerce_dict(data.get("metadata")),
            version=str(data.get("version") or PROBLEM_MODEL_EVOLUTION_VERSION),
        )


@dataclass(frozen=True)
class ProblemResidual:
    kind: str
    evidence_ref: str
    magnitude: float
    suggested_operator: str
    affected_model_hash: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind or "unknown_residual"))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or ""))
        object.__setattr__(self, "magnitude", _bounded_float(self.magnitude, default=0.0))
        object.__setattr__(self, "suggested_operator", str(self.suggested_operator or "refine"))
        object.__setattr__(self, "affected_model_hash", str(self.affected_model_hash or ""))
        object.__setattr__(self, "description", str(self.description or self.kind))
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralProposal:
    operator: str
    source_model_hash: str
    proposed_model: ProblemModelHypothesis
    residual_refs: tuple[str, ...] = ()
    evidence_basis_hashes: tuple[str, ...] = ()
    expected_information_gain: float = 0.0
    predicted_validation_gain: float = 0.0
    complexity_delta: float = 0.0
    provenance_ref: str = ""
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", str(self.operator or self.proposed_model.proposal_operator or "refine"))
        object.__setattr__(self, "source_model_hash", str(self.source_model_hash or ""))
        object.__setattr__(self, "residual_refs", _str_tuple(self.residual_refs))
        object.__setattr__(self, "evidence_basis_hashes", _str_tuple(self.evidence_basis_hashes))
        object.__setattr__(self, "expected_information_gain", _bounded_float(self.expected_information_gain, default=0.0))
        object.__setattr__(self, "predicted_validation_gain", _bounded_float(self.predicted_validation_gain, default=0.0))
        object.__setattr__(self, "complexity_delta", max(0.0, float(self.complexity_delta)))
        object.__setattr__(self, "provenance_ref", str(self.provenance_ref or ""))
        key = self.idempotency_key or stable_hash(
            {
                "operator": self.operator,
                "source": self.source_model_hash,
                "model": self.proposed_model.model_hash(),
                "residual_refs": list(self.residual_refs),
            }
        )
        object.__setattr__(self, "idempotency_key", "sp:" + key if not str(key).startswith("sp:") else str(key))

    def proposal_hash(self) -> str:
        return "proposal:" + stable_hash(self.stable_payload())

    def stable_payload(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "source_model_hash": self.source_model_hash,
            "proposed_model_hash": self.proposed_model.model_hash(),
            "residual_refs": list(self.residual_refs),
            "evidence_basis_hashes": list(self.evidence_basis_hashes),
            "expected_information_gain": self.expected_information_gain,
            "predicted_validation_gain": self.predicted_validation_gain,
            "complexity_delta": self.complexity_delta,
            "provenance_ref": self.provenance_ref,
            "idempotency_key": self.idempotency_key,
            "score": self.score(),
        }

    def score(self, *, complexity_weight: float = 0.10) -> float:
        return self.predicted_validation_gain + self.expected_information_gain - complexity_weight * self.complexity_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "source_model_hash": self.source_model_hash,
            "proposed_model": self.proposed_model.to_dict(),
            "proposed_model_hash": self.proposed_model.model_hash(),
            "residual_refs": list(self.residual_refs),
            "evidence_basis_hashes": list(self.evidence_basis_hashes),
            "expected_information_gain": self.expected_information_gain,
            "predicted_validation_gain": self.predicted_validation_gain,
            "complexity_delta": self.complexity_delta,
            "provenance_ref": self.provenance_ref,
            "idempotency_key": self.idempotency_key,
            "proposal_hash": self.proposal_hash(),
            "score": self.score(),
        }


@dataclass(frozen=True)
class ProblemModelValidation:
    model_hash: str
    parent_model_hash: str = ""
    frozen_model_hash: str = ""
    validation_evidence_refs: tuple[str, ...] = ()
    trusted_verifier_refs: tuple[str, ...] = ()
    predictive_gain: float = 0.0
    complexity_penalty: float = 0.0
    parent_delta_penalty: float = 0.0
    falsification_survived: bool = False
    calibration_status: str = "unknown"
    promoted: bool = False
    reason_codes: tuple[str, ...] = ()
    validation_score: float = 0.0
    validation_model_version: str = PROBLEM_MODEL_EVOLUTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_hash(self) -> str:
        return "pmv:" + stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProblemModelLedgerEvent:
    sequence: int
    event_type: str
    event_id: str = ""
    model_hash: str = ""
    idempotency_key: str = ""
    target_model_hash: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    created_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        event_type = str(self.event_type or "")
        if event_type not in PROBLEM_MODEL_EVENT_TYPES:
            raise ValueError(f"unknown problem model event type: {event_type}")
        object.__setattr__(self, "sequence", max(1, int(self.sequence or 1)))
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "payload", coerce_dict(self.payload))
        if not self.event_id:
            object.__setattr__(self, "event_id", problem_model_event_id(self))

    def replay_payload(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("created_at_utc", None)
        data.pop("event_id", None)
        return data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "ProblemModelLedgerEvent" | None:
        data = coerce_dict(raw)
        if not data.get("event_type"):
            return None
        return cls(
            sequence=int(data.get("sequence") or 1),
            event_type=str(data.get("event_type") or ""),
            event_id=str(data.get("event_id") or ""),
            model_hash=str(data.get("model_hash") or ""),
            idempotency_key=str(data.get("idempotency_key") or ""),
            target_model_hash=str(data.get("target_model_hash") or ""),
            payload=coerce_dict(data.get("payload")),
            reason=str(data.get("reason") or ""),
            created_at_utc=str(data.get("created_at_utc") or utc_now()),
        )


@dataclass(frozen=True)
class ProblemModelLedgerReplay:
    cursor: int
    active_models: tuple[ProblemModelHypothesis, ...] = ()
    active_model_hashes: tuple[str, ...] = ()
    promoted_model_hashes: tuple[str, ...] = ()
    retired_model_hashes: tuple[str, ...] = ()
    rejected_events: tuple[dict[str, Any], ...] = ()
    validation_results: tuple[ProblemModelValidation, ...] = ()
    ledger_replay_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "active_models": [model.to_dict() for model in self.active_models],
            "active_model_hashes": list(self.active_model_hashes),
            "promoted_model_hashes": list(self.promoted_model_hashes),
            "retired_model_hashes": list(self.retired_model_hashes),
            "rejected_events": [dict(item) for item in self.rejected_events],
            "validation_results": [item.to_dict() for item in self.validation_results],
            "ledger_replay_hash": self.ledger_replay_hash,
        }


@dataclass
class ProblemModelLedger:
    ledger_id: str = "problem-model-ledger:v1"
    events: list[ProblemModelLedgerEvent] = field(default_factory=list)
    created_at_utc: str = field(default_factory=utc_now)
    version: str = PROBLEM_MODEL_LEDGER_VERSION

    @property
    def cursor(self) -> int:
        return max((event.sequence for event in self.events), default=0)

    def append_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        model_hash: str = "",
        idempotency_key: str = "",
        target_model_hash: str = "",
        reason: str = "",
    ) -> ProblemModelLedgerEvent:
        event = ProblemModelLedgerEvent(
            sequence=self.cursor + 1,
            event_type=event_type,
            model_hash=str(model_hash or ""),
            idempotency_key=str(idempotency_key or ""),
            target_model_hash=str(target_model_hash or ""),
            payload=coerce_dict(payload),
            reason=str(reason or ""),
        )
        self.events.append(event)
        return event

    def add_model(self, model: ProblemModelHypothesis | dict[str, Any], *, idempotency_key: str = "") -> ProblemModelLedgerEvent:
        item = ProblemModelHypothesis.from_dict(model)
        if item is None:
            return self.reject_model(model, reason="malformed_problem_model")
        model_hash = item.model_hash()
        key = str(idempotency_key or item.provenance_ref or model_hash)
        replay = self.replay()
        if model_hash in replay.active_model_hashes or any(event.idempotency_key == key for event in self.events if key):
            return self.append_event(
                MODEL_DEDUPLICATED,
                model_hash=model_hash,
                idempotency_key=key,
                payload={"model": item.to_dict(), "duplicate_of": model_hash},
                reason="duplicate_problem_model_or_idempotency_key",
            )
        return self.append_event(MODEL_ADDED, model_hash=model_hash, idempotency_key=key, payload={"model": item.to_dict()})

    def reject_model(self, raw: Any, *, reason: str, provenance_ref: str = "") -> ProblemModelLedgerEvent:
        return self.append_event(
            MODEL_REJECTED,
            payload={"raw_hash": stable_hash(raw), "raw": _small_raw(raw), "provenance_ref": str(provenance_ref or "")},
            reason=reason,
        )

    def supersede_model(
        self,
        target_model_hash: str,
        model: ProblemModelHypothesis,
        *,
        idempotency_key: str = "",
        reason: str = "problem_model_superseded",
    ) -> ProblemModelLedgerEvent:
        return self.append_event(
            MODEL_SUPERSEDED,
            model_hash=model.model_hash(),
            idempotency_key=idempotency_key or model.provenance_ref or model.model_hash(),
            target_model_hash=str(target_model_hash or ""),
            payload={"model": model.to_dict()},
            reason=reason,
        )

    def retire_model(self, model_hash: str, *, reason: str = "problem_model_retired") -> ProblemModelLedgerEvent:
        return self.append_event(MODEL_RETIRED, target_model_hash=str(model_hash or ""), reason=reason)

    def record_validation(self, validation: ProblemModelValidation) -> ProblemModelLedgerEvent:
        return self.append_event(
            MODEL_VALIDATED,
            model_hash=validation.model_hash,
            target_model_hash=validation.parent_model_hash,
            payload={"validation": validation.to_dict(), "validation_hash": validation.validation_hash()},
            reason="problem_model_validation_recorded",
        )

    def promote_model(self, validation: ProblemModelValidation) -> ProblemModelLedgerEvent:
        if not validation.promoted:
            return self.append_event(
                MODEL_REJECTED,
                model_hash=validation.model_hash,
                payload={"validation": validation.to_dict(), "validation_hash": validation.validation_hash()},
                reason="problem_model_promotion_gate_failed",
            )
        return self.append_event(
            MODEL_PROMOTED,
            model_hash=validation.model_hash,
            target_model_hash=validation.parent_model_hash,
            payload={"validation": validation.to_dict(), "validation_hash": validation.validation_hash()},
            reason="problem_model_predictive_promotion_gate_passed",
        )

    def record_decision_bound(self, *, decision_type: str, snapshot: "ProblemModelSnapshot", decision_payload: dict[str, Any] | None = None) -> ProblemModelLedgerEvent:
        return self.append_event(
            MODEL_DECISION_BOUND,
            payload={
                "decision_type": str(decision_type or "problem_model_decision"),
                "snapshot": snapshot.to_dict(),
                "decision": coerce_dict(decision_payload),
            },
            reason="problem_model_decision_bound_to_pinned_snapshot",
        )

    def replay(self, *, cursor: int | None = None) -> ProblemModelLedgerReplay:
        active: dict[str, ProblemModelHypothesis] = {}
        promoted: list[str] = []
        retired: list[str] = []
        rejected: list[dict[str, Any]] = []
        validations: list[ProblemModelValidation] = []
        selected_events = [event for event in sorted(self.events, key=lambda item: item.sequence) if cursor is None or event.sequence <= cursor]
        for event in selected_events:
            if event.event_type == MODEL_ADDED:
                model = ProblemModelHypothesis.from_dict(event.payload.get("model"))
                if model is None:
                    rejected.append({"event_id": event.event_id, "sequence": event.sequence, "reason": "malformed_added_problem_model"})
                    continue
                active[event.model_hash or model.model_hash()] = model
            elif event.event_type == MODEL_SUPERSEDED:
                if event.target_model_hash:
                    active.pop(event.target_model_hash, None)
                    retired.append(event.target_model_hash)
                model = ProblemModelHypothesis.from_dict(event.payload.get("model"))
                if model is None:
                    rejected.append({"event_id": event.event_id, "sequence": event.sequence, "reason": "malformed_superseding_problem_model"})
                    continue
                active[event.model_hash or model.model_hash()] = model
            elif event.event_type == MODEL_RETIRED:
                if event.target_model_hash:
                    active.pop(event.target_model_hash, None)
                    retired.append(event.target_model_hash)
            elif event.event_type == MODEL_VALIDATED:
                validation = _validation_from_any(event.payload.get("validation"))
                if validation is not None:
                    validations.append(validation)
            elif event.event_type == MODEL_PROMOTED:
                if event.model_hash:
                    promoted.append(event.model_hash)
                validation = _validation_from_any(event.payload.get("validation"))
                if validation is not None:
                    validations.append(validation)
            elif event.event_type == MODEL_REJECTED:
                rejected.append({"event_id": event.event_id, "sequence": event.sequence, "reason": event.reason, **event.payload})
        active_hashes = tuple(active.keys())
        return ProblemModelLedgerReplay(
            cursor=max((event.sequence for event in selected_events), default=0),
            active_models=tuple(active[item] for item in active_hashes),
            active_model_hashes=active_hashes,
            promoted_model_hashes=tuple(dict.fromkeys(promoted)),
            retired_model_hashes=tuple(dict.fromkeys(retired)),
            rejected_events=tuple(rejected),
            validation_results=tuple(validations),
            ledger_replay_hash=stable_hash([event.replay_payload() for event in selected_events]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ledger_id": self.ledger_id,
            "created_at_utc": self.created_at_utc,
            "events": [event.to_dict() for event in sorted(self.events, key=lambda item: item.sequence)],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ProblemModelLedger":
        if isinstance(raw, ProblemModelLedger):
            return raw
        data = coerce_dict(raw)
        events = tuple(
            event
            for event in (ProblemModelLedgerEvent.from_dict(item) for item in data.get("events", []))
            if event is not None
        )
        return cls(
            ledger_id=str(data.get("ledger_id") or "problem-model-ledger:v1"),
            events=list(events),
            created_at_utc=str(data.get("created_at_utc") or utc_now()),
            version=str(data.get("version") or PROBLEM_MODEL_LEDGER_VERSION),
        )


@dataclass(frozen=True)
class ProblemModelSnapshot:
    active_models: tuple[ProblemModelHypothesis, ...]
    ledger_cursor: int = 0
    active_model_hashes: tuple[str, ...] = ()
    promoted_model_hashes: tuple[str, ...] = ()
    ledger_replay_hash: str = ""
    update_model_version: str = PROBLEM_MODEL_EVOLUTION_VERSION
    materialized_at_utc: str = field(default_factory=utc_now)
    version: str = "problem-model-snapshot/v1"

    def __post_init__(self) -> None:
        hashes = self.active_model_hashes or tuple(model.model_hash() for model in self.active_models)
        object.__setattr__(self, "active_model_hashes", _str_tuple(hashes))
        object.__setattr__(self, "promoted_model_hashes", _str_tuple(self.promoted_model_hashes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "active_models": [model.to_dict() for model in self.active_models],
            "active_model_hashes": list(self.active_model_hashes),
            "promoted_model_hashes": list(self.promoted_model_hashes),
            "ledger_cursor": int(self.ledger_cursor),
            "ledger_replay_hash": self.ledger_replay_hash,
            "update_model_version": self.update_model_version,
            "model_space_hash": self.model_space_hash(),
            "materialized_at_utc": self.materialized_at_utc,
        }

    def stable_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("materialized_at_utc", None)
        return payload

    def snapshot_hash(self) -> str:
        return "pms:" + stable_hash(self.stable_payload())

    def model_space_hash(self) -> str:
        return "space:" + stable_hash(sorted(self.active_model_hashes))

    def decision_trace(self, *, decision_type: str) -> dict[str, Any]:
        return {
            "problem_model_ledger_cursor": int(self.ledger_cursor),
            "problem_model_snapshot_hash": self.snapshot_hash(),
            "problem_model_space_hash": self.model_space_hash(),
            "problem_model_update_model_version": self.update_model_version,
            "problem_model_decision_trace_ref": f"problem-model-decision:{decision_type}:{self.snapshot_hash()[4:20]}:{self.ledger_cursor}",
        }


@dataclass(frozen=True)
class ProblemModelPrediction:
    model_hash: str
    action_id: str
    predicted_outcome: str
    probability: float = 1.0
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_hash", str(self.model_hash or ""))
        object.__setattr__(self, "action_id", str(self.action_id or ""))
        object.__setattr__(self, "predicted_outcome", str(self.predicted_outcome or "unknown"))
        object.__setattr__(self, "probability", _bounded_float(self.probability, default=1.0))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDiscriminationAction:
    action_id: str
    competing_model_hashes: tuple[str, ...]
    expected_information_gain: float
    cost: float = 0.0
    predicted_outcomes: dict[str, list[str]] = field(default_factory=dict)
    decision_trace: dict[str, Any] = field(default_factory=dict)

    def acquisition_score(self) -> float:
        return self.expected_information_gain / max(1e-9, 1.0 + self.cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "competing_model_hashes": list(self.competing_model_hashes),
            "expected_information_gain": self.expected_information_gain,
            "cost": self.cost,
            "predicted_outcomes": {key: list(value) for key, value in sorted(self.predicted_outcomes.items())},
            "acquisition_score": self.acquisition_score(),
            "decision_trace": dict(self.decision_trace),
        }


def initial_problem_model_from_latent_state(
    state: LatentProblemState,
    *,
    model_id: str = "latent-problem-model:initial",
    provenance_ref: str = "latent_problem_state",
) -> ProblemModelHypothesis:
    objectives = tuple(
        ProblemObjective(
            id=intent.id,
            statement=intent.statement,
            utility_dimensions=intent.utility_dimensions,
            hard_constraints=intent.hard_constraints,
            evidence_refs=tuple(intent.representation_refs + intent.evaluator_refs),
            posterior_hint=intent.posterior,
        )
        for intent in state.intents
    )
    constraints = tuple(dict.fromkeys(item for intent in state.intents for item in intent.hard_constraints))
    observable_mappings = {
        objective.id: tuple(dict.fromkeys(objective.utility_dimensions))
        for objective in objectives
    }
    unknown_regions = tuple(f"unmodeled_region_for:{intent.id}" for intent in state.intents if intent.uncertainty >= 0.5)
    return ProblemModelHypothesis(
        id=model_id,
        statement="Initial problem model lifted from the latent objective posterior.",
        objectives=objectives,
        constraints=constraints,
        observable_mappings=observable_mappings,
        causal_mechanisms=("candidate_mutation -> trial_observation -> improvement_certificate",),
        subproblems=("intent_disambiguation", "candidate_materialization", "verified_improvement"),
        unknown_regions=unknown_regions,
        unknown_mass=max(0.05, min(1.0, state.posterior_entropy() or max((intent.uncertainty for intent in state.intents), default=0.1))),
        proposal_operator="initial_from_latent_state",
        evidence_basis_hashes=(state.state_hash(),),
        validation_rules=(
            "freeze_structure_before_validation",
            "require_heldout_or_future_evidence",
            "require_trusted_m5_verifier_before_solved",
        ),
        falsification_conditions=("contradictory_verified_evidence", "prediction_residual_persists"),
        complexity_score=0.0,
        provenance_ref=provenance_ref,
        metadata={"latent_state_hash": state.state_hash()},
    )


def materialize_problem_model_snapshot(ledger: ProblemModelLedger | None) -> ProblemModelSnapshot:
    replay = (ledger or ProblemModelLedger()).replay()
    return ProblemModelSnapshot(
        active_models=replay.active_models,
        active_model_hashes=replay.active_model_hashes,
        promoted_model_hashes=replay.promoted_model_hashes,
        ledger_cursor=replay.cursor,
        ledger_replay_hash=replay.ledger_replay_hash,
    )


def detect_problem_residuals(
    model: ProblemModelHypothesis,
    *,
    evidence: list[PreferenceEvidence | dict[str, Any]] | None = None,
    certificates: list[Any] | None = None,
    archive_observations: list[Any] | None = None,
    min_magnitude: float = 0.20,
) -> tuple[ProblemResidual, ...]:
    """Find places where the current problem model is explaining evidence poorly."""

    residuals: list[ProblemResidual] = []
    objective_ids = {objective.id for objective in model.objectives}
    for raw in evidence or []:
        item = _preference_evidence_from_any(raw)
        if item is None:
            continue
        total = item.support + item.contradiction
        conflict = min(item.support, item.contradiction)
        if item.intent_id not in objective_ids:
            residuals.append(
                ProblemResidual(
                    kind="unknown_intent_evidence",
                    evidence_ref=item.evidence_ref,
                    magnitude=max(min_magnitude, total * item.weight * item.confidence),
                    suggested_operator="birth",
                    affected_model_hash=model.model_hash(),
                    description=f"Evidence targets intent not represented by current problem model: {item.intent_id}",
                    metadata={"intent_id": item.intent_id, "source_type": item.source_type},
                )
            )
        elif conflict >= min_magnitude:
            residuals.append(
                ProblemResidual(
                    kind="contradictory_preference_evidence",
                    evidence_ref=item.evidence_ref,
                    magnitude=conflict * item.weight * item.confidence,
                    suggested_operator="split",
                    affected_model_hash=model.model_hash(),
                    description=f"Objective {item.intent_id} has simultaneous support and contradiction.",
                    metadata={"intent_id": item.intent_id, "source_type": item.source_type},
                )
            )
        elif item.contradiction * item.weight * item.confidence >= min_magnitude:
            residuals.append(
                ProblemResidual(
                    kind="negative_evidence_residual",
                    evidence_ref=item.evidence_ref,
                    magnitude=item.contradiction * item.weight * item.confidence,
                    suggested_operator="refine",
                    affected_model_hash=model.model_hash(),
                    description=f"Objective {item.intent_id} needs conditions or constraints after negative evidence.",
                    metadata={"intent_id": item.intent_id, "source_type": item.source_type},
                )
            )
    for raw in certificates or []:
        data = coerce_dict(raw if not hasattr(raw, "to_dict") else raw.to_dict())
        failures = _str_list(data.get("critical_failures") or data.get("improvement_critical_failures"))
        if failures:
            residuals.append(
                ProblemResidual(
                    kind="certificate_failure_residual",
                    evidence_ref=str(data.get("certificate_hash") or data.get("improvement_certificate_hash") or stable_hash(data)),
                    magnitude=max(min_magnitude, min(1.0, 0.15 * len(failures))),
                    suggested_operator="refine",
                    affected_model_hash=model.model_hash(),
                    description="Verified-improvement attempt failed under current problem model assumptions.",
                    metadata={"critical_failures": failures[:8]},
                )
            )
    for raw in archive_observations or []:
        data = coerce_dict(raw)
        discontinuity = _bounded_float(data.get("pareto_discontinuity") or data.get("cluster_distance") or data.get("topological_distance"), default=0.0)
        if discontinuity >= min_magnitude:
            residuals.append(
                ProblemResidual(
                    kind="pareto_discontinuity_residual",
                    evidence_ref=str(data.get("evidence_ref") or data.get("archive_ref") or stable_hash(data)),
                    magnitude=discontinuity,
                    suggested_operator="split",
                    affected_model_hash=model.model_hash(),
                    description="Pareto archive shows separated latent niches that should not collapse into one model.",
                    metadata={"niche_hint": str(data.get("niche_id") or "")},
                )
            )
    if model.unknown_mass >= 0.45:
        residuals.append(
            ProblemResidual(
                kind="high_unknown_mass",
                evidence_ref="problem_model:unknown_mass",
                magnitude=model.unknown_mass,
                suggested_operator="birth",
                affected_model_hash=model.model_hash(),
                description="Current problem model reserves high unmodeled mass.",
                metadata={"unknown_regions": list(model.unknown_regions)},
            )
        )
    return tuple(sorted((item for item in residuals if item.magnitude >= min_magnitude), key=lambda item: (-item.magnitude, item.kind, item.evidence_ref)))


def propose_structural_models(
    model: ProblemModelHypothesis,
    residuals: tuple[ProblemResidual, ...] | list[ProblemResidual],
    *,
    max_proposals: int = 6,
) -> tuple[StructuralProposal, ...]:
    proposals: list[StructuralProposal] = []
    seen_models: set[str] = set()
    for residual in sorted(residuals, key=lambda item: (-item.magnitude, item.kind, item.evidence_ref)):
        if len(proposals) >= max(0, int(max_proposals or 0)):
            break
        proposed = _apply_structural_operator(model, residual)
        if proposed is None:
            continue
        proposed_hash = proposed.model_hash()
        if proposed_hash in seen_models or proposed_hash == model.model_hash():
            continue
        seen_models.add(proposed_hash)
        proposals.append(
            StructuralProposal(
                operator=residual.suggested_operator,
                source_model_hash=model.model_hash(),
                proposed_model=proposed,
                residual_refs=(residual.evidence_ref,),
                evidence_basis_hashes=(stable_hash(residual.to_dict()),),
                expected_information_gain=min(1.0, 0.30 + 0.40 * residual.magnitude),
                predicted_validation_gain=min(1.0, 0.20 + 0.50 * residual.magnitude),
                complexity_delta=max(0.0, proposed.complexity_score - model.complexity_score),
                provenance_ref=residual.evidence_ref,
            )
        )
    return tuple(sorted(proposals, key=lambda item: (item.score(), item.expected_information_gain, item.proposal_hash()), reverse=True))


def validate_problem_model_promotion(
    model: ProblemModelHypothesis,
    *,
    parent_model: ProblemModelHypothesis | None = None,
    frozen_model_hash: str = "",
    validation_evidence_refs: tuple[str, ...] = (),
    trusted_verifier_refs: tuple[str, ...] = (),
    predictive_gain: float = 0.0,
    complexity_penalty: float | None = None,
    parent_delta_penalty: float | None = None,
    falsification_survived: bool = False,
    calibration_status: str = "unknown",
    min_predictive_gain: float = 0.05,
    min_validation_score: float = 0.02,
) -> ProblemModelValidation:
    model_hash = model.model_hash()
    parent_hash = parent_model.model_hash() if parent_model is not None else (model.parent_model_hashes[0] if model.parent_model_hashes else "")
    complexity = compute_problem_model_complexity(model)
    parent_complexity = compute_problem_model_complexity(parent_model) if parent_model is not None else max(0.0, complexity - 1.0)
    complexity_penalty = max(0.0, float(complexity_penalty if complexity_penalty is not None else 0.02 * complexity))
    parent_delta_penalty = max(0.0, float(parent_delta_penalty if parent_delta_penalty is not None else 0.03 * max(0.0, complexity - parent_complexity)))
    score = float(predictive_gain) - complexity_penalty - parent_delta_penalty
    reasons: list[str] = []
    if frozen_model_hash and frozen_model_hash != model_hash:
        reasons.append("model_changed_after_freeze")
    if not validation_evidence_refs:
        reasons.append("missing_validation_evidence")
    if not trusted_verifier_refs:
        reasons.append("missing_trusted_verifier")
    if predictive_gain < min_predictive_gain:
        reasons.append("predictive_gain_below_threshold")
    if score < min_validation_score:
        reasons.append("complexity_corrected_gain_below_threshold")
    if not falsification_survived:
        reasons.append("falsification_not_survived")
    if str(calibration_status or "unknown") in {"failed", "overconfident"}:
        reasons.append("calibration_failed")
    return ProblemModelValidation(
        model_hash=model_hash,
        parent_model_hash=parent_hash,
        frozen_model_hash=frozen_model_hash or model_hash,
        validation_evidence_refs=_str_tuple(validation_evidence_refs),
        trusted_verifier_refs=_str_tuple(trusted_verifier_refs),
        predictive_gain=float(predictive_gain),
        complexity_penalty=complexity_penalty,
        parent_delta_penalty=parent_delta_penalty,
        falsification_survived=bool(falsification_survived),
        calibration_status=str(calibration_status or "unknown"),
        promoted=not reasons,
        reason_codes=tuple(reasons),
        validation_score=score,
    )


def select_model_discrimination_action(
    predictions: list[ProblemModelPrediction | dict[str, Any]],
    *,
    cost_by_action: dict[str, float] | None = None,
    snapshot: ProblemModelSnapshot | None = None,
    min_information_gain: float = 0.01,
) -> ModelDiscriminationAction | None:
    parsed = tuple(item for item in (_prediction_from_any(raw) for raw in predictions) if item is not None and item.action_id and item.model_hash)
    if not parsed:
        return None
    costs = {str(key): _bounded_float(value, default=0.0) for key, value in coerce_dict(cost_by_action).items()}
    by_action: dict[str, list[ProblemModelPrediction]] = {}
    for item in parsed:
        by_action.setdefault(item.action_id, []).append(item)
    actions: list[ModelDiscriminationAction] = []
    for action_id, items in by_action.items():
        model_hashes = tuple(dict.fromkeys(item.model_hash for item in items))
        outcomes_by_model: dict[str, list[str]] = {}
        for item in items:
            outcomes_by_model.setdefault(item.model_hash, []).append(item.predicted_outcome)
        if len(model_hashes) < 2:
            continue
        outcome_weights: dict[str, float] = {}
        for item in items:
            outcome_weights[item.predicted_outcome] = outcome_weights.get(item.predicted_outcome, 0.0) + item.probability
        total = sum(outcome_weights.values()) or 1.0
        entropy = -sum((weight / total) * math.log(weight / total) for weight in outcome_weights.values() if weight > 0)
        normalized_entropy = entropy / math.log(max(2, len(outcome_weights)))
        disagreement_bonus = 1.0 if len(outcome_weights) > 1 else 0.0
        information_gain = normalized_entropy * disagreement_bonus
        if information_gain < min_information_gain:
            continue
        trace = snapshot.decision_trace(decision_type="model_discrimination") if snapshot is not None else {}
        actions.append(
            ModelDiscriminationAction(
                action_id=action_id,
                competing_model_hashes=model_hashes,
                expected_information_gain=information_gain,
                cost=costs.get(action_id, 0.0),
                predicted_outcomes=outcomes_by_model,
                decision_trace=trace,
            )
        )
    if not actions:
        return None
    return max(actions, key=lambda item: (item.acquisition_score(), item.expected_information_gain, item.action_id))


def compute_problem_model_complexity(model: ProblemModelHypothesis | None) -> float:
    if model is None:
        return 0.0
    return float(
        len(model.objectives) * 1.0
        + len(model.constraints) * 0.4
        + sum(len(values) for values in model.observable_mappings.values()) * 0.25
        + len(model.causal_mechanisms) * 0.7
        + len(model.subproblems) * 0.45
        + len(model.unknown_regions) * 0.25
        + len(model.validation_rules) * 0.35
        + len(model.falsification_conditions) * 0.35
    )


def problem_model_event_id(event: ProblemModelLedgerEvent) -> str:
    return "pme:" + stable_hash(event.replay_payload())


def _apply_structural_operator(model: ProblemModelHypothesis, residual: ProblemResidual) -> ProblemModelHypothesis | None:
    operator = residual.suggested_operator
    residual_hash = stable_hash(residual.to_dict())[:12]
    parent_hash = model.model_hash()
    evidence_basis = tuple(dict.fromkeys((*model.evidence_basis_hashes, stable_hash(residual.to_dict()))))
    metadata = dict(model.metadata) | {"last_structural_residual": residual.to_dict()}
    base_kwargs = {
        "id": f"{model.id}:{operator}:{residual_hash}",
        "statement": f"{model.statement} | structural {operator} from {residual.kind}",
        "parent_model_hashes": (parent_hash,),
        "proposal_operator": operator,
        "evidence_basis_hashes": evidence_basis,
        "provenance_ref": residual.evidence_ref,
        "metadata": metadata,
        "niche_id": model.niche_id,
    }
    if operator == "birth":
        new_id = str(residual.metadata.get("intent_id") or f"unknown_objective_{residual_hash}")
        objectives = (*model.objectives, ProblemObjective(id=new_id, statement=f"Newly modeled objective from residual: {residual.description}", evidence_refs=(residual.evidence_ref,), posterior_hint=0.05))
        unknown_regions = tuple(item for item in model.unknown_regions if item != new_id)
        return ProblemModelHypothesis(
            objectives=objectives,
            constraints=model.constraints,
            observable_mappings=dict(model.observable_mappings) | {new_id: (new_id,)},
            causal_mechanisms=model.causal_mechanisms,
            subproblems=(*model.subproblems, f"validate_new_objective:{new_id}"),
            unknown_regions=unknown_regions,
            unknown_mass=max(0.0, model.unknown_mass - 0.15),
            validation_rules=(*model.validation_rules, "new_objective_requires_future_certificate"),
            falsification_conditions=(*model.falsification_conditions, f"new_objective_not_reproduced:{new_id}"),
            **base_kwargs,
        )
    if operator == "split":
        intent_id = str(residual.metadata.get("intent_id") or "latent_niche")
        niche = str(residual.metadata.get("niche_hint") or f"niche:{intent_id}:{residual_hash}")
        return ProblemModelHypothesis(
            objectives=model.objectives,
            constraints=model.constraints,
            observable_mappings=model.observable_mappings,
            causal_mechanisms=model.causal_mechanisms,
            subproblems=(*model.subproblems, f"split_niche:{niche}"),
            unknown_regions=(*model.unknown_regions, f"disputed_region:{intent_id}"),
            unknown_mass=min(1.0, model.unknown_mass + 0.05),
            validation_rules=(*model.validation_rules, f"compare_predictions_across:{niche}"),
            falsification_conditions=(*model.falsification_conditions, f"niche_collapse_without_predictive_gain:{niche}"),
            **base_kwargs,
        )
    if operator == "retire":
        return ProblemModelHypothesis(
            objectives=model.objectives,
            constraints=(*model.constraints, f"retire_if_residual_persists:{residual.kind}"),
            observable_mappings=model.observable_mappings,
            causal_mechanisms=model.causal_mechanisms,
            subproblems=model.subproblems,
            unknown_regions=model.unknown_regions,
            unknown_mass=model.unknown_mass,
            validation_rules=(*model.validation_rules, "retirement_requires_dominating_replacement"),
            falsification_conditions=model.falsification_conditions,
            **base_kwargs,
        )
    return ProblemModelHypothesis(
        objectives=model.objectives,
        constraints=(*model.constraints, f"condition_from_residual:{residual.kind}"),
        observable_mappings=model.observable_mappings,
        causal_mechanisms=(*model.causal_mechanisms, f"mechanism_refinement:{residual.kind}"),
        subproblems=(*model.subproblems, f"explain_residual:{residual.kind}"),
        unknown_regions=model.unknown_regions,
        unknown_mass=model.unknown_mass,
        validation_rules=(*model.validation_rules, "refinement_requires_heldout_predictive_gain"),
        falsification_conditions=(*model.falsification_conditions, f"refinement_fails_on:{residual.kind}"),
        **base_kwargs,
    )


def _preference_evidence_from_any(raw: PreferenceEvidence | dict[str, Any] | None) -> PreferenceEvidence | None:
    if isinstance(raw, PreferenceEvidence):
        return raw
    data = coerce_dict(raw)
    intent_id = str(data.get("intent_id") or "").strip()
    if not intent_id:
        return None
    try:
        return PreferenceEvidence(
            intent_id=intent_id,
            support=_bounded_float(data.get("support"), default=0.0),
            contradiction=_bounded_float(data.get("contradiction"), default=0.0),
            weight=_bounded_float(data.get("weight"), default=1.0, upper=1_000_000.0),
            evidence_ref=str(data.get("evidence_ref") or ""),
            source_type=str(data.get("source_type") or "unknown"),
            provenance_ref=str(data.get("provenance_ref") or ""),
            confidence=_bounded_float(data.get("confidence"), default=1.0),
            calibration=str(data.get("calibration") or "uncalibrated"),
            metadata=coerce_dict(data.get("metadata")),
        )
    except Exception:
        return None


def _prediction_from_any(raw: ProblemModelPrediction | dict[str, Any]) -> ProblemModelPrediction | None:
    if isinstance(raw, ProblemModelPrediction):
        return raw
    data = coerce_dict(raw)
    if not data.get("model_hash") or not data.get("action_id"):
        return None
    return ProblemModelPrediction(
        model_hash=str(data.get("model_hash") or ""),
        action_id=str(data.get("action_id") or ""),
        predicted_outcome=str(data.get("predicted_outcome") or data.get("outcome") or "unknown"),
        probability=_bounded_float(data.get("probability"), default=1.0),
        evidence_ref=str(data.get("evidence_ref") or ""),
    )


def _validation_from_any(raw: ProblemModelValidation | dict[str, Any] | None) -> ProblemModelValidation | None:
    if isinstance(raw, ProblemModelValidation):
        return raw
    data = coerce_dict(raw)
    if not data.get("model_hash"):
        return None
    return ProblemModelValidation(
        model_hash=str(data.get("model_hash") or ""),
        parent_model_hash=str(data.get("parent_model_hash") or ""),
        frozen_model_hash=str(data.get("frozen_model_hash") or ""),
        validation_evidence_refs=tuple(_str_list(data.get("validation_evidence_refs"))),
        trusted_verifier_refs=tuple(_str_list(data.get("trusted_verifier_refs"))),
        predictive_gain=float(data.get("predictive_gain") or 0.0),
        complexity_penalty=float(data.get("complexity_penalty") or 0.0),
        parent_delta_penalty=float(data.get("parent_delta_penalty") or 0.0),
        falsification_survived=bool(data.get("falsification_survived")),
        calibration_status=str(data.get("calibration_status") or "unknown"),
        promoted=bool(data.get("promoted")),
        reason_codes=tuple(_str_list(data.get("reason_codes"))),
        validation_score=float(data.get("validation_score") or 0.0),
        validation_model_version=str(data.get("validation_model_version") or PROBLEM_MODEL_EVOLUTION_VERSION),
    )


def _dedupe_objectives(objectives: tuple[ProblemObjective, ...]) -> list[ProblemObjective]:
    out: dict[str, ProblemObjective] = {}
    for objective in objectives:
        out.setdefault(objective.id, objective)
    return list(out.values())


def _coerce_mapping_tuple(value: Any) -> dict[str, tuple[str, ...]]:
    data = coerce_dict(value)
    return {str(key): _str_tuple(raw) for key, raw in sorted(data.items()) if str(key)}


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_str_list(value)))


def _bounded_float(value: Any, *, default: float = 0.0, upper: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed != parsed:
        parsed = default
    return max(0.0, min(float(upper), parsed))


def _small_raw(raw: Any) -> Any:
    if hasattr(raw, "to_dict"):
        return raw.to_dict()
    if isinstance(raw, dict):
        return {str(key): value for key, value in list(raw.items())[:20]}
    return str(raw)[:1000]


__all__ = [
    "MODEL_ADDED",
    "MODEL_DEDUPLICATED",
    "MODEL_DECISION_BOUND",
    "MODEL_PROMOTED",
    "MODEL_REJECTED",
    "MODEL_RETIRED",
    "MODEL_SUPERSEDED",
    "MODEL_VALIDATED",
    "PROBLEM_MODEL_EVENT_TYPES",
    "PROBLEM_MODEL_EVOLUTION_VERSION",
    "ModelDiscriminationAction",
    "ProblemModelHypothesis",
    "ProblemModelLedger",
    "ProblemModelLedgerEvent",
    "ProblemModelLedgerReplay",
    "ProblemModelPrediction",
    "ProblemModelSnapshot",
    "ProblemModelValidation",
    "ProblemObjective",
    "ProblemResidual",
    "StructuralProposal",
    "compute_problem_model_complexity",
    "detect_problem_residuals",
    "initial_problem_model_from_latent_state",
    "materialize_problem_model_snapshot",
    "problem_model_event_id",
    "propose_structural_models",
    "select_model_discrimination_action",
    "validate_problem_model_promotion",
]
