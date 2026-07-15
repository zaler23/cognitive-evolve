"""Receipts attached to the authoritative generation-plan evidence path."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.mutation import MutationOperator, TYPED_MOVE_KINDS
from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash

from ._serde import stable_hash


class ReceiptValidationError(ValueError):
    """A receipt cannot enter the generation-plan evidence path."""


DONOR_ROLES = frozenset(
    {
        "representation_donor",
        "repair_pattern_donor",
        "mechanism_fragment_donor",
    }
)


@dataclass(frozen=True)
class InterventionReceipt:
    receipt_id: str
    diagnosed_pressure: dict[str, str]
    intervention_type: str
    target: dict[str, str]
    recipient_branch_slot_ids: list[str]
    produced_candidate_ids: list[str]
    outcome_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterventionReceipt":
        if not isinstance(data, dict):
            raise ReceiptValidationError("intervention receipt must be an object")
        pressure = _fixed_text_map(data.get("diagnosed_pressure"), ("stagnation_type", "diagnosis_ref"), "diagnosed_pressure")
        target = _fixed_text_map(data.get("target"), ("axis", "family", "action", "slot"), "target", allow_empty=True)
        receipt = cls(
            receipt_id=_required_text(data.get("receipt_id"), "receipt_id"),
            diagnosed_pressure=pressure,
            intervention_type=_required_text(data.get("intervention_type"), "intervention_type"),
            target=target,
            recipient_branch_slot_ids=_text_list(data.get("recipient_branch_slot_ids"), "recipient_branch_slot_ids", required=True),
            produced_candidate_ids=_text_list(data.get("produced_candidate_ids"), "produced_candidate_ids"),
            outcome_refs=_text_list(data.get("outcome_refs"), "outcome_refs"),
        )
        if target["slot"] and target["slot"] not in receipt.recipient_branch_slot_ids:
            raise ReceiptValidationError("target.slot must reference a recipient branch slot")
        return receipt


@dataclass(frozen=True)
class TransferReceipt:
    source_relations: list[str]
    target_relations: list[str]
    mapping: list[dict[str, str]]
    preserved_invariant: str
    predicted_break_condition: str
    probe_ref: str
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransferReceipt":
        if not isinstance(data, dict):
            raise ReceiptValidationError("transfer receipt must be an object")
        raw_mapping = data.get("mapping")
        if not isinstance(raw_mapping, list) or not raw_mapping:
            raise ReceiptValidationError("transfer receipt mapping must be a non-empty element mapping")
        mapping = [
            _fixed_text_map(item, ("source", "target"), f"mapping[{index}]")
            for index, item in enumerate(raw_mapping)
        ]
        return cls(
            source_relations=_text_list(data.get("source_relations"), "source_relations", required=True),
            target_relations=_text_list(data.get("target_relations"), "target_relations", required=True),
            mapping=mapping,
            preserved_invariant=_required_text(data.get("preserved_invariant"), "preserved_invariant"),
            predicted_break_condition=_required_text(data.get("predicted_break_condition"), "predicted_break_condition"),
            probe_ref=_required_text(data.get("probe_ref"), "probe_ref"),
            artifact_hash=_required_text(data.get("artifact_hash"), "artifact_hash"),
        )


@dataclass(frozen=True)
class BlendReceipt:
    receipt_id: str
    candidate_id: str
    primary_parent_id: str
    donor_parent_id: str
    donor_role: str
    generic_space_mapping: list[dict[str, str]]
    retained_from_primary: list[str]
    borrowed_from_donor: list[str]
    structural_correspondence: list[str]
    emergent_delta: list[str]
    incompatibilities: list[str]
    unresolved_obligations: list[str]
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlendReceipt":
        if not isinstance(data, dict):
            raise ReceiptValidationError("blend receipt must be an object")
        role = _required_text(data.get("donor_role"), "donor_role")
        if role not in DONOR_ROLES:
            raise ReceiptValidationError(f"unsupported donor_role: {role}")
        raw_mapping = data.get("generic_space_mapping")
        if not isinstance(raw_mapping, list) or not raw_mapping:
            raise ReceiptValidationError("blend receipt generic_space_mapping must be a non-empty element mapping")
        mapping = [
            _fixed_text_map(item, ("primary", "donor", "child"), f"generic_space_mapping[{index}]")
            for index, item in enumerate(raw_mapping)
        ]
        return cls(
            receipt_id=_required_text(data.get("receipt_id"), "receipt_id"),
            candidate_id=_required_text(data.get("candidate_id"), "candidate_id"),
            primary_parent_id=_required_text(data.get("primary_parent_id"), "primary_parent_id"),
            donor_parent_id=_required_text(data.get("donor_parent_id"), "donor_parent_id"),
            donor_role=role,
            generic_space_mapping=mapping,
            retained_from_primary=_text_list(data.get("retained_from_primary"), "retained_from_primary"),
            borrowed_from_donor=_text_list(data.get("borrowed_from_donor"), "borrowed_from_donor", required=True),
            structural_correspondence=_text_list(data.get("structural_correspondence"), "structural_correspondence"),
            emergent_delta=_text_list(data.get("emergent_delta"), "emergent_delta"),
            incompatibilities=_text_list(data.get("incompatibilities"), "incompatibilities"),
            unresolved_obligations=_text_list(data.get("unresolved_obligations"), "unresolved_obligations"),
            artifact_hash=_required_text(data.get("artifact_hash"), "artifact_hash"),
        )


@dataclass(frozen=True)
class MoveReceipt:
    receipt_id: str
    candidate_id: str
    parent_id: str
    move_kind: str
    input_domain: str
    declared_invariant_refs: list[str]
    expected_delta: dict[str, Any]
    actual_delta: dict[str, Any]
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveReceipt":
        if not isinstance(data, dict):
            raise ReceiptValidationError("move receipt must be an object")
        move_kind = _required_text(data.get("move_kind"), "move_kind")
        if move_kind not in TYPED_MOVE_KINDS:
            raise ReceiptValidationError(f"move receipt has unsupported move_kind: {move_kind}")
        input_domain = _required_text(data.get("input_domain"), "input_domain")
        if input_domain not in {"dict", "project_patch", "proof"}:
            raise ReceiptValidationError(f"move receipt has unsupported input_domain: {input_domain}")
        actual_delta = data.get("actual_delta")
        if not isinstance(actual_delta, dict) or not actual_delta:
            raise ReceiptValidationError("move receipt actual_delta must be a non-empty object")
        expected_delta = data.get("expected_delta")
        if not isinstance(expected_delta, dict):
            raise ReceiptValidationError("move receipt expected_delta must be an object")
        return cls(
            receipt_id=_required_text(data.get("receipt_id"), "receipt_id"),
            candidate_id=_required_text(data.get("candidate_id"), "candidate_id"),
            parent_id=_required_text(data.get("parent_id"), "parent_id"),
            move_kind=move_kind,
            input_domain=input_domain,
            declared_invariant_refs=_text_list(data.get("declared_invariant_refs"), "declared_invariant_refs"),
            expected_delta=dict(expected_delta),
            actual_delta=dict(actual_delta),
            artifact_hash=_required_text(data.get("artifact_hash"), "artifact_hash"),
        )


@dataclass(frozen=True)
class TransferCreditDecision:
    eligible: bool
    productive_credit: bool
    reason: str


def append_intervention_receipt(
    generation_plan: dict[str, Any],
    raw_receipt: InterventionReceipt | dict[str, Any],
) -> InterventionReceipt:
    try:
        receipt = raw_receipt if isinstance(raw_receipt, InterventionReceipt) else InterventionReceipt.from_dict(raw_receipt)
        known_slots = _known_branch_slot_ids(generation_plan)
        unknown = [slot_id for slot_id in receipt.recipient_branch_slot_ids if slot_id not in known_slots]
        if unknown:
            raise ReceiptValidationError(f"intervention receipt references unknown branch slot: {unknown[0]}")
    except ReceiptValidationError as exc:
        _audit_rejection(generation_plan, "intervention", str(exc))
        raise
    receipts = generation_plan.setdefault("intervention_receipts", [])
    if not any(isinstance(item, dict) and item.get("receipt_id") == receipt.receipt_id for item in receipts):
        receipts.append(receipt.to_dict())
    return receipt


def append_transfer_receipt(
    generation_plan: dict[str, Any],
    raw_receipt: TransferReceipt | dict[str, Any],
    *,
    artifact: Any | None = None,
) -> TransferReceipt:
    try:
        receipt = raw_receipt if isinstance(raw_receipt, TransferReceipt) else TransferReceipt.from_dict(raw_receipt)
        if artifact is not None and receipt.artifact_hash != stable_artifact_hash(artifact):
            raise ReceiptValidationError("transfer receipt artifact_hash does not match the produced artifact")
    except ReceiptValidationError as exc:
        _audit_rejection(generation_plan, "transfer", str(exc))
        raise
    receipts = generation_plan.setdefault("transfer_receipts", [])
    if not any(isinstance(item, dict) and item.get("artifact_hash") == receipt.artifact_hash for item in receipts):
        receipts.append(receipt.to_dict())
    return receipt


def canonical_blend_receipt(
    raw_receipt: dict[str, Any],
    *,
    candidate: Any,
    primary: Any,
    donor: Any,
    donor_role: str,
) -> BlendReceipt:
    if donor_role not in DONOR_ROLES:
        raise ReceiptValidationError(f"unsupported donor_role: {donor_role}")
    core = {
        **dict(raw_receipt),
        "candidate_id": str(getattr(candidate, "id", "")),
        "primary_parent_id": str(getattr(primary, "id", "")),
        "donor_parent_id": str(getattr(donor, "id", "")),
        "donor_role": donor_role,
        "artifact_hash": stable_artifact_hash(getattr(candidate, "artifact", None)),
    }
    core.pop("receipt_id", None)
    receipt = BlendReceipt.from_dict(
        {"receipt_id": "blend-" + stable_hash(core)[:20], **core}
    )
    primary_text = _candidate_material_text(primary)
    donor_text = _candidate_material_text(donor)
    child_text = _candidate_material_text(candidate)
    if not any(
        item in donor_text and item in child_text and item not in primary_text
        for item in receipt.borrowed_from_donor
    ):
        raise ReceiptValidationError("blend receipt has no engine-observable material donor contribution")
    return receipt


def append_blend_receipt(
    generation_plan: dict[str, Any],
    raw_receipt: BlendReceipt | dict[str, Any],
    *,
    artifact: Any | None = None,
) -> BlendReceipt:
    try:
        receipt = raw_receipt if isinstance(raw_receipt, BlendReceipt) else BlendReceipt.from_dict(raw_receipt)
        if artifact is not None and receipt.artifact_hash != stable_artifact_hash(artifact):
            raise ReceiptValidationError("blend receipt artifact_hash does not match the produced artifact")
    except ReceiptValidationError as exc:
        _audit_rejection(generation_plan, "blend", str(exc))
        raise
    receipts = generation_plan.setdefault("blend_receipts", [])
    if not any(isinstance(item, dict) and item.get("receipt_id") == receipt.receipt_id for item in receipts):
        receipts.append(receipt.to_dict())
    return receipt


def append_move_receipt(
    generation_plan: dict[str, Any],
    raw_receipt: MoveReceipt | dict[str, Any],
    *,
    artifact: Any | None = None,
) -> MoveReceipt:
    try:
        receipt = raw_receipt if isinstance(raw_receipt, MoveReceipt) else MoveReceipt.from_dict(raw_receipt)
        if artifact is not None and receipt.artifact_hash != stable_artifact_hash(artifact):
            raise ReceiptValidationError("move receipt artifact_hash does not match the produced artifact")
    except ReceiptValidationError as exc:
        _audit_rejection(generation_plan, "move", str(exc))
        raise
    receipts = generation_plan.setdefault("move_receipts", [])
    if not any(isinstance(item, dict) and item.get("receipt_id") == receipt.receipt_id for item in receipts):
        receipts.append(receipt.to_dict())
    return receipt


def intervention_receipts(
    source: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    receipt_id: str = "",
    slot_id: str = "",
    candidate_id: str = "",
    outcome_ref: str = "",
) -> list[InterventionReceipt]:
    found: list[InterventionReceipt] = []
    seen: set[str] = set()
    for plan in _generation_plans(source):
        for raw in plan.get("intervention_receipts", []):
            receipt = InterventionReceipt.from_dict(raw)
            if receipt.receipt_id in seen:
                continue
            if receipt_id and receipt.receipt_id != receipt_id:
                continue
            if slot_id and slot_id not in receipt.recipient_branch_slot_ids:
                continue
            if candidate_id and candidate_id not in receipt.produced_candidate_ids:
                continue
            if outcome_ref and outcome_ref not in receipt.outcome_refs:
                continue
            seen.add(receipt.receipt_id)
            found.append(receipt)
    return found


def transfer_receipts(
    source: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    artifact_hash: str = "",
    probe_ref: str = "",
) -> list[TransferReceipt]:
    found: list[TransferReceipt] = []
    seen: set[str] = set()
    for plan in _generation_plans(source):
        for raw in plan.get("transfer_receipts", []):
            receipt = TransferReceipt.from_dict(raw)
            if receipt.artifact_hash in seen:
                continue
            if artifact_hash and receipt.artifact_hash != artifact_hash:
                continue
            if probe_ref and receipt.probe_ref != probe_ref:
                continue
            seen.add(receipt.artifact_hash)
            found.append(receipt)
    return found


def transfer_credit_decision(
    receipt: TransferReceipt,
    *,
    probe_survived: bool,
    credited_artifact_hashes: set[str],
) -> TransferCreditDecision:
    if not probe_survived:
        return TransferCreditDecision(eligible=True, productive_credit=False, reason="probe_not_survived")
    if receipt.artifact_hash in credited_artifact_hashes:
        return TransferCreditDecision(eligible=True, productive_credit=False, reason="duplicate_artifact_hash")
    credited_artifact_hashes.add(receipt.artifact_hash)
    return TransferCreditDecision(eligible=True, productive_credit=True, reason="invariant_probe_survived")


def transfer_credit_for_candidate(
    candidate: Any,
    *,
    credited_artifact_hashes: set[str],
) -> TransferCreditDecision | None:
    if not _requires_transfer_receipt(candidate):
        return None
    metadata = getattr(candidate, "metadata", {})
    raw = metadata.get("transfer_receipt") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return TransferCreditDecision(eligible=False, productive_credit=False, reason="invalid_or_missing_receipt")
    try:
        receipt = TransferReceipt.from_dict(raw)
    except ReceiptValidationError:
        return TransferCreditDecision(eligible=False, productive_credit=False, reason="invalid_or_missing_receipt")
    if receipt.artifact_hash != stable_artifact_hash(getattr(candidate, "artifact", None)):
        return TransferCreditDecision(eligible=False, productive_credit=False, reason="artifact_hash_mismatch")
    return transfer_credit_decision(
        receipt,
        probe_survived=_named_probe_survived(candidate, receipt.probe_ref),
        credited_artifact_hashes=credited_artifact_hashes,
    )


def record_transfer_receipts(
    generation_plan: dict[str, Any],
    candidates: Iterable[Any],
) -> list[TransferReceipt]:
    recorded: list[TransferReceipt] = []
    for candidate in candidates:
        if not _requires_transfer_receipt(candidate):
            continue
        metadata = getattr(candidate, "metadata", {})
        raw = metadata.get("transfer_receipt") if isinstance(metadata, dict) else None
        if not isinstance(raw, dict):
            _audit_rejection(
                generation_plan,
                "transfer",
                "transfer candidate is missing transfer_receipt",
                candidate_id=str(getattr(candidate, "id", "")),
            )
            continue
        try:
            canonical = append_transfer_receipt(generation_plan, raw, artifact=getattr(candidate, "artifact", None))
        except ReceiptValidationError:
            continue
        metadata["transfer_receipt"] = canonical.to_dict()
        recorded.append(canonical)
    return recorded


def record_blend_receipts(
    generation_plan: dict[str, Any],
    candidates: Iterable[Any],
) -> list[BlendReceipt]:
    recorded: list[BlendReceipt] = []
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        rejection = str(metadata.get("blend_receipt_rejection") or "")
        if rejection:
            _audit_rejection(
                generation_plan,
                "blend",
                rejection,
                candidate_id=str(getattr(candidate, "id", "")),
            )
        raw = metadata.get("blend_receipt")
        if not isinstance(raw, dict):
            continue
        try:
            canonical = append_blend_receipt(
                generation_plan,
                raw,
                artifact=getattr(candidate, "artifact", None),
            )
        except ReceiptValidationError:
            continue
        metadata["blend_receipt"] = canonical.to_dict()
        recorded.append(canonical)
    return recorded


def record_move_receipts(
    generation_plan: dict[str, Any],
    candidates: Iterable[Any],
) -> list[MoveReceipt]:
    recorded: list[MoveReceipt] = []
    contracts = generation_plan.setdefault("move_contracts", [])
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        contract = metadata.get("move_contract")
        if isinstance(contract, dict) and not any(
            isinstance(item, dict) and item == contract
            for item in contracts
        ):
            contracts.append(dict(contract))
        raw = metadata.get("move_receipt")
        if not isinstance(raw, dict):
            continue
        try:
            canonical = append_move_receipt(
                generation_plan,
                raw,
                artifact=getattr(candidate, "artifact", None),
            )
        except ReceiptValidationError:
            continue
        metadata["move_receipt"] = canonical.to_dict()
        recorded.append(canonical)
    return recorded


def record_reproduction_receipts(
    generation_plan: dict[str, Any],
    *,
    diagnosis: Any,
    mutation_plans: list[Any],
    offspring: list[Any],
    outcomes: list[dict[str, Any]],
) -> list[InterventionReceipt]:
    """Record reproduction facts without changing the intervention itself."""

    record_blend_receipts(generation_plan, offspring)
    record_move_receipts(generation_plan, offspring)
    record_transfer_receipts(generation_plan, offspring)

    if not _has_diagnosed_pressure(diagnosis):
        return []
    diagnosis_payload = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else dict(diagnosis or {})
    diagnosis_payload.pop("created_at", None)
    pressure = {
        "stagnation_type": str(diagnosis_payload.get("stagnation_type") or "None"),
        "diagnosis_ref": "diagnosis-" + stable_hash(diagnosis_payload)[:20],
    }
    plan_by_id: dict[str, Any] = {}
    for plan in mutation_plans:
        plan_metadata = getattr(plan, "metadata", None)
        if isinstance(plan_metadata, dict):
            plan_by_id[str(plan_metadata.get("plan_id") or plan_metadata.get("id") or "")] = plan
    outcomes_by_candidate: dict[str, list[str]] = {}
    for outcome in outcomes:
        candidate_id = str(outcome.get("candidate_id") or "")
        if candidate_id:
            outcomes_by_candidate.setdefault(candidate_id, []).append(_outcome_ref(outcome))
    recorded: list[InterventionReceipt] = []
    recommended = [str(item) for item in diagnosis_payload.get("recommended_actions", []) if str(item)]
    for candidate in offspring:
        metadata = getattr(candidate, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        slot_id = str(metadata.get("branch_slot_id") or "")
        if not slot_id:
            continue
        plan = plan_by_id.get(str(metadata.get("plan_id") or ""))
        action = str(getattr(plan, "operator", "") or (recommended[0] if recommended else "continue"))
        raw_search_space = metadata.get("search_space")
        search_space = raw_search_space if isinstance(raw_search_space, dict) else {}
        core = {
            "diagnosed_pressure": pressure,
            "intervention_type": "diagnosis_guided_reproduction",
            "target": {
                "axis": str(search_space.get("seed_axis") or ""),
                "family": str(search_space.get("family_id") or ""),
                "action": action,
                "slot": slot_id,
            },
            "recipient_branch_slot_ids": [slot_id],
            "produced_candidate_ids": [str(getattr(candidate, "id", ""))],
            "outcome_refs": list(dict.fromkeys(outcomes_by_candidate.get(str(getattr(candidate, "id", "")), []))),
        }
        receipt = append_intervention_receipt(
            generation_plan,
            {"receipt_id": "intervention-" + stable_hash(core)[:20], **core},
        )
        recorded.append(receipt)
    return recorded


def _requires_transfer_receipt(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    raw_search_space = metadata.get("search_space")
    search_space = raw_search_space if isinstance(raw_search_space, dict) else {}
    return bool(
        isinstance(metadata.get("transfer_receipt"), dict)
        or str(metadata.get("mutation_operator") or "") == MutationOperator.TRANSFER
        or MutationOperator.TRANSFER in getattr(candidate, "mutation_history", [])
        or str(search_space.get("seed_axis") or "") == "cross_domain_transfer"
    )


def _named_probe_survived(candidate: Any, probe_ref: str) -> bool:
    metadata = getattr(candidate, "metadata", {})
    payloads = [item for item in getattr(candidate, "verification_trace", []) if isinstance(item, dict)]
    verification_result = getattr(candidate, "verification_result", {})
    if isinstance(verification_result, dict) and verification_result:
        payloads.append(verification_result)
    if isinstance(metadata, dict):
        for key in ("offspring_verification", "evidence_records"):
            value = metadata.get(key)
            if isinstance(value, dict):
                payloads.append(value)
            elif isinstance(value, list):
                payloads.extend(item for item in value if isinstance(item, dict))
    for payload in payloads:
        raw_nested = payload.get("metadata")
        nested = raw_nested if isinstance(raw_nested, dict) else {}
        refs = {
            str(payload.get(key) or "")
            for key in ("probe_ref", "result_ref", "id")
        } | {
            str(nested.get(key) or "")
            for key in ("probe_ref", "result_ref", "id")
        }
        if probe_ref not in refs:
            continue
        status = str(payload.get("status") or payload.get("validation_status") or "").lower()
        if payload.get("passed") is True or status in {"passed", "survived", "success", "verified"}:
            return True
    return False


def _has_diagnosed_pressure(diagnosis: Any) -> bool:
    data = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else dict(diagnosis or {})
    actions = [str(item) for item in data.get("recommended_actions", []) if str(item)]
    return bool(
        data.get("stagnation_detected")
        or data.get("over_explored_families")
        or data.get("under_explored_families")
        or data.get("prematurely_culled_genes")
        or any(action != "continue" for action in actions)
    )


def _outcome_ref(outcome: dict[str, Any]) -> str:
    for key in ("probe_ref", "evaluator_ref", "result_ref", "id"):
        value = str(outcome.get(key) or "")
        if value:
            return value
    return "outcome-" + stable_hash(outcome)[:20]


def _known_branch_slot_ids(generation_plan: dict[str, Any]) -> set[str]:
    allocation = generation_plan.get("productive_branch_allocation")
    allocation = allocation if isinstance(allocation, dict) else {}
    return {
        str(item.get("slot_id") or "")
        for item in allocation.get("slots", [])
        if isinstance(item, dict) and item.get("slot_id")
    }


def _generation_plans(source: Any) -> Iterable[dict[str, Any]]:
    items = [source] if isinstance(source, dict) else source
    for item in items:
        if not isinstance(item, dict):
            continue
        plan = item.get("generation_plan")
        if isinstance(plan, dict):
            yield dict(plan)
        else:
            yield item


def _audit_rejection(
    generation_plan: dict[str, Any],
    receipt_type: str,
    reason: str,
    *,
    candidate_id: str = "",
) -> None:
    event = {"receipt_type": receipt_type, "status": "rejected", "reason": reason}
    if candidate_id:
        event["candidate_id"] = candidate_id
    generation_plan.setdefault("receipt_audit", []).append(event)


def _artifact_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _candidate_material_text(candidate: Any) -> str:
    patch_set = getattr(candidate, "patch_set", None)
    material = {"artifact": getattr(candidate, "artifact", None)}
    if isinstance(patch_set, list):
        material["patch_set"] = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in patch_set
        ]
    return _artifact_text(material)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ReceiptValidationError(f"receipt field {field_name} must be non-empty")
    return text


def _text_list(value: Any, field_name: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ReceiptValidationError(f"receipt field {field_name} must be an array")
    items = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if required and not items:
        raise ReceiptValidationError(f"receipt field {field_name} must be non-empty")
    return items


def _fixed_text_map(
    value: Any,
    keys: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ReceiptValidationError(f"receipt field {field_name} must be an object")
    result = {key: str(value.get(key) or "").strip() for key in keys}
    if not allow_empty:
        missing = [key for key, item in result.items() if not item]
        if missing:
            raise ReceiptValidationError(f"receipt field {field_name}.{missing[0]} must be non-empty")
    return result


__all__ = [
    "BlendReceipt",
    "DONOR_ROLES",
    "InterventionReceipt",
    "MoveReceipt",
    "ReceiptValidationError",
    "TransferCreditDecision",
    "TransferReceipt",
    "append_blend_receipt",
    "append_intervention_receipt",
    "append_move_receipt",
    "append_transfer_receipt",
    "canonical_blend_receipt",
    "intervention_receipts",
    "record_blend_receipts",
    "record_move_receipts",
    "record_reproduction_receipts",
    "record_transfer_receipts",
    "transfer_credit_decision",
    "transfer_credit_for_candidate",
    "transfer_receipts",
]
