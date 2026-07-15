from __future__ import annotations

from typing import Any

from ..nexus.request_context import get_llm_stage
from .session import current_llm_session
from .utils import now_iso


def record_event(
    request_type: str,
    response: dict[str, Any],
    status: dict[str, Any],
    *,
    usage: dict[str, int] | None = None,
    usage_provenance: str | None = None,
    estimated_cost_usd: float | None = None,
    attempts: int = 1,
    retry_history: list[dict[str, Any]] | None = None,
    governor: dict[str, Any] | None = None,
    error_type: str | None = None,
    cache_replayed: bool = False,
    physical_call_id: str = "",
    sampling: dict[str, Any] | None = None,
    logical_call_id: str = "",
    request_hash: str = "",
    idempotency_key: str = "",
    run_id: str = "",
    round_id: str = "",
    step_id: str = "",
    slot_ids: list[str] | None = None,
    candidate_bindings: list[dict[str, str]] | None = None,
    intervention_ref: str | None = None,
) -> None:
    usage = usage or {}
    event = {
        "time": now_iso(),
        "request_type": request_type,
        "stage": get_llm_stage() or "unscoped",
        "provider": status.get("provider"),
        "model": status.get("model"),
        "model_profile_id": status.get("model_profile_id"),
        "llm_call_identity": status.get("llm_call_identity"),
        "test_provider_only": status.get("test_provider_only", False),
        "confidence": response.get("confidence"),
        "attempts": attempts,
        "usage": usage,
        "usage_provenance": usage_provenance or ("unspecified" if usage else "unavailable"),
        "estimated_cost_usd": estimated_cost_usd,
    }
    for key, value in {
        "logical_call_id": logical_call_id,
        "request_hash": request_hash,
        "idempotency_key": idempotency_key,
        "run_id": run_id,
        "round_id": round_id,
        "step_id": step_id,
    }.items():
        if value:
            event[key] = value
    if slot_ids:
        event["slot_ids"] = list(dict.fromkeys(str(item) for item in slot_ids if str(item)))
    if candidate_bindings:
        event["candidate_bindings"] = [dict(item) for item in candidate_bindings]
    if intervention_ref:
        event["intervention_ref"] = str(intervention_ref)
    if sampling:
        event["sampling"] = dict(sampling)
    if retry_history:
        event["retry_history"] = retry_history
    if governor:
        event["governor"] = governor
    if error_type:
        event["error_type"] = error_type
    if physical_call_id:
        event["physical_call_id"] = physical_call_id
    if cache_replayed:
        event["cache_replayed"] = True
    current_llm_session().record(event)


_TOTAL_FIELDS = (
    "logical_calls",
    "physical_calls",
    "remote_attempts",
    "cache_hits",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "unpriced_remote_attempts",
    "estimated_cost_usd",
)
_INTEGER_TOTAL_FIELDS = _TOTAL_FIELDS[:-1]
_UNATTRIBUTED = "__unattributed__"


def transport_cost_attribution(payload: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Extract opaque attribution references at the transport boundary."""

    slot_ids: list[str] = []
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            branch_slots = value.get("branch_slots")
            if isinstance(branch_slots, list):
                for slot in branch_slots:
                    if isinstance(slot, dict) and str(slot.get("slot_id") or ""):
                        slot_ids.append(str(slot["slot_id"]))
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    slot_ids = list(dict.fromkeys(slot_ids))

    candidate_bindings: list[dict[str, str]] = []
    response_candidates: list[dict[str, Any]] = []
    for key in ("offspring", "candidates", "children"):
        values = response.get(key)
        if isinstance(values, list):
            response_candidates.extend(item for item in values if isinstance(item, dict))
    request_candidates: list[dict[str, Any]] = []
    if not response_candidates:
        for key in ("candidates", "population", "parents"):
            values = payload.get(key)
            if isinstance(values, list):
                request_candidates.extend(item for item in values if isinstance(item, dict))
    for item in response_candidates or request_candidates:
        candidate_id = str(item.get("id") or item.get("candidate_id") or "")
        if not candidate_id:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        slot_id = str(metadata.get("branch_slot_id") or item.get("slot_id") or "")
        if not slot_id and len(slot_ids) == 1:
            slot_id = slot_ids[0]
        candidate_bindings.append({"candidate_id": candidate_id, "slot_id": slot_id})

    intervention_ref = payload.get("intervention_ref")
    if intervention_ref is None and isinstance(payload.get("metadata"), dict):
        intervention_ref = payload["metadata"].get("intervention_ref")
    return {
        "slot_ids": slot_ids,
        "candidate_bindings": list(
            {
                (item["candidate_id"], item["slot_id"]): item
                for item in candidate_bindings
            }.values()
        ),
        "intervention_ref": str(intervention_ref) if intervention_ref is not None else None,
    }


def build_round_cost_ledger(
    events: list[dict[str, Any]],
    *,
    budget_history: list[dict[str, Any]] | None = None,
    existing_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project transport telemetry into a conservative round/slot/candidate ledger."""

    history = [item for item in (budget_history or []) if isinstance(item, dict)]
    rounds: dict[str, dict[str, Any]] = {}
    for existing in (existing_ledger or {}).get("rounds", []):
        if isinstance(existing, dict) and existing.get("round") is not None:
            _restore_round(rounds, existing)
    for record in history:
        existing = record.get("cost_ledger")
        if isinstance(existing, dict) and existing.get("round") is not None:
            _restore_round(rounds, existing)

    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("type") == "round_cost_ledger":
            continue
        round_id = str(event.get("round_id") or "0")
        round_acc = rounds.setdefault(round_id, _new_round(round_id))
        physical_id = str(event.get("physical_call_id") or "")
        logical_id = str(event.get("logical_call_id") or physical_id or f"event-{index}")
        cache_replayed = event.get("cache_replayed") is True
        contribution = _new_totals()
        if logical_id not in round_acc["logical_call_ids"]:
            round_acc["logical_call_ids"].add(logical_id)
            contribution["logical_calls"] = 1
        if cache_replayed:
            if logical_id not in round_acc["cache_replay_logical_call_ids"]:
                round_acc["cache_replay_logical_call_ids"].add(logical_id)
                contribution["cache_hits"] = 1
        else:
            physical_key = physical_id or (logical_id if int(event.get("attempts") or 0) > 0 else "")
            if physical_key and physical_key not in round_acc["physical_call_ids"]:
                round_acc["physical_call_ids"].add(physical_key)
                usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                contribution.update(
                    {
                        "physical_calls": 1,
                        "remote_attempts": max(0, int(event.get("attempts") or 0)),
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                        "unpriced_remote_attempts": (
                            max(0, int(event.get("attempts") or 0))
                            if event.get("estimated_cost_usd") is None
                            else 0
                        ),
                        "estimated_cost_usd": float(event.get("estimated_cost_usd") or 0.0),
                    }
                )
        targets = _event_targets(event)
        shares = _split_totals(contribution, len(targets))
        intervention_ref = str(event.get("intervention_ref") or "")
        for (slot_id, candidate_id), share in zip(targets, shares):
            _add_share(round_acc, slot_id, candidate_id, intervention_ref, share)
        _record_event_truncations(round_acc, event, logical_id)

    history_by_round: dict[str, list[dict[str, Any]]] = {}
    for record in history:
        history_by_round.setdefault(str(record.get("round") or "0"), []).append(record)
    serialized = [
        _serialize_round(round_acc, history_by_round.get(round_id, []))
        for round_id, round_acc in sorted(rounds.items(), key=lambda item: _round_sort_key(item[0]))
    ]
    return {
        "schema_version": "round-cost-ledger/v1",
        "source": "llm_session_transport_telemetry_and_budget_history",
        "allocation_policy": "deterministic_conservative_split_with_explicit_unattributed_residual",
        "rounds": serialized,
    }


def attach_round_cost_ledger(budget_history: list[dict[str, Any]], ledger: dict[str, Any]) -> None:
    by_round = {
        str(item.get("round")): item
        for item in ledger.get("rounds", [])
        if isinstance(item, dict) and item.get("round") is not None
    }
    for record in budget_history:
        if not isinstance(record, dict):
            continue
        round_record = by_round.get(str(record.get("round")))
        if round_record is not None:
            record["cost_ledger"] = round_record


def _new_totals() -> dict[str, int | float]:
    return {**{key: 0 for key in _INTEGER_TOTAL_FIELDS}, "estimated_cost_usd": 0.0}


def _copy_totals(value: Any) -> dict[str, int | float]:
    data = value if isinstance(value, dict) else {}
    return {
        **{key: int(data.get(key) or 0) for key in _INTEGER_TOTAL_FIELDS},
        "estimated_cost_usd": float(data.get("estimated_cost_usd") or 0.0),
    }


def _add_totals(target: dict[str, int | float], addition: dict[str, int | float]) -> None:
    for key in _INTEGER_TOTAL_FIELDS:
        target[key] = int(target[key]) + int(addition[key])
    target["estimated_cost_usd"] = float(target["estimated_cost_usd"]) + float(addition["estimated_cost_usd"])


def _new_slot(slot_id: str) -> dict[str, Any]:
    return {"slot_id": slot_id, "totals": _new_totals(), "candidates": {}, "intervention_refs": set()}


def _new_round(round_id: str) -> dict[str, Any]:
    return {
        "round_id": round_id,
        "totals": _new_totals(),
        "slots": {},
        "logical_call_ids": set(),
        "physical_call_ids": set(),
        "cache_replay_logical_call_ids": set(),
        "truncation_event_ids": set(),
        "transport_truncation_count": 0,
        "intervention_refs": set(),
        "existing_observations": {},
    }


def _restore_round(rounds: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    round_id = str(record.get("round") or "0")
    if round_id in rounds:
        return
    restored = _new_round(round_id)
    restored["totals"] = _copy_totals(record.get("totals"))
    for key in ("logical_call_ids", "physical_call_ids", "cache_replay_logical_call_ids", "truncation_event_ids"):
        restored[key].update(str(item) for item in record.get(key, []) if str(item))
    restored["transport_truncation_count"] = int(record.get("transport_truncation_count") or 0)
    restored["existing_observations"] = dict(record.get("observations") or {})
    for slot_record in [*(record.get("slots") or []), record.get("unattributed")]:
        if not isinstance(slot_record, dict):
            continue
        slot_id = str(slot_record.get("slot_id") or _UNATTRIBUTED)
        slot = _new_slot(slot_id)
        slot["totals"] = _copy_totals(slot_record.get("totals"))
        for candidate_record in [*(slot_record.get("candidates") or []), slot_record.get("unattributed")]:
            if not isinstance(candidate_record, dict):
                continue
            candidate_id = str(candidate_record.get("candidate_id") or _UNATTRIBUTED)
            intervention_ref = str(candidate_record.get("intervention_ref") or "")
            slot["candidates"][(candidate_id, intervention_ref)] = _copy_totals(candidate_record.get("totals"))
            if intervention_ref:
                slot["intervention_refs"].add(intervention_ref)
                restored["intervention_refs"].add(intervention_ref)
        restored["slots"][slot_id] = slot
    rounds[round_id] = restored


def _event_targets(event: dict[str, Any]) -> list[tuple[str, str]]:
    slots = [str(item) for item in event.get("slot_ids", []) if str(item)]
    bindings = event.get("candidate_bindings") if isinstance(event.get("candidate_bindings"), list) else []
    targets = []
    for binding in bindings:
        if not isinstance(binding, dict) or not str(binding.get("candidate_id") or ""):
            continue
        slot_id = str(binding.get("slot_id") or (slots[0] if len(slots) == 1 else _UNATTRIBUTED))
        targets.append((slot_id, str(binding["candidate_id"])))
    if not targets:
        targets = [(slot_id, _UNATTRIBUTED) for slot_id in slots]
    return list(dict.fromkeys(targets or [(_UNATTRIBUTED, _UNATTRIBUTED)]))


def _split_totals(totals: dict[str, int | float], parts: int) -> list[dict[str, int | float]]:
    count = max(1, parts)
    shares = [_new_totals() for _ in range(count)]
    for key in _INTEGER_TOTAL_FIELDS:
        quotient, remainder = divmod(int(totals[key]), count)
        for index in range(count):
            shares[index][key] = quotient + int(index < remainder)
    cost = float(totals["estimated_cost_usd"])
    per_part = cost / count
    for index in range(count - 1):
        shares[index]["estimated_cost_usd"] = per_part
    shares[-1]["estimated_cost_usd"] = cost - per_part * (count - 1)
    return shares


def _add_share(
    round_acc: dict[str, Any],
    slot_id: str,
    candidate_id: str,
    intervention_ref: str,
    share: dict[str, int | float],
) -> None:
    slot = round_acc["slots"].setdefault(slot_id, _new_slot(slot_id))
    candidate = slot["candidates"].setdefault((candidate_id, intervention_ref), _new_totals())
    _add_totals(round_acc["totals"], share)
    _add_totals(slot["totals"], share)
    _add_totals(candidate, share)
    if intervention_ref:
        round_acc["intervention_refs"].add(intervention_ref)
        slot["intervention_refs"].add(intervention_ref)


def _record_event_truncations(round_acc: dict[str, Any], event: dict[str, Any], logical_id: str) -> None:
    categories = [str(event.get("error_type") or "")]
    categories.extend(
        str(item.get("category") or "")
        for item in event.get("retry_history", [])
        if isinstance(item, dict)
    )
    for index, category in enumerate(categories):
        if category != "truncated_response":
            continue
        key = f"{logical_id}:{index}:truncated_response"
        if key not in round_acc["truncation_event_ids"]:
            round_acc["truncation_event_ids"].add(key)
            round_acc["transport_truncation_count"] += 1


def _serialize_round(round_acc: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    named_slots = [
        _serialize_slot(slot)
        for slot_id, slot in sorted(round_acc["slots"].items())
        if slot_id != _UNATTRIBUTED
    ]
    residual_slot = _serialize_slot(round_acc["slots"].get(_UNATTRIBUTED, _new_slot(_UNATTRIBUTED)))
    observations = _round_observations(round_acc, history)
    value: dict[str, Any] = {
        "round": int(round_acc["round_id"]) if round_acc["round_id"].isdigit() else round_acc["round_id"],
        "totals": _rounded_totals(round_acc["totals"]),
        "slots": named_slots,
        "unattributed": residual_slot,
        "observations": observations,
        "logical_call_ids": sorted(round_acc["logical_call_ids"]),
        "physical_call_ids": sorted(round_acc["physical_call_ids"]),
        "cache_replay_logical_call_ids": sorted(round_acc["cache_replay_logical_call_ids"]),
        "truncation_event_ids": sorted(round_acc["truncation_event_ids"]),
        "transport_truncation_count": int(round_acc["transport_truncation_count"]),
    }
    _attach_intervention_refs(value, round_acc["intervention_refs"])
    return value


def _serialize_slot(slot: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    residual_totals = _new_totals()
    residual_refs: set[str] = set()
    for (candidate_id, intervention_ref), totals in sorted(slot["candidates"].items()):
        if candidate_id == _UNATTRIBUTED:
            _add_totals(residual_totals, totals)
            if intervention_ref:
                residual_refs.add(intervention_ref)
            continue
        candidate = {"candidate_id": candidate_id, "totals": _rounded_totals(totals)}
        if intervention_ref:
            candidate["intervention_ref"] = intervention_ref
        candidates.append(candidate)
    residual: dict[str, Any] = {"candidate_id": _UNATTRIBUTED, "totals": _rounded_totals(residual_totals)}
    _attach_intervention_refs(residual, residual_refs)
    value: dict[str, Any] = {
        "slot_id": slot["slot_id"],
        "totals": _rounded_totals(slot["totals"]),
        "candidates": candidates,
        "unattributed": residual,
    }
    _attach_intervention_refs(value, slot["intervention_refs"])
    return value


def _round_observations(round_acc: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, int]:
    valid_children: set[str] = set()
    qualified_survivors = 0
    truncations = int(round_acc["transport_truncation_count"])
    blast_radius = 0
    for record in history:
        plan = record.get("generation_plan") if isinstance(record.get("generation_plan"), dict) else {}
        valid_children.update(str(item) for item in plan.get("offspring_ids", []) if str(item))
        qualified_survivors = max(
            qualified_survivors,
            int(record.get("evaluator_qualified_survivors") or 0),
        )
        harvest = plan.get("offspring_harvest") if isinstance(plan.get("offspring_harvest"), dict) else {}
        truncations += int(harvest.get("reservoir_truncated_count") or 0)
        failed_slots = harvest.get("failed_slot_ids") if isinstance(harvest.get("failed_slot_ids"), list) else harvest.get("slot_errors")
        if isinstance(failed_slots, list):
            blast_radius = max(blast_radius, len(failed_slots))
    existing = round_acc["existing_observations"]
    return {
        "physical_calls": int(round_acc["totals"]["physical_calls"]),
        "unique_valid_children": max(len(valid_children), int(existing.get("unique_valid_children") or 0)),
        "evaluator_qualified_survivors": max(qualified_survivors, int(existing.get("evaluator_qualified_survivors") or 0)),
        "truncation_count": max(truncations, int(existing.get("truncation_count") or 0)),
        "partial_failure_blast_radius": max(blast_radius, int(existing.get("partial_failure_blast_radius") or 0)),
    }


def _rounded_totals(totals: dict[str, int | float]) -> dict[str, int | float]:
    return {
        **{key: int(totals[key]) for key in _INTEGER_TOTAL_FIELDS},
        "estimated_cost_usd": round(float(totals["estimated_cost_usd"]), 12),
    }


def _attach_intervention_refs(target: dict[str, Any], refs: set[str]) -> None:
    if len(refs) == 1:
        target["intervention_ref"] = next(iter(refs))
    elif len(refs) > 1:
        target["intervention_refs"] = sorted(refs)


def _round_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


__all__ = [
    "attach_round_cost_ledger",
    "build_round_cost_ledger",
    "record_event",
    "transport_cost_attribution",
]
