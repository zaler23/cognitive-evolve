"""Application ledger for research effect channels.

Research signals are proposals until a downstream runtime component actually
consumes them.  This module provides stable effect keys and append-only
application records so trace/ablation can distinguish produced effects from
real decision changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import json_ready, utc_now

_EFFECT_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "verification_obligations": ("id",),
    "archive_directives": ("kind", "descriptor"),
    "budget_directives": ("target", "reason"),
    "context_transforms": ("view_hash",),
    "candidate_transforms": ("candidate_id", "kind"),
    "contract_delta_proposals": ("delta_id",),
}


@dataclass(frozen=True)
class EffectApplicationRecord:
    channel: str
    effect_key: str
    concept_id: str = ""
    round: int = 0
    changed: bool = False
    consumer: str = ""
    reason: str = ""
    effect: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effect_key(channel: str, item: Any) -> str:
    """Return a stable key for an effect item in *channel*.

    The selected fields intentionally mirror ``merge_research_signals`` de-dupe
    keys.  The final digest is canonical across tuples/lists and float repr
    variations.
    """

    data = _item_dict(item)
    fields = _EFFECT_KEY_FIELDS.get(str(channel), ())
    if fields:
        selected = {field: data.get(field) for field in fields}
    else:
        selected = data
    payload = {"channel": str(channel), "identity": _canonicalize(selected)}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{channel}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def target_effect_key(channel: str, item: Any, *, target: str) -> str:
    """Return a target-scoped key for persistent effects such as context views."""

    base = effect_key(channel, item)
    target_text = json.dumps(_canonicalize(target), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{base}:target:{hashlib.sha256(target_text.encode('utf-8')).hexdigest()[:16]}"


def already_consumed(consumed: list[str] | set[str] | tuple[str, ...], key: str) -> bool:
    return str(key) in {str(item) for item in consumed or []}


def append_consumed(consumed: list[str], key: str, *, limit: int = 2000) -> list[str]:
    values = [str(item) for item in consumed or [] if str(item or "").strip()]
    if key not in values:
        values.append(str(key))
    return values[-max(1, int(limit or 2000)) :]


def record_effect_application(
    state: Any,
    *,
    channel: str,
    item: Any,
    concept_id: str = "",
    round_index: int = 0,
    changed: bool = False,
    consumer: str = "",
    reason: str = "",
    result: dict[str, Any] | None = None,
    key: str | None = None,
    consume: bool = True,
    limit: int = 1000,
) -> dict[str, Any]:
    """Append an application record to ``state`` and optionally mark consumed."""

    effect = _item_dict(item)
    resolved_key = str(key or effect_key(channel, effect))
    record = EffectApplicationRecord(
        channel=str(channel),
        effect_key=resolved_key,
        concept_id=str(concept_id or effect.get("origin") or effect.get("source") or ""),
        round=int(round_index or effect.get("round_index") or 0),
        changed=bool(changed),
        consumer=str(consumer or ""),
        reason=str(reason or ""),
        effect=effect,
        result=dict(result or {}),
    ).to_dict()
    applications = [dict(item) for item in getattr(state, "effect_applications", []) if isinstance(item, dict)]
    applications.append(record)
    setattr(state, "effect_applications", applications[-max(1, int(limit or 1000)) :])
    if consume:
        setattr(state, "consumed_effect_keys", append_consumed(list(getattr(state, "consumed_effect_keys", []) or []), resolved_key))
    return record


def _canonicalize(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _item_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        try:
            value = item.to_dict()
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}
    if isinstance(item, dict):
        return dict(item)
    try:
        return dict(item)
    except Exception:
        return {}


__all__ = [
    "EffectApplicationRecord",
    "already_consumed",
    "append_consumed",
    "effect_key",
    "record_effect_application",
    "target_effect_key",
]
