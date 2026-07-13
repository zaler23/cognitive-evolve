"""Compatibility re-export for stable JSON helpers.

New code should import from :mod:`cognitive_evolve_runtime.core.serialization`.
"""
from __future__ import annotations

from cognitive_evolve_runtime.core.serialization import (
    coerce_dict,
    coerce_str_list,
    json_ready,
    stable_hash,
    stable_json,
    utc_now,
)

__all__ = ["coerce_dict", "coerce_str_list", "json_ready", "stable_hash", "stable_json", "utc_now"]
