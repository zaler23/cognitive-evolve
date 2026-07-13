"""Verification strength ladder."""
from __future__ import annotations

from enum import IntEnum


class VerificationStrength(IntEnum):
    NONE = 0
    ADVERSARIAL = 1
    DECOMPOSED = 2
    EMPIRICAL = 3
    FORMAL = 4
    EXECUTABLE = 5

    @classmethod
    def from_value(cls, value: object) -> "VerificationStrength":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().upper()
            if key in cls.__members__:
                return cls[key]
        try:
            return cls(int(value))
        except Exception:
            return cls.NONE


__all__ = ["VerificationStrength"]
