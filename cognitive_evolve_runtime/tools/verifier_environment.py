"""Hermetic-ish local verifier environment helpers."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTHORIZATION", "COOKIE", "SESSION")
HERMETIC_PROTECTED_KEYS = {"PATH", "PYTHONPATH", "COGEV_HERMETIC_TEST"}


def _is_secret_env_key(key: str) -> bool:
    upper = str(key).upper()
    return any(marker in upper for marker in SECRET_MARKERS)


@dataclass
class VerifierEnvironment:
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    hermetic: bool = True

    @classmethod
    def for_path(cls, cwd: str | Path, *, extra_env: dict[str, str] | None = None, hermetic: bool = True) -> "VerifierEnvironment":
        if hermetic:
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "COGEV_HERMETIC_TEST": os.environ.get("COGEV_HERMETIC_TEST", "1")}
        else:
            env = {k: v for k, v in os.environ.items() if not _is_secret_env_key(k)}
        env.update(
            {
                str(k): str(v)
                for k, v in (extra_env or {}).items()
                if not _is_secret_env_key(str(k)) and (not hermetic or str(k) not in HERMETIC_PROTECTED_KEYS)
            }
        )
        return cls(cwd=str(Path(cwd)), env=env, hermetic=hermetic)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["VerifierEnvironment"]
