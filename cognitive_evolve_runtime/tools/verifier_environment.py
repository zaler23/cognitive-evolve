"""Hermetic-ish local verifier environment helpers."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


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
            env = {k: v for k, v in os.environ.items() if not any(marker in k for marker in SECRET_MARKERS)}
        env.update(extra_env or {})
        return cls(cwd=str(Path(cwd)), env=env, hermetic=hermetic)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["VerifierEnvironment"]
