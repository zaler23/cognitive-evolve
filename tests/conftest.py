from __future__ import annotations

import os
import shutil
from pathlib import Path


def pytest_configure() -> None:
    # Unit tests are hermetic by default: no user-home .env/config loading and
    # no recursive local test runner unless a test explicitly opts in.
    os.environ.setdefault("COGEV_HERMETIC_TEST", "1")
    os.environ.setdefault("COGEV_RESEARCH_DEFAULT_ENABLED", "0")
    _remove_repo_pycache()


def _remove_repo_pycache() -> None:
    root = Path(__file__).resolve().parents[1]
    pycache = root / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache, ignore_errors=True)
