from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def pytest_configure() -> None:
    # Unit tests are hermetic by default: no user-home .env/config loading and
    # no recursive local test runner unless a test explicitly opts in.
    sys.dont_write_bytecode = True
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("COGEV_HERMETIC_TEST", "1")
    os.environ.setdefault("COGEV_RESEARCH_DEFAULT_ENABLED", "0")
    _remove_repo_pycache()


def _remove_repo_pycache() -> None:
    root = Path(__file__).resolve().parents[1]
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)
    for pyc in root.rglob("*.pyc"):
        if pyc.is_file():
            pyc.unlink(missing_ok=True)
