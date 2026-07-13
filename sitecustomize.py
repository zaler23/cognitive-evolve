"""Project-local Python startup defaults for reproducible test runs."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

if any(part.endswith("pytest") or part.endswith("pytest.__main__") for part in sys.argv[:1]) or "pytest" in " ".join(sys.argv[:2]):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def _cleanup_startup_pycache() -> None:
    root_cache = Path(__file__).resolve().parent / "__pycache__"
    if root_cache.exists():
        shutil.rmtree(root_cache, ignore_errors=True)


atexit.register(_cleanup_startup_pycache)
