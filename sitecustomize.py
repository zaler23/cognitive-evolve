"""Project-local Python startup defaults for reproducible test runs."""
from __future__ import annotations

import os
import sys

if any(part.endswith("pytest") or part.endswith("pytest.__main__") for part in sys.argv[:1]) or "pytest" in " ".join(sys.argv[:2]):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
