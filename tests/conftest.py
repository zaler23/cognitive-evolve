from __future__ import annotations

import os


def pytest_configure() -> None:
    # Unit tests are hermetic by default: no user-home .env/config loading and
    # no recursive local test runner unless a test explicitly opts in.
    os.environ.setdefault("COGEV_HERMETIC_TEST", "1")
