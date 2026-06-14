from __future__ import annotations

import os
from pathlib import Path

from cognitive_evolve_runtime.api.config import load_service_env


def test_api_key_not_loaded_from_home_env_in_hermetic_tests(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir()
    (runtime_root / ".env").write_text("COGEV_LLM_API_KEY=forbidden-home-key\nCOGEV_SERVER_API_KEY=forbidden-service-key\n", encoding="utf-8")

    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.delenv("COGEV_ENV_FILE", raising=False)
    monkeypatch.delenv("COGEV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("COGEV_SERVER_API_KEY", raising=False)

    loaded = load_service_env()

    assert loaded is None
    assert os.environ.get("COGEV_LLM_API_KEY") is None
    assert os.environ.get("COGEV_SERVER_API_KEY") is None
