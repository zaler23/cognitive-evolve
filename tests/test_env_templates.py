from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_evolve_runtime.config_templates import available_env_profiles, render_env_template, write_env_template


def test_env_templates_are_generic_and_bridge_free() -> None:
    assert set(available_env_profiles()) == {"fixture", "local", "production"}
    for profile in available_env_profiles():
        text = render_env_template(profile)
        lowered = text.lower()
        assert "cogev_llm_provider" in lowered
        assert ("anti" + "gravity") not in lowered
        assert ("gemi" + "ni-3.5") not in lowered
        assert ("codex" + "-local") not in lowered
        assert ("local" + "-bridge") not in lowered
        assert "/users/" not in lowered


def test_write_env_template_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    written = write_env_template(target, profile="fixture")
    assert written == target.resolve()
    assert "COGEV_LLM_PROVIDER=fixture" in target.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_env_template(target, profile="local")

    write_env_template(target, profile="local", force=True)
    assert "COGEV_LLM_PROVIDER=litellm" in target.read_text(encoding="utf-8")
