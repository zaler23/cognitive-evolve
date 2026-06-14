from __future__ import annotations

import os
from pathlib import Path

from cognitive_evolve_runtime.configuration import config_candidates, config_resolution_diagnostics, load_layered_config, parse_simple_yaml


def test_config_hermetic_no_home_env(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "home-runtime"
    home_cfg = runtime_root / ".cogev" / "config.yaml"
    home_cfg.parent.mkdir(parents=True)
    home_cfg.write_text("provider:\n  model: forbidden-home-model\n", encoding="utf-8")

    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.delenv("COGEV_CONFIG_FILE", raising=False)
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    assert home_cfg not in config_candidates()
    assert load_layered_config() is None
    assert "COGEV_LLM_MODEL" not in __import__("os").environ


def test_explicit_tmp_config_allowed_in_hermetic_mode(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider:\n  model: fixture-model\n", encoding="utf-8")
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_CONFIG_FILE", str(cfg))
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    assert load_layered_config() == cfg.resolve()
    assert __import__("os").environ["COGEV_LLM_MODEL"] == "fixture-model"


def test_evolution_config_uses_current_nexus_env_names(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
        evolution:
          safety_rounds: 180
          min_candidates: 9
          branch_factor: 7
          mutation_branch_factor: 1
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_CONFIG_FILE", str(cfg))
    for name in ("COGEV_NEXUS_SAFETY_MAX_ROUNDS", "COGEV_NEXUS_MIN_CANDIDATES", "COGEV_NEXUS_BRANCH_FACTOR", "COGEV_MUTATION_BRANCH_FACTOR"):
        monkeypatch.delenv(name, raising=False)

    assert load_layered_config() == cfg.resolve()
    environ = __import__("os").environ
    assert environ["COGEV_NEXUS_SAFETY_MAX_ROUNDS"] == "180"
    assert environ["COGEV_NEXUS_MIN_CANDIDATES"] == "9"
    assert environ["COGEV_NEXUS_BRANCH_FACTOR"] == "7"
    assert "COGEV_MUTATION_BRANCH_FACTOR" not in environ
    for name in ("COGEV_NEXUS_SAFETY_MAX_ROUNDS", "COGEV_NEXUS_MIN_CANDIDATES", "COGEV_NEXUS_BRANCH_FACTOR"):
        os.environ.pop(name, None)


def test_legacy_yaml_mutation_branch_factor_maps_to_current_env(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("evolution:\n  mutation_branch_factor: 4\n", encoding="utf-8")
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_CONFIG_FILE", str(cfg))
    monkeypatch.delenv("COGEV_NEXUS_BRANCH_FACTOR", raising=False)
    monkeypatch.delenv("COGEV_MUTATION_BRANCH_FACTOR", raising=False)

    assert load_layered_config() == cfg.resolve()
    environ = __import__("os").environ
    assert environ["COGEV_NEXUS_BRANCH_FACTOR"] == "4"
    assert "COGEV_MUTATION_BRANCH_FACTOR" not in environ
    os.environ.pop("COGEV_NEXUS_BRANCH_FACTOR", None)


def test_simple_yaml_parser_supports_deeper_nested_mappings() -> None:
    data = parse_simple_yaml(
        """
        provider:
          model: fixture-model
          retry:
            policy:
              max_attempts: 3
        server:
          require_auth: false
        """
    )

    assert data["provider"]["model"] == "fixture-model"
    assert data["provider"]["retry"]["policy"]["max_attempts"] == 3
    assert data["server"]["require_auth"] is False


def test_config_resolution_diagnostics_redacts_secrets_and_reports_legacy(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_LLM_API_KEY", "sk-secret-value")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS", "60")

    diagnostics = config_resolution_diagnostics()

    assert diagnostics["active_env_keys"]["COGEV_LLM_API_KEY"] == "[REDACTED]"
    assert "COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS" in diagnostics["legacy_profile_keys_present"]
    assert diagnostics["secrets_redacted"] is True
