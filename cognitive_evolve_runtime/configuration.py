"""Layered runtime configuration with a small YAML-compatible reader.

Environment variables remain the final source of truth.  This module lets new
operators set the common provider/server/evolution/budget knobs in one
``.cogev/config.yaml`` file without needing all 70+ low-level ``COGEV_*`` vars.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

CONFIG_ENV_MAP = {
    ("provider", "primary"): "COGEV_LLM_PROVIDER",
    ("provider", "model"): "COGEV_LLM_MODEL",
    ("provider", "api_key"): "COGEV_LLM_API_KEY",
    ("provider", "api_base"): "COGEV_LLM_API_BASE",
    ("provider", "fixture"): "COGEV_LLM_FIXTURE",
    ("provider", "temperature"): "COGEV_LLM_TEMPERATURE",
    ("provider", "max_tokens"): "COGEV_LLM_MAX_TOKENS",
    ("provider", "timeout_seconds"): "COGEV_LLM_TIMEOUT",
    ("server", "host"): "COGEV_SERVER_HOST",
    ("server", "port"): "COGEV_SERVER_PORT",
    ("server", "public_base_url"): "COGEV_SERVER_PUBLIC_BASE_URL",
    ("server", "api_key"): "COGEV_SERVER_API_KEY",
    ("server", "api_keys"): "COGEV_SERVER_API_KEYS",
    ("server", "require_auth"): "COGEV_SERVER_REQUIRE_AUTH",
    ("server", "task_root"): "COGEV_API_TASK_ROOT",
    ("evolution", "profile"): "COGEV_EVOLUTION_PROFILE",
    ("evolution", "stop_policy"): "COGEV_STOP_POLICY",
    ("evolution", "min_rounds_before_stop"): "COGEV_MIN_ROUNDS_BEFORE_STOP",
    ("evolution", "safety_rounds"): "COGEV_NEXUS_SAFETY_MAX_ROUNDS",
    ("evolution", "min_candidates"): "COGEV_NEXUS_MIN_CANDIDATES",
    ("evolution", "branch_factor"): "COGEV_NEXUS_BRANCH_FACTOR",
    # Backward-compatible YAML key only; the legacy COGEV_MUTATION_BRANCH_FACTOR
    # env var is intentionally not emitted by config files.
    ("evolution", "mutation_branch_factor"): "COGEV_NEXUS_BRANCH_FACTOR",
    ("budget", "max_usd"): "COGEV_LLM_BUDGET_USD",
    ("budget", "max_tokens"): "COGEV_LLM_MAX_TOKENS",
    ("budget", "stage_strict"): "COGEV_LLM_STAGE_BUDGET_STRICT",
}


@dataclass(frozen=True)
class ConfigSource:
    path: Path
    kind: str
    explicit: bool = False

    @property
    def allowed_in_hermetic_test(self) -> bool:
        if self.explicit:
            return True
        # Hermetic tests may read only cwd/repo-local config candidates, never
        # user-home runtime defaults.  Tests can point COGEV_CONFIG_FILE at a
        # tmp_path config explicitly.
        return self.kind in {"cwd", "repo"}


def hermetic_test_enabled() -> bool:
    return os.environ.get("COGEV_HERMETIC_TEST", "").strip().lower() in {"1", "true", "yes", "on"} or "PYTEST_CURRENT_TEST" in os.environ


def runtime_root() -> Path:
    return Path(os.environ.get("COGEV_RUNTIME_ROOT", Path.home() / ".cognitive-evolve")).expanduser()


def config_sources() -> list[ConfigSource]:
    explicit = os.environ.get("COGEV_CONFIG_FILE", "").strip()
    if explicit:
        return [ConfigSource(Path(explicit).expanduser(), "explicit", explicit=True)]
    repo_root = Path(__file__).resolve().parents[1]
    sources = [
        ConfigSource(runtime_root() / ".cogev" / "config.yaml", "runtime_root"),
        ConfigSource(Path.cwd() / ".cogev" / "config.yaml", "cwd"),
        ConfigSource(repo_root / ".cogev" / "config.yaml", "repo"),
    ]
    if hermetic_test_enabled():
        return [source for source in sources if source.allowed_in_hermetic_test]
    return sources


def config_candidates() -> list[Path]:
    return [source.path for source in config_sources()]


def load_layered_config(path: Path | str | None = None, *, override: bool = False) -> Path | None:
    """Load a small nested config file into ``os.environ``.

    The parser intentionally supports only the simple YAML subset used by the
    documented config: indentation-based nested mappings and scalar values.  It
    avoids a mandatory PyYAML dependency while still accepting ``${ENV_VAR}``
    references.
    """

    if path is not None:
        paths = [Path(path).expanduser()]
    else:
        paths = config_candidates()
    for candidate in paths:
        if not candidate.exists():
            continue
        data = parse_simple_yaml(candidate.read_text(encoding="utf-8"))
        apply_config(data, override=override)
        return candidate.resolve()
    return None


def config_resolution_diagnostics() -> dict[str, Any]:
    """Return non-secret configuration source and precedence diagnostics."""

    sources = [
        {
            "path": str(source.path),
            "kind": source.kind,
            "exists": source.path.exists(),
            "explicit": source.explicit,
            "allowed_in_hermetic_test": source.allowed_in_hermetic_test,
        }
        for source in config_sources()
    ]
    active_env = {}
    for env_name in sorted(set(CONFIG_ENV_MAP.values())):
        if env_name in os.environ:
            active_env[env_name] = _diagnostic_value(env_name, os.environ.get(env_name, ""))
    legacy_profile = sorted(
        name
        for name in os.environ
        if name.startswith("COGEV_NEXUS_PROFILE_") and (name.endswith("_ROUNDS") or name.endswith("_CANDIDATES"))
    )
    return {
        "precedence": [
            "explicit process environment",
            "COGEV_CONFIG_FILE when set",
            "runtime/cwd/repo .cogev/config.yaml defaults when env key is unset",
            "code defaults",
        ],
        "config_sources": sources,
        "active_env_keys": active_env,
        "legacy_profile_keys_present": legacy_profile,
        "legacy_profile_semantics": "legacy *_ROUNDS and *_CANDIDATES are ignored by default unless explicit compatibility flags are enabled",
        "secrets_redacted": True,
    }


def apply_config(data: dict[str, Any], *, override: bool = False) -> dict[str, str]:
    applied: dict[str, str] = {}
    for (section, key), env_name in CONFIG_ENV_MAP.items():
        section_data = data.get(section)
        if not isinstance(section_data, dict) or key not in section_data:
            continue
        if section == "evolution" and key == "mutation_branch_factor" and "branch_factor" in section_data:
            continue
        if not override and os.environ.get(env_name, ""):
            continue
        value = _stringify(section_data[key])
        os.environ[env_name] = value
        applied[env_name] = value
    return applied


def parse_simple_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            return loaded
        raise ValueError("config YAML must decode to an object")
    except ImportError:
        pass
    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else data
        if not raw_value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _parse_scalar(raw_value)
    return data


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"\'')
    value = _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if "," in value and not value.startswith("http"):
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _diagnostic_value(name: str, value: str) -> str:
    if any(token in name for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
        return "[REDACTED]" if value else ""
    return value


__all__ = [
    "CONFIG_ENV_MAP",
    "ConfigSource",
    "apply_config",
    "config_candidates",
    "config_sources",
    "hermetic_test_enabled",
    "load_layered_config",
    "config_resolution_diagnostics",
    "parse_simple_yaml",
    "runtime_root",
]
