"""Safe environment-file templates for source installs and deployments.

The public repository never ships a real ``.env``.  It ships generic example
profiles and a small generator that copies one of those profiles to an operator
chosen path.  Model access remains provider-generic: LiteLLM, direct
OpenAI-compatible HTTP, or fixture-only tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvTemplate:
    profile: str
    filename: str
    description: str


_ENV_TEMPLATES: dict[str, EnvTemplate] = {
    "local": EnvTemplate(
        profile="local",
        filename=".env.example",
        description="loopback development with a generic LiteLLM/OpenAI-compatible upstream placeholder",
    ),
    "production": EnvTemplate(
        profile="production",
        filename=".env.production.example",
        description="shared deployment template with public-bind safety defaults and explicit trusted CORS",
    ),
    "fixture": EnvTemplate(
        profile="fixture",
        filename=".env.fixture.example",
        description="deterministic fixture provider for tests and demos only",
    ),
}

_FALLBACK_TEMPLATES: dict[str, str] = {
    "local": """# CognitiveEvolve generic local configuration\nCOGEV_LLM_PROVIDER=litellm\nCOGEV_LLM_MODEL=provider/model-id\nCOGEV_LLM_API_BASE=https://your-provider.example/v1\nCOGEV_LLM_API_KEY=replace-with-your-upstream-model-api-key\nCOGEV_SERVER_HOST=127.0.0.1\nCOGEV_SERVER_PORT=8765\nCOGEV_SERVER_PUBLIC_BASE_URL=http://127.0.0.1:8765/v1\nCOGEV_SERVER_REQUIRE_AUTH=true\nCOGEV_SERVER_API_KEY=ce-local-dev-key-change-me\nCOGEV_STOP_POLICY=adaptive_until_solved\nCOGEV_ADAPTIVE_ENABLED=false\nCOGEV_SPATIAL_MODE=observe\nCOGEV_EXTERNAL_EVALUATOR_COMMAND=\nCOGEV_EXTERNAL_EVALUATOR_TIMEOUT=30\n""",
    "production": """# CognitiveEvolve generic production configuration\nCOGEV_LLM_PROVIDER=litellm\nCOGEV_LLM_MODEL=provider/model-id\nCOGEV_LLM_API_BASE=https://your-provider.example/v1\nCOGEV_LLM_API_KEY=replace-with-your-upstream-model-api-key\nCOGEV_SERVER_HOST=0.0.0.0\nCOGEV_SERVER_PORT=8765\nCOGEV_SERVER_PUBLIC_BASE_URL=https://cognitive-evolve.example/v1\nCOGEV_SERVER_REQUIRE_AUTH=true\nCOGEV_SERVER_API_KEY=replace-with-high-entropy-service-api-key\nCOGEV_CORS_ALLOW_ORIGINS=https://your-frontend.example\nCOGEV_STOP_POLICY=adaptive_until_solved\nCOGEV_ADAPTIVE_ENABLED=false\nCOGEV_SPATIAL_MODE=observe\nCOGEV_EXTERNAL_EVALUATOR_COMMAND=\nCOGEV_EXTERNAL_EVALUATOR_TIMEOUT=30\n""",
    "fixture": """# CognitiveEvolve deterministic fixture configuration\nCOGEV_LLM_PROVIDER=fixture\nCOGEV_LLM_FIXTURE=tests/fixtures/llm_fixture.json\nCOGEV_SERVER_HOST=127.0.0.1\nCOGEV_SERVER_PORT=8765\nCOGEV_SERVER_PUBLIC_BASE_URL=http://127.0.0.1:8765/v1\nCOGEV_SERVER_REQUIRE_AUTH=true\nCOGEV_SERVER_API_KEY=ce-local-dev-key-change-me\n""",
}


def available_env_profiles() -> tuple[str, ...]:
    """Return supported environment template profiles."""

    return tuple(sorted(_ENV_TEMPLATES))


def env_template_info(profile: str) -> EnvTemplate:
    """Return metadata for a known env template profile."""

    key = _normalize_profile(profile)
    return _ENV_TEMPLATES[key]


def render_env_template(profile: str = "local") -> str:
    """Render a safe generic ``.env`` template.

    The source checkout templates are the source of truth.  Embedded fallbacks
    keep installed console scripts usable if a packaging frontend omits root
    example files.
    """

    key = _normalize_profile(profile)
    info = _ENV_TEMPLATES[key]
    source_path = _repo_root() / info.filename
    if source_path.exists():
        return source_path.read_text(encoding="utf-8")
    return _FALLBACK_TEMPLATES[key]


def write_env_template(output: Path | str, *, profile: str = "local", force: bool = False) -> Path:
    """Write a selected env template to ``output`` without overwriting by default."""

    target = Path(output).expanduser()
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing env file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_env_template(profile), encoding="utf-8")
    return target.resolve()


def _normalize_profile(profile: str) -> str:
    key = str(profile or "local").strip().lower().replace("_", "-")
    aliases = {
        "dev": "local",
        "development": "local",
        "prod": "production",
        "test": "fixture",
        "demo": "fixture",
    }
    key = aliases.get(key, key)
    if key not in _ENV_TEMPLATES:
        choices = ", ".join(available_env_profiles())
        raise ValueError(f"Unknown env template profile {profile!r}; choose one of: {choices}")
    return key


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


__all__ = [
    "EnvTemplate",
    "available_env_profiles",
    "env_template_info",
    "render_env_template",
    "write_env_template",
]
