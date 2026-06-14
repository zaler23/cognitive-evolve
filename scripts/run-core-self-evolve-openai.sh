#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${COGEV_PROJECT_DIR:-$SCRIPT_DIR}"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="$(command -v python3)"
fi
ENV_FILE="${COGEV_ENV_FILE:-$HOME/.cognitive-evolve/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
export COGEV_LLM_PROVIDER="${COGEV_LLM_PROVIDER:-litellm}"
export COGEV_LLM_TEMPERATURE="${COGEV_CORE_SELF_EVOLVE_TEMPERATURE:-${COGEV_LLM_TEMPERATURE:-0.7}}"
export COGEV_LLM_TIMEOUT="${COGEV_LLM_TIMEOUT:-900}"
export COGEV_LLM_MAX_TOKENS="${COGEV_LLM_MAX_TOKENS:-8192}"
exec "$VENV_PY" "$PROJECT_DIR/scripts/run-core-self-evolve-openai.py" --project-dir "$PROJECT_DIR" "$@"
