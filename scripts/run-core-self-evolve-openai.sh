#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${COGEV_PROJECT_DIR:-$SCRIPT_DIR}"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="$(command -v python3)"
fi
ENV_FILE="${COGEV_ENV_FILE:-$HOME/.cognitive-evolve/.env}"
load_env_file() {
  local env_file="$1"
  local line key value first last
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*($|#) ]] && continue
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    value="${value%$'\r'}"
    if [[ ${#value} -ge 2 ]]; then
      first="${value:0:1}"
      last="${value: -1}"
      if [[ ( "$first" == "'" && "$last" == "'" ) || ( "$first" == '"' && "$last" == '"' ) ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    export "$key=$value"
  done < "$env_file"
}
if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi
export COGEV_LLM_PROVIDER="${COGEV_LLM_PROVIDER:-litellm}"
export COGEV_LLM_TEMPERATURE="${COGEV_CORE_SELF_EVOLVE_TEMPERATURE:-${COGEV_LLM_TEMPERATURE:-0.7}}"
export COGEV_LLM_TIMEOUT="${COGEV_LLM_TIMEOUT:-900}"
exec "$VENV_PY" "$PROJECT_DIR/scripts/run-core-self-evolve-openai.py" --project-dir "$PROJECT_DIR" "$@"
