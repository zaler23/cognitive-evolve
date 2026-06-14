#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.13 || command -v python3 || true)}"
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
VENV_DIR="${COGEV_VENV_DIR:-$CACHE_ROOT/cognitive-evolve/venv}"
RUNTIME_ROOT="${COGEV_RUNTIME_ROOT:-$HOME/.cognitive-evolve}"
ENV_FILE="${COGEV_ENV_FILE:-$RUNTIME_ROOT/.env}"
LOG_DIR="${COGEV_LOG_DIR:-$RUNTIME_ROOT/logs}"
SERVICE_PORT="${COGEV_SERVICE_PORT:-8765}"
STOP_EXISTING_PORT="${COGEV_STOP_EXISTING_PORT:-0}"
VENV_MARKER=".cogev-managed-venv"
SETUP_ONLY="false"

case "${1:-}" in
  --setup-only) SETUP_ONLY="true" ;;
  --help|-h)
    cat <<HELP
CognitiveEvolve 2.0 OpenAI-compatible API launcher

Usage:
  ./scripts/start-cognitive-evolve-api.sh              setup env, then start API
  ./scripts/start-cognitive-evolve-api.sh --setup-only setup env only

Configure upstream model access with COGEV_LLM_PROVIDER, COGEV_LLM_MODEL,
COGEV_LLM_API_BASE, and COGEV_LLM_API_KEY before issuing model-backed requests.

Defaults:
  - Virtual environment: ${COGEV_VENV_DIR:-$CACHE_ROOT/cognitive-evolve/venv}
  - Runtime root:        ${COGEV_RUNTIME_ROOT:-$HOME/.cognitive-evolve}
  - Service port:        ${COGEV_SERVICE_PORT:-8765}

Safety:
  The launcher will not stop an existing listener on the service port unless
  COGEV_STOP_EXISTING_PORT=1 is set explicitly.
HELP
    exit 0
    ;;
esac

mkdir -p "$RUNTIME_ROOT" "$LOG_DIR" "$(dirname "$VENV_DIR")"
cd "$PROJECT_DIR"

echo "== CognitiveEvolve 2.0 OpenAI-compatible API =="
echo "Project: $PROJECT_DIR"
echo "Virtual environment: $VENV_DIR"
echo "Runtime root: $RUNTIME_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)
PY
then
  echo "Python >=3.10 and <3.15 is required." >&2
  exit 1
fi

venv_is_usable() {
  [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)
PY
}

if ! venv_is_usable; then
  if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/$VENV_MARKER" ]]; then
    echo "Virtual environment path exists but is not a CognitiveEvolve-managed venv: $VENV_DIR" >&2
    echo "Set COGEV_VENV_DIR to another path or remove that directory manually." >&2
    exit 1
  fi
  echo "Preparing Python virtual environment..."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  date > "$VENV_DIR/$VENV_MARKER"
fi

if [[ ! -f "$VENV_DIR/.cogev-installed" ]] || [[ "$PROJECT_DIR/pyproject.toml" -nt "$VENV_DIR/.cogev-installed" ]]; then
  echo "Installing/updating project dependencies..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
  "$VENV_DIR/bin/python" -m pip install "$PROJECT_DIR" >/dev/null
  date > "$VENV_DIR/.cogev-installed"
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "No env file found at $ENV_FILE; continuing with current environment variables."
fi

if [[ "$SETUP_ONLY" == "true" ]]; then
  echo "Environment is ready."
  exit 0
fi

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill || true
    sleep 1
  fi
}

port_listener_pids() {
  lsof -tiTCP:"$SERVICE_PORT" -sTCP:LISTEN 2>/dev/null || true
}

EXISTING_PIDS="$(port_listener_pids)"
if [[ -n "$EXISTING_PIDS" ]]; then
  if [[ "$STOP_EXISTING_PORT" == "1" ]]; then
    echo "Stopping existing listener(s) on port $SERVICE_PORT because COGEV_STOP_EXISTING_PORT=1."
    stop_port "$SERVICE_PORT"
  else
    echo "Port $SERVICE_PORT is already in use." >&2
    echo "Stop that process yourself, choose COGEV_SERVICE_PORT, or set COGEV_STOP_EXISTING_PORT=1." >&2
    exit 1
  fi
fi

API_LOG="$LOG_DIR/cogev-api.out.log"
: > "$API_LOG"

echo "Starting CognitiveEvolve API on port $SERVICE_PORT..."
"$VENV_DIR/bin/python" "$PROJECT_DIR/scripts/cogev.py" api serve >"$API_LOG" 2>&1 &
API_PID=$!
echo "$API_PID" > "$RUNTIME_ROOT/cogev-api.pid"

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup INT TERM

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null 2>&1; then
    echo "Started."
    echo "Base URL: http://127.0.0.1:$SERVICE_PORT/v1"
    echo "Model: cognitive-evolve-one-shot-deep / cognitive-evolve-one-shot-exhaustive"
    echo "Logs: $LOG_DIR"
    wait "$API_PID"
    exit $?
  fi
  sleep 1
done

echo "Startup failed. API log:" >&2
tail -80 "$API_LOG" >&2 || true
cleanup
exit 1
