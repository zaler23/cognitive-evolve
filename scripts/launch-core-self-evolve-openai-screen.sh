#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${COGEV_PROJECT_DIR:-$SCRIPT_DIR}"
RUNTIME_ROOT="${COGEV_RUNTIME_ROOT:-$HOME/.cognitive-evolve}"
API_HEALTH_URL="${COGEV_API_HEALTH_URL:-http://127.0.0.1:8765/health}"
UPSTREAM_HEALTH_URL="${COGEV_UPSTREAM_HEALTH_URL:-}"
REQUIRE_UPSTREAM_HEALTH="false"
LABEL="runner-preflight-validation"
NOT_BEFORE=""
RUN_DIR=""
DRY_RUN="false"
RESUME="false"
INCLUDE_TESTS="false"
MAX_ROUNDS="48"
BRANCH_FACTOR="4"
INITIAL_CANDIDATES="16"
MIN_ROUNDS="8"
SERVICES_SCREEN=""
RUNNER_SCREEN=""

usage() {
  cat <<HELP
Launch CognitiveEvolve core self-evolution under screen through a configured
OpenAI-compatible upstream.

Usage:
  $0 [options]

Options:
  --label NAME                 run label; default: runner-preflight-validation
  --not-before ISO             do not start services or model calls before this time
  --run-dir PATH               explicit run directory
  --resume                     resume the run-dir checkpoint
  --include-tests              run project tests in patch sandboxes
  --max-rounds N               default: 48
  --branch-factor N            default: 4
  --initial-candidates N       default: 16
  --min-rounds-before-stop N   default: 8
  --require-upstream-health    require --upstream-health-url before model calls
  --upstream-health-url URL    operator-supplied upstream health endpoint
  --dry-run                    print the launch plan without starting services/runner
  -h, --help                   show this help
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --not-before) NOT_BEFORE="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --resume) RESUME="true"; shift ;;
    --include-tests) INCLUDE_TESTS="true"; shift ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --branch-factor) BRANCH_FACTOR="$2"; shift 2 ;;
    --initial-candidates) INITIAL_CANDIDATES="$2"; shift 2 ;;
    --min-rounds-before-stop) MIN_ROUNDS="$2"; shift 2 ;;
    --require-upstream-health) REQUIRE_UPSTREAM_HEALTH="true"; shift ;;
    --upstream-health-url) UPSTREAM_HEALTH_URL="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

safe_label="$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9_-' '-' | sed 's/^-*//;s/-*$//')"
if [[ -z "$safe_label" ]]; then
  safe_label="core-self-evolve"
fi
stamp="$(date '+%Y%m%d-%H%M%S')"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$RUNTIME_ROOT/.cogev/api-runs/self-evolve-core-openai-${safe_label}-${stamp}"
fi
SERVICES_SCREEN="cogev_${safe_label}_services"
RUNNER_SCREEN="cogev_${safe_label}_runner_${stamp}"

before_not_before() {
  [[ -z "$NOT_BEFORE" ]] && return 1
  python3 - "$NOT_BEFORE" <<'PY'
from __future__ import annotations
from datetime import datetime, timezone
import sys
text = sys.argv[1]
if text.endswith('Z'):
    text = text[:-1] + '+00:00'
target = datetime.fromisoformat(text)
if target.tzinfo is None:
    target = target.replace(tzinfo=timezone.utc)
raise SystemExit(0 if datetime.now(timezone.utc) < target.astimezone(timezone.utc) else 1)
PY
}

runner_args=(
  --project-dir "$PROJECT_DIR"
  --run-dir "$RUN_DIR"
  --label "$LABEL"
  --max-rounds "$MAX_ROUNDS"
  --round-safety-limit "$MAX_ROUNDS"
  --branch-factor "$BRANCH_FACTOR"
  --initial-candidates "$INITIAL_CANDIDATES"
  --min-rounds-before-stop "$MIN_ROUNDS"
)
if [[ -n "$NOT_BEFORE" ]]; then
  runner_args+=(--not-before "$NOT_BEFORE")
fi
if [[ "$RESUME" == "true" ]]; then
  runner_args+=(--resume)
fi
if [[ "$INCLUDE_TESTS" == "true" ]]; then
  runner_args+=(--include-tests)
fi
if [[ "$REQUIRE_UPSTREAM_HEALTH" == "true" ]]; then
  runner_args+=(--require-upstream-health --upstream-health-url "$UPSTREAM_HEALTH_URL")
fi

cat <<PLAN
CognitiveEvolve OpenAI-compatible self-evolution launch plan
Project: $PROJECT_DIR
Run dir: $RUN_DIR
Services screen: $SERVICES_SCREEN
Runner screen: $RUNNER_SCREEN
API health: $API_HEALTH_URL
Upstream health: ${UPSTREAM_HEALTH_URL:-not configured}
Require upstream health: $REQUIRE_UPSTREAM_HEALTH
Not before: ${NOT_BEFORE:-none}
Dry run: $DRY_RUN
PLAN

mkdir -p "$RUN_DIR"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: no service, screen, or model call will be started."
  printf 'Runner command:'
  printf ' %q' "$PROJECT_DIR/scripts/run-core-self-evolve-openai.sh" "${runner_args[@]}"
  printf '\n'
  exit 0
fi

if before_not_before; then
  echo "Waiting for not-before gate: $NOT_BEFORE"
  "$PROJECT_DIR/scripts/run-core-self-evolve-openai.sh" "${runner_args[@]}"
  exit 75
fi

if [[ "$REQUIRE_UPSTREAM_HEALTH" == "true" && -z "$UPSTREAM_HEALTH_URL" ]]; then
  echo "--require-upstream-health requires --upstream-health-url." >&2
  exit 2
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for background launch." >&2
  exit 2
fi

screen -S "$SERVICES_SCREEN" -X quit >/dev/null 2>&1 || true
project_q="$(printf '%q' "$PROJECT_DIR")"
screen -dmS "$SERVICES_SCREEN" bash -lc "cd $project_q && ./scripts/start-cognitive-evolve-api.sh"

for _ in {1..60}; do
  if curl -fsS "$API_HEALTH_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "$API_HEALTH_URL" >/dev/null 2>&1; then
  echo "API service did not become healthy in time" >&2
  exit 1
fi

if [[ "$REQUIRE_UPSTREAM_HEALTH" == "true" ]]; then
  for _ in {1..10}; do
    if curl -fsS "$UPSTREAM_HEALTH_URL" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  if ! curl -fsS "$UPSTREAM_HEALTH_URL" >/dev/null 2>&1; then
    echo "upstream health endpoint did not become healthy in time" >&2
    exit 1
  fi
fi

runner_log="$RUN_DIR/self-evolve-runner.out.log"
: > "$runner_log"
runner_cmd=""
printf -v runner_cmd '%q ' "$PROJECT_DIR/scripts/run-core-self-evolve-openai.sh" "${runner_args[@]}"
runner_log_q="$(printf '%q' "$runner_log")"
screen -dmS "$RUNNER_SCREEN" bash -lc "cd $project_q && $runner_cmd >> $runner_log_q 2>&1"

cat <<DONE
Started CognitiveEvolve OpenAI-compatible self-evolution.
Run dir: $RUN_DIR
Services screen: $SERVICES_SCREEN
Runner screen: $RUNNER_SCREEN
Runner log: $runner_log
DONE
