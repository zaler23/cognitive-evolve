#!/bin/zsh
set -e
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"
exec ./scripts/start-cognitive-evolve-api.sh
