#!/usr/bin/env bash
set -euo pipefail

export COPYFILE_DISABLE=1

cd "$(dirname "$0")/.."

name="cognitive-evolve-v2.0.0"
out_dir="dist"
out="$out_dir/${name}-public-clean.tar.gz"
tmp="$(mktemp -d)"
check_tmp="$(mktemp -d)"
trap 'rm -rf "$tmp" "$check_tmp"' EXIT

mkdir -p "$tmp/$name" "$out_dir"
rm -f "$out_dir"/*public-clean.tar.gz "$out_dir"/*public-clean.tar.gz.sha256

rsync -a ./ "$tmp/$name/" \
  --exclude '._*' \
  --exclude '.DS_Store' \
  --exclude '__MACOSX/' \
  --exclude '.git/' \
  --exclude '.collab/' \
  --exclude '.local-provider-cache/' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude '.venv/' \
  --exclude 'node_modules/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude '*.egg-info/' \
  --exclude 'runtime/' \
  --exclude '.cogev/tasks/' \
  --exclude '.cogev/api-runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'htmlcov/' \
  --exclude '.coverage' \
  --exclude 'coverage.xml' \
  --exclude 'package-lock.json'

tar_extra=()
if tar --help 2>&1 | grep -q -- '--disable-copyfile'; then
  tar_extra+=(--disable-copyfile)
fi
if tar --help 2>&1 | grep -q -- '--no-xattrs'; then
  tar_extra+=(--no-xattrs)
fi
if tar --help 2>&1 | grep -q -- '--no-acls'; then
  tar_extra+=(--no-acls)
fi

tar "${tar_extra[@]}" -C "$tmp" -czf "$out" "$name"
sha256_value="$(shasum -a 256 "$out" | awk '{print $1}')"
printf '%s  %s\n' "$sha256_value" "$(basename "$out")" > "$out.sha256"

tar -xzf "$out" -C "$check_tmp"
root="$check_tmp/$name"

for pattern in '._*' '__MACOSX' '.git' '.env' '.env.local' '.pytest_cache' '__pycache__' '.coverage'; do
  found="$(find "$root" -name "$pattern" -print -quit)"
  if [[ -n "$found" ]]; then
    echo "package_clean failed: forbidden artifact found: $found" >&2
    exit 1
  fi
done

compile_targets=()
for rel in cognitive_evolve_runtime scripts tests; do
  if [[ -e "$root/$rel" ]]; then
    compile_targets+=("$root/$rel")
  fi
done
python3 -m compileall -q "${compile_targets[@]}"

echo "$out"
