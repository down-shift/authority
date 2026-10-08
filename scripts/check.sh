#!/usr/bin/env bash
# The mandatory local gate. Must pass before any commit / auto-merge.
#
# The pre-existing code predates ruff, so lint/format are enforced only on
# Python files changed relative to the base (default: merge-base with
# origin/main, plus untracked files). Touch a file, leave it ruff-clean.
# Tests always run in full. GATE_BASE overrides the base ref (CI sets it).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

base="${GATE_BASE:-origin/main}"
mb="$(git merge-base HEAD "$base" 2>/dev/null || git rev-parse HEAD)"
files="$( { git diff --name-only --diff-filter=ACMR "$mb" -- '*.py'
            git ls-files --others --exclude-standard -- '*.py'; } | sort -u)"

if [ -n "$files" ]; then
  echo "ruff on changed files:"; printf '  %s\n' $files
  # shellcheck disable=SC2086
  uv run --extra dev --extra engines ruff check $files
  # shellcheck disable=SC2086
  uv run --extra dev --extra engines ruff format --check $files
else
  echo "ruff: no changed Python files vs $base"
fi
uv run --extra dev --extra engines pytest -q
echo "gate: OK"
