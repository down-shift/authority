#!/usr/bin/env bash
# Idempotent environment bootstrap for the autonomous loop. Cross-platform
# (Linux GPU nodes / macOS laptop). Safe to re-run. See docs/AUTONOMY.md.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# 1. uv (package/venv manager)
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. project venv + package from uv.lock (inference extras on GPU nodes)
if command -v nvidia-smi >/dev/null 2>&1; then
  uv sync --extra dev --extra inference --extra quantization || {
    echo "WARN: inference extras failed to install; falling back to dev only"
    uv sync --extra dev
  }
else
  uv sync --extra dev
fi

# 3. working dirs (outputs/ is git-ignored except .gitkeep)
mkdir -p outputs

# 4. soft prerequisites (warn, don't fail)
if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || echo "WARN: gh not authenticated — run 'gh auth login' for PR/auto-merge"
else
  echo "WARN: gh CLI missing — PR/auto-merge unavailable"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU(s): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd', ' -)"
  { [ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; } \
    || echo "WARN: no Hugging Face token — gated models (e.g. google/gemma-3-*) will fail to download"
else
  echo "NOTE: no GPU on this machine — inference steps ([GPU: …]) must run on a model-capable node"
fi

echo "bootstrap: OK"
