#!/usr/bin/env bash
# Phase 1 baseline environment capture.
#
# Default mode: query the project's .venv via `uv pip` (matches how uv builds
# the venv — without a bundled pip executable). For a non-uv setup, point
# PYTHON_BIN at the right interpreter; pip freeze then falls back to
# `python -m pip` if available, otherwise `uv pip`.
#
# Usage:
#   bash bench/env/capture.sh                 # uses .venv (uv-managed)
#   PYTHON_BIN=apple_env/bin/python bash bench/env/capture.sh
#
# Writes bench/env/baseline-<gitsha>.txt with: this repo's git sha,
# ComfyUI's git sha (override path with COMFY_DIR=...), python + macOS
# versions, the resolved git commit of python-coreml-stable-diffusion as
# installed, and the full freeze of the active venv (for fresh-venv
# reproducibility).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SHA_SHORT="$(git rev-parse --short HEAD)"
OUT_DIR="bench/env"
OUT="${OUT_DIR}/baseline-${SHA_SHORT}.txt"
mkdir -p "$OUT_DIR"

COMFY_DIR="${COMFY_DIR:-$(cd ../.. && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "PYTHON_BIN not executable: $PYTHON_BIN" >&2
  exit 1
fi

# Prefer `python -m pip` (works in classic venvs). Fall back to `uv pip`,
# which queries any interpreter without needing pip installed in it.
run_freeze() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip freeze
  elif command -v uv >/dev/null 2>&1; then
    uv pip freeze --python "$PYTHON_BIN"
  else
    echo "neither pip nor uv available for freeze"
  fi
}

run_show_msd() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip show python-coreml-stable-diffusion 2>/dev/null || echo "not-installed"
  elif command -v uv >/dev/null 2>&1; then
    uv pip show --python "$PYTHON_BIN" python-coreml-stable-diffusion 2>/dev/null || echo "not-installed"
  else
    echo "neither pip nor uv available for show"
  fi
}

{
  echo "# Phase 1 baseline environment capture"
  echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_sha: $(git rev-parse HEAD)"
  echo "repo_branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "comfyui_dir: ${COMFY_DIR}"
  echo "comfyui_sha: $(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python_bin: ${PYTHON_BIN}"
  echo "python_version: $("$PYTHON_BIN" --version 2>&1)"
  echo "platform_machine: $(uname -m)"
  echo "platform_uname: $(uname -a)"
  if command -v sw_vers >/dev/null 2>&1; then
    echo "macos_product: $(sw_vers -productName)"
    echo "macos_version: $(sw_vers -productVersion)"
    echo "macos_build: $(sw_vers -buildVersion)"
  fi
  echo
  echo "## python-coreml-stable-diffusion (resolved)"
  run_show_msd
  echo
  echo "## python-coreml-stable-diffusion installed git metadata"
  DIST_INFO="$("$PYTHON_BIN" -c "import importlib.metadata as m; d=m.distribution('python-coreml-stable-diffusion'); print(d._path)" 2>/dev/null || true)"
  if [ -n "${DIST_INFO}" ] && [ -f "${DIST_INFO}/direct_url.json" ]; then
    cat "${DIST_INFO}/direct_url.json"
    echo
  else
    echo "direct_url.json not found"
  fi
  echo
  echo "## coremltools version"
  "$PYTHON_BIN" -c "import coremltools; print('coremltools', coremltools.__version__)" 2>&1 || true
  echo
  echo "## torch version"
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__)" 2>&1 || true
  echo
  echo "## numpy version"
  "$PYTHON_BIN" -c "import numpy; print('numpy', numpy.__version__)" 2>&1 || true
  echo
  echo "## freeze"
  run_freeze
} > "$OUT"

echo "wrote $OUT"
