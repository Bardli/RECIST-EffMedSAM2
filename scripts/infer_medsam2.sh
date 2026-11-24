#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

has_flag() {
  local name=$1; shift
  for arg in "$@"; do
    case "$arg" in
      "$name"|"$name"=*) return 0 ;;
    esac
  done
  return 1
}

MODEL="${MODEL:-medsam2_box}"
INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/data/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/medsam2}"
ASSETS_DIR="${ASSETS_DIR:-$REPO_ROOT/assets}"
DEVICE="${DEVICE:-cpu}"
NUM_THREADS="${NUM_THREADS:-1}"
VERIFY_SHA256="${VERIFY_SHA256:-1}"
PLOTS="${PLOTS:-1}"

if has_flag "-h" "$@" || has_flag "--help" "$@"; then
  cat <<EOF
Usage: bash scripts/infer_medsam2.sh [recist-benchmark options...]

Run MedSAM2 (FLARE25 RECIST baseline) inference with per-case reload, resource profiling, and CSV reporting.

Defaults:
  --models         $MODEL
  --input-dir      $INPUT_DIR
  --output-dir     $OUTPUT_DIR
  --assets-dir     $ASSETS_DIR
  --device         $DEVICE
  --num-threads    $NUM_THREADS
  --verify-sha256  $([[ "$VERIFY_SHA256" != "0" ]] && echo enabled || echo disabled)
  --plots          $([[ "$PLOTS" != "0" ]] && echo enabled || echo disabled)

Environment overrides:
  PYTHON, MODEL, INPUT_DIR, WINDOW_DIR, OUTPUT_DIR, ASSETS_DIR, DEVICE, NUM_THREADS, VERIFY_SHA256=0, PLOTS=0
EOF
  exit 0
fi

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv-sam/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv-sam/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
has_flag "--models-config" "$@" || EXTRA_ARGS+=(--models-config "$REPO_ROOT/configs/models.json")
has_flag "--models" "$@" || EXTRA_ARGS+=(--models "$MODEL")
has_flag "--input-dir" "$@" || EXTRA_ARGS+=(--input-dir "$INPUT_DIR")
has_flag "--output-dir" "$@" || EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
has_flag "--assets-dir" "$@" || EXTRA_ARGS+=(--assets-dir "$ASSETS_DIR")
has_flag "--device" "$@" || EXTRA_ARGS+=(--device "$DEVICE")
has_flag "--num-threads" "$@" || EXTRA_ARGS+=(--num-threads "$NUM_THREADS")
has_flag "--sam-python" "$@" || EXTRA_ARGS+=(--sam-python "$PYTHON")
if [[ "$VERIFY_SHA256" != "0" ]] && ! has_flag "--verify-sha256" "$@"; then
  EXTRA_ARGS+=(--verify-sha256)
fi
if [[ "$PLOTS" != "0" ]] && ! has_flag "--plots" "$@"; then
  EXTRA_ARGS+=(--plots)
fi

if [[ -z "${WINDOW_DIR:-}" ]] && ! has_flag "--input-dir" "$@" && ! has_flag "--window-dir" "$@" && ! has_flag "--no-window-dir" "$@"; then
  if [[ "$INPUT_DIR" == "$REPO_ROOT/data/inputs" ]] && [[ -d "$REPO_ROOT/data/window_twins" ]] \
    && find "$REPO_ROOT/data/window_twins" -maxdepth 2 -name '*.npz' -print -quit | grep -q .; then
    WINDOW_DIR="$REPO_ROOT/data/window_twins"
  fi
fi
if [[ -n "${WINDOW_DIR:-}" ]] && ! has_flag "--window-dir" "$@" && ! has_flag "--no-window-dir" "$@"; then
  EXTRA_ARGS+=(--window-dir "$WINDOW_DIR")
fi

exec "$PYTHON" -m recist_effmedsam2.runner.run_benchmark "${EXTRA_ARGS[@]}" "$@"
