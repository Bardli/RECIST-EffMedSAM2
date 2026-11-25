#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ASSETS_DIR=${1:-"$REPO_ROOT/assets"}
MANIFEST="$REPO_ROOT/configs/weights.sha256"

mkdir -p \
  "$ASSETS_DIR/MedSAM2-RECIST/checkpoints" \
  "$ASSETS_DIR/RECIST-EffTiny-m8/checkpoints" \
  "$ASSETS_DIR/nninteractive_hf_models/models/nnInteractive_v1.0/fold_0"

download_file() {
  local url=$1
  local dest=$2
  if [[ -f "$dest" ]]; then
    echo "[skip] already exists: $dest"
    return 0
  fi
  echo "[download] $url -> $dest"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$dest.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest.part" "$url"
  else
    echo "Error: curl or wget is required." >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

# 1. MedSAM2 RECIST checkpoint (pinned commit)
MEDSAM2_REV="e4a6f35edd7e091619cbc0750f462f1574e23955"
download_file \
  "https://huggingface.co/wanglab/MedSAM2/resolve/${MEDSAM2_REV}/medsam2_FLARE25_RECIST_baseline.pt" \
  "$ASSETS_DIR/MedSAM2-RECIST/checkpoints/medsam2_FLARE25_RECIST_baseline.pt"

# 2. nnInteractive v1.0 official release (pinned commit)
NN_REV="3f308d751c00644e4fde6f09c600264b393b21b5"
for rel in \
  "LICENSE" \
  "dataset.json" \
  "inference_session_class.json" \
  "plans.json" \
  "fold_0/checkpoint_final.pth"; do
  download_file \
    "https://huggingface.co/MIC-DKFZ/nnInteractive/resolve/${NN_REV}/nnInteractive_v1.0/${rel}" \
    "$ASSETS_DIR/nninteractive_hf_models/models/nnInteractive_v1.0/${rel}"
done

echo ""
echo "[verify] Checking downloaded public weights against $MANIFEST..."
(
  cd "$ASSETS_DIR"
  grep -E 'MedSAM2-RECIST|nninteractive_hf_models' "$MANIFEST" | sha256sum -c -
)

EFF_TINY="$ASSETS_DIR/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt"
echo ""
if [[ -f "$EFF_TINY" ]]; then
  echo "[verify] Checking RECIST-EffTiny-m8 checkpoint..."
  (
    cd "$ASSETS_DIR"
    grep 'RECIST-EffTiny-m8' "$MANIFEST" | sha256sum -c -
  )
else
  echo "[note] Place RECIST-EffTiny-m8 epoch 40 checkpoint at:"
  echo "       $EFF_TINY"
  echo "       and verify with:"
  echo "       (cd \"$ASSETS_DIR\" && grep RECIST-EffTiny-m8 \"$MANIFEST\" | sha256sum -c -)"
fi
