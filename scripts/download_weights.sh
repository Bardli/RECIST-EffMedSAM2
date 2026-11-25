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

# 1. RECIST-EffMedSAM2 Tiny (m8, epoch 40) checkpoint (GitHub Release)
EFF_RELEASE_TAG="weights-v1"
download_file \
  "https://github.com/Bardli/RECIST-EffMedSAM2/releases/download/${EFF_RELEASE_TAG}/RECIST-EffTiny-m8-epoch40.pt" \
  "$ASSETS_DIR/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt"

# 2. MedSAM2 RECIST checkpoint (pinned commit)
MEDSAM2_REV="e4a6f35edd7e091619cbc0750f462f1574e23955"
download_file \
  "https://huggingface.co/wanglab/MedSAM2/resolve/${MEDSAM2_REV}/medsam2_FLARE25_RECIST_baseline.pt" \
  "$ASSETS_DIR/MedSAM2-RECIST/checkpoints/medsam2_FLARE25_RECIST_baseline.pt"

# 3. nnInteractive v1.0 official release (pinned commit)
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
echo "[verify] Checking downloaded checkpoints against $MANIFEST..."
(
  cd "$ASSETS_DIR"
  grep -E 'RECIST-EffTiny-m8|MedSAM2-RECIST|nninteractive_hf_models' "$MANIFEST" | sha256sum -c -
)

EFF_SMALL="$ASSETS_DIR/RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt"
if [[ -f "$EFF_SMALL" ]]; then
  echo ""
  echo "[verify] Checking optional RECIST-EffSmall-m8 checkpoint..."
  (
    cd "$ASSETS_DIR"
    grep 'RECIST-EffSmall-m8' "$MANIFEST" | sha256sum -c -
  )
fi
