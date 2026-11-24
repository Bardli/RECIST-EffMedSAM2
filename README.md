# RECIST-EffMedSAM2

A clean, standalone inference and efficiency-benchmarking suite for **3D CT lesion segmentation from 2D RECIST annotations** across **RECIST-EffMedSAM2**, **MedSAM2**, and **nnInteractive** — designed for effortless CPU/GPU reproduction in native Python without private containers.

*Prefer letting an AI coding agent handle the setup?* Open the cloned folder in your agent and tell it to follow [`SKILL.md`](SKILL.md) — it will create the virtual environments, download and verify checkpoints, run a 1-case smoke test, and check the output CSVs for you.

---

## Installation

**Install only what you need:**

- **`.venv-sam`** — for **RECIST-EffMedSAM2** and **MedSAM2**.
- **`.venv-nn`** — for **nnInteractive** (`box`, `point`, and `scribble`).

Both `requirements/sam.txt` and `requirements/nninteractive.txt` already include the core runner, profiler, and evaluator dependencies (`requirements/core.txt`), so no separate base environment is required.

First clone the repository:

```bash
git clone https://github.com/Bardli/RECIST-EffMedSAM2.git
cd RECIST-EffMedSAM2
```

> [!NOTE]
> For GPU environments, pick the PyTorch install command matching your CUDA driver on the [PyTorch homepage](https://pytorch.org/get-started/locally/) before installing the remaining packages below.

##### Option A — SAM / RECIST-EffMedSAM2 environment (`.venv-sam`)

```bash
python3 -m venv .venv-sam
source .venv-sam/bin/activate
pip install --upgrade pip

# CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/sam.txt
# Editable install preserves upstream EfficientTAM Hydra configs:
git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e .venv-sam/src/MedSAM2

# Or on CUDA GPU (after installing CUDA PyTorch from pytorch.org):
# pip install -r requirements/sam.txt
# git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
# git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
# pip install --no-build-isolation -e .venv-sam/src/MedSAM2

pip install -e . --no-deps
deactivate
```

##### Option B — nnInteractive environment (`.venv-nn`)

```bash
python3 -m venv .venv-nn
source .venv-nn/bin/activate
pip install --upgrade pip

# Use --index-url https://download.pytorch.org/whl/cpu for CPU-only, or your CUDA wheel for GPU:
pip install torch torchvision
pip install -r requirements/nninteractive.txt
pip install -e . --no-deps
deactivate
```

---

## Getting Started

### 1. Download checkpoints

Download the public `MedSAM2` FLARE25 RECIST baseline and `nnInteractive_v1.0` checkpoints into `./assets` and verify them against [`configs/weights.sha256`](configs/weights.sha256):

```bash
bash scripts/download_weights.sh        # defaults to ./assets (or pass a custom path)
```

Place the `RECIST-EffTiny-m8` checkpoint at `assets/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` (and optionally `RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt` if running `--model-scope all-six`). Passing `--verify-sha256` checks every file hash automatically before inference starts.

### 2. Prepare input NPZs

By default, all inference scripts use repository-relative folders (all ignored by `.gitignore`):

```text
RECIST-EffMedSAM2/
├── data/
│   ├── inputs/           # place your case *.npz files here
│   └── window_twins/     # optional separate pre-windowed SAM twins
├── assets/               # downloaded checkpoints
└── runs/                 # predictions, metrics, plots, and summary CSVs
```

Each `.npz` file stores one 3D CT scan in `(Z, Y, X)` order:
- `imgs`: preserved HU CT volume (`intensity_encoding="nifti_stored_scaled_v1"` with `hu_slope`/`hu_intercept`), raw unencoded HU, or a pre-windowed/normalized volume (`float32` in `[0, 1]` / `[0, 255]`, or legacy `uint8`).
- `recist`: integer `(Z, Y, X)` mask with non-zero lesion IDs drawn on each lesion's axial RECIST slice.
- `spacing`: float `(3,)` physical voxel spacing `[sx, sy, sz]` in mm.
- `gts` *(optional)*: integer `(Z, Y, X)` ground-truth mask for Dice and volume evaluation.

> [!IMPORTANT]
> **Do not quantize raw HU images down to `uint8` if full precision is desired:**
> - **`nnInteractive`** expects raw unwindowed Hounsfield Units (passed straight through as `float32`).
> - **`RECIST-EffMedSAM2` & `MedSAM2`** window HU volumes continuously in `float32` (`L40/W400`) and keep `float32` inputs at full precision through bilinear resizing (legacy `uint8` twins remain supported for backward compatibility).

### 3. Run inference

Three standalone shell scripts mirror the three model families. Each script automatically picks `.venv-sam` or `.venv-nn`, reloads the model fresh per case, profiles CPU/RAM/GPU usage, and writes summary CSVs to `runs/`:

```bash
# 1) RECIST-EffMedSAM2 (Tiny by default; pass --limit 1 for a fast single-case check)
bash scripts/infer_effmedsam2.sh --limit 1

# 2) MedSAM2 (FLARE25 RECIST baseline)
bash scripts/infer_medsam2.sh

# 3) nnInteractive (runs box, 3-point, and scribble configs; CPU by default)
bash scripts/infer_nninteractive.sh
```

Common one-flag switches work across all three scripts:

```bash
# Run on physical GPU 2, enable moving-centroid postprocessing, or override paths:
bash scripts/infer_effmedsam2.sh --device cuda --gpu-index 2 --postprocess moving_centroid
bash scripts/infer_nninteractive.sh --models nninteractive_box_hu --device cuda --gpu-index 0
```

To run the entire 5-model paper benchmark in one sweep and generate combined comparison tables:

```bash
.venv-sam/bin/recist-benchmark --verify-sha256 --plots
```

<details>
<summary>Custom paths & standalone Python CLIs</summary>

```bash
# Explicit input/output/window-twin folders (adds --recursive for nested cancer subfolders)
bash scripts/infer_effmedsam2.sh \
  --input-dir /path/to/hu_inputs \
  --window-dir /path/to/window_twins \
  --output-dir runs/effmedsam2_cohort \
  --recursive \
  --postprocess moving_centroid

# Run Small variant instead of Tiny
MODEL=effsmall_m8_ep40_box bash scripts/infer_effmedsam2.sh

# Single-process Python backend CLIs (without the outer benchmark coordinator)
.venv-sam/bin/recist-sam-infer -i /path/to/case.npz -o runs/sam_single \
  --backend eff --model-size tiny --weights assets/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt \
  --device cpu --num-threads 1 --prompt box --postprocess moving_centroid --disable-image-encoder-compile

.venv-nn/bin/recist-nninteractive-infer -i /path/to/case.npz -o runs/nn_single \
  --checkpoint assets/nninteractive_hf_models/models/nnInteractive_v1.0 \
  --device cpu --num-threads 1 --prompt-mode 3pt_25 --interaction batch
```

</details>

For runtime/memory metric definitions, output CSV schemas, `recist-eval`, and offline `pytest` verification, see [`docs/metrics_and_evaluation.md`](docs/metrics_and_evaluation.md).

---

## Supported Model Configurations

| Model Family | Script | Prompt Derived from RECIST | Model ID (`configs/models.json`) | Scope |
|---|---|---|---|---|
| **RECIST-EffMedSAM2** (Tiny m8, ep40) | `scripts/infer_effmedsam2.sh` | Square box from SVD diameter | `efftiny_m8_ep40_box` | Paper (5 configs) |
| **MedSAM2** | `scripts/infer_medsam2.sh` | Square box from SVD diameter | `medsam2_box` | Paper (5 configs) |
| **nnInteractive-box** | `scripts/infer_nninteractive.sh` | 2D box on RECIST slice | `nninteractive_box_hu` | Paper (5 configs) |
| **nnInteractive-point** | `scripts/infer_nninteractive.sh` | 3 points (25% / 50% / 75% along line) | `nninteractive_3pt_25_hu` | Paper (5 configs) |
| **nnInteractive-scribble** | `scripts/infer_nninteractive.sh` | Skeletonized line, endpoints inset 3 px | `nninteractive_line_skel_in3_hu` | Paper (5 configs) |
| EfficientMedSAM2-Small (Small m8, ep40) | `MODEL=effsmall_m8_ep40_box scripts/infer_effmedsam2.sh` | Square box from SVD diameter | `effsmall_m8_ep40_box` | Supplementary (`--model-scope all-six`) |

---

## License

This codebase is released under the **Apache-2.0** license (see [`LICENSE`](LICENSE)). Third-party pre-trained checkpoints downloaded by `scripts/download_weights.sh` retain their upstream licenses — in particular, **`nnInteractive_v1.0` is licensed under `CC BY-NC-SA 4.0`** (non-commercial), and `MedSAM2` follows its upstream FLARE25 release terms.

## Citation

> Manuscript / arXiv citation coming soon.

## Acknowledgments

Built upon [MedSAM2](https://github.com/bowang-lab/MedSAM2), [EfficientTAM](https://github.com/yformer/EfficientTAM), [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), and [FLARE-Efficiency](https://github.com/medfm-flare/FLARE-Efficiency).
