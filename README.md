# RECIST-EffMedSAM2

A clean, standalone inference and efficiency-benchmarking repository for **3D CT lesion segmentation from 2D RECIST annotations** across **EffMedSAM2**, **MedSAM2**, and **nnInteractive** — organized as simple standalone Python scripts (`infer_effmedsam2.py`, `infer_medsam2.py`, `infer_nninteractive.py`, and `merge_csvs.py`) with no package installation or wrapper shell layers required.

*Prefer letting an AI coding agent handle the setup?* Open the cloned folder in your agent and tell it to follow [`SKILL.md`](SKILL.md) — it will create the virtual environments, download and verify checkpoints, run a 1-case smoke test, and audit the output CSVs for you.

---

## Installation

**Install only what you need:**

- **`.venv-sam`** — for **`infer_effmedsam2.py`** and **`infer_medsam2.py`**.
- **`.venv-nn`** — for **`infer_nninteractive.py`**.

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`uv venv` + `uv pip`) for fast environment setup and automatic Python 3.11 management *(standard `python3 -m venv` + `pip` remains compatible with all `requirements/*.txt` files on offline/HPC systems without `uv`)*. Both `requirements/sam.txt` and `requirements/nninteractive.txt` include `requirements/core.txt`, and every script imports directly from the repository root (`utils/`), so no `pip install -e .` is needed.

First clone the repository:

```bash
git clone https://github.com/Bardli/RECIST-EffMedSAM2.git
cd RECIST-EffMedSAM2
```

> [!NOTE]
> For GPU environments, pick the PyTorch install command matching your CUDA driver on the [PyTorch homepage](https://pytorch.org/get-started/locally/) before installing the remaining packages below.

##### Option A — SAM / EffMedSAM2 environment (`.venv-sam`)

```bash
uv venv --python 3.11 .venv-sam

# CPU-only:
uv pip install --python .venv-sam/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-sam/bin/python -r requirements/sam.txt
# Editable install preserves upstream EfficientTAM Hydra configs:
git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
SAM2_BUILD_CUDA=0 uv pip install --python .venv-sam/bin/python --no-build-isolation -e .venv-sam/src/MedSAM2

# Or on CUDA GPU (after installing CUDA PyTorch from pytorch.org via `uv pip install`):
# uv pip install --python .venv-sam/bin/python -r requirements/sam.txt
# git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
# git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
# uv pip install --python .venv-sam/bin/python --no-build-isolation -e .venv-sam/src/MedSAM2
```

##### Option B — nnInteractive environment (`.venv-nn`)

```bash
uv venv --python 3.11 .venv-nn

# Use --index-url https://download.pytorch.org/whl/cpu for CPU-only, or your CUDA wheel for GPU:
uv pip install --python .venv-nn/bin/python torch torchvision
uv pip install --python .venv-nn/bin/python -r requirements/nninteractive.txt
```

---

## Getting Started

### 1. Download checkpoints

Download and SHA-256-verify all three default checkpoints — `RECIST-EffMedSAM2 Tiny` (from the [GitHub Release `weights-v1`](https://github.com/Bardli/RECIST-EffMedSAM2/releases/tag/weights-v1)), `MedSAM2`, and `nnInteractive_v1.0` — into `./assets` against [`configs/weights.sha256`](configs/weights.sha256):

```bash
bash scripts/download_weights.sh        # defaults to ./assets (or pass a custom path)
```

*(The optional `--model-size small` variant, `assets/RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt`, is not included in the default download and can be passed locally via `--checkpoint` if needed.)*

### 2. Prepare input NPZs

By default, every script reads and writes relative to the repository root (all ignored by `.gitignore`):

```text
RECIST-EffMedSAM2/
├── data/
│   ├── inputs/           # default --input-dir (*.npz)
│   └── window_twins/     # optional pre-windowed twins for EffMedSAM2 / MedSAM2
├── assets/               # checkpoints
└── runs/                 # per-model predictions, metrics JSONs, plots, and results.csv
```

Each `.npz` file stores one 3D CT scan in `(Z, Y, X)` order:
- `imgs`: preserved HU CT volume (`intensity_encoding="nifti_stored_scaled_v1"` with `hu_slope`/`hu_intercept`), raw unencoded HU, or a pre-windowed/normalized volume (`float32` in `[0, 1]` / `[0, 255]`, or legacy `uint8`).
- `recist`: integer `(Z, Y, X)` mask with non-zero lesion IDs drawn on each lesion's axial RECIST slice.
- `spacing`: float `(3,)` physical voxel spacing `[sx, sy, sz]` in mm.
- `gts` *(optional)*: integer `(Z, Y, X)` ground-truth mask for Dice and volume evaluation.

> [!IMPORTANT]
> **Keep CT intensities at full precision whenever possible:**
> - **`nnInteractive`** expects raw unwindowed Hounsfield Units (passed straight through as `float32`).
> - **`EffMedSAM2` & `MedSAM2`** window HU volumes continuously in `float32` (`L40/W400`) and resize `float32` inputs via bilinear `cv2.resize` without `uint8` truncation (legacy `uint8` twins remain supported for compatibility).

### 3. Run inference & merge CSVs

Each model has one standalone Python script that reloads the model fresh per case, runs serial single-threaded inference (`--num-threads 1`) by default, records explicit per-phase timings and RAM/GPU telemetry, saves final prediction NPZs, and writes **one per-model `results.csv`** (one RECIST lesion per row):

```bash
# 1) RECIST-EffMedSAM2 (outputs runs/EffMedSAM2/results.csv)
.venv-sam/bin/python infer_effmedsam2.py -i data/inputs -o runs/EffMedSAM2 --device cpu --plots

# 2) MedSAM2 (outputs runs/MedSAM2/results.csv)
.venv-sam/bin/python infer_medsam2.py -i data/inputs -o runs/MedSAM2 --device cpu --plots

# 3) nnInteractive (box, 3-point, or scribble)
.venv-nn/bin/python infer_nninteractive.py -i data/inputs -o runs/nnInteractive_box --prompt-mode box --device cpu --plots
.venv-nn/bin/python infer_nninteractive.py -i data/inputs -o runs/nnInteractive_point --prompt-mode 3pt_25 --device cpu --plots
.venv-nn/bin/python infer_nninteractive.py -i data/inputs -o runs/nnInteractive_scribble --prompt-mode line --device cpu --plots
```

Finally, concatenate the per-model `results.csv` files directly into **one combined `merged_results.csv`** (preserving the exact same columns without pivoting):

```bash
.venv-sam/bin/python merge_csvs.py -i runs/EffMedSAM2 runs/MedSAM2 runs/nnInteractive_box -o runs/merged/merged_results.csv
```

<details>
<summary>Common options: GPU execution, smoke tests, custom paths, and Small variant</summary>

```bash
# Fast 1-case smoke test on physical GPU 2
.venv-sam/bin/python infer_effmedsam2.py -i data/inputs -o runs/eff_smoke --device cuda --gpu-index 2 --limit 1 --plots

# Custom dataset root with nested folders + pre-windowed companion dir
.venv-sam/bin/python infer_effmedsam2.py \
  -i /path/to/hu_inputs \
  --window-dir /path/to/window_twins \
  -o runs/eff_cohort \
  --recursive \
  --postprocess moving_centroid \
  --plots

# EffMedSAM2-Small variant (disables moving-centroid postprocessing with --postprocess none if desired)
.venv-sam/bin/python infer_effmedsam2.py --model-size small -i data/inputs -o runs/EffMedSAM2_small

# Auto-discover all model folders under runs/ when merging
.venv-sam/bin/python merge_csvs.py --runs-dir runs -o runs/merged/merged_results.csv
```

</details>

For lymph-node inputs, see [which RECIST axis each model actually uses](docs/nodal_prompts.md). The released commands use the supplied prompt; a Tiny clipping exemption does not construct a long axis.

#### Exempt specific Tiny lesions from moving-centroid clipping

Exemption skips **only** the postprocessing disk clip: inference, storage, and same-ID
scoring still run. Pass original positive lesion IDs, not zero-based channel indices.
Without either flag the existing behavior is unchanged. No organ class is inferred.

For a single input file (IDs `2` and `19` are examples):
```bash
.venv-sam/bin/python infer_effmedsam2.py -i data/inputs/case.npz -o runs/tiny_exempt \
  --model-size tiny --postprocess moving_centroid --exempt-ids 2 19
```

For multiple cases use `--exempt-manifest exemptions.json` **instead of** `--exempt-ids`.
The JSON maps the SHA256 of each **actual primary input NPZ** to the IDs exempt in that
file. Use an explicit empty list for a case with no exemptions. Do not hash a window
twin, a NIfTI, or the parent of a repacked input. Compute digests for example with
`sha256sum data/inputs/*.npz`. Manifest shape (replace the sample key with a real digest):

```json
{"schema":"recist-exempt-v1","inputs":{"0000000000000000000000000000000000000000000000000000000000000000":[2,19]}}
```

Unknown input hashes or unknown/unprompted IDs cause the case to fail rather than silently
clipping or scoring it. The two options require Tiny + moving centroid. Direct IDs cannot
be applied directory-wide. Use separate output roots for same-named physical inputs.
A manifest does **not** select lesions for inference or change CT window, box, or prompt.
Keep private cohort manifests outside this public repository.

When a policy is supplied, the prediction NPZ provenance and per-case metrics JSON
record `postprocess_exempt_ids`, per-ID `postprocess_actions` (including
`exempt_by_id` and removed voxels), the policy source and, for manifests, its SHA256.
The native `results.csv` keeps the same columns and includes exempt lesions.

For runtime/memory column definitions, merged CSV layout, and offline `pytest` verification, see [`docs/metrics_and_evaluation.md`](docs/metrics_and_evaluation.md).

---

## Supported Model Configurations

| Model Name | Script | Prompt Derived from RECIST | Postprocessing |
|---|---|---|---|
| `EffMedSAM2` | [`infer_effmedsam2.py`](infer_effmedsam2.py) | Square box from SVD diameter (Z-slab `1.6x`) | `moving_centroid` (default) |
| `MedSAM2` | [`infer_medsam2.py`](infer_medsam2.py) | Square box from SVD diameter (full Z-volume) | None |
| `nnInteractive_box` | [`infer_nninteractive.py`](infer_nninteractive.py) | 2D box on RECIST slice (`--prompt-mode box`) | None |
| `nnInteractive_point` | [`infer_nninteractive.py`](infer_nninteractive.py) | 3 points center-out 25% (`--prompt-mode 3pt_25`) | None |
| `nnInteractive_scribble` | [`infer_nninteractive.py`](infer_nninteractive.py) | Skeletonized line inset 3 px (`--prompt-mode line`) | None |
| `EffMedSAM2_small` | [`infer_effmedsam2.py`](infer_effmedsam2.py) (`--model-size small`) | Square box from SVD diameter (Z-slab `1.6x`) | `moving_centroid` (configurable) |

---

## License

This codebase is released under the **Apache-2.0** license (see [`LICENSE`](LICENSE)). Third-party pre-trained checkpoints downloaded by `scripts/download_weights.sh` retain their upstream licenses — in particular, **`nnInteractive_v1.0` is licensed under `CC BY-NC-SA 4.0`** (non-commercial), and `MedSAM2` follows its upstream release terms.

## Citation

> Manuscript / arXiv citation coming soon.

## Acknowledgments

Built upon [MedSAM2](https://github.com/bowang-lab/MedSAM2), [EfficientTAM](https://github.com/yformer/EfficientTAM), [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), and [FLARE-Efficiency](https://github.com/medfm-flare/FLARE-Efficiency).
