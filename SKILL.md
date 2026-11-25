---
name: reproduce-recist-benchmark
description: >-
  End-to-end reproduction runbook and strict execution policy for
  RECIST-EffMedSAM2. Use when an agent is asked to set up uv environments, download
  and verify weights, validate HU and normalized/windowed CT NPZs, run standalone
  per-model Python inference scripts, merge model CSVs, or audit efficiency outputs.
---

# Reproducing RECIST-EffMedSAM2 Experiments

Follow **Part A** sequentially to reproduce experiments from a clean clone, while enforcing every invariant in **Part B**.

---

## Part A — End-to-End Reproduction Runbook

### Step 0: Confirm Inputs Once
Ask the user once **only** for details they have not already specified:
1. **Input NPZ directory:** (default `data/inputs/`; optional pre-windowed SAM twins in `data/window_twins/` or `--window-dir`). Ask whether subdirectories should be scanned (`--recursive`).
2. **Execution device:** `cpu` (default) or `cuda` with explicit `--gpu-index N`.
3. **Models to run:** any subset or all of `infer_effmedsam2.py`, `infer_medsam2.py`, and `infer_nninteractive.py` (`--prompt-mode box | 3pt_25 | line`).
4. **Postprocessing:** for `infer_effmedsam2.py`, `--postprocess moving_centroid` (default) or `none`.
5. **Rollout gate:** confirm stopping after a **1-case smoke test (`--limit 1`)** for review before running the full cohort (default), unless the user explicitly asks for an unattended full run.

### Step 1: Preflight Check
Inspect the host environment before creating venvs:
```bash
uv --version || python3 --version   # prefer uv (auto-provisions Python 3.11); python3 >= 3.10 fallback on offline/HPC hosts
free -h                             # check host RAM before launching multi-GB 3D volumes
df -h .                             # check disk space for venvs, checkpoints, and outputs
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "No NVIDIA GPU tooling detected"
```
If `--device cuda` was requested, confirm the physical GPU index `--gpu-index N` exists in `nvidia-smi -L`.

### Step 2: Create Virtual Environments (`uv venv` + `uv pip`)
Skip `.venv-sam` or `.venv-nn` only if the user excluded that model family. No `pip install -e .` is required for this repository (`utils/` is imported directly from the repo root). *(If `uv` is unavailable on an offline/HPC host, fall back to `python3 -m venv` + `pip`.)*

1. **SAM / EffMedSAM2 environment (`.venv-sam`):**
   *(Install the pinned `MedSAM2` checkout editable `-e` so upstream EfficientTAM Hydra config files are preserved; `requirements/sam.txt` includes `setuptools` and `wheel` for `--no-build-isolation`.)*
   - **CPU-only:**
     ```bash
     uv venv --python 3.11 .venv-sam
     uv pip install --python .venv-sam/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
     uv pip install --python .venv-sam/bin/python -r requirements/sam.txt
     git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
     git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
     SAM2_BUILD_CUDA=0 uv pip install --python .venv-sam/bin/python --no-build-isolation -e .venv-sam/src/MedSAM2
     ```
   - **CUDA GPU:**
     ```bash
     uv venv --python 3.11 .venv-sam
     uv pip install --python .venv-sam/bin/python torch torchvision
     uv pip install --python .venv-sam/bin/python -r requirements/sam.txt
     git clone https://github.com/Bardli/MedSAM2.git .venv-sam/src/MedSAM2
     git -C .venv-sam/src/MedSAM2 checkout --detach 333419e40353e55f4e4b191cb6cd3a238bd0b180
     uv pip install --python .venv-sam/bin/python --no-build-isolation -e .venv-sam/src/MedSAM2
     ```

2. **nnInteractive environment (`.venv-nn`):**
   ```bash
   uv venv --python 3.11 .venv-nn
   # For CPU-only add: --index-url https://download.pytorch.org/whl/cpu
   uv pip install --python .venv-nn/bin/python torch torchvision
   uv pip install --python .venv-nn/bin/python -r requirements/nninteractive.txt
   ```

### Step 3: Offline Self-Test
Verify resource profiling, CT input loading, prompt geometry, final-only postprocessing, standalone CLIs, single-CSV Option A reporting, and metric-grouped merging before running real checkpoints:
```bash
uv pip install --python .venv-sam/bin/python pytest
.venv-sam/bin/pytest -v
```
All offline tests must pass before proceeding (`tests/test_cuda_cold_start.py` skips automatically when CUDA PyTorch/hardware is unavailable).

### Step 4: Download & Verify Checkpoints
Download and SHA-256-verify all three default checkpoints (`RECIST-EffTiny-m8`, `MedSAM2`, and `nnInteractive_v1.0`) into `./assets` against `configs/weights.sha256`:
```bash
bash scripts/download_weights.sh
```
Confirm `assets/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` (and the `MedSAM2` / `nnInteractive_v1.0` files) passed verification. Only the optional `--model-size small` variant (`assets/RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt`) requires a locally supplied checkpoint path via `--checkpoint` if explicitly requested.

### Step 5: Validate Input NPZs (HU & Normalized/Windowed Channels)
Inspect at least one input `.npz` (without leaking patient IDs or CT arrays) to confirm keys, shape, dtype, and `utils/ct_inputs.py` compatibility:
- Expected arrays: `imgs` `(Z, Y, X)`, `recist` integer `(Z, Y, X)`, `spacing` `(3,)` `[sx, sy, sz]` in mm, and optional `gts` (or `gts_hidden`).
- Both dataset channels are supported in `utils/ct_inputs.py`:
  - **HU channel:** `intensity_encoding="nifti_stored_scaled_v1"` with `hu_slope`/`hu_intercept` (decoded to `float64` HU by `raw_hu()`/`nninteractive_image()`, and continuously windowed to `float32 [0.0, 1.0]` L40/W400 by `abdomen_image()` without `uint8` truncation), or unencoded raw `imgs` passed through by `nninteractive_image(..., require_raw=False)`.
  - **Normalized / pre-windowed channel (`float32` or legacy `uint8`):** `float32` in `[0.0, 1.0]` or `[0.0, 255.0]` (kept at full `float32` precision through bilinear `cv2.resize`), or legacy `uint8` in `[0, 255]`, accepted by `abdomen_image(..., require_raw=False)` or supplied alongside HU inputs via `data/window_twins/` / `--window-dir`.
- Pass `--require-raw-hu` only when strictly enforcing `nifti_stored_scaled_v1` HU decoding and rejecting pre-windowed inputs.

### Step 6: Run 1-Case Smoke Test & Audit Telemetry
Run one case per selected method with strict `--num-threads 1` and `--plots`:
```bash
.venv-sam/bin/python infer_effmedsam2.py -i "<INPUT_DIR>" -o runs/smoke_eff --device cpu --num-threads 1 --limit 1 --plots
```
Audit `runs/smoke_eff/results.csv`:
- `case_status == ok`.
- `total_inference_time_s` equals `model_loading_time_s + model_inference_time_s + postprocessing_time_s` (and on multi-lesion CTs, `model_loading_time_s`, `data_loading_time_s`, `image_preprocessing_time_s`, `output_saving_time_s`, and `end_to_end_wall_time_including_io_s` are recorded on the 1st lesion row and `0.0` on subsequent lesion rows of the same CT).
- `end_to_end_wall_time_including_io_s >= total_inference_time_s` on the 1st lesion row.
- `peak_ram_mb > 0`.
- No intermediate `post_*` columns or `predictions_postprocessed/` directories are produced; `predicted_voxels`, `predicted_volume_ml`, and `dice_score` reflect the final saved output.
- **On `--device cpu`:**
  - If `gpu_monitor_status == measured`: confirm `gpu_sanity_ok == True` and `peak_gpu_memory_mb == 0.0`.
  - If running on a CPU-only machine without NVIDIA tooling (`gpu_monitor_status == unavailable`), confirm `peak_gpu_memory_mb` and `gpu_sanity_ok` are blank (not `0.0`).
  - Never accept `gpu_monitor_status == query_failed` / `partial` as zero VRAM.
- **On `--device cuda --gpu-index N`:** confirm `gpu_monitor_status == measured` and nonzero GPU usage on physical card `N`.

Unless the user explicitly requested an unattended full run in Step 0, share the 1-case `results.csv` summary and wait for confirmation before launching the full dataset.

### Step 7: Full Run, CSV Merge & Final Report
Run each requested script without `--limit`, then merge the per-model `results.csv` files into one metric-grouped `merged_results.csv` with `merge_csvs.py`:
```bash
.venv-sam/bin/python infer_effmedsam2.py -i "<INPUT_DIR>" -o runs/EffMedSAM2 --device "<cpu|cuda>" --num-threads 1 --plots
.venv-sam/bin/python infer_medsam2.py -i "<INPUT_DIR>" -o runs/MedSAM2 --device "<cpu|cuda>" --num-threads 1 --plots
.venv-nn/bin/python infer_nninteractive.py -i "<INPUT_DIR>" -o runs/nnInteractive_box --prompt-mode box --device "<cpu|cuda>" --num-threads 1 --plots

.venv-sam/bin/python merge_csvs.py -i runs/EffMedSAM2 runs/MedSAM2 runs/nnInteractive_box -o runs/merged/merged_results.csv
```
Summarize observed results from:
- `runs/merged/merged_results.csv`
- Per-model `<output-dir>/results.csv` files (explicitly flagging any `failed(...)` rows if present).

---

## Part B — Strict Execution, HU & Repo Hygiene Policy

### 1. Execution & Model-Loading Invariants
1. **Fresh model reload per case:** Model construction and checkpoint loading execute inside every case loop iteration, are counted in `total_inference_time_s` (`model_loading_time_s + model_inference_time_s + postprocessing_time_s`), and explicitly clean up (`del`, `gc.collect()`, `empty_cache()`) after each case.
2. **Postprocessing inside profiler & final-only output:** `moving_centroid` postprocessing executes inside `CaseProfiler.phase("postprocessing_time")`, and only the final prediction NPZ and final Dice/volume columns are written.
3. **RECIST prompt alignment:** Slice discovery (`find_recist_slices`) and 2D prompt extraction run inside `image_preprocessing_time_s` so `model_inference_time_s` and per-lesion timing cover pure inference symmetrically.
3. **Serial single-threaded default (`--num-threads 1`):** Sets `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`, and `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS_NUM_THREADS=1` before importing PyTorch to keep host RAM bounded and timings stable.
4. **Strict CPU isolation & physical GPU mapping:**
   - `--device cpu`: forces `CUDA_VISIBLE_DEVICES=""` and `NVIDIA_VISIBLE_DEVICES="void"` before importing PyTorch and asserts `torch.cuda.is_available() == False`, while querying **all** physical GPUs via NVML.
   - `--device cuda --gpu-index N`: sets `CUDA_VISIBLE_DEVICES=N` before importing PyTorch (calling `reset_cuda_peaks(torch_device)` on GPU cases) and passes physical index `N` to `CaseProfiler` so NVML monitors physical card `N`.
5. **Failure telemetry & `gpu_monitor_status` semantics:** Stale `<case>.metrics.json` is cleared before each case; failed cases record `status="failed(exception)"` and observed RAM/GPU metrics. Never conflate unmeasured/failed NVML queries (`None` -> blank cell; `gpu_monitor_status` in `{"unavailable", "query_failed", "partial"}`) with a verified `0.0 MiB` measurement (`gpu_monitor_status == "measured"`).

### 2. Input Channels: HU & Normalized/Windowed
- **HU channel:** `nifti_stored_scaled_v1` reconstructs `float64` HU for `nnInteractive` and windows continuously to `float32 [0.0, 1.0]` (`L40/W400`) for `EffMedSAM2`/`MedSAM2`; direct unencoded `imgs` passes through to `nninteractive_image(..., require_raw=False)`.
- **Normalized / pre-windowed channel (`float32` or legacy `uint8`):** `float32` in `[0.0, 1.0]` or `[0.0, 255.0]` (resized in `float32` via bilinear `cv2.resize`) or legacy `uint8 [0, 255]`.

### 3. Public Repo Hygiene
- Never commit patient CT scans, real case identifiers, private manifests, model weights (`.pt`/`.pth`), `.npz`/`.nii.gz` files, outputs (`runs/`), secrets, or internal filesystem/cluster paths.
- Before pushing changes, run `pytest -v`, `python -m compileall -q`, and a repository grep check for private paths or credentials.
