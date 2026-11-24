---
name: reproduce-recist-benchmark
description: >-
  End-to-end one-click reproduction runbook and strict execution policy for
  RECIST-EffMedSAM2. Use when an agent is asked to set up environments, download
  and verify weights, validate raw-HU CT NPZs, run a 1-case smoke test or full
  CPU/GPU efficiency benchmark, audit reports CSVs, or modify runner/profiler/backend code.
---

# Reproducing RECIST-EffMedSAM2 Experiments

Follow **Part A** sequentially to reproduce the benchmark from a clean clone, while enforcing every invariant in **Part B**.

---

## Part A — End-to-End One-Click Reproduction Runbook

### Step 0: Confirm Inputs Once
Ask the user once **only** for details they have not already specified:
1. **Input NPZ directory:** (default `data/inputs/`; optional pre-windowed SAM twins in `data/window_twins/` or `--window-dir`). Ask whether subdirectories should be scanned (`--recursive`).
2. **Execution device:** `cpu` (default) or `cuda` with explicit `--gpu-index N`.
3. **Model scope:** `paper5` (default: `efftiny_m8_ep40_box`, `medsam2_box`, `nninteractive_box_hu`, `nninteractive_3pt_25_hu`, `nninteractive_line_skel_in3_hu`), `all-six`, or a specific `--models` subset.
4. **Postprocessing:** `--postprocess none` (default) or `--postprocess moving_centroid`.
5. **Rollout gate:** confirm stopping after a **1-case smoke test (`--limit 1`)** for review before running the full cohort (default), unless the user explicitly asks for an unattended full run.

### Step 1: Preflight Check
Inspect the host environment before creating venvs:
```bash
python3 --version          # requires Python >= 3.10
free -h                    # check host RAM before launching multi-GB 3D volumes
df -h .                    # check disk space for venvs, checkpoints, and outputs
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "No NVIDIA GPU tooling detected"
```
If `--device cuda` was requested, confirm the physical GPU index `--gpu-index N` exists in `nvidia-smi -L`.

### Step 2: Create Virtual Environments
Skip `.venv-sam` or `.venv-nn` only if the user excluded that model family.

1. **Core package (`.venv` — runner, profiler, prompts, evaluator, pytest):**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -r requirements/core.txt pytest
   .venv/bin/pip install -e .
   ```

2. **SAM / EfficientMedSAM2 environment (`.venv-sam`):**
   - **CPU-only:**
     ```bash
     python3 -m venv .venv-sam
     .venv-sam/bin/pip install --upgrade pip
     .venv-sam/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     .venv-sam/bin/pip install -r requirements/sam.txt
     SAM2_BUILD_CUDA=0 .venv-sam/bin/pip install --no-build-isolation \
       "git+https://github.com/Bardli/MedSAM2.git@333419e40353e55f4e4b191cb6cd3a238bd0b180"
     .venv-sam/bin/pip install -e . --no-deps
     ```
   - **CUDA GPU:**
     ```bash
     python3 -m venv .venv-sam
     .venv-sam/bin/pip install --upgrade pip
     .venv-sam/bin/pip install torch torchvision
     .venv-sam/bin/pip install -r requirements/sam.txt
     .venv-sam/bin/pip install --no-build-isolation \
       "git+https://github.com/Bardli/MedSAM2.git@333419e40353e55f4e4b191cb6cd3a238bd0b180"
     .venv-sam/bin/pip install -e . --no-deps
     ```

3. **nnInteractive environment (`.venv-nn`):**
   ```bash
   python3 -m venv .venv-nn
   .venv-nn/bin/pip install --upgrade pip
   # For CPU-only add: --index-url https://download.pytorch.org/whl/cpu
   .venv-nn/bin/pip install torch torchvision
   .venv-nn/bin/pip install -r requirements/nninteractive.txt
   .venv-nn/bin/pip install -e . --no-deps
   ```

### Step 3: Offline Self-Test
Verify the runner, FLARE-style profiler, direct raw-HU loader, independent-instance storage, and evaluator before running real checkpoints:
```bash
.venv/bin/pytest -v
```
All tests must pass before proceeding.

### Step 4: Download & Verify Checkpoints
Download the public `MedSAM2` FLARE25 RECIST baseline and `nnInteractive_v1.0` checkpoints into `./assets` and check SHA-256 checksums against `configs/weights.sha256`:
```bash
bash scripts/download_weights.sh
```
Check whether `assets/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` (and `assets/RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt` if `all-six` was requested) is present.
- If a checkpoint is missing, **stop and inform the user**: either obtain its path / copy it into `assets/`, or omit that model via `--models` until supplied.
- Always pass `--verify-sha256` on benchmark runs so every asset hash in `configs/models.json` / `configs/weights.sha256` is strictly enforced.

### Step 5: Validate Input NPZs (Direct Raw HU)
Inspect at least one input `.npz` (without leaking patient IDs or CT arrays) to confirm keys, shape, dtype, and `ct_inputs.py` compatibility:
- Expected arrays: `imgs` `(Z, Y, X)`, `recist` integer `(Z, Y, X)`, `spacing` `(3,)` `[sx, sy, sz]` in mm, and optional `gts` (or `gts_hidden`).
- Confirm `imgs` stores raw CT Hounsfield Units directly (`int16`/`int32`/`float32`/`float64`, **no** `hu_slope`/`hu_intercept` reconstruction needed), or provide separate `uint8` SAM twins via `--window-dir` / `data/window_twins/`.
- Verify with `nninteractive_image(data, require_raw=True)` and `abdomen_image(data, require_raw=...)` in `.venv/bin/python` on a single header/case check. Pass `--require-raw-hu` on `recist-benchmark` whenever raw HU inputs are expected (mandatory for valid `nninteractive_*` paper runs).

### Step 6: Run 1-Case Smoke Test & Audit Telemetry
Run one case first with strict `--num-threads 1`, `--verify-sha256`, and `--plots`:
```bash
.venv/bin/recist-benchmark \
  --input-dir "<INPUT_DIR>" \
  --output-dir runs/smoke_test \
  --device cpu \
  --num-threads 1 \
  --limit 1 \
  --verify-sha256 \
  --plots
```
*(Add `--recursive`, `--window-dir`, `--dataset`, `--models`, `--postprocess moving_centroid`, `--require-raw-hu`, or `--device cuda --gpu-index N` as configured in Step 0.)*

Audit `runs/smoke_test/reports/case_metrics.csv`:
- `status == ok` and `monitor_source == worker`.
- `runtime_s` equals `model_load_s + inference_s + postprocess_s`, and `wall_total_s >= runtime_s`.
- `MaxRAM_MB > 0` and `ram_baseline_mb > 0`.
- **On `--device cpu`:**
  - If `gpu_monitor_status == measured`: confirm `gpu_sanity_ok == True`, `MaxGPU_Mem_MB == 0.0`, and `gpu_nvml_peak_mb == 0.0`.
  - If running on a CPU-only machine without NVIDIA tooling (`gpu_monitor_status == unavailable`), confirm `MaxGPU_Mem_MB` and `gpu_sanity_ok` are blank (not `0.0`).
  - Never accept `gpu_monitor_status == query_failed` / `partial` as zero VRAM.
- **On `--device cuda --gpu-index N`:** confirm `gpu_monitor_status == measured` and nonzero GPU usage on physical card `N`.
- If `gts` are present: confirm `mean_dice > 0`; if `--postprocess moving_centroid` is enabled, confirm `postprocess_s > 0` and `mean_post_dice` is populated.

Unless the user explicitly requested an unattended full run in Step 0, share the 1-case CSV summary and wait for confirmation before launching the full dataset.

### Step 7: Full Benchmark & Final Report
Run without `--limit` to an explicit cohort `--output-dir`:
```bash
.venv/bin/recist-benchmark \
  --input-dir "<INPUT_DIR>" \
  --output-dir runs/benchmark_full \
  --device "<cpu|cuda>" \
  --num-threads 1 \
  --verify-sha256 \
  --plots
```
Summarize results directly from observed output files only:
- `reports/model_summary.csv` (mean Dice, mean postprocessed Dice if enabled, `runtime_s`, breakdown phases, `MaxRAM_MB`, `MaxGPU_Mem_MB`, `RankGPU_Mem_MB`, AUC metrics).
- `reports/case_metrics.csv` and `reports/lesion_scores.csv`, explicitly flagging any `failed(...)` / OOM cases along with their captured `wall_total_s`, `MaxRAM_MB`, and `monitor_source`.

---

## Part B — Strict Execution, HU & Repo Hygiene Policy

### 1. Execution & Model-Loading Invariants
Each `(model, case)` pair runs sequentially in an isolated Python subprocess (`recist_effmedsam2.runner.run_case`):

1. **Fresh model reload per case:** Model construction and checkpoint loading execute inside every case run and are counted in `runtime_s` (`runtime_s = model_load_s + inference_s + postprocess_s`). Never replace this with a persistent model daemon unless explicitly instructed for a separate study.
2. **Postprocessing inside profiler scope:** When `--postprocess moving_centroid` is enabled, clipping and postprocessed NPZ saving + scoring execute inside the active `CaseProfiler` window so `postprocess_s`, `runtime_s`, `wall_total_s`, and `MaxRAM_MB` capture the complete workflow.
3. **Serial single-threaded default (`--num-threads 1`):** Sets `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`, and `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS_NUM_THREADS=1` before importing PyTorch to stabilize CPU measurements and keep host RAM bounded.
4. **Strict CPU isolation & physical GPU mapping:**
   - `--device cpu`: forces `CUDA_VISIBLE_DEVICES=""` and `NVIDIA_VISIBLE_DEVICES="void"` prior to Python startup and asserts `torch.cuda.is_available() == False`, while querying **all** physical GPUs via NVML to verify zero hardware VRAM usage.
   - `--device cuda --gpu-index N`: sets `CUDA_VISIBLE_DEVICES=N` (logical `cuda:0` inside the worker) and passes physical index `N` explicitly to NVML so VRAM sampling monitors physical card `N` rather than defaulting to GPU `0`.
5. **Failure & OOM telemetry:** Both the worker (`try ... except BaseException`) and an external parent `ProcessTreeMonitor` capture RAM/CPU/GPU samples and write atomic JSON receipts. Stale `<case>.metrics.json` is cleared before worker launch so a crashed or `SIGKILL`/OOM-killed worker can never leave behind an earlier `ok` receipt.
6. **GPU monitoring status semantics:** Never conflate unmeasured/failed NVML queries (`None` -> blank CSV cell; `gpu_monitor_status` in `{"unavailable", "query_failed", "partial"}`) with a verified `0.0 MiB` measurement (`gpu_monitor_status == "measured"`).

### 2. Direct Raw-HU Input Convention
`src/recist_effmedsam2/ct_inputs.py` reads raw Hounsfield Units **directly** from original CT NPZs without slope/intercept reconstruction:
- **Direct raw HU (primary path):** NPZ `imgs` is a 3D signed-integer (`int16`/`int32`) or floating-point (`float32`/`float64`) array storing raw CT HU directly, with no `intensity_encoding`, `hu_slope`, or `hu_intercept` keys.
  - `nninteractive_image()` passes raw HU through unmodified.
  - `abdomen_image()` windows raw HU directly to L40/W400 (`[-160, 240]` -> `uint8 [0, 255]`) at input load time unless a separate pre-windowed `uint8` twin is supplied via `--window-dir` / `data/window_twins/`.
- **Legacy compatibility:** NPZs explicitly carrying `intensity_encoding="nifti_stored_scaled_v1"` with valid `hu_slope`/`hu_intercept` remain supported if encountered.
- **Strict mode (`--require-raw-hu`):** Rejects pre-windowed `uint8` inputs while accepting direct raw HU (`int16`/`float32`) natively.

### 3. Public Repo Hygiene
- Never commit patient CT scans, real case identifiers, private manifests, model weights (`.pt`/`.pth`), `.npz`/`.nii.gz` files, outputs (`runs/`), secrets, or internal filesystem/cluster paths.
- Before pushing any code or skill changes, run `.venv/bin/pytest -v`, `python -m compileall -q src tests`, and a repository grep check for private paths or credentials.
