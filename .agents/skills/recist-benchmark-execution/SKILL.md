---
name: recist-benchmark-execution
description: >-
  Execution, model-loading, efficiency profiling, and direct raw-HU input policy
  for RECIST-EffMedSAM2. Use when running recist-benchmark or standalone backends,
  modifying runner/profiler/backend code, or auditing efficiency CSV outputs.
---

# RECIST-EffMedSAM2 Execution & Benchmark Policy

Follow these rules whenever executing benchmarks, modifying the runner/profiler/backends, or reviewing performance reports in this repository.

## 1. Execution & Model-Loading Policy

Each `(model, case)` pair runs sequentially in an isolated Python subprocess (`recist_effmedsam2.runner.run_case`):

1. **Fresh model reload per case:** Model construction and checkpoint loading execute inside every case run and are counted in `runtime_s` (`runtime_s = model_load_s + inference_s + postprocess_s`). This reflects single-scan clinical deployment and prevents cross-case memory or allocator state from distorting peak RAM or VRAM measurements.
2. **Postprocessing inside profiler scope:** When `--postprocess moving_centroid` is enabled, clipping and postprocessed NPZ saving + scoring execute inside the active `CaseProfiler` window so `postprocess_s`, `runtime_s`, `wall_total_s`, and `MaxRAM_MB` capture the complete workflow.
3. **Serial single-threaded default (`--num-threads 1`):** Sets `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`, and `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS_NUM_THREADS=1` before importing PyTorch to stabilize CPU measurements and keep host RAM bounded. Override only intentionally via `--num-threads N`.
4. **Strict CPU isolation & physical GPU mapping:**
   - `--device cpu`: forces `CUDA_VISIBLE_DEVICES=""` and `NVIDIA_VISIBLE_DEVICES="void"` prior to Python startup and asserts `torch.cuda.is_available() == False`, while querying **all** physical GPUs via NVML to verify zero hardware VRAM usage.
   - `--device cuda --gpu-index N`: sets `CUDA_VISIBLE_DEVICES=N` (logical `cuda:0` inside the worker) and passes physical index `N` explicitly to NVML so VRAM sampling monitors physical card `N` rather than defaulting to GPU `0`.
5. **Failure & OOM telemetry:** Both the worker (`try ... except BaseException`) and an external parent `ProcessTreeMonitor` capture RAM/CPU/GPU samples and write atomic JSON receipts. Before spawning a worker, stale `<case>.metrics.json` is cleared so a crashed or `SIGKILL`/OOM-killed worker can never leave behind an earlier `ok` receipt. `case_metrics.csv` preserves `status=failed(...)`, `wall_total_s`, and observed peak RAM/GPU usage.
6. **GPU monitoring status semantics:** Never conflate unmeasured/failed NVML queries (`None` -> blank CSV cell, `gpu_monitor_status` = `unavailable` / `query_failed` / `partial`) with a verified `0.0 MiB` measurement (`gpu_monitor_status == "measured"`). On CPU runs, `gpu_sanity_ok` is `True` only when `gpu_monitor_status == "measured"`, VRAM is `0.0`, and CUDA is unavailable.

## 2. Direct Raw-HU Input Convention

`src/recist_effmedsam2/ct_inputs.py` reads raw Hounsfield Units **directly** from original CT NPZs without slope/intercept reconstruction:

- **Direct raw HU (primary path):** NPZ `imgs` is a 3D signed-integer (`int16`/`int32`) or floating-point (`float32`/`float64`) array storing raw CT HU directly, with no `intensity_encoding`, `hu_slope`, or `hu_intercept` keys.
  - `nninteractive_image()` passes the raw HU volume through unmodified.
  - `abdomen_image()` windows raw HU directly to L40/W400 (`[-160, 240]` -> `uint8 [0, 255]`) at input load time unless a separate pre-windowed `uint8` twin is provided via `--window-dir` or `data/window_twins/`.
- **Legacy compatibility:** NPZs explicitly carrying `intensity_encoding="nifti_stored_scaled_v1"` with valid `hu_slope`/`hu_intercept` are still decoded for backward compatibility.
- **Strict mode (`--require-raw-hu`):** Rejects pre-windowed `uint8` inputs (`raw_hu()`, `abdomen_image(..., require_raw=True)`, `nninteractive_image(..., require_raw=True)`), while accepting direct raw HU (`int16`/`float32`) inputs natively.

## 3. Public Repo Hygiene & Pre-Push Verification

- Never commit patient CT scans, real case identifiers, private manifests, model weights (`.pt`/`.pth`), `.npz`/`.nii.gz` files, secrets, or internal server/cluster paths.
- Before committing or pushing changes:
  1. Run offline verification: `pytest -v` and `python -m compileall -q src tests`.
  2. Run a privacy scan over tracked/working files (including `.agents/skills/`) to ensure zero internal paths or secrets are present.
