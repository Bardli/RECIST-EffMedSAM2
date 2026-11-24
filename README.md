# RECIST-EffMedSAM2

Clean, standalone inference and efficiency-benchmarking suite for 3D CT lesion segmentation from 2D RECIST annotations.

- **Unified CPU & GPU execution** via native Python (`--device cpu` or `--device cuda --gpu-index N`), without private Docker dependencies.
- **Per-case model reload & serial single-threaded default** (`--num-threads 1`) matching single-scan clinical usage and minimizing peak RAM.
- **Embedded FLARE-Efficiency resource profiling** (aligned with [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency)) recording phase-level runtime breakdown, process-tree RAM (`psutil` + `ru_maxrss`), and PID-scoped GPU memory (`nvidia-ml-py` + PyTorch CUDA stats) with automatic `GPU ≈ 0` sanity checking in CPU mode.
- **Independent-instance evaluation** (`independent_instances_v2`) scoring each prompted lesion channel without cross-lesion overlap overwriting or best-overlap rematching.

---

## Supported Model Configurations

| Method | Prompt Derived from RECIST | Model ID (`configs/models.json`) | Default Scope |
|---|---|---|---|
| **RECIST-EffMedSAM2 (Tiny m8, ep40)** | Square box from SVD diameter | `efftiny_m8_ep40_box` | Paper (5 configs) |
| **MedSAM2** | Square box from SVD diameter | `medsam2_box` | Paper (5 configs) |
| **nnInteractive-box** | 2D box on RECIST slice | `nninteractive_box_hu` | Paper (5 configs) |
| **nnInteractive-point** | 3 points (25% / 50% / 75% along line) | `nninteractive_3pt_25_hu` | Paper (5 configs) |
| **nnInteractive-scribble** | Skeletonized line, endpoints inset 3 px | `nninteractive_line_skel_in3_hu` | Paper (5 configs) |
| EfficientMedSAM2-Small (Small m8, ep40) | Square box from SVD diameter | `effsmall_m8_ep40_box` | Supplementary (`--model-scope all-six`) |

---

## Execution & Model-Loading Policy

Each `(model, case)` pair runs sequentially in an isolated Python subprocess (`recist_effmedsam2.runner.run_case`):

1. **Fresh model reload per case:** Model construction and checkpoint loading execute inside every case run and are counted in `runtime_s` (`runtime_s = model_load_s + inference_s`). This reflects single-scan clinical deployment and prevents cross-case memory/allocator state from distorting peak RAM or VRAM measurements.
2. **Serial single-threaded default (`--num-threads 1`):** Sets `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`, and `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS_NUM_THREADS=1` before importing PyTorch to stabilize CPU measurements and keep host RAM bounded. Override via `--num-threads N`.
3. **Strict CPU isolation:** When `--device cpu` is selected, the runner forces `CUDA_VISIBLE_DEVICES=""` and `NVIDIA_VISIBLE_DEVICES="void"` prior to Python startup and verifies `torch.cuda.is_available() == False` inside the worker.

---

## Installation

SAM/EfficientTAM and nnInteractive can share one environment (if dependencies are compatible) or live in two separate virtual environments passed via `--sam-python` and `--nn-python`.

### 1. Core package (runner, profiler, prompt utilities, evaluator)

```bash
git clone https://github.com/Bardli/RECIST-EffMedSAM2.git
cd RECIST-EffMedSAM2

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/core.txt
pip install -e .
```

### 2. SAM / EfficientMedSAM2 environment (CPU or GPU)

**CPU-only:**
```bash
python3 -m venv .venv-sam
source .venv-sam/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/sam.txt
SAM2_BUILD_CUDA=0 pip install --no-build-isolation \
  "git+https://github.com/Bardli/MedSAM2.git@333419e40353e55f4e4b191cb6cd3a238bd0b180"
pip install -e . --no-deps
```

**CUDA GPU:**
```bash
python3 -m venv .venv-sam
source .venv-sam/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements/sam.txt
pip install --no-build-isolation \
  "git+https://github.com/Bardli/MedSAM2.git@333419e40353e55f4e4b191cb6cd3a238bd0b180"
pip install -e . --no-deps
```

### 3. nnInteractive environment (CPU or GPU)

```bash
python3 -m venv .venv-nn
source .venv-nn/bin/activate
pip install --upgrade pip
# Use --index-url https://download.pytorch.org/whl/cpu for CPU-only, or default CUDA wheel for GPU:
pip install torch torchvision
pip install -r requirements/nninteractive.txt
pip install -e . --no-deps
```

---

## Weights Download & Checksum Verification

Download the public Hugging Face checkpoints (`MedSAM2` FLARE25 RECIST baseline and `nnInteractive_v1.0`) and verify SHA-256 checksums against `configs/weights.sha256`:

```bash
export ASSETS=/path/to/model_assets
bash scripts/download_weights.sh "$ASSETS"
```

Expected layout under `$ASSETS`:

| Model | Relative Path under `$ASSETS/` | SHA-256 |
|---|---|---|
| `medsam2_box` | `MedSAM2-RECIST/checkpoints/medsam2_FLARE25_RECIST_baseline.pt` | `890e6e10924984c7b704a1984362c53a56dc0ee27c796a34eb49c83a1dc4a144` |
| `nninteractive_*` | `nninteractive_hf_models/models/nnInteractive_v1.0/` (`fold_0/checkpoint_final.pth`, `plans.json`, `dataset.json`, `inference_session_class.json`) | Verified in `configs/weights.sha256` |
| `efftiny_m8_ep40_box` | `RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` | `0b377887f5587cbfd42dbc739a1711f30db276f3fd4c849233df2142ee4f5f5d` |
| `effsmall_m8_ep40_box` (optional) | `RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt` | `423793cec97e9ec93b70c25fbff1c8267d1a1f58deed254b79e16caf7b489fb3` |

Place the fine-tuned `RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` checkpoint into `$ASSETS` and pass `--verify-sha256` to enforce strict pre-run verification.

---

## Input NPZ Format

Each input `.npz` file represents one 3D CT case with arrays in `(Z, Y, X)` voxel order and `(X, Y, Z)` physical spacing:

- `imgs`:
  - **Raw HU (`nifti_stored_scaled_v1`):** stored CT array accompanied by `intensity_encoding="nifti_stored_scaled_v1"`, `hu_slope`, and `hu_intercept`. `ct_inputs.py` reconstructs float64 HU for nnInteractive and windows to L40/W400 (`[-160, 240]` -> `uint8 [0, 255]`) at the SAM model boundary.
  - **Pre-windowed SAM input:** `uint8` 3D array in `[0, 255]`. If SAM models require a separate pre-windowed directory while nnInteractive reads raw HU, pass `--window-dir /path/to/windowed_npz`.
- `recist`: integer `(Z, Y, X)` label array with non-zero lesion IDs drawn on each lesion's axial RECIST slice.
- `spacing`: float `(3,)` physical spacing `[sx, sy, sz]` in mm.
- `gts` (or `gts_hidden`, optional): integer `(Z, Y, X)` ground-truth mask for Dice/volume evaluation (never used as a prompt source for RECIST modes).

---

## Running Inference

### 1. Single-case CPU smoke test (`--limit 1`)

```bash
recist-benchmark \
  --input-dir /path/to/dataset/cancer_type \
  --output-dir runs/cpu_smoke \
  --assets-dir "$ASSETS" \
  --sam-python .venv-sam/bin/python \
  --nn-python .venv-nn/bin/python \
  --device cpu \
  --num-threads 1 \
  --limit 1 \
  --verify-sha256 \
  --plots
```

### 2. Full single cancer-type CPU run (with optional EAY window twins)

```bash
recist-benchmark \
  --input-dir /path/to/hu_inputs/colon_cancer \
  --window-dir /path/to/window_twins/colon_cancer \
  --output-dir runs/eay_colon_cpu \
  --assets-dir "$ASSETS" \
  --sam-python .venv-sam/bin/python \
  --nn-python .venv-nn/bin/python \
  --device cpu \
  --num-threads 1 \
  --verify-sha256 \
  --plots
```

### 3. Running a single model or switching to GPU mode

```bash
# Single model on CPU
recist-benchmark \
  --input-dir /path/to/data \
  --output-dir runs/efftiny_cpu \
  --assets-dir "$ASSETS" \
  --models efftiny_m8_ep40_box \
  --sam-python .venv-sam/bin/python \
  --device cpu \
  --num-threads 1

# GPU execution (same codebase, --device cuda --gpu-index 0)
recist-benchmark \
  --input-dir /path/to/data \
  --output-dir runs/paper5_gpu \
  --assets-dir "$ASSETS" \
  --sam-python .venv-sam/bin/python \
  --nn-python .venv-nn/bin/python \
  --device cuda \
  --gpu-index 0 \
  --num-threads 1
```

### 4. Standalone backend CLIs

Both backends also expose standalone CLIs that automatically reload the model per case and emit `.metrics.json` + `.npz`:

```bash
# SAM / EfficientMedSAM2
recist-sam-infer \
  -i /path/to/case.npz \
  -o runs/sam_single \
  --backend eff \
  --model-size tiny \
  --weights "$ASSETS/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt" \
  --device cpu \
  --num-threads 1 \
  --prompt box \
  --disable-image-encoder-compile

# nnInteractive
recist-nninteractive-infer \
  -i /path/to/case.npz \
  -o runs/nn_single \
  --checkpoint "$ASSETS/nninteractive_hf_models/models/nnInteractive_v1.0" \
  --device cpu \
  --num-threads 1 \
  --prompt-mode 3pt_25 \
  --interaction batch
```

Pass `--postprocess moving_centroid` to `recist-benchmark` (or `run_case`) to additionally compute slice-centered RECIST physical-radius clipping for `eff*` models into `predictions_postprocessed/` and `post_*` CSV columns without altering raw predictions.

---

## Performance & Resource Metrics

Every case execution runs an embedded `CaseProfiler` (`src/recist_effmedsam2/profiling/monitor.py`) that mirrors [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency) sampling (`0.1 s` default interval) and AUC integration, scoped strictly to the active worker process tree so shared workstations do not contaminate CPU or memory readings.

### Runtime Breakdown (seconds via `time.perf_counter()`, CUDA-synchronized on GPU)

| Metric | Included in `runtime_s`? | Definition |
|---|---|---|
| `model_load_s` | **Yes** | Network instantiation + checkpoint loading onto target device |
| `inference_s` | **Yes** | Pure model inference across prompted lesions (state init/reset, prompt application, forward/reverse propagation or AutoZoom); **excludes** NPZ loading/decompression, preprocessing, and output saving |
| `runtime_s` | **Total** | `model_load_s + inference_s` |
| `data_load_s` | No (audited separately) | NPZ disk load & decompression |
| `preprocess_s` | No (audited separately) | HU decoding / L40-W400 windowing / RGB resize & normalization, or nnInteractive `set_image` preprocessing |
| `postprocess_s` | No (audited separately) | Optional moving-centroid clipping (`0.0` when disabled) |
| `save_s` | No (audited separately) | Streamed independent-instance compressed NPZ write & CRC check |
| `wall_total_s` | No (audited separately) | Complete case worker wall-clock time |

Per-lesion inference durations are also preserved under `per_lesion_inference_s` in each `<case>.metrics.json`.

### RAM, CPU & GPU Memory Metrics

| Metric | Unit | Definition |
|---|---|---|
| `ram_baseline_mb` | MiB | Process-tree RSS right before model loading |
| `MaxRAM_MB` | MiB | `max(sampled_peak_rss_mb, resource.getrusage.ru_maxrss_mb, ram_baseline_mb)` |
| `ram_inference_peak_mb` | MiB | Peak process-tree RSS observed during the `inference` phase |
| `ram_delta_mb` | MiB | `MaxRAM_MB - ram_baseline_mb` |
| `AUC_RAM_Time_MB_s` | MiB·s | Trapezoidal area under process-tree RSS vs. time |
| `MaxCPU_Utilization_pct` | % | Peak normalized process-tree CPU utilization (0–100%) |
| `AUC_CPU_Time_pct_s` | %·s | Trapezoidal area under CPU utilization vs. time |
| `MaxGPU_Mem_MB` | MiB | `max(PID-scoped NVML peak, PyTorch peak reserved)` |
| `gpu_nvml_peak_mb` | MiB | Peak PID-scoped GPU memory queried via `pynvml` (`nvidia-ml-py`, fallback `nvidia-smi`) |
| `AUC_GPU_Time_MB_s` | MiB·s | Trapezoidal area under PID-scoped GPU memory vs. time |
| `RankGPU_Mem_MB` | MiB | `round(max(0.0, MaxGPU_Mem_MB - gpu_tolerance_mb))` (default tolerance: `4096.0 MiB`) |
| `gpu_torch_peak_allocated_mb` | MiB | `torch.cuda.max_memory_allocated()` (`0.0` on CPU) |
| `gpu_torch_peak_reserved_mb` | MiB | `torch.cuda.max_memory_reserved()` (`0.0` on CPU) |
| `gpu_sanity_ok` | bool | In CPU mode: `True` iff `MaxGPU_Mem_MB == 0.0`, `gpu_torch_peak_allocated_mb == 0.0`, `gpu_torch_peak_reserved_mb == 0.0`, and `not torch.cuda.is_available()` |

---

## Output Layout & Evaluation CSVs

After inference completes, `compute_metrics.py` automatically writes three CSV reports under `<output-dir>/reports/`:

```text
<output-dir>/
├── run_manifest.json
├── <model_name>/
│   ├── predictions/<case_id>.npz          # independent_instances_v2: segs[N,Z,Y,X], instance_ids, prediction_status, spacing
│   ├── metrics/<case_id>.metrics.json     # full breakdown, resource samples, host info, per-lesion counts
│   ├── logs/<case_id>.log
│   └── plots/                             # optional CPU/RAM/GPU-vs-time PNGs (--plots)
└── reports/
    ├── case_metrics.csv                   # 1 row per (model, case): timing breakdown, RAM, GPU, sanity check, mean Dice
    ├── lesion_scores.csv                  # 1 row per (model, case, lesion): independent same-ID Dice, volumes (mL), timings
    └── model_summary.csv                  # 1 row per (model, device): aggregated Dice & efficiency averages
```

To re-evaluate or recompute rankings with a different GPU tolerance without re-running inference:

```bash
recist-eval --run-dir runs/eay_colon_cpu --verify-from-npz --gpu-tolerance-mb 4096
```

---

## Running Tests

Offline unit and end-to-end synthetic tests validate phase breakdown math, `runtime_s = model_load_s + inference_s`, process-tree RAM tracking, CPU `gpu_sanity_ok == True` (even on shared GPU machines), independent-instance storage, prompt geometry, moving-centroid postprocessing, and CSV generation without requiring real CT scans or checkpoints:

```bash
pytest -v
```
