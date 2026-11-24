# RECIST-EffMedSAM2

Clean, standalone inference and efficiency-benchmarking suite for 3D CT lesion segmentation from 2D RECIST annotations.

- **Unified CPU & GPU execution** via native Python (`--device cpu` or `--device cuda --gpu-index N`), without private Docker dependencies.
- **Per-case model reload & serial single-threaded default** (`--num-threads 1`) matching single-scan clinical usage and minimizing peak RAM.
- **Embedded FLARE-Efficiency resource profiling** (aligned with [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency)) recording phase-level runtime breakdown (including postprocessing), process-tree RAM (`psutil` + `ru_maxrss`), PID-scoped GPU memory (`nvidia-ml-py` + PyTorch CUDA stats) mapped to the true physical GPU index, explicit `gpu_monitor_status` (`measured` vs `unavailable`/`query_failed`), and parent-side fallback monitoring so OOM/`SIGKILL`/crash runs still preserve resource telemetry.
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

## One-Click Reproduction with an AI Agent

This repository ships with a built-in reproduction skill ([`.agents/skills/reproduce-recist-benchmark/SKILL.md`](.agents/skills/reproduce-recist-benchmark/SKILL.md), mirrored at [`.claude/skills/reproduce-recist-benchmark/SKILL.md`](.claude/skills/reproduce-recist-benchmark/SKILL.md) and referenced by [`AGENTS.md`](AGENTS.md)) for **Antigravity, Claude Code, Codex, and Cursor**:

```bash
git clone https://github.com/Bardli/RECIST-EffMedSAM2.git
cd RECIST-EffMedSAM2
```

Open the cloned directory in your coding agent and give it a single instruction, for example:

> Use the `reproduce-recist-benchmark` skill to create `.venv` / `.venv-sam` / `.venv-nn`, run `pytest -v`, download & verify checkpoints, validate the raw-HU NPZs under `/path/to/data`, run a 1-case CPU smoke test, and audit `case_metrics.csv` before running the full cohort.

The skill enforces the benchmark's exact model-loading policy (fresh per-case reload, in-profiler postprocessing, `--num-threads 1`), direct raw-HU handling, CPU/GPU NVML sanity checks, and CSV audit gates automatically.

---

## Installation

SAM/EfficientTAM and nnInteractive can share one environment (if dependencies are compatible) or live in the default two virtual environments (`.venv-sam` and `.venv-nn`, auto-detected by `recist-benchmark`).

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

### 2. SAM / EfficientMedSAM2 environment (`.venv-sam`, CPU or GPU)

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

### 3. nnInteractive environment (`.venv-nn`, CPU or GPU)

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

## Default Directory Layout & Weights

By default, `recist-benchmark` and `scripts/download_weights.sh` use repository-relative paths (all ignored by `.gitignore`):

```text
RECIST-EffMedSAM2/
├── data/
│   ├── inputs/           # default --input-dir (*.npz)
│   └── window_twins/     # optional, auto-detected for SAM when data/inputs/ is used
├── assets/               # default --assets-dir
├── .venv-sam/            # auto-detected --sam-python
├── .venv-nn/             # auto-detected --nn-python
└── runs/
    └── benchmark/        # default --output-dir
```

Download the public Hugging Face checkpoints (`MedSAM2` FLARE25 RECIST baseline and `nnInteractive_v1.0`) into `assets/` and verify SHA-256 checksums against `configs/weights.sha256`:

```bash
bash scripts/download_weights.sh        # defaults to ./assets
# or specify a custom directory:
bash scripts/download_weights.sh /path/to/model_assets
```

Expected layout under `assets/` (or custom `$ASSETS`):

| Model | Relative Path under `assets/` | SHA-256 |
|---|---|---|
| `medsam2_box` | `MedSAM2-RECIST/checkpoints/medsam2_FLARE25_RECIST_baseline.pt` | `890e6e10924984c7b704a1984362c53a56dc0ee27c796a34eb49c83a1dc4a144` |
| `nninteractive_*` | `nninteractive_hf_models/models/nnInteractive_v1.0/` (`fold_0/checkpoint_final.pth`, `plans.json`, `dataset.json`, `inference_session_class.json`) | Verified in `configs/weights.sha256` |
| `efftiny_m8_ep40_box` | `RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` | `0b377887f5587cbfd42dbc739a1711f30db276f3fd4c849233df2142ee4f5f5d` |
| `effsmall_m8_ep40_box` (optional) | `RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt` | `423793cec97e9ec93b70c25fbff1c8267d1a1f58deed254b79e16caf7b489fb3` |

Place `RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt` into `assets/` and pass `--verify-sha256` to enforce strict pre-run verification.

---

## Input NPZ Format

Each input `.npz` file represents one 3D CT case with arrays in `(Z, Y, X)` voxel order and `(X, Y, Z)` physical spacing:

- `imgs`:
  - **Raw HU (default):** 3D signed-integer (`int16` / `int32`) or floating-point (`float32` / `float64`) array storing raw CT Hounsfield Units directly from the original CT volume, with **no** `hu_slope`/`hu_intercept` reconstruction. `nnInteractive` uses the raw HU values directly; SAM/EfficientMedSAM2 applies L40/W400 windowing (`[-160, 240]` -> `uint8 [0, 255]`) in `ct_inputs.py` unless pre-windowed twins are provided.
  - **Pre-windowed SAM input (`uint8`):** 3D `uint8` array in `[0, 255]`. If SAM models should use separate pre-windowed volumes while `nnInteractive` reads raw HU, place twins in `data/window_twins/` (auto-detected) or pass `--window-dir /path/to/windowed_npz`.
  - *(Legacy compatibility: archives explicitly carrying `intensity_encoding="nifti_stored_scaled_v1"` with `hu_slope`/`hu_intercept` remain optionally supported.)*
- `recist`: integer `(Z, Y, X)` label array with non-zero lesion IDs drawn on each lesion's axial RECIST slice.
- `spacing`: float `(3,)` physical spacing `[sx, sy, sz]` in mm.
- `gts` (or `gts_hidden`, optional): integer `(Z, Y, X)` ground-truth mask for Dice/volume evaluation (never used as a prompt source for RECIST modes).

---

## Running Inference

### 1. Zero-flag Quick Start (using default `data/inputs/`, `assets/`, `.venv-sam/`, `.venv-nn/`)

```bash
# Single-case CPU smoke test
recist-benchmark --limit 1 --verify-sha256 --plots

# Full default CPU run on data/inputs/
recist-benchmark --verify-sha256 --plots
```

### 2. Explicit paths (single cancer-type CPU run, with optional postprocessing)

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
  --postprocess moving_centroid \
  --verify-sha256 \
  --plots
```

### 3. Running a single model or switching to GPU (including non-0 physical GPU)

```bash
# Single model on CPU
recist-benchmark \
  --input-dir /path/to/data \
  --output-dir runs/efftiny_cpu \
  --models efftiny_m8_ep40_box \
  --device cpu \
  --num-threads 1

# GPU execution on physical GPU 2 (same codebase; NVML monitors physical GPU 2)
recist-benchmark \
  --input-dir /path/to/data \
  --output-dir runs/paper5_gpu2 \
  --device cuda \
  --gpu-index 2 \
  --num-threads 1
```

### 4. Standalone backend CLIs

Both backends also expose standalone CLIs (defaulting to `data/inputs/` -> `runs/sam/` or `runs/nninteractive/`):

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
  --postprocess moving_centroid \
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

---

## Performance & Resource Metrics

Every case execution runs an embedded `CaseProfiler` (and parent `ProcessTreeMonitor`) in `src/recist_effmedsam2/profiling/monitor.py` that mirrors [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency) sampling (`0.1 s` default interval) and AUC integration, scoped strictly to the active worker process tree.

### Runtime Breakdown (seconds via `time.perf_counter()`, CUDA-synchronized on GPU)

| Metric | Included in `runtime_s`? | Definition |
|---|---|---|
| `model_load_s` | **Yes** | Network instantiation + checkpoint loading onto target device |
| `inference_s` | **Yes** | Pure model inference across prompted lesions (state init/reset, prompt application, forward/reverse propagation or AutoZoom); **excludes** NPZ loading/decompression, preprocessing, and output saving |
| `postprocess_s` | **Yes** | Optional postprocessing (`moving_centroid`, inside `CaseProfiler`; `0.0` when `none`) |
| `runtime_s` | **Total** | `model_load_s + inference_s + postprocess_s` |
| `data_load_s` | No (audited separately) | NPZ disk load, decompression, and raw-HU / L40-W400 input loading |
| `preprocess_s` | No (audited separately) | SAM RGB resize & ImageNet normalization, or nnInteractive `set_image` preprocessing |
| `save_s` | No (audited separately) | Streamed independent-instance compressed NPZ write (raw + postprocessed) & CRC check |
| `wall_total_s` | No (audited separately) | Complete end-to-end case worker wall-clock time across all phases |

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
| `MaxGPU_Mem_MB` | MiB | `max(PID-scoped NVML peak, PyTorch peak reserved)`, or blank (`None`) if GPU memory could not be queried |
| `gpu_nvml_peak_mb` | MiB | Peak PID-scoped GPU memory queried via `pynvml` (`nvidia-ml-py`, fallback `nvidia-smi`), or blank (`None`) if unmeasured/failed |
| `AUC_GPU_Time_MB_s` | MiB·s | Trapezoidal area under PID-scoped GPU memory vs. time (`None` if unmeasured) |
| `RankGPU_Mem_MB` | MiB | `round(max(0.0, MaxGPU_Mem_MB - gpu_tolerance_mb))` (default tolerance: `4096.0 MiB`; blank if `MaxGPU_Mem_MB` is blank) |
| `gpu_torch_peak_allocated_mb` | MiB | `torch.cuda.max_memory_allocated()` (`0.0` on CPU) |
| `gpu_torch_peak_reserved_mb` | MiB | `torch.cuda.max_memory_reserved()` (`0.0` on CPU) |
| `gpu_monitor_status` | str | `measured` (all driver queries succeeded), `partial`, `query_failed`, or `unavailable` (no NVIDIA tooling/GPU) |
| `gpu_sanity_ok` | bool/blank | On CPU: `True` only when `gpu_monitor_status == "measured"` and VRAM == `0.0` and CUDA is unavailable; `False` on `query_failed`/`partial`/non-zero/CUDA available; blank (`None`) when `unavailable` |

---

## Output Layout & Evaluation CSVs

After inference completes, `compute_metrics.py` automatically writes three CSV reports under `<output-dir>/reports/`:

```text
<output-dir>/
├── run_manifest.json
├── <model_name>/
│   ├── predictions/<case_id>.npz                # independent_instances_v2 raw predictions
│   ├── predictions_postprocessed/<case_id>.npz  # optional postprocessed predictions (--postprocess moving_centroid)
│   ├── metrics/<case_id>.metrics.json           # full breakdown, resource samples, host info, per-lesion counts
│   ├── logs/<case_id>.log
│   └── plots/                                   # optional CPU/RAM/GPU-vs-time PNGs (--plots)
└── reports/
    ├── case_metrics.csv                         # 1 row per (model, case): timing breakdown, RAM, GPU, monitor status, mean Dice
    ├── lesion_scores.csv                        # 1 row per (model, case, lesion): independent same-ID Dice, volumes (mL), timings
    └── model_summary.csv                        # 1 row per (model, device): aggregated Dice & efficiency averages
```

To re-evaluate or recompute rankings with a different GPU tolerance without re-running inference:

```bash
recist-eval --run-dir runs/benchmark --verify-from-npz --gpu-tolerance-mb 4096
```

---

## Running Tests

Offline unit and end-to-end synthetic tests validate phase breakdown math (`runtime_s = model_load_s + inference_s + postprocess_s`), in-profiler postprocessing, exception and `SIGKILL` failure resource capture, `gpu_monitor_status` (`measured` vs `query_failed` vs `unavailable`), non-0 `CUDA_VISIBLE_DEVICES` -> physical GPU resolution, independent-instance storage, prompt geometry, and CSV generation:

```bash
pytest -v
```
