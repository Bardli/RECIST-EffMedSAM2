# Performance, Resource Metrics & CSV Layout

For installation, input NPZ format, and step-by-step inference instructions, see [`../README.md`](../README.md).

---

## Performance & Resource Metrics

Every case run wraps all phases inside `CaseProfiler` (`utils/monitor.py`), aligned with [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency) sampling (`0.1 s` default interval) and trapezoidal AUC integration, scoped strictly to the active process tree and mapped to the true physical GPU index.

When postprocessing (such as `moving_centroid` in `infer_effmedsam2.py`) is enabled, postprocessing executes before final NPZ saving and only the final prediction NPZ and final metrics are saved (no intermediate raw-vs-postprocessed NPZ files or `post_*` CSV columns). RECIST slice discovery and prompt geometry extraction run inside `image_preprocessing_time_s`, so `model_inference_time_s` captures pure model inference cleanly.

### RECIST Diameter & Runtime Breakdown

- **`ground_truth_recist_length_mm` & `predicted_recist_length_mm`**: Exact maximal 2D axial Euclidean Feret diameter (in mm, using in-plane XY spacing `(sx, sy)`) across all slices of the lesion's 3D ground-truth / final predicted mask (`0.0` for empty predictions; blank if `gt` was omitted). The 2D prompt line length (`recist_prompt_length_mm`) is also saved per lesion in `metrics/<case_id>.metrics.json`.

Runtime breakdown columns (seconds via `time.perf_counter()`, CUDA-synchronized on GPU):

| CSV Column | Included in `total_model_time_load_infer_post_s`? | Definition |
|---|---|---|
| `model_loading_time_s` | **Yes** | Model construction and checkpoint weight loading onto target device |
| `model_inference_time_s` | **Yes** | Pure model inference for this lesion (state init/reset, prompt application, forward/reverse propagation or AutoZoom); excludes disk load, preprocessing, and saving |
| `postprocessing_time_s` | **Yes** | Final mask postprocessing (`moving_centroid` per lesion; `0.0` when `none`) |
| `total_model_time_load_infer_post_s` | **Total** | `model_loading_time_s + model_inference_time_s + postprocessing_time_s` |
| `data_loading_time_s` | No (audited separately) | NPZ disk load, decompression, and intensity windowing lookup |
| `image_preprocessing_time_s` | No (audited separately) | SAM bilinear resize / ImageNet normalization or nnInteractive `set_image`, plus RECIST prompt extraction |
| `output_saving_time_s` | No (audited separately) | Streamed independent-instance compressed NPZ write & CRC check |
| `total_end_to_end_time_including_io_s` | No (audited separately) | Complete external end-to-end case time across all phases (load + preprocess + model load + inference + postprocess + save) |

#### Multi-RECIST Case-Level Rule (No Double-Counting)

Each row in `results.csv` represents **one lesion**:
- For a **single-lesion CT**, that single row records all phase timings, `ram_auc_mb_s`, and `gpu_memory_auc_mb_s` directly.
- For a **multi-lesion CT** (same model loaded once per CT volume):
  - Case-level shared timings (`model_loading_time_s`, `data_loading_time_s`, `image_preprocessing_time_s`, `output_saving_time_s`, `total_end_to_end_time_including_io_s`) are recorded on the **first prompted lesion row** of that CT and set to **`0.0`** on subsequent lesion rows of the same CT, while each lesion row records its own `model_inference_time_s`, `postprocessing_time_s`, `ground_truth_recist_length_mm`, and `predicted_recist_length_mm`.
  - **`ram_auc_mb_s`** and **`gpu_memory_auc_mb_s`** are integrated over each subsequent prompted lesion's own execution time span (inference + postprocessing), and the first prompted lesion row holds the remainder (`case_total_auc - sum(other_lesions)` = shared load/IO/preprocess/save overhead + first lesion's execution span), while unprompted (`no_recist`) rows receive `0.0` (or blank `""` if GPU memory was unmeasured). Every active lesion gets its real positive AUC while summing across rows still equals the exact cohort total without double-counting.

### RAM & GPU Memory Metrics (Human-Readable CSVs vs. Full JSON Receipts)

- **`results.csv` & `merged_results.csv` (human-facing):** Keep only essential identification, `ground_truth_recist_length_mm` / `predicted_recist_length_mm` (2 decimals in mm), segmentation Dice/volumes (4 decimals), clean phase timings (4 decimals), and RAM/VRAM peaks + AUC scores (1 decimal):
  - `peak_ram_mb` (`MaxRAM_MB` in JSON, MiB): peak process-tree RSS (`sampled_peak_rss_mb`, plus `ru_maxrss` on the first case in a run, bounded below by `ram_baseline_mb`)
  - `ram_auc_mb_s` (`AUC_RAM_Time_MB_s` in JSON, MiB·s): trapezoidal area under the process-tree RSS vs. time curve, matching FLARE-Efficiency's `AUC_RAM_Time` metric
  - `peak_gpu_memory_mb` (`MaxGPU_Mem_MB` in JSON, MiB): `max(PID-scoped NVML peak, PyTorch peak reserved)`, or blank if GPU memory could not be queried
  - `gpu_memory_auc_mb_s` (`AUC_GPU_Time_MB_s` in JSON, MiB·s): trapezoidal area under the PID-scoped GPU memory vs. time curve, matching FLARE-Efficiency's `AUC_GPU_Time` metric (`0.0` on verified `--device cpu`; blank if GPU memory could not be queried)
- **`metrics/<case_id>.metrics.json` (machine/diagnostic receipts):** Preserve full-precision unrounded floats alongside all diagnostic, status, and profiling fields removed from CSV (`status`, `error`, `postprocess`, `recist_id`, `recist_prompt_length_mm`, per-lesion `prediction_status` and raw voxel counts, `MaxCPU_Utilization_pct`, `ram_baseline_mb`, `ram_inference_peak_mb`, `ram_delta_mb`, `AUC_CPU_Time_pct_s`, `gpu_nvml_peak_mb`, `RankGPU_Mem_MB`, `gpu_torch_peak_allocated_mb`, `gpu_torch_peak_reserved_mb`, `gpu_monitor_status`, `gpu_sanity_ok`, and host hardware metadata).

---

## Single Per-Model CSV (`results.csv`) & Concatenated Merged CSV (`merged_results.csv`)

Each standalone inference script produces **one concise CSV (`<output-dir>/results.csv`, 22 columns)** alongside predictions, JSON telemetry receipts, and optional plots:

```text
<output-dir>/
├── predictions/<case_id>.npz          # final independent_instances_v2 predictions (no intermediate raw+post duplicates)
├── metrics/<case_id>.metrics.json     # full-precision diagnostic telemetry, status/errors, voxel counts, host info
├── plots/                             # optional CPU/RAM/GPU-vs-time PNGs (--plots)
└── results.csv                        # 22 human-readable columns, 1 row per lesion
```

`results.csv` columns (22):
`dataset`, `case_id`, `lesion_id`, `model_name`, `device`, `dice_score`, `ground_truth_recist_length_mm`, `predicted_recist_length_mm`, `ground_truth_volume_ml`, `predicted_volume_ml`, `total_model_time_load_infer_post_s`, `model_loading_time_s`, `model_inference_time_s`, `postprocessing_time_s`, `data_loading_time_s`, `image_preprocessing_time_s`, `output_saving_time_s`, `total_end_to_end_time_including_io_s`, `peak_ram_mb`, `ram_auc_mb_s`, `peak_gpu_memory_mb`, `gpu_memory_auc_mb_s`.

Running `merge_csvs.py` performs a **plain vertical concatenation** of the per-model `results.csv` files into **one combined `merged_results.csv`** with the exact same 22 columns (no pivoting or column prefixes), while printing a compact per-model summary to stdout.

### Subset Nodal Outputs (`infer_nodal.py`)

When running confirmed short-axis nodal inference with [`infer_nodal.py`](../infer_nodal.py) (`tiny` or `medsam2`; see [`nodal_prompts.md`](nodal_prompts.md)), the output folder retains the exact 22-column `results.csv` schema but contains **only the manifest-selected `confirmed_short_axis_ids`** for that CT (non-selected lesion IDs are omitted from predictions and CSV rows, never zero-filled). Always join subset tables downstream by `(case_id, lesion_id)`, never by row position. Both `metrics/<case_id>.metrics.json` and the saved prediction NPZ metadata record `nodal_protocol` (`"confirmed-nodal-short-to-long-1.25-v1"`), `nodal_manifest_sha256`, `window_sha256`, `confirmed_short_axis_ids`, and `nodal_geometry` (`source_endpoints_xy`, `nominal_endpoints_xy`, `endpoints_xy`, `short_mm`, `nominal_mm`, `chord_mm`, `endpoint_clamped`, `box_xyxy`; plus `z_bounds_inclusive` and `postprocess_actions` for `tiny`).

---

## Running Tests

Offline unit and end-to-end synthetic tests validate phase timing math (`total_model_time_load_infer_post_s = model_loading_time_s + model_inference_time_s + postprocessing_time_s`), GT/predicted RECIST Feret diameter, FLARE-Efficiency `ram_auc_mb_s` and `gpu_memory_auc_mb_s` with multi-RECIST zeroing, human-readable 22-column CSV formatting alongside full JSON receipts, final-only postprocessing, case-scoped Tiny exemptions (`tests/test_exemptions.py`), 1.25× nodal prompt geometry/manifest/subset output (`tests/test_nodal.py`), `gpu_monitor_status` (`measured` vs `query_failed` vs `unavailable`), physical GPU target resolution, independent-instance storage, prompt geometry, standalone script CLIs, and vertical CSV concatenation:

```bash
uv pip install --python .venv-sam/bin/python pytest
.venv-sam/bin/pytest -v
```

### Real GPU cold-start acceptance test

From the checkout, install `pytest` in `.venv-sam` and run on an allocated GPU (for example, physical GPU 2):

```bash
uv pip install --python .venv-sam/bin/python pytest
CUDA_VISIBLE_DEVICES=2 RECIST_REQUIRE_CUDA=1 .venv-sam/bin/python -m pytest -q -s tests/test_cuda_cold_start.py
```

On a scheduler-managed node, keep its assigned `CUDA_VISIBLE_DEVICES` instead. The test starts a fresh Python interpreter, checks that the CPU path leaves CUDA uninitialized, then resets real CUDA peak counters and verifies a real tensor allocation. Without `RECIST_REQUIRE_CUDA=1`, GPU-less test environments skip this test; with it, absent hardware or a missing CUDA-enabled PyTorch fails.
