# Performance, Resource Metrics & CSV Layout

For installation, input NPZ format, and step-by-step inference instructions, see [`../README.md`](../README.md).

---

## Performance & Resource Metrics

Every case run wraps all phases inside `CaseProfiler` (`utils/monitor.py`), aligned with [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency) sampling (`0.1 s` default interval) and AUC integration, scoped strictly to the active process tree and mapped to the true physical GPU index.

When postprocessing (such as `moving_centroid` in `infer_effmedsam2.py`) is enabled, postprocessing executes before final NPZ saving and only the final prediction NPZ and final metrics are saved (no intermediate raw-vs-postprocessed NPZ files or `post_*` CSV columns). RECIST slice discovery and prompt geometry extraction run inside `image_preprocessing_time_s`, so `model_inference_time_s` captures pure model inference cleanly.

### Runtime Breakdown (seconds via `time.perf_counter()`, CUDA-synchronized on GPU)

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

#### Multi-RECIST Timing Rule (No Double-Counting)

Each row in `results.csv` represents **one RECIST lesion**:
- For a **single-lesion CT**, that single row records all phase timings directly.
- For a **multi-lesion CT** (same model loaded once per CT volume), case-level shared phases (`model_loading_time_s`, `data_loading_time_s`, `image_preprocessing_time_s`, `output_saving_time_s`, `total_end_to_end_time_including_io_s`) are recorded on the **first lesion row** of that CT and set to **`0.0`** on subsequent lesion rows of the same CT, while each lesion row records its own `model_inference_time_s` and `postprocessing_time_s`. Summing any time column across rows gives the true cohort total without double-counting.

### Human-Readable CSVs vs. Full JSON Receipts

- **`results.csv` & `merged_results.csv` (human-facing):** Keep only essential identification, segmentation scores/volumes (rounded to 4 decimal places), clean phase timings (4 decimals), and peak RAM/VRAM (1 decimal in MiB):
  - `peak_ram_mb` (`MaxRAM_MB` in JSON)
  - `peak_gpu_memory_mb` (`MaxGPU_Mem_MB` in JSON, blank when GPU memory could not be queried)
- **`metrics/<case_id>.metrics.json` (machine/diagnostic receipts):** Preserve full-precision unrounded floats alongside all diagnostic, status, and profiling fields removed from CSV (`status`, `error`, `postprocess`, `recist_id`, per-lesion `prediction_status` and raw voxel counts, `MaxCPU_Utilization_pct`, `ram_baseline_mb`, `ram_inference_peak_mb`, `ram_delta_mb`, `AUC_RAM_Time_MB_s`, `AUC_CPU_Time_pct_s`, `gpu_nvml_peak_mb`, `AUC_GPU_Time_MB_s`, `RankGPU_Mem_MB`, `gpu_torch_peak_allocated_mb`, `gpu_torch_peak_reserved_mb`, `gpu_monitor_status`, `gpu_sanity_ok`, and host hardware metadata).

---

## Single Per-Model CSV (`results.csv`) & Merged Comparison CSV (`merged_results.csv`)

Each standalone inference script produces **one concise CSV (`<output-dir>/results.csv`, 18 columns)** alongside predictions, JSON telemetry receipts, and optional plots:

```text
<output-dir>/
├── predictions/<case_id>.npz          # final independent_instances_v2 predictions (no intermediate raw+post duplicates)
├── metrics/<case_id>.metrics.json     # full-precision diagnostic telemetry, status/errors, voxel counts, host info
├── plots/                             # optional CPU/RAM/GPU-vs-time PNGs (--plots)
└── results.csv                        # 18 human-readable columns, 1 row per RECIST lesion
```

`results.csv` columns (18):
`dataset`, `case_id`, `lesion_id`, `model_name`, `device`, `dice_score`, `ground_truth_volume_ml`, `predicted_volume_ml`, `total_model_time_load_infer_post_s`, `model_loading_time_s`, `model_inference_time_s`, `postprocessing_time_s`, `data_loading_time_s`, `image_preprocessing_time_s`, `output_saving_time_s`, `total_end_to_end_time_including_io_s`, `peak_ram_mb`, `peak_gpu_memory_mb`.

Running `merge_csvs.py` merges independent `results.csv` files horizontally into **one compact wide table (`merged_results.csv`)**, organized **by metric** so all models sit directly side-by-side and exposing **only one time column per model** (`<Model>_total_model_time_load_infer_post_s`):

1. **Shared lesion metadata:** `dataset`, `case_id`, `lesion_id`, `ground_truth_volume_ml`
2. **Dice scores across models:** `<ModelA>_dice_score`, `<ModelB>_dice_score`, ...
3. **Predicted volumes across models:** `<ModelA>_predicted_volume_ml`, `<ModelB>_predicted_volume_ml`, ...
4. **Total model time across models (load + infer + postprocess):** `<ModelA>_total_model_time_load_infer_post_s`, `<ModelB>_total_model_time_load_infer_post_s`, ...
5. **Peak host RAM across models:** `<ModelA>_peak_ram_mb`, `<ModelB>_peak_ram_mb`, ...
6. **Peak GPU VRAM across models:** `<ModelA>_peak_gpu_memory_mb`, `<ModelB>_peak_gpu_memory_mb`, ...

*(If the same `model_name` is merged across different devices such as `cpu` and `cuda`, column prefixes automatically disambiguate to `<ModelName>_cpu_*` and `<ModelName>_cuda_*`.)*

---

## Running Tests

Offline unit and end-to-end synthetic tests validate phase timing math (`total_model_time_load_infer_post_s = model_loading_time_s + model_inference_time_s + postprocessing_time_s`), multi-RECIST zeroing, human-readable 18-column CSV trimming alongside full JSON receipts, final-only postprocessing, `gpu_monitor_status` (`measured` vs `query_failed` vs `unavailable`), physical GPU target resolution, independent-instance storage, prompt geometry, standalone script CLIs, and metric-grouped CSV merging:

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
