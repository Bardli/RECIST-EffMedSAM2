# Performance, Resource Metrics & CSV Layout

For installation, input NPZ format, and step-by-step inference instructions, see [`../README.md`](../README.md).

---

## Performance & Resource Metrics

Every case run wraps all phases inside `CaseProfiler` (`utils/monitor.py`), aligned with [`medfm-flare/FLARE-Efficiency`](https://github.com/medfm-flare/FLARE-Efficiency) sampling (`0.1 s` default interval) and AUC integration, scoped strictly to the active process tree and mapped to the true physical GPU index.

When postprocessing (such as `moving_centroid` in `infer_effmedsam2.py`) is enabled, postprocessing executes before final NPZ saving and only the final prediction NPZ and final metrics are saved (no intermediate raw-vs-postprocessed NPZ files or `post_*` CSV columns).

### Runtime Breakdown (seconds via `time.perf_counter()`, CUDA-synchronized on GPU)

| Column | Included in `total_runtime_s`? | Definition |
|---|---|---|
| `model_loading_time_s` | **Yes** | Model construction and checkpoint weight loading onto target device |
| `model_inference_time_s` | **Yes** | Pure model inference across prompted lesions (state init/reset, prompt application, forward/reverse propagation or AutoZoom); excludes disk load, preprocessing, and saving |
| `postprocessing_time_s` | **Yes** | Final mask postprocessing (`moving_centroid` inside `CaseProfiler`; `0.0` when `none`) |
| `total_runtime_s` | **Total** | `model_loading_time_s + model_inference_time_s + postprocessing_time_s` |
| `single_lesion_inference_time_s` | Lesion CSV only | Pure inference time spent on that individual lesion (`lesion_scores.csv`) |
| `data_loading_time_s` | No (audited separately) | NPZ disk load, decompression, and intensity windowing lookup |
| `image_preprocessing_time_s` | No (audited separately) | SAM bilinear resize / ImageNet normalization, or nnInteractive `set_image` preprocessing |
| `output_saving_time_s` | No (audited separately) | Streamed independent-instance compressed NPZ write & CRC check |
| `end_to_end_wall_time_including_io_s` | No (audited separately) | Complete external end-to-end case wall-clock time across all phases (load + preprocess + model load + inference + postprocess + save) |

### RAM, CPU & GPU Memory Metrics

| Metric | Unit | Definition |
|---|---|---|
| `ram_baseline_mb` | MiB | Process-tree RSS right before model loading |
| `MaxRAM_MB` | MiB | Peak process-tree RSS (`sampled_peak_rss_mb`, plus `ru_maxrss` on the first case in a run, bounded below by `ram_baseline_mb`) |
| `ram_inference_peak_mb` | MiB | Peak process-tree RSS observed during the `model_inference_time` phase |
| `ram_delta_mb` | MiB | `MaxRAM_MB - ram_baseline_mb` |
| `AUC_RAM_Time_MB_s` | MiB·s | Trapezoidal area under process-tree RSS vs. time |
| `MaxCPU_Utilization_pct` | % | Peak normalized process-tree CPU utilization (0–100%) |
| `AUC_CPU_Time_pct_s` | %·s | Trapezoidal area under CPU utilization vs. time |
| `MaxGPU_Mem_MB` | MiB | `max(PID-scoped NVML peak, PyTorch peak reserved)`, or blank if GPU memory could not be queried |
| `gpu_nvml_peak_mb` | MiB | Peak PID-scoped GPU memory queried via `pynvml` (`nvidia-ml-py`, fallback `nvidia-smi`), or blank if unmeasured/failed |
| `AUC_GPU_Time_MB_s` | MiB·s | Trapezoidal area under PID-scoped GPU memory vs. time (blank if unmeasured) |
| `RankGPU_Mem_MB` | MiB | `round(max(0.0, MaxGPU_Mem_MB - gpu_tolerance_mb))` (default tolerance: `4096.0 MiB`; blank if `MaxGPU_Mem_MB` is blank) |
| `gpu_torch_peak_allocated_mb` | MiB | `torch.cuda.max_memory_allocated()` (`0.0` on CPU) |
| `gpu_torch_peak_reserved_mb` | MiB | `torch.cuda.max_memory_reserved()` (`0.0` on CPU) |
| `gpu_monitor_status` | str | `measured` (all driver queries succeeded), `partial`, `query_failed`, or `unavailable` (no NVIDIA tooling/GPU) |
| `gpu_sanity_ok` | bool/blank | On CPU: `True` only when `gpu_monitor_status == "measured"` and VRAM == `0.0` and CUDA is unavailable; `False` on `query_failed`/`partial`/non-zero/CUDA available; blank when `unavailable` |

---

## Per-Model Outputs & Merged Comparison CSVs

Each standalone script writes final predictions, raw metric receipts, and independent per-model CSVs under `<output-dir>`:

```text
<output-dir>/
├── predictions/<case_id>.npz          # final independent_instances_v2 predictions (no intermediate raw+post duplicates)
├── metrics/<case_id>.metrics.json     # full timing breakdown, resource samples, host info
├── plots/                             # optional CPU/RAM/GPU-vs-time PNGs (--plots)
├── case_metrics.csv                   # 1 row per case (model_name, clear timing breakdown, RAM/GPU, mean_dice)
└── lesion_scores.csv                  # 1 row per lesion (GT info, final pred_voxels / pred_volume_ml / dice, single_lesion_inference_time_s)
```

Running `merge_csvs.py` combines the independent per-model CSVs horizontally into wide side-by-side comparison tables where every model's prediction, Dice, volume, and timing columns carry its model prefix (for example `EffMedSAM2_dice`, `MedSAM2_pred_volume_ml`, `nnInteractive_box_model_inference_time_s`):

```text
runs/merged/
├── merged_case_metrics.csv            # 1 row per (dataset, case_id) with <ModelName>_* columns
├── merged_lesion_scores.csv           # 1 row per (dataset, case_id, gt_id) with shared GT + <ModelName>_* columns
└── model_summary.csv                  # 1 row per (model_name, device) with aggregated Dice & efficiency averages
```

*(If the same `model_name` is merged across different devices such as `cpu` and `cuda`, column prefixes automatically disambiguate to `<ModelName>_cpu_*` and `<ModelName>_cuda_*`.)*

---

## Running Tests

Offline unit and end-to-end synthetic tests validate phase timing math (`total_runtime_s = model_loading_time_s + model_inference_time_s + postprocessing_time_s`), final-only postprocessing, `gpu_monitor_status` (`measured` vs `query_failed` vs `unavailable`), physical GPU target resolution, independent-instance storage, prompt geometry, standalone script CLIs, and wide CSV merging:

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
