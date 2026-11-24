# Performance, Resource Metrics & Evaluation

For installation, input NPZ format, and step-by-step inference instructions, see [`../README.md`](../README.md).

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
| `data_load_s` | No (audited separately) | NPZ disk load & decompression |
| `preprocess_s` | No (audited separately) | HU decoding / L40-W400 windowing / RGB resize & normalization, or nnInteractive `set_image` preprocessing |
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
