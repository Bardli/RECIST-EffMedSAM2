# Agent Instructions for RECIST-EffMedSAM2

When asked to set up, verify, or reproduce RECIST-EffMedSAM2 CPU/GPU inference or efficiency benchmarks, read and execute the bundled repository skill:

- Canonical skill: [`.agents/skills/reproduce-recist-benchmark/SKILL.md`](.agents/skills/reproduce-recist-benchmark/SKILL.md)
- Claude Code mirror: [`.claude/skills/reproduce-recist-benchmark/SKILL.md`](.claude/skills/reproduce-recist-benchmark/SKILL.md)

## Non-Negotiable Repository Invariants

- **Fresh per-case model reload:** Each `(model, case)` runs in an isolated subprocess with model construction/loading inside `runtime_s = model_load_s + inference_s + postprocess_s`.
- **In-profiler postprocessing:** `moving_centroid` postprocessing executes inside `CaseProfiler` so `postprocess_s`, `runtime_s`, `wall_total_s`, and `MaxRAM_MB` stay consistent.
- **Serial single-threaded default:** Preserve `--num-threads 1` for CPU and GPU benchmark comparability unless explicitly overridden.
- **CPU isolation & physical GPU mapping:** Verify `--device cpu` with `gpu_monitor_status == measured` & `gpu_sanity_ok == True` when NVIDIA tooling is available (never confuse `None`/`unavailable`/`query_failed` with `0.0 MiB`); pass `--gpu-index N` explicitly for non-0 GPU NVML monitoring.
- **Direct raw HU (no reconstruction):** Raw-HU NPZs store `int16`/`float32` HU directly in `imgs` (no `hu_slope`/`hu_intercept` reconstruction required); `ct_inputs.py` windows raw HU to L40/W400 only when pre-windowed SAM twins are not supplied.
- **Public repo hygiene:** Never commit patient data, checkpoints, `.npz`/`.nii.gz`, `runs/`, or private filesystem paths.
