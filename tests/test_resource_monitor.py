"""Tests for embedded FLARE-Efficiency resource monitor, clear phase timings, and GPU mapping."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from utils.monitor import (
    CaseProfiler,
    PidGpuSampler,
    Samples,
    get_process_max_rss_mb,
    plot_case,
    ranking_gpu_memory,
    resolve_physical_gpu_targets,
    summarize_samples,
)


def test_summarize_samples_math_and_cpu_sanity():
    samples = Samples(
        interval=1.0,
        seconds=[0.0, 1.0, 2.0],
        cpu_percent=[20.0, 40.0, 20.0],
        ram_mb=[100.0, 200.0, 100.0],
        gpu_mb=[0.0, 0.0, 0.0],
    )
    timings = {
        "model_loading_time_s": 0.4,
        "model_inference_time_s": 1.2,
        "postprocessing_time_s": 0.25,
        "data_loading_time_s": 0.3,
        "image_preprocessing_time_s": 0.2,
        "output_saving_time_s": 0.1,
        "end_to_end_wall_time_including_io_s": 2.45,
    }
    m = summarize_samples(
        samples,
        timings,
        device="cpu",
        ram_baseline_mb=90.0,
        ru_maxrss_mb=210.0,
        inference_spans=[(0.5, 1.5)],
        torch_cuda_stats={
            "cuda_available": False,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        },
        gpu_tolerance_mb=4096.0,
        gpu_monitor_status="measured",
        gpu_monitor_backend="pynvml",
    )
    assert np.isclose(m["total_runtime_s"], 1.85)
    assert np.isclose(m["model_loading_time_s"], 0.4)
    assert np.isclose(m["model_inference_time_s"], 1.2)
    assert np.isclose(m["postprocessing_time_s"], 0.25)
    assert np.isclose(m["end_to_end_wall_time_including_io_s"], 2.45)
    assert m["MaxRAM_MB"] == 210.0
    assert m["ram_inference_peak_mb"] == 200.0
    assert m["ram_delta_mb"] == 120.0
    assert np.isclose(m["AUC_RAM_Time_MB_s"], 300.0)
    assert np.isclose(m["AUC_CPU_Time_pct_s"], 60.0)
    assert m["MaxGPU_Mem_MB"] == 0.0
    assert m["RankGPU_Mem_MB"] == 0.0
    assert m["gpu_monitor_status"] == "measured"
    assert m["gpu_sanity_ok"] is True


def test_gpu_unmeasured_vs_zero_and_sanity():
    assert ranking_gpu_memory(3000.0, 4096.0) == 0.0
    assert ranking_gpu_memory(6000.0, 4096.0) == 1904.0

    unmeasured_samples = Samples(
        interval=1.0,
        seconds=[0.0, 1.0],
        cpu_percent=[10.0, 10.0],
        ram_mb=[128.0, 128.0],
        gpu_mb=[None, None],
    )
    m_unavail = summarize_samples(
        unmeasured_samples,
        {"model_loading_time_s": 0.1, "model_inference_time_s": 0.9},
        device="cpu",
        gpu_monitor_status="unavailable",
        gpu_monitor_backend="none",
    )
    assert m_unavail["MaxGPU_Mem_MB"] is None
    assert m_unavail["gpu_nvml_peak_mb"] is None
    assert m_unavail["AUC_GPU_Time_MB_s"] is None
    assert m_unavail["RankGPU_Mem_MB"] is None
    assert m_unavail["gpu_sanity_ok"] is None

    m_failed = summarize_samples(
        unmeasured_samples,
        {"model_loading_time_s": 0.1, "model_inference_time_s": 0.9},
        device="cpu",
        gpu_monitor_status="query_failed",
        gpu_monitor_backend="pynvml",
        gpu_queries_failed=2,
    )
    assert m_failed["MaxGPU_Mem_MB"] is None
    assert m_failed["gpu_sanity_ok"] is False

    sampler = PidGpuSampler.__new__(PidGpuSampler)
    sampler.gpu_targets = None
    sampler.backend = "none"
    sampler.gpu_name = None
    sampler.monitored_physical_gpus = []
    sampler.queries_attempted = 0
    sampler.queries_succeeded = 0
    sampler.queries_failed = 0
    sampler.last_error = None
    sampler._pynvml = None
    sampler._handles = []
    sampler._nvidia_smi = None
    sampler._target_resolution_failed = False
    assert sampler.read_mb({12345}) is None
    assert sampler.status == "unavailable"


def test_resolve_physical_gpu_targets():
    assert resolve_physical_gpu_targets("cpu", None, "2,3") is None
    assert resolve_physical_gpu_targets("cuda:0", 2, "2") == [2]
    assert resolve_physical_gpu_targets("cuda:0", None, "2,3") == [2]
    assert resolve_physical_gpu_targets("cuda:1", None, "2,3") == [3]
    assert resolve_physical_gpu_targets("cuda:0", None, "GPU-1234-uuid") == ["GPU-1234-uuid"]
    assert resolve_physical_gpu_targets("cuda:0", None, "") == []


def test_case_profiler_cpu_with_postprocess_and_plots(tmp_path: Path):
    with CaseProfiler(device="cpu", sample_interval=0.03) as prof:
        with prof.phase("model_loading_time"):
            buf = np.ones((512, 512), dtype=np.float32)
            time.sleep(0.04)
        with prof.phase("data_loading_time"):
            time.sleep(0.02)
        with prof.phase("image_preprocessing_time"):
            time.sleep(0.02)
        with prof.phase("model_inference_time"):
            time.sleep(0.05)
            prof.record_lesion_time(1, 0.045)
        with prof.phase("postprocessing_time"):
            time.sleep(0.03)
        with prof.phase("output_saving_time"):
            time.sleep(0.02)
        del buf
        record = prof.finalize(extra_metadata={"case_id": "synth_01", "model_name": "EffMedSAM2"})

    m = record["metrics"]
    assert get_process_max_rss_mb() > 0.0
    assert m["MaxRAM_MB"] > 0.0
    assert m["ram_inference_peak_mb"] > 0.0
    assert m["postprocessing_time_s"] >= 0.02
    assert np.isclose(
        m["total_runtime_s"],
        m["model_loading_time_s"] + m["model_inference_time_s"] + m["postprocessing_time_s"],
        atol=1e-5,
    )
    assert m["end_to_end_wall_time_including_io_s"] > m["total_runtime_s"]
    assert "postprocessing_time" in record["phase_spans"]
    if m["gpu_monitor_status"] == "measured":
        assert m["MaxGPU_Mem_MB"] == 0.0
        assert m["gpu_nvml_peak_mb"] == 0.0
        assert m["gpu_sanity_ok"] is True

    plots = plot_case(record, tmp_path / "plots")
    assert len(plots) >= 2
    for p in plots:
        assert p.is_file() and p.stat().st_size > 0

    rec_no_gpu = dict(record)
    rec_no_gpu["samples"] = dict(record["samples"], gpu_mb=[None] * len(record["samples"]["seconds"]))
    plots2 = plot_case(rec_no_gpu, tmp_path / "plots_nogpu")
    assert len(plots2) == 2
