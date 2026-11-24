"""Tests for embedded FLARE-Efficiency resource monitor and phase breakdown."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from recist_effmedsam2.profiling.monitor import (
    CaseProfiler,
    Samples,
    get_process_max_rss_mb,
    ranking_gpu_memory,
    summarize_samples,
)
from recist_effmedsam2.profiling.report import plot_case


def test_summarize_samples_math_and_cpu_sanity():
    samples = Samples(
        interval=1.0,
        seconds=[0.0, 1.0, 2.0],
        cpu_percent=[20.0, 40.0, 20.0],
        ram_mb=[100.0, 200.0, 100.0],
        gpu_mb=[0.0, 0.0, 0.0],
    )
    timings = {
        "model_load_s": 0.4,
        "inference_s": 1.2,
        "data_load_s": 0.3,
        "preprocess_s": 0.2,
        "save_s": 0.1,
        "wall_total_s": 2.2,
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
    )
    assert np.isclose(m["runtime_s"], 1.6)
    assert np.isclose(m["model_load_s"], 0.4)
    assert np.isclose(m["inference_s"], 1.2)
    assert m["MaxRAM_MB"] == 210.0
    assert m["ram_inference_peak_mb"] == 200.0
    assert m["ram_delta_mb"] == 120.0
    assert np.isclose(m["AUC_RAM_Time_MB_s"], 300.0)
    assert np.isclose(m["AUC_CPU_Time_pct_s"], 60.0)
    assert m["MaxGPU_Mem_MB"] == 0.0
    assert m["RankGPU_Mem_MB"] == 0.0
    assert m["gpu_sanity_ok"] is True


def test_gpu_tolerance_ranking_and_cpu_sanity_failure():
    samples = Samples(
        interval=1.0,
        seconds=[0.0, 1.0],
        cpu_percent=[10.0, 10.0],
        ram_mb=[128.0, 128.0],
        gpu_mb=[6000.0, 6000.0],
    )
    assert ranking_gpu_memory(3000.0, 4096.0) == 0.0
    assert ranking_gpu_memory(6000.0, 4096.0) == 1904.0

    # If a CPU run somehow shows GPU usage or CUDA availability, sanity check must fail
    m_cpu_bad = summarize_samples(
        samples,
        {"model_load_s": 0.1, "inference_s": 0.9},
        device="cpu",
        torch_cuda_stats={
            "cuda_available": True,
            "peak_allocated_mb": 12.0,
            "peak_reserved_mb": 20.0,
        },
    )
    assert m_cpu_bad["gpu_sanity_ok"] is False
    assert m_cpu_bad["MaxGPU_Mem_MB"] == 6000.0
    assert m_cpu_bad["RankGPU_Mem_MB"] == 1904.0


def test_case_profiler_cpu_and_plots(tmp_path: Path):
    with CaseProfiler(device="cpu", sample_interval=0.03) as prof:
        with prof.phase("model_load"):
            buf = np.ones((512, 512), dtype=np.float32)
            time.sleep(0.05)
        with prof.phase("data_load"):
            time.sleep(0.03)
        with prof.phase("preprocess"):
            time.sleep(0.02)
        with prof.phase("inference"):
            time.sleep(0.06)
            prof.record_lesion_time(1, 0.055)
        with prof.phase("save"):
            time.sleep(0.02)
        del buf
        record = prof.finalize(extra_metadata={"case_id": "synth_01", "model": "unit_test"})

    m = record["metrics"]
    assert get_process_max_rss_mb() > 0.0
    assert m["MaxRAM_MB"] > 0.0
    assert m["ram_inference_peak_mb"] > 0.0
    assert np.isclose(m["runtime_s"], m["model_load_s"] + m["inference_s"], atol=1e-5)
    assert m["wall_total_s"] > m["runtime_s"]
    assert m["MaxGPU_Mem_MB"] == 0.0
    assert m["gpu_nvml_peak_mb"] == 0.0
    assert m["gpu_sanity_ok"] is True
    assert record["per_lesion_inference_s"]["1"] == 0.055

    plots = plot_case(record, tmp_path / "plots")
    assert len(plots) == 3
    for p in plots:
        assert p.is_file() and p.stat().st_size > 0
