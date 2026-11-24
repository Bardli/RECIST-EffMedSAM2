"""End-to-end subprocess benchmark runner and CSV evaluation tests."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def _make_synthetic_case(
    hu_path: Path,
    win_path: Path | None = None,
    seed_offset: int = 0,
) -> None:
    shape = (8, 48, 48)
    hu = np.full(shape, -100, dtype=np.int16)
    hu[:, 12:36, 12:36] = 50 + seed_offset

    gt = np.zeros(shape, dtype=np.uint16)
    recist = np.zeros(shape, dtype=np.uint16)
    gt[3:6, 16:28, 16:28] = 1
    cv2.line(recist[4], (16, 22), (27, 22), 1, thickness=2)

    gt[2:5, 30:38, 30:38] = 3
    cv2.line(recist[3], (30, 34), (37, 34), 3, thickness=2)

    spacing = np.array([1.0, 1.0, 2.5], dtype=np.float32)
    np.savez_compressed(
        hu_path,
        imgs=hu,
        gts=gt,
        recist=recist,
        spacing=spacing,
        intensity_encoding=np.array("nifti_stored_scaled_v1"),
        hu_slope=np.array(1.0),
        hu_intercept=np.array(0.0),
    )
    if win_path is not None:
        win = ((np.clip(hu.astype(np.float64), -160.0, 240.0) + 160.0) / 400.0 * 255.0).astype(
            np.uint8
        )
        np.savez_compressed(
            win_path,
            imgs=win,
            gts=gt,
            recist=recist,
            spacing=spacing,
        )


def test_end_to_end_cpu_benchmark_and_csvs(tmp_path: Path):
    in_dir = tmp_path / "inputs" / "colon_cancer"
    win_dir = tmp_path / "window_twins" / "colon_cancer"
    in_dir.mkdir(parents=True)
    win_dir.mkdir(parents=True)
    _make_synthetic_case(in_dir / "case_001.npz", win_dir / "case_001.npz", seed_offset=0)
    _make_synthetic_case(in_dir / "case_002.npz", win_dir / "case_002.npz", seed_offset=10)

    out_dir = tmp_path / "run_cpu"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    # 1. Run all 5 paper configurations on 1 case (--limit 1) in CPU mode via subprocesses
    cmd = [
        sys.executable,
        "-m",
        "recist_effmedsam2.runner.run_benchmark",
        "--input-dir",
        str(in_dir),
        "--window-dir",
        str(win_dir),
        "--output-dir",
        str(out_dir),
        "--device",
        "cpu",
        "--num-threads",
        "1",
        "--limit",
        "1",
        "--sample-interval",
        "0.02",
        "--backend-override",
        "mock",
        "--postprocess",
        "moving_centroid",
        "--plots",
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout + "\n" + res.stderr

    case_csv = out_dir / "reports" / "case_metrics.csv"
    lesion_csv = out_dir / "reports" / "lesion_scores.csv"
    summary_csv = out_dir / "reports" / "model_summary.csv"
    assert case_csv.is_file()
    assert lesion_csv.is_file()
    assert summary_csv.is_file()

    with case_csv.open() as f:
        case_rows = list(csv.DictReader(f))
    assert len(case_rows) == 5  # 5 paper configs * 1 case
    for r in case_rows:
        assert r["dataset"] == "colon_cancer"
        assert r["case_id"] == "case_001"
        assert r["device"] == "cpu"
        assert r["num_threads"] == "1"
        assert r["status"] == "ok"
        assert r["gpu_sanity_ok"] == "True"
        assert float(r["MaxGPU_Mem_MB"]) == 0.0
        assert float(r["gpu_nvml_peak_mb"]) == 0.0
        assert float(r["MaxRAM_MB"]) > 0.0
        assert np.isclose(
            float(r["runtime_s"]),
            float(r["model_load_s"]) + float(r["inference_s"]),
            atol=1e-4,
        )
        assert float(r["wall_total_s"]) >= float(r["runtime_s"])
        assert float(r["mean_dice"]) > 0.0

    eff_row = next(r for r in case_rows if r["model"] == "efftiny_m8_ep40_box")
    assert eff_row["mean_post_dice"] != ""

    with lesion_csv.open() as f:
        lesion_rows = list(csv.DictReader(f))
    assert len(lesion_rows) == 10  # 5 models * 2 lesions
    for lr in lesion_rows:
        assert float(lr["gt_volume_ml"]) > 0.0
        assert float(lr["pred_volume_ml"]) > 0.0
        assert lr["score_protocol"] == "independent_same_id_raw_v2"

    # Check plot artifacts exist
    assert (out_dir / "efftiny_m8_ep40_box" / "plots" / "case_001_RAM-Time.png").is_file()
    assert (out_dir / "efftiny_m8_ep40_box" / "plots" / "case_001_GPU-Time.png").is_file()

    # 2. Re-run standalone evaluation with --verify-from-npz
    eval_cmd = [
        sys.executable,
        "-m",
        "recist_effmedsam2.evaluation.compute_metrics",
        "--run-dir",
        str(out_dir),
        "--verify-from-npz",
    ]
    eval_res = subprocess.run(eval_cmd, env=env, capture_output=True, text=True, check=False)
    assert eval_res.returncode == 0, eval_res.stderr

    # 3. Verify --dry-run produces clean CPU and GPU commands for all-six models
    for device in ("cpu", "cuda"):
        dry_cmd = [
            sys.executable,
            "-m",
            "recist_effmedsam2.runner.run_benchmark",
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(tmp_path / f"dry_{device}"),
            "--assets-dir",
            str(tmp_path / "assets"),
            "--model-scope",
            "all-six",
            "--device",
            device,
            "--dry-run",
        ]
        dry_res = subprocess.run(dry_cmd, env=env, capture_output=True, text=True, check=False)
        assert dry_res.returncode == 0, dry_res.stderr
        lines = [line for line in dry_res.stdout.splitlines() if line.startswith("[dry-run]")]
        assert len(lines) == 12  # 6 models * 2 cases
        if device == "cpu":
            assert all("CUDA_VISIBLE_DEVICES=''" in ln for ln in lines)
        else:
            assert all("CUDA_VISIBLE_DEVICES=0" in ln for ln in lines)
