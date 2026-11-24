"""End-to-end subprocess benchmark runner, failure capture, GPU mapping, and default layout tests."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import cv2
import numpy as np

from recist_effmedsam2.storage import load_independent_prediction


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
    yy, xx = np.ogrid[:48, :48]
    gt[3:6] = np.where((yy - 22) ** 2 + (xx - 22) ** 2 <= 4.5**2, 1, gt[3:6])
    cv2.line(recist[4], (16, 22), (28, 22), 1, thickness=2)

    gt[2:5] = np.where((yy - 34) ** 2 + (xx - 34) ** 2 <= 3.0**2, 3, gt[2:5])
    cv2.line(recist[3], (30, 34), (38, 34), 3, thickness=2)

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


def test_end_to_end_cpu_benchmark_with_inprofiler_postprocess(tmp_path: Path):
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

    # Check postprocessed NPZ schema and in-profiler postprocess phase for efftiny
    eff_rec = json.loads((out_dir / "efftiny_m8_ep40_box" / "metrics" / "case_001.metrics.json").read_text())
    assert "postprocess" in eff_rec["phase_spans"]
    assert eff_rec["metrics"]["postprocess_s"] > 0.0
    post_npz_path = out_dir / "efftiny_m8_ep40_box" / "predictions_postprocessed" / "case_001.npz"
    assert post_npz_path.is_file()
    loaded_post = load_independent_prediction(post_npz_path)
    assert loaded_post["segs"].shape[0] == 2

    with case_csv.open() as f:
        case_rows = list(csv.DictReader(f))
    assert len(case_rows) == 5  # 5 paper configs * 1 case
    for r in case_rows:
        assert r["dataset"] == "colon_cancer"
        assert r["case_id"] == "case_001"
        assert r["device"] == "cpu"
        assert r["num_threads"] == "1"
        assert r["status"] == "ok"
        assert r["monitor_source"] == "worker"
        assert float(r["MaxRAM_MB"]) > 0.0
        assert np.isclose(
            float(r["runtime_s"]),
            float(r["model_load_s"]) + float(r["inference_s"]) + float(r["postprocess_s"]),
            atol=1e-4,
        )
        assert float(r["wall_total_s"]) >= float(r["runtime_s"])
        assert float(r["mean_dice"]) > 0.0
        if r["gpu_monitor_status"] == "measured":
            assert r["gpu_sanity_ok"] == "True"
            assert float(r["MaxGPU_Mem_MB"]) == 0.0
            assert float(r["gpu_nvml_peak_mb"]) == 0.0

    eff_row = next(r for r in case_rows if r["model"] == "efftiny_m8_ep40_box")
    assert eff_row["mean_post_dice"] != ""
    assert float(eff_row["mean_post_dice"]) > float(eff_row["mean_dice"])
    assert float(eff_row["postprocess_s"]) > 0.0
    assert loaded_post["segs"][0, 4, 0, 0] == 0
    assert eff_rec["postprocessed_rows"][0]["post_removed_voxels"] > 0

    with lesion_csv.open() as f:
        lesion_rows = list(csv.DictReader(f))
    assert len(lesion_rows) == 10  # 5 models * 2 lesions
    for lr in lesion_rows:
        assert float(lr["gt_volume_ml"]) > 0.0
        assert float(lr["pred_volume_ml"]) > 0.0
        assert lr["postprocess_s"] != ""
        assert lr["wall_total_s"] != ""
        assert lr["score_protocol"] == "independent_same_id_raw_v2"

    # Check plot artifacts exist
    assert (out_dir / "efftiny_m8_ep40_box" / "plots" / "case_001_RAM-Time.png").is_file()

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


def test_failure_resource_capture_exception_and_sigkill(tmp_path: Path):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir(parents=True)
    _make_synthetic_case(in_dir / "case_fail.npz")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    # 1. Python exception inside worker: stale 'ok' JSON must be overwritten with failed(1)
    out_exc = tmp_path / "run_exc"
    stale_dir = out_exc / "efftiny_m8_ep40_box" / "metrics"
    stale_dir.mkdir(parents=True)
    (stale_dir / "case_fail.metrics.json").write_text(json.dumps({"status": "ok"}))

    cmd_exc = [
        sys.executable,
        "-m",
        "recist_effmedsam2.runner.run_benchmark",
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_exc),
        "--models",
        "efftiny_m8_ep40_box",
        "--device",
        "cpu",
        "--sample-interval",
        "0.02",
        "--backend-override",
        "mock",
        "--mock-fail-mode",
        "exception",
        "--overwrite",
    ]
    res_exc = subprocess.run(cmd_exc, env=env, capture_output=True, text=True, check=False)
    assert res_exc.returncode == 1
    rec_exc = json.loads((stale_dir / "case_fail.metrics.json").read_text())
    assert rec_exc["status"] == "failed(1)"
    assert "Simulated mock inference exception" in rec_exc["error"]
    assert rec_exc["metrics"]["MaxRAM_MB"] > 0.0

    # 2. Hard SIGKILL (simulates Linux OOM killer): parent ProcessTreeMonitor must capture RAM + wall time
    out_kill = tmp_path / "run_kill"
    cmd_kill = [
        sys.executable,
        "-m",
        "recist_effmedsam2.runner.run_benchmark",
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_kill),
        "--models",
        "efftiny_m8_ep40_box",
        "--device",
        "cpu",
        "--sample-interval",
        "0.02",
        "--backend-override",
        "mock",
        "--mock-fail-mode",
        "sigkill",
        "--plots",
    ]
    res_kill = subprocess.run(cmd_kill, env=env, capture_output=True, text=True, check=False)
    assert res_kill.returncode == 1
    rec_kill = json.loads(
        (out_kill / "efftiny_m8_ep40_box" / "metrics" / "case_fail.metrics.json").read_text()
    )
    assert rec_kill["status"] == "failed(-9)"
    assert rec_kill["monitor_source"] == "parent_process_monitor"
    assert rec_kill["metrics"]["MaxRAM_MB"] > 0.0
    assert rec_kill["metrics"]["wall_total_s"] > 0.0
    assert (out_kill / "efftiny_m8_ep40_box" / "plots" / "case_fail_RAM-Time.png").is_file()

    with (out_kill / "reports" / "case_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed(-9)"
    assert rows[0]["monitor_source"] == "parent_process_monitor"
    assert float(rows[0]["MaxRAM_MB"]) > 0.0


def test_non_zero_gpu_dispatch_and_default_layout(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    # Set up default repo-relative layout in tmp_path
    (tmp_path / "data" / "inputs").mkdir(parents=True)
    (tmp_path / "data" / "window_twins").mkdir(parents=True)
    _make_synthetic_case(
        tmp_path / "data" / "inputs" / "c1.npz",
        tmp_path / "data" / "window_twins" / "c1.npz",
    )

    # Create fake default .venv-sam and .venv-nn python stubs to verify auto-detection
    for vname in (".venv-sam", ".venv-nn"):
        py_stub = tmp_path / vname / "bin" / "python"
        py_stub.parent.mkdir(parents=True)
        py_stub.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} \"$@\"\n")
        py_stub.chmod(py_stub.stat().st_mode | stat.S_IEXEC)

    # Zero-flag default run (using default data/inputs, data/window_twins, runs/benchmark, .venv-sam/.venv-nn)
    res_default = subprocess.run(
        [
            sys.executable,
            "-m",
            "recist_effmedsam2.runner.run_benchmark",
            "--models-config",
            str(repo_root / "configs" / "models.json"),
            "--models",
            "efftiny_m8_ep40_box",
            "nninteractive_box_hu",
            "--backend-override",
            "mock",
            "--sample-interval",
            "0.02",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_default.returncode == 0, res_default.stdout + "\n" + res_default.stderr
    manifest = json.loads((tmp_path / "runs" / "benchmark" / "run_manifest.json").read_text())
    assert manifest["sam_python"] == str((tmp_path / ".venv-sam" / "bin" / "python").resolve()) or manifest["sam_python"].endswith(".venv-sam/bin/python")
    assert manifest["nn_python"].endswith(".venv-nn/bin/python")
    assert Path(manifest["window_dir"]).name == "window_twins"
    assert (tmp_path / "runs" / "benchmark" / "reports" / "case_metrics.csv").is_file()

    # Non-0 physical GPU dispatch (--device cuda --gpu-index 2) passes --gpu-index 2 + CUDA_VISIBLE_DEVICES=2
    dry_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "recist_effmedsam2.runner.run_benchmark",
            "--models-config",
            str(repo_root / "configs" / "models.json"),
            "--model-scope",
            "all-six",
            "--device",
            "cuda",
            "--gpu-index",
            "2",
            "--dry-run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_res.returncode == 0, dry_res.stderr
    lines = [ln for ln in dry_res.stdout.splitlines() if ln.startswith("[dry-run]")]
    assert len(lines) == 6
    assert all("CUDA_VISIBLE_DEVICES=2" in ln for ln in lines)
    assert all("--device cuda:0" in ln for ln in lines)
    assert all("--gpu-index 2" in ln for ln in lines)
