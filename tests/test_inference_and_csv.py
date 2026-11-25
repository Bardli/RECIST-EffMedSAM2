"""Tests for standalone script CLIs, final-only postprocessing, and wide model-prefixed CSV merging."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from utils.csv_io import CASE_COLUMNS, LESION_COLUMNS, write_model_csvs
from utils.monitor import CaseProfiler
from utils.postprocess import finalize_and_save_predictions
from utils.storage import IndependentPredictionWriter, load_independent_prediction


def test_standalone_scripts_help_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script in (
        "infer_effmedsam2.py",
        "infer_medsam2.py",
        "infer_nninteractive.py",
        "merge_csvs.py",
    ):
        res = subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"{script} --help failed: {res.stderr}"

    syntax = subprocess.run(
        ["bash", "-n", str(repo_root / "scripts" / "download_weights.sh")],
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_final_only_postprocessing_saves_single_npz_and_no_intermediate_dirs(tmp_path: Path):
    shape = (8, 48, 48)
    gt = np.zeros(shape, dtype=np.uint16)
    recist = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.ogrid[:48, :48]
    gt[3:6] = np.where((yy - 22) ** 2 + (xx - 22) ** 2 <= 4.5**2, 1, 0)
    cv2.line(recist[4], (16, 22), (28, 22), 1, thickness=2)
    spacing = np.array([1.0, 1.0, 2.5], dtype=np.float32)

    out_dir = tmp_path / "EffMedSAM2"
    out_npz = out_dir / "predictions" / "c1.npz"

    with CaseProfiler(device="cpu", sample_interval=0.02) as prof:
        with prof.phase("model_loading_time"):
            time.sleep(0.02)
        with IndependentPredictionWriter(shape=shape, recist=recist, spacing=spacing, gt=gt, scratch=tmp_path) as writer:
            with prof.phase("model_inference_time"):
                writer.mark_attempted(1)
                pred = np.zeros(shape, dtype=np.uint8)
                pred[3:6, 17:27, 17:27] = 1
                pred[4, 0, 0] = 1  # far false positive removed by moving_centroid
                writer.write_volume(1, pred)
                prof.record_lesion_time(1, 0.01)

            receipt = finalize_and_save_predictions(
                writer,
                out_npz,
                profiler=prof,
                postprocess="moving_centroid",
                scratch=tmp_path,
            )

        rec = prof.finalize(
            extra_metadata={
                "dataset": "colon",
                "case_id": "c1",
                "model_name": "EffMedSAM2",
                "device": "cpu",
                "postprocess": "moving_centroid",
                "num_threads": 1,
                "n_lesions": len(receipt["instance_ids"]),
                "shape_zyx": list(shape),
                "spacing_xyz": spacing.tolist(),
                "rows": receipt["rows"],
                "status": "ok",
                "error": "",
            }
        )

    assert out_npz.is_file()
    assert not (out_dir / "predictions_postprocessed").exists()
    loaded = load_independent_prediction(out_npz)
    assert loaded["segs"][0, 4, 0, 0] == 0
    assert rec["metrics"]["postprocessing_time_s"] > 0.0
    assert np.isclose(
        rec["metrics"]["total_runtime_s"],
        rec["metrics"]["model_loading_time_s"]
        + rec["metrics"]["model_inference_time_s"]
        + rec["metrics"]["postprocessing_time_s"],
        atol=1e-5,
    )


def _make_record(
    *,
    dataset: str,
    case_id: str,
    model_name: str,
    device: str,
    postprocess: str,
    load_s: float,
    inf_s: float,
    post_s: float,
    wall_s: float,
    dice1: float,
    pred_vox1: int,
    ram_mb: float,
    gpu_mb: float | None,
) -> dict:
    total_rt = load_s + inf_s + post_s
    return {
        "dataset": dataset,
        "case_id": case_id,
        "model_name": model_name,
        "device": device,
        "physical_gpu_index": 0 if device == "cuda" else None,
        "num_threads": 1,
        "postprocess": postprocess,
        "status": "ok",
        "error": "",
        "n_lesions": 1,
        "shape_zyx": [8, 48, 48],
        "spacing_xyz": [1.0, 1.0, 2.0],
        "per_lesion_inference_time_s": {"1": round(inf_s, 6)},
        "rows": [
            {
                "channel_index": 0,
                "gt_id": 1,
                "recist_id": 1,
                "prediction_id": 1,
                "status": "predicted",
                "gt_voxels": 100,
                "pred_voxels": pred_vox1,
                "intersection": int(round(dice1 * (100 + pred_vox1) / 2.0)),
                "dice": dice1,
            }
        ],
        "metrics": {
            "total_runtime_s": round(total_rt, 6),
            "model_loading_time_s": load_s,
            "model_inference_time_s": inf_s,
            "postprocessing_time_s": post_s,
            "data_loading_time_s": 0.05,
            "image_preprocessing_time_s": 0.04,
            "output_saving_time_s": 0.03,
            "end_to_end_wall_time_including_io_s": wall_s,
            "MaxRAM_MB": ram_mb,
            "ram_baseline_mb": 100.0,
            "ram_inference_peak_mb": ram_mb - 10.0,
            "ram_delta_mb": ram_mb - 100.0,
            "AUC_RAM_Time_MB_s": ram_mb * wall_s,
            "MaxCPU_Utilization_pct": 85.0,
            "AUC_CPU_Time_pct_s": 85.0 * wall_s,
            "MaxGPU_Mem_MB": gpu_mb,
            "gpu_nvml_peak_mb": gpu_mb,
            "AUC_GPU_Time_MB_s": (gpu_mb * wall_s) if gpu_mb is not None else None,
            "RankGPU_Mem_MB": max(0.0, gpu_mb - 4096.0) if gpu_mb is not None else None,
            "gpu_torch_peak_allocated_mb": (gpu_mb * 0.7) if (device == "cuda" and gpu_mb is not None) else 0.0,
            "gpu_torch_peak_reserved_mb": gpu_mb if (device == "cuda" and gpu_mb is not None) else 0.0,
            "gpu_monitor_status": "measured" if gpu_mb is not None else "unavailable",
            "gpu_sanity_ok": True if gpu_mb == 0.0 or device == "cuda" else None,
        },
    }


def test_per_model_csvs_and_wide_merge_cli(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    eff_cpu_dir = tmp_path / "runs" / "EffMedSAM2_cpu"
    eff_gpu_dir = tmp_path / "runs" / "EffMedSAM2_cuda"
    med_dir = tmp_path / "runs" / "MedSAM2"
    nn_dir = tmp_path / "runs" / "nnInteractive_box"

    eff_cpu_csvs = write_model_csvs(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="EffMedSAM2",
                device="cpu",
                postprocess="moving_centroid",
                load_s=0.4,
                inf_s=1.2,
                post_s=0.1,
                wall_s=1.82,
                dice1=0.88,
                pred_vox1=102,
                ram_mb=900.0,
                gpu_mb=0.0,
            )
        ],
        eff_cpu_dir,
    )
    write_model_csvs(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="EffMedSAM2",
                device="cuda",
                postprocess="moving_centroid",
                load_s=0.3,
                inf_s=0.2,
                post_s=0.08,
                wall_s=0.70,
                dice1=0.89,
                pred_vox1=101,
                ram_mb=1200.0,
                gpu_mb=2200.0,
            )
        ],
        eff_gpu_dir,
    )
    write_model_csvs(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="MedSAM2",
                device="cpu",
                postprocess="none",
                load_s=0.6,
                inf_s=2.4,
                post_s=0.0,
                wall_s=3.12,
                dice1=0.84,
                pred_vox1=110,
                ram_mb=1500.0,
                gpu_mb=0.0,
            )
        ],
        med_dir,
    )
    write_model_csvs(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="nnInteractive_box",
                device="cpu",
                postprocess="none",
                load_s=0.5,
                inf_s=3.5,
                post_s=0.0,
                wall_s=4.12,
                dice1=0.91,
                pred_vox1=98,
                ram_mb=2100.0,
                gpu_mb=0.0,
            )
        ],
        nn_dir,
    )

    # Verify per-model CSV headers exclude post_* intermediates and use clear time names
    with eff_cpu_csvs["case_metrics"].open() as f:
        case_hdr = csv.DictReader(f).fieldnames or []
    assert case_hdr == CASE_COLUMNS
    assert not any(c.startswith("post_") or c == "mean_post_dice" for c in case_hdr)
    assert "total_runtime_s" in case_hdr
    assert "model_loading_time_s" in case_hdr
    assert "model_inference_time_s" in case_hdr
    assert "end_to_end_wall_time_including_io_s" in case_hdr

    with eff_cpu_csvs["lesion_scores"].open() as f:
        lesion_hdr = csv.DictReader(f).fieldnames or []
    assert lesion_hdr == LESION_COLUMNS
    assert not any(c.startswith("post_") for c in lesion_hdr)

    # Run merge_csvs.py without PYTHONPATH
    merged_dir = tmp_path / "runs" / "merged"
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    merge_res = subprocess.run(
        [
            sys.executable,
            str(repo_root / "merge_csvs.py"),
            "-i",
            str(eff_cpu_dir),
            str(eff_gpu_dir),
            str(med_dir),
            str(nn_dir),
            "-o",
            str(merged_dir),
        ],
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge_res.returncode == 0, merge_res.stderr

    merged_lesions_csv = merged_dir / "merged_lesion_scores.csv"
    merged_cases_csv = merged_dir / "merged_case_metrics.csv"
    summary_csv = merged_dir / "model_summary.csv"
    assert merged_lesions_csv.is_file()
    assert merged_cases_csv.is_file()
    assert summary_csv.is_file()

    with merged_lesions_csv.open() as f:
        lrows = list(csv.DictReader(f))
    assert len(lrows) == 1
    row = lrows[0]
    assert row["case_id"] == "case_01"
    assert row["gt_volume_ml"] == "0.2"
    # Duplicate model_name on cpu/cuda auto-disambiguates; unique models use <ModelName>_
    assert row["EffMedSAM2_cpu_dice"] == "0.88"
    assert row["EffMedSAM2_cuda_dice"] == "0.89"
    assert row["MedSAM2_dice"] == "0.84"
    assert row["nnInteractive_box_dice"] == "0.91"
    assert float(row["EffMedSAM2_cpu_pred_volume_ml"]) > 0.0
    assert float(row["MedSAM2_pred_volume_ml"]) > 0.0
    assert float(row["nnInteractive_box_single_lesion_inference_time_s"]) == 3.5
    assert np.isclose(float(row["EffMedSAM2_cpu_total_runtime_s"]), 1.7)
    assert np.isclose(float(row["MedSAM2_end_to_end_wall_time_including_io_s"]), 3.12)
    assert not any("_post_" in k for k in row)

    with merged_cases_csv.open() as f:
        crows = list(csv.DictReader(f))
    assert len(crows) == 1
    assert crows[0]["EffMedSAM2_cpu_mean_dice"] == "0.88"
    assert crows[0]["nnInteractive_box_mean_dice"] == "0.91"
    assert float(crows[0]["EffMedSAM2_cuda_MaxGPU_Mem_MB"]) == 2200.0

    with summary_csv.open() as f:
        srows = list(csv.DictReader(f))
    assert len(srows) == 4
    assert "mean_total_runtime_s" in srows[0]
    assert "mean_model_loading_time_s" in srows[0]
    assert "mean_model_inference_time_s" in srows[0]
    assert "mean_end_to_end_wall_time_including_io_s" in srows[0]
