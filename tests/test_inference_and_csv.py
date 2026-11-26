"""Tests for standalone script CLIs, human-readable 22-col CSV, JSON diagnostics, and plain merge."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from utils.csv_io import RESULT_COLUMNS, write_model_csv
from utils.monitor import CaseProfiler
from utils.postprocess import finalize_and_save_predictions
from utils.storage import IndependentPredictionWriter, load_independent_prediction

DIAGNOSTIC_ONLY_COLUMNS = {
    "recist_id",
    "postprocess",
    "case_status",
    "prediction_status",
    "ground_truth_voxels",
    "predicted_voxels",
    "peak_cpu_utilization_pct",
    "gpu_monitor_status",
    "gpu_sanity_ok",
    "error_message",
}


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
    assert rec["per_lesion_postprocessing_time_s"]["1"] > 0.0
    assert rec["rows"][0]["ground_truth_recist_length_mm"] > 0.0
    assert rec["rows"][0]["predicted_recist_length_mm"] > 0.0
    assert rec["rows"][0]["recist_prompt_length_mm"] > 0.0
    assert np.isclose(
        rec["metrics"]["total_model_time_load_infer_post_s"],
        rec["metrics"]["model_loading_time_s"]
        + rec["metrics"]["model_inference_time_s"]
        + rec["metrics"]["postprocessing_time_s"],
        atol=1e-5,
    )
    # Full diagnostic fields remain preserved in the JSON receipt record
    assert rec["status"] == "ok"
    assert rec["postprocess"] == "moving_centroid"
    assert rec["rows"][0]["recist_id"] == 1
    assert rec["rows"][0]["gt_voxels"] > 0
    assert rec["rows"][0]["pred_voxels"] > 0
    assert "MaxCPU_Utilization_pct" in rec["metrics"]
    assert "AUC_RAM_Time_MB_s" in rec["metrics"]
    assert "AUC_GPU_Time_MB_s" in rec["metrics"]
    assert "gpu_monitor_status" in rec["metrics"]
    assert "gpu_sanity_ok" in rec["metrics"]


def _make_record(
    *,
    dataset: str,
    case_id: str,
    model_name: str,
    device: str,
    postprocess: str,
    load_s: float,
    lesions: list[tuple[int, float, float, float, int, float, float]],
    wall_s: float,
    ram_mb: float,
    gpu_mb: float | None,
    status: str = "ok",
    error: str = "",
) -> dict:
    # lesions: [(gt_id, gt_recist_mm, pred_recist_mm, dice, pred_voxels, inf_s, post_s)]
    inf_total = round(sum(x[5] for x in lesions), 6)
    post_total = round(sum(x[6] for x in lesions), 6)
    total_inf = round(load_s + inf_total + post_total, 6)
    return {
        "dataset": dataset,
        "case_id": case_id,
        "model_name": model_name,
        "device": device,
        "physical_gpu_index": 0 if device == "cuda" else None,
        "num_threads": 1,
        "postprocess": postprocess,
        "status": status,
        "error": error,
        "n_lesions": len(lesions),
        "shape_zyx": [8, 48, 48],
        "spacing_xyz": [1.0, 1.0, 2.0],
        "per_lesion_inference_time_s": {str(gid): round(inf_s, 6) for gid, _, _, _, _, inf_s, _ in lesions},
        "per_lesion_postprocessing_time_s": {str(gid): round(post_s, 6) for gid, _, _, _, _, _, post_s in lesions},
        "rows": [
            {
                "channel_index": i,
                "gt_id": gid,
                "recist_id": gid,
                "ground_truth_recist_length_mm": gt_len,
                "predicted_recist_length_mm": pred_len,
                "recist_prompt_length_mm": gt_len,
                "prediction_id": gid,
                "status": "predicted",
                "gt_voxels": 100,
                "pred_voxels": pvox,
                "intersection": int(round(dice * (100 + pvox) / 2.0)),
                "dice": dice,
            }
            for i, (gid, gt_len, pred_len, dice, pvox, _, _) in enumerate(lesions)
        ],
        "metrics": {
            "total_model_time_load_infer_post_s": total_inf,
            "model_loading_time_s": load_s,
            "model_inference_time_s": inf_total,
            "postprocessing_time_s": post_total,
            "data_loading_time_s": 0.05123,
            "image_preprocessing_time_s": 0.04123,
            "output_saving_time_s": 0.03123,
            "total_end_to_end_time_including_io_s": wall_s,
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


def test_single_csv_option_a_and_vertical_concat_merge_cli(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    eff_cpu_dir = tmp_path / "runs" / "EffMedSAM2_cpu"
    eff_gpu_dir = tmp_path / "runs" / "EffMedSAM2_cuda"
    med_dir = tmp_path / "runs" / "MedSAM2"
    nn_dir = tmp_path / "runs" / "nnInteractive_box"

    # CPU measured (0.0 VRAM), GPU multi-lesion (Option A on ram_auc & gpu_memory_auc), unavailable GPU (blank), failed
    eff_cpu_csv = write_model_csv(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="EffMedSAM2",
                device="cpu",
                postprocess="moving_centroid",
                load_s=0.41234,
                lesions=[(1, 22.456, 21.984, 0.881234, 102, 1.23456, 0.10004)],
                wall_s=1.82345,
                ram_mb=900.16,
                gpu_mb=0.0,
            ),
            _make_record(
                dataset="colon",
                case_id="case_02",
                model_name="EffMedSAM2",
                device="cpu",
                postprocess="moving_centroid",
                load_s=0.5,
                lesions=[(1, 18.2, 17.5, 0.85, 95, 1.1, 0.12), (2, 31.7, 30.2, 0.90, 104, 0.8, 0.08)],
                wall_s=2.72,
                ram_mb=950.0,
                gpu_mb=None,  # unavailable -> peak_gpu_memory_mb & gpu_memory_auc_mb_s must stay blank ""
            ),
            _make_record(
                dataset="colon",
                case_id="case_03_err",
                model_name="EffMedSAM2",
                device="cpu",
                postprocess="moving_centroid",
                load_s=0.2,
                lesions=[],
                wall_s=0.25,
                ram_mb=400.0,
                gpu_mb=0.0,
                status="failed(exception)",
                error="RuntimeError('bad input')",
            ),
        ],
        eff_cpu_dir,
    )
    assert eff_cpu_csv == eff_cpu_dir / "results.csv"
    assert not (eff_cpu_dir / "case_metrics.csv").exists()
    assert not (eff_cpu_dir / "lesion_scores.csv").exists()

    eff_gpu_csv = write_model_csv(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="EffMedSAM2",
                device="cuda",
                postprocess="moving_centroid",
                load_s=0.3,
                lesions=[(1, 22.456, 22.1, 0.89, 101, 0.2, 0.08), (2, 15.0, 14.5, 0.86, 80, 0.15, 0.06)],
                wall_s=0.90,
                ram_mb=1200.0,
                gpu_mb=2200.04,
            ),
        ],
        eff_gpu_dir,
    )
    write_model_csv(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="MedSAM2",
                device="cpu",
                postprocess="none",
                load_s=0.6,
                lesions=[(1, 22.456, 23.4, 0.84, 110, 2.4, 0.0)],
                wall_s=3.12,
                ram_mb=1500.0,
                gpu_mb=0.0,
            ),
        ],
        med_dir,
    )
    write_model_csv(
        [
            _make_record(
                dataset="colon",
                case_id="case_01",
                model_name="nnInteractive_box",
                device="cpu",
                postprocess="none",
                load_s=0.5,
                lesions=[(1, 22.456, 22.3, 0.91, 98, 3.5, 0.0)],
                wall_s=4.12,
                ram_mb=2100.0,
                gpu_mb=0.0,
            ),
        ],
        nn_dir,
    )

    with eff_cpu_csv.open() as f:
        reader = csv.DictReader(f)
        eff_hdr = reader.fieldnames or []
        eff_rows = list(reader)
    assert eff_hdr == RESULT_COLUMNS
    assert len(eff_hdr) == 22
    assert "ground_truth_recist_length_mm" in eff_hdr
    assert "predicted_recist_length_mm" in eff_hdr
    assert "gpu_memory_auc_mb_s" in eff_hdr
    assert not (set(eff_hdr) & DIAGNOSTIC_ONLY_COLUMNS)
    assert not any(
        c.startswith("post_")
        or c
        in (
            "total_runtime_s",
            "single_lesion_inference_time_s",
            "total_inference_time_s",
            "end_to_end_wall_time_including_io_s",
            "recist_length_mm",
        )
        for c in eff_hdr
    )
    assert len(eff_rows) == 4  # 1 (case_01) + 2 (case_02) + 1 (case_03_err fallback)

    # Single-lesion case: GT & predicted RECIST lengths, AUCs, 4-decimal timings, 1-decimal RAM/VRAM
    c1 = eff_rows[0]
    assert c1["ground_truth_recist_length_mm"] == "22.46"
    assert c1["predicted_recist_length_mm"] == "21.98"
    assert c1["dice_score"] == "0.8812"
    assert c1["model_loading_time_s"] == "0.4123"
    assert c1["model_inference_time_s"] == "1.2346"
    assert c1["postprocessing_time_s"] == "0.1"
    assert np.isclose(
        float(c1["total_model_time_load_infer_post_s"]),
        float(c1["model_loading_time_s"])
        + float(c1["model_inference_time_s"])
        + float(c1["postprocessing_time_s"]),
        atol=1e-9,
    )
    assert c1["peak_ram_mb"] == "900.2"
    assert c1["ram_auc_mb_s"] == str(round(900.16 * 1.82345, 1))
    assert c1["peak_gpu_memory_mb"] == "0.0"
    assert c1["gpu_memory_auc_mb_s"] == "0.0"

    # Unavailable GPU case (case_02): peak_gpu_memory_mb & gpu_memory_auc_mb_s stay blank "" (never 0.0)
    # Multi-lesion RAM AUC: both active lesions receive positive AUC and sum == case AUC total
    c2_l1, c2_l2 = eff_rows[1], eff_rows[2]
    assert c2_l1["ground_truth_recist_length_mm"] == "18.2"
    assert c2_l1["predicted_recist_length_mm"] == "17.5"
    assert c2_l2["ground_truth_recist_length_mm"] == "31.7"
    assert c2_l2["predicted_recist_length_mm"] == "30.2"
    assert c2_l1["peak_gpu_memory_mb"] == ""
    assert c2_l1["gpu_memory_auc_mb_s"] == ""
    assert c2_l2["peak_gpu_memory_mb"] == ""
    assert c2_l2["gpu_memory_auc_mb_s"] == ""
    assert float(c2_l1["ram_auc_mb_s"]) > 0.0
    assert float(c2_l2["ram_auc_mb_s"]) > 0.0
    assert np.isclose(
        float(c2_l1["ram_auc_mb_s"]) + float(c2_l2["ram_auc_mb_s"]),
        round(950.0 * 2.72, 1),
    )
    assert float(c2_l2["model_loading_time_s"]) == 0.0

    # CUDA multi-lesion case: both lesions have positive RAM & GPU AUC, sums equal case totals
    with eff_gpu_csv.open() as f:
        gpu_rows = list(csv.DictReader(f))
    assert gpu_rows[0]["peak_gpu_memory_mb"] == "2200.0"
    assert float(gpu_rows[0]["gpu_memory_auc_mb_s"]) > 0.0
    assert float(gpu_rows[1]["gpu_memory_auc_mb_s"]) > 0.0
    assert np.isclose(
        float(gpu_rows[0]["gpu_memory_auc_mb_s"]) + float(gpu_rows[1]["gpu_memory_auc_mb_s"]),
        round(2200.04 * 0.90, 1),
    )
    assert float(gpu_rows[0]["ram_auc_mb_s"]) > 0.0
    assert float(gpu_rows[1]["ram_auc_mb_s"]) > 0.0
    assert np.isclose(
        float(gpu_rows[0]["ram_auc_mb_s"]) + float(gpu_rows[1]["ram_auc_mb_s"]),
        round(1200.0 * 0.90, 1),
    )

    # Failed case fallback row
    c3 = eff_rows[3]
    assert c3["case_id"] == "case_03_err"
    assert c3["lesion_id"] == ""
    assert c3["ground_truth_recist_length_mm"] == ""
    assert c3["predicted_recist_length_mm"] == ""
    assert c3["ram_auc_mb_s"] == "100.0"
    assert c3["gpu_memory_auc_mb_s"] == "0.0"

    # Merge by plain vertical concatenation (no pivoting or prefix rewriting)
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

    merged_csv = merged_dir / "merged_results.csv"
    assert merged_csv.is_file()
    assert not (merged_dir / "merged_lesion_scores.csv").exists()
    assert not (merged_dir / "merged_case_metrics.csv").exists()
    assert not (merged_dir / "model_summary.csv").exists()

    with merged_csv.open() as f:
        mreader = csv.DictReader(f)
        mhdr = mreader.fieldnames or []
        mrows = list(mreader)

    # Exact same 22 columns as results.csv
    assert mhdr == RESULT_COLUMNS
    # 4 (eff_cpu) + 2 (eff_gpu) + 1 (med) + 1 (nn) = 8 concatenated rows in order
    assert len(mrows) == 8
    assert [(r["model_name"], r["device"], r["case_id"], r["lesion_id"]) for r in mrows] == [
        ("EffMedSAM2", "cpu", "case_01", "1"),
        ("EffMedSAM2", "cpu", "case_02", "1"),
        ("EffMedSAM2", "cpu", "case_02", "2"),
        ("EffMedSAM2", "cpu", "case_03_err", ""),
        ("EffMedSAM2", "cuda", "case_01", "1"),
        ("EffMedSAM2", "cuda", "case_01", "2"),
        ("MedSAM2", "cpu", "case_01", "1"),
        ("nnInteractive_box", "cpu", "case_01", "1"),
    ]
    assert mrows[4]["peak_gpu_memory_mb"] == "2200.0"
    assert mrows[6]["predicted_recist_length_mm"] == "23.4"
    assert mrows[7]["ground_truth_recist_length_mm"] == "22.46"

    # Also check direct .csv path via -o out/custom.csv
    custom_csv = tmp_path / "custom_out" / "compare.csv"
    res2 = subprocess.run(
        [
            sys.executable,
            str(repo_root / "merge_csvs.py"),
            "-i",
            str(eff_cpu_dir / "results.csv"),
            str(med_dir / "results.csv"),
            "-o",
            str(custom_csv),
        ],
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, res2.stderr
    assert custom_csv.is_file()


def test_multilesion_profiler_end_to_end_ram_auc_per_lesion_and_no_recist(tmp_path: Path):
    shape = (8, 48, 48)
    gt = np.zeros(shape, dtype=np.uint16)
    recist = np.zeros(shape, dtype=np.uint16)
    spacing = np.array([1.0, 1.0, 2.0], dtype=np.float32)

    # Lesion 1: GT without RECIST (no_recist -> active time & AUC must be 0.0, shared load goes to first prompted)
    gt[2:4, 2:8, 2:8] = 1
    # Lesion 2 & Lesion 3: GT + RECIST lines
    gt[2:5, 12:22, 12:22] = 2
    cv2.line(recist[3], (12, 17), (21, 17), 2, thickness=2)
    gt[2:5, 28:38, 28:38] = 3
    cv2.line(recist[3], (28, 33), (37, 33), 3, thickness=2)

    out_dir = tmp_path / "multi_eff"
    out_npz = out_dir / "predictions" / "multi.npz"

    with CaseProfiler(device="cpu", sample_interval=0.02) as prof:
        with prof.phase("model_loading_time"):
            time.sleep(0.03)
        with IndependentPredictionWriter(shape=shape, recist=recist, spacing=spacing, gt=gt, scratch=tmp_path) as writer:
            with prof.phase("model_inference_time"):
                for lab in (2, 3):
                    with prof.lesion_inference(lab):
                        writer.mark_attempted(lab)
                        pred = (gt == lab).astype(np.uint8)
                        writer.write_volume(lab, pred)
                        time.sleep(0.04)

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
                "case_id": "multi",
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

    csv_path = write_model_csv([rec], out_dir)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["lesion_id"] for r in rows] == ["1", "2", "3"]
    # Lesion 1 (no_recist) gets 0.0 for all timings and AUC; Lesion 2 & Lesion 3 both get positive RAM AUC
    r1, r2, r3 = rows
    assert float(r1["model_loading_time_s"]) == 0.0
    assert float(r1["ram_auc_mb_s"]) == 0.0
    assert float(r2["model_loading_time_s"]) > 0.0
    assert float(r3["model_loading_time_s"]) == 0.0
    assert float(r2["ram_auc_mb_s"]) > 0.0
    assert float(r3["ram_auc_mb_s"]) > 0.0
    assert np.isclose(
        float(r1["ram_auc_mb_s"]) + float(r2["ram_auc_mb_s"]) + float(r3["ram_auc_mb_s"]),
        round(rec["metrics"]["AUC_RAM_Time_MB_s"], 1),
        atol=0.11,
    )
