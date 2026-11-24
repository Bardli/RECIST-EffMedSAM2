"""Compute independent same-ID Dice, predicted volumes (mL), and efficiency CSVs.

Reads per-case ``metrics/*.metrics.json`` (and optionally recomputes or fills
GT-matched Dice/volumes directly from ``predictions/*.npz`` and source NPZs).
Writes:
  - ``reports/case_metrics.csv``
  - ``reports/lesion_scores.csv``
  - ``reports/model_summary.csv``
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from ..profiling.monitor import DEFAULT_GPU_TOLERANCE_MB, ranking_gpu_memory
from ..storage import load_independent_prediction

CASE_COLUMNS = [
    "dataset",
    "case_id",
    "model",
    "device",
    "num_threads",
    "status",
    "n_lesions",
    "shape_zyx",
    "spacing_xyz",
    "runtime_s",
    "model_load_s",
    "inference_s",
    "data_load_s",
    "preprocess_s",
    "postprocess_s",
    "save_s",
    "wall_total_s",
    "MaxRAM_MB",
    "ram_baseline_mb",
    "ram_inference_peak_mb",
    "ram_delta_mb",
    "AUC_RAM_Time_MB_s",
    "MaxCPU_Utilization_pct",
    "AUC_CPU_Time_pct_s",
    "MaxGPU_Mem_MB",
    "gpu_nvml_peak_mb",
    "AUC_GPU_Time_MB_s",
    "RankGPU_Mem_MB",
    "gpu_torch_peak_allocated_mb",
    "gpu_torch_peak_reserved_mb",
    "gpu_sanity_ok",
    "mean_dice",
    "mean_post_dice",
    "source_sha256",
    "prediction_sha256",
]

LESION_COLUMNS = [
    "dataset",
    "case_id",
    "model",
    "device",
    "gt_id",
    "recist_id",
    "prediction_id",
    "channel_index",
    "status",
    "gt_voxels",
    "pred_voxels",
    "intersection",
    "dice",
    "gt_volume_ml",
    "pred_volume_ml",
    "post_pred_voxels",
    "post_intersection",
    "post_dice",
    "post_pred_volume_ml",
    "post_action",
    "lesion_inference_s",
    "runtime_s",
    "model_load_s",
    "inference_s",
    "MaxRAM_MB",
    "MaxGPU_Mem_MB",
    "gpu_sanity_ok",
    "score_protocol",
]


def _voxel_volume_ml(spacing_xyz: list[float] | np.ndarray) -> float:
    sp = np.asarray(spacing_xyz, dtype=np.float64).ravel()
    if len(sp) != 3:
        return float("nan")
    return float(sp[0] * sp[1] * sp[2] / 1000.0)


def _recompute_rows_from_npz(
    pred_npz: Path,
    source_npz: Path,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Independently verify/compute same-ID Dice from saved ``[N, Z, Y, X]`` NPZ and source GT."""
    pred = load_independent_prediction(pred_npz)
    with np.load(source_npz, allow_pickle=False) as src:
        gt = None
        for key in ("gts", "gts_hidden"):
            if key in src.files:
                gt = np.asarray(src[key])
                break
        spacing = [float(s) for s in src["spacing"]]
    segs = pred["segs"]
    ids = pred["instance_ids"]
    statuses = pred["prediction_status"]
    rows: list[dict[str, Any]] = []
    for k, lab in enumerate(ids):
        mask = segs[k] > 0
        pred_voxels = int(mask.sum())
        if gt is not None:
            g = gt == lab
            ng = int(g.sum())
            ni = int((g & mask).sum())
            denom = ng + pred_voxels
            dice = (2.0 * ni / denom) if denom > 0 else 1.0
        else:
            ng = None
            ni = None
            dice = None
        rows.append(
            dict(
                channel_index=k,
                gt_id=int(lab),
                recist_id=int(lab) if statuses[k] != "no_recist" else None,
                prediction_id=int(lab),
                status=statuses[k],
                gt_voxels=ng,
                pred_voxels=pred_voxels,
                intersection=ni,
                dice=dice,
                score_protocol="independent_same_id_raw_v2",
            )
        )
    return rows, spacing


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(
    run_dir: Path,
    *,
    models: list[str] | None = None,
    gpu_tolerance_mb: float | None = None,
    verify_from_npz: bool = False,
) -> dict[str, Path]:
    """Aggregate all per-case records under ``run_dir`` into report CSVs."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    skip_dirs = {"reports", "logs", "scratch"}
    model_dirs = sorted(
        p
        for p in run_dir.iterdir()
        if p.is_dir() and p.name not in skip_dirs and (p / "metrics").is_dir()
    )
    if models:
        wanted = set(models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    case_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []
    summary_buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for model_dir in model_dirs:
        for mf in sorted((model_dir / "metrics").glob("*.json")):
            record = json.loads(mf.read_text())
            metrics = dict(record.get("metrics", {}))
            if gpu_tolerance_mb is not None and "MaxGPU_Mem_MB" in metrics:
                metrics["RankGPU_Mem_MB"] = ranking_gpu_memory(
                    metrics["MaxGPU_Mem_MB"],
                    gpu_tolerance_mb,
                )

            status = record.get("status", "ok")
            model_name = record.get("model", model_dir.name)
            dataset = record.get("dataset", "")
            case_id = record.get("case_id", mf.stem.replace(".metrics", ""))
            device = record.get("device", "")
            spacing = record.get("spacing_xyz", [])

            rows = list(record.get("rows", []))
            if (
                verify_from_npz
                and status == "ok"
                and record.get("prediction_npz")
                and record.get("source_npz")
            ):
                rows, spacing = _recompute_rows_from_npz(
                    Path(record["prediction_npz"]),
                    Path(record["source_npz"]),
                )

            post_by_id = {
                int(r["gt_id"]): r for r in record.get("postprocessed_rows", [])
            }
            lesion_times = record.get("per_lesion_inference_s", {})
            voxel_ml = _voxel_volume_ml(spacing) if spacing else float("nan")

            dices = [float(r["dice"]) for r in rows if r.get("dice") is not None]
            post_dices = [
                float(post_by_id[int(r["gt_id"])]["post_dice"])
                for r in rows
                if int(r["gt_id"]) in post_by_id
                and post_by_id[int(r["gt_id"])].get("post_dice") is not None
            ]

            case_row: dict[str, Any] = {
                "dataset": dataset,
                "case_id": case_id,
                "model": model_name,
                "device": device,
                "num_threads": record.get("num_threads", ""),
                "status": status,
                "n_lesions": record.get("n_lesions", len(rows)),
                "shape_zyx": json.dumps(record.get("shape_zyx", [])),
                "spacing_xyz": json.dumps(spacing),
                "mean_dice": round(mean(dices), 6) if dices else "",
                "mean_post_dice": round(mean(post_dices), 6) if post_dices else "",
                "source_sha256": record.get("source_sha256", ""),
                "prediction_sha256": record.get("prediction_sha256", ""),
            }
            for col in CASE_COLUMNS:
                if col not in case_row:
                    case_row[col] = metrics.get(col, "")
            case_rows.append(case_row)
            summary_buckets.setdefault((model_name, device), []).append(case_row)

            for r in rows:
                gid = int(r["gt_id"])
                pp = post_by_id.get(gid, {})
                gt_vox = r.get("gt_voxels")
                pred_vox = r.get("pred_voxels")
                pp_vox = pp.get("post_pred_voxels")
                gt_vol_ml = (
                    round(gt_vox * voxel_ml, 6)
                    if (gt_vox is not None and np.isfinite(voxel_ml))
                    else ""
                )
                pred_vol_ml = (
                    round(pred_vox * voxel_ml, 6)
                    if (pred_vox is not None and np.isfinite(voxel_ml))
                    else ""
                )
                pp_vol_ml = (
                    round(pp_vox * voxel_ml, 6)
                    if (pp_vox is not None and np.isfinite(voxel_ml))
                    else ""
                )
                lesion_rows.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "model": model_name,
                        "device": device,
                        "gt_id": gid,
                        "recist_id": r.get("recist_id", ""),
                        "prediction_id": r.get("prediction_id", gid),
                        "channel_index": r.get("channel_index", ""),
                        "status": r.get("status", ""),
                        "gt_voxels": gt_vox if gt_vox is not None else "",
                        "pred_voxels": pred_vox if pred_vox is not None else "",
                        "intersection": r.get("intersection", "") if r.get("intersection") is not None else "",
                        "dice": round(float(r["dice"]), 6) if r.get("dice") is not None else "",
                        "gt_volume_ml": gt_vol_ml,
                        "pred_volume_ml": pred_vol_ml,
                        "post_pred_voxels": pp_vox if pp_vox is not None else "",
                        "post_intersection": pp.get("post_intersection", "") if pp.get("post_intersection") is not None else "",
                        "post_dice": (
                            round(float(pp["post_dice"]), 6)
                            if pp.get("post_dice") is not None
                            else ""
                        ),
                        "post_pred_volume_ml": pp_vol_ml,
                        "post_action": pp.get("post_action", ""),
                        "lesion_inference_s": lesion_times.get(str(gid), ""),
                        "runtime_s": metrics.get("runtime_s", ""),
                        "model_load_s": metrics.get("model_load_s", ""),
                        "inference_s": metrics.get("inference_s", ""),
                        "MaxRAM_MB": metrics.get("MaxRAM_MB", ""),
                        "MaxGPU_Mem_MB": metrics.get("MaxGPU_Mem_MB", ""),
                        "gpu_sanity_ok": metrics.get("gpu_sanity_ok", ""),
                        "score_protocol": r.get("score_protocol", "independent_same_id_raw_v2"),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for (model_name, device), rows_group in sorted(summary_buckets.items()):
        ok_rows = [r for r in rows_group if r["status"] == "ok"]
        model_lesions = [
            float(lr["dice"])
            for lr in lesion_rows
            if lr["model"] == model_name and lr["device"] == device and lr["dice"] != ""
        ]

        def _vals(col: str) -> list[float]:
            return [float(r[col]) for r in ok_rows if r.get(col, "") != ""]

        ram_vals = _vals("MaxRAM_MB")
        gpu_vals = _vals("MaxGPU_Mem_MB")
        rt_vals = _vals("runtime_s")
        load_vals = _vals("model_load_s")
        inf_vals = _vals("inference_s")
        auc_ram_vals = _vals("AUC_RAM_Time_MB_s")
        cpu_vals = _vals("MaxCPU_Utilization_pct")
        auc_gpu_vals = _vals("AUC_GPU_Time_MB_s")
        rank_gpu_vals = _vals("RankGPU_Mem_MB")
        sanity_vals = [bool(r["gpu_sanity_ok"]) for r in ok_rows if r.get("gpu_sanity_ok") != ""]

        summary_rows.append(
            {
                "model": model_name,
                "device": device,
                "cases_total": len(rows_group),
                "cases_ok": len(ok_rows),
                "cases_failed": len(rows_group) - len(ok_rows),
                "lesions_total": len(
                    [lr for lr in lesion_rows if lr["model"] == model_name and lr["device"] == device]
                ),
                "mean_dice": round(mean(model_lesions), 6) if model_lesions else "",
                "median_dice": round(median(model_lesions), 6) if model_lesions else "",
                "std_dice": round(pstdev(model_lesions), 6) if len(model_lesions) > 1 else ("0.0" if model_lesions else ""),
                "mean_runtime_s": round(mean(rt_vals), 4) if rt_vals else "",
                "mean_model_load_s": round(mean(load_vals), 4) if load_vals else "",
                "mean_inference_s": round(mean(inf_vals), 4) if inf_vals else "",
                "mean_MaxRAM_MB": round(mean(ram_vals), 2) if ram_vals else "",
                "max_MaxRAM_MB": round(max(ram_vals), 2) if ram_vals else "",
                "mean_AUC_RAM_Time_MB_s": round(mean(auc_ram_vals), 2) if auc_ram_vals else "",
                "mean_MaxCPU_Utilization_pct": round(mean(cpu_vals), 2) if cpu_vals else "",
                "mean_MaxGPU_Mem_MB": round(mean(gpu_vals), 2) if gpu_vals else "",
                "mean_AUC_GPU_Time_MB_s": round(mean(auc_gpu_vals), 2) if auc_gpu_vals else "",
                "mean_RankGPU_Mem_MB": round(mean(rank_gpu_vals), 2) if rank_gpu_vals else "",
                "all_gpu_sanity_ok": all(sanity_vals) if sanity_vals else "",
            }
        )

    reports_dir = run_dir / "reports"
    case_csv = reports_dir / "case_metrics.csv"
    lesion_csv = reports_dir / "lesion_scores.csv"
    summary_csv = reports_dir / "model_summary.csv"

    _write_csv(case_csv, CASE_COLUMNS, case_rows)
    _write_csv(lesion_csv, LESION_COLUMNS, lesion_rows)
    summary_cols = (
        list(summary_rows[0].keys())
        if summary_rows
        else [
            "model",
            "device",
            "cases_total",
            "cases_ok",
            "cases_failed",
            "lesions_total",
            "mean_dice",
            "median_dice",
            "std_dice",
            "mean_runtime_s",
            "mean_model_load_s",
            "mean_inference_s",
            "mean_MaxRAM_MB",
            "max_MaxRAM_MB",
            "mean_AUC_RAM_Time_MB_s",
            "mean_MaxCPU_Utilization_pct",
            "mean_MaxGPU_Mem_MB",
            "mean_AUC_GPU_Time_MB_s",
            "mean_RankGPU_Mem_MB",
            "all_gpu_sanity_ok",
        ]
    )
    _write_csv(summary_csv, summary_cols, summary_rows)
    print(f"Wrote {case_csv} ({len(case_rows)} case(s))")
    print(f"Wrote {lesion_csv} ({len(lesion_rows)} lesion(s))")
    print(f"Wrote {summary_csv} ({len(summary_rows)} model group(s))")
    return {
        "case_metrics": case_csv,
        "lesion_scores": lesion_csv,
        "model_summary": summary_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="Benchmark output root")
    ap.add_argument("--models", nargs="+", default=None, help="Optional subset of model folders")
    ap.add_argument(
        "--gpu-tolerance-mb",
        type=float,
        default=None,
        help=f"Override RankGPU_Mem allowance in MiB (FLARE default: {DEFAULT_GPU_TOLERANCE_MB:.0f})",
    )
    ap.add_argument(
        "--verify-from-npz",
        action="store_true",
        help="Recompute Dice/counts directly from saved prediction and source NPZ files",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    evaluate_run(
        args.run_dir,
        models=args.models,
        gpu_tolerance_mb=args.gpu_tolerance_mb,
        verify_from_npz=args.verify_from_npz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
