"""Per-model CSV reporting and wide multi-model CSV merging."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

CASE_COLUMNS = [
    "dataset",
    "case_id",
    "model_name",
    "device",
    "physical_gpu_index",
    "num_threads",
    "postprocess",
    "status",
    "error",
    "n_lesions",
    "shape_zyx",
    "spacing_xyz",
    "mean_dice",
    "total_runtime_s",
    "model_loading_time_s",
    "model_inference_time_s",
    "postprocessing_time_s",
    "data_loading_time_s",
    "image_preprocessing_time_s",
    "output_saving_time_s",
    "end_to_end_wall_time_including_io_s",
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
    "gpu_monitor_status",
    "gpu_sanity_ok",
]

LESION_COLUMNS = [
    "dataset",
    "case_id",
    "gt_id",
    "recist_id",
    "gt_voxels",
    "gt_volume_ml",
    "model_name",
    "device",
    "postprocess",
    "prediction_status",
    "pred_voxels",
    "pred_volume_ml",
    "intersection",
    "dice",
    "single_lesion_inference_time_s",
    "model_loading_time_s",
    "model_inference_time_s",
    "postprocessing_time_s",
    "total_runtime_s",
    "end_to_end_wall_time_including_io_s",
    "MaxRAM_MB",
    "MaxGPU_Mem_MB",
    "gpu_monitor_status",
    "gpu_sanity_ok",
]

MERGED_CASE_SHARED = ["dataset", "case_id", "n_lesions", "shape_zyx", "spacing_xyz"]
MERGED_CASE_MODEL_FIELDS = [
    "device",
    "status",
    "mean_dice",
    "total_runtime_s",
    "model_loading_time_s",
    "model_inference_time_s",
    "postprocessing_time_s",
    "data_loading_time_s",
    "image_preprocessing_time_s",
    "output_saving_time_s",
    "end_to_end_wall_time_including_io_s",
    "MaxRAM_MB",
    "AUC_RAM_Time_MB_s",
    "MaxCPU_Utilization_pct",
    "MaxGPU_Mem_MB",
    "RankGPU_Mem_MB",
    "gpu_monitor_status",
    "gpu_sanity_ok",
]

MERGED_LESION_SHARED = ["dataset", "case_id", "gt_id", "recist_id", "gt_voxels", "gt_volume_ml"]
MERGED_LESION_MODEL_FIELDS = [
    "prediction_status",
    "pred_voxels",
    "pred_volume_ml",
    "intersection",
    "dice",
    "single_lesion_inference_time_s",
    "model_loading_time_s",
    "model_inference_time_s",
    "postprocessing_time_s",
    "total_runtime_s",
    "end_to_end_wall_time_including_io_s",
    "MaxRAM_MB",
    "MaxGPU_Mem_MB",
]

SUMMARY_COLUMNS = [
    "model_name",
    "device",
    "cases_total",
    "cases_ok",
    "cases_failed",
    "lesions_total",
    "mean_dice",
    "median_dice",
    "std_dice",
    "mean_total_runtime_s",
    "mean_model_loading_time_s",
    "mean_model_inference_time_s",
    "mean_postprocessing_time_s",
    "mean_end_to_end_wall_time_including_io_s",
    "mean_MaxRAM_MB",
    "max_MaxRAM_MB",
    "mean_AUC_RAM_Time_MB_s",
    "mean_MaxCPU_Utilization_pct",
    "mean_MaxGPU_Mem_MB",
    "mean_AUC_GPU_Time_MB_s",
    "mean_RankGPU_Mem_MB",
    "gpu_monitor_statuses",
    "all_gpu_sanity_ok",
]


def _fmt_opt(val: Any) -> Any:
    return "" if val is None else val


def _voxel_volume_ml(spacing_xyz: list[float] | np.ndarray) -> float:
    sp = np.asarray(spacing_xyz, dtype=np.float64).ravel()
    if len(sp) != 3:
        return float("nan")
    return float(sp[0] * sp[1] * sp[2] / 1000.0)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def records_to_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert per-case JSON records into flat case and lesion CSV row dicts."""
    case_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []

    for record in records:
        metrics = dict(record.get("metrics", {}))
        status = record.get("status", "ok")
        model_name = record.get("model_name") or record.get("model", "")
        dataset = record.get("dataset", "")
        case_id = record.get("case_id", "")
        device = record.get("device", "")
        postprocess = record.get("postprocess", "none")
        spacing = record.get("spacing_xyz", [])
        rows = list(record.get("rows", []))
        lesion_times = record.get("per_lesion_inference_time_s", {})
        voxel_ml = _voxel_volume_ml(spacing) if spacing else float("nan")
        dices = [float(r["dice"]) for r in rows if r.get("dice") is not None]

        case_row: dict[str, Any] = {
            "dataset": dataset,
            "case_id": case_id,
            "model_name": model_name,
            "device": device,
            "physical_gpu_index": _fmt_opt(record.get("physical_gpu_index")),
            "num_threads": _fmt_opt(record.get("num_threads")),
            "postprocess": postprocess,
            "status": status,
            "error": record.get("error", ""),
            "n_lesions": record.get("n_lesions", len(rows)),
            "shape_zyx": json.dumps(record.get("shape_zyx", [])) if record.get("shape_zyx") else "",
            "spacing_xyz": json.dumps(spacing) if spacing else "",
            "mean_dice": round(mean(dices), 6) if dices else "",
        }
        for col in CASE_COLUMNS:
            if col not in case_row:
                case_row[col] = _fmt_opt(metrics.get(col))
        case_rows.append(case_row)

        for r in rows:
            gid = int(r["gt_id"])
            gt_vox = r.get("gt_voxels")
            pred_vox = r.get("pred_voxels")
            gt_vol_ml = round(gt_vox * voxel_ml, 6) if (gt_vox is not None and np.isfinite(voxel_ml)) else ""
            pred_vol_ml = round(pred_vox * voxel_ml, 6) if (pred_vox is not None and np.isfinite(voxel_ml)) else ""
            lesion_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "gt_id": gid,
                    "recist_id": _fmt_opt(r.get("recist_id")),
                    "gt_voxels": _fmt_opt(gt_vox),
                    "gt_volume_ml": gt_vol_ml,
                    "model_name": model_name,
                    "device": device,
                    "postprocess": postprocess,
                    "prediction_status": r.get("status", ""),
                    "pred_voxels": _fmt_opt(pred_vox),
                    "pred_volume_ml": pred_vol_ml,
                    "intersection": _fmt_opt(r.get("intersection")),
                    "dice": round(float(r["dice"]), 6) if r.get("dice") is not None else "",
                    "single_lesion_inference_time_s": lesion_times.get(str(gid), ""),
                    "model_loading_time_s": _fmt_opt(metrics.get("model_loading_time_s")),
                    "model_inference_time_s": _fmt_opt(metrics.get("model_inference_time_s")),
                    "postprocessing_time_s": _fmt_opt(metrics.get("postprocessing_time_s")),
                    "total_runtime_s": _fmt_opt(metrics.get("total_runtime_s")),
                    "end_to_end_wall_time_including_io_s": _fmt_opt(
                        metrics.get("end_to_end_wall_time_including_io_s")
                    ),
                    "MaxRAM_MB": _fmt_opt(metrics.get("MaxRAM_MB")),
                    "MaxGPU_Mem_MB": _fmt_opt(metrics.get("MaxGPU_Mem_MB")),
                    "gpu_monitor_status": metrics.get("gpu_monitor_status", ""),
                    "gpu_sanity_ok": _fmt_opt(metrics.get("gpu_sanity_ok")),
                }
            )

    return case_rows, lesion_rows


def write_model_csvs(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Write independent per-model ``case_metrics.csv`` and ``lesion_scores.csv``."""
    out_dir = Path(output_dir).resolve()
    case_rows, lesion_rows = records_to_rows(records)
    case_csv = out_dir / "case_metrics.csv"
    lesion_csv = out_dir / "lesion_scores.csv"
    _write_csv(case_csv, CASE_COLUMNS, case_rows)
    _write_csv(lesion_csv, LESION_COLUMNS, lesion_rows)
    return {"case_metrics": case_csv, "lesion_scores": lesion_csv}


def _load_input_csvs(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_case_rows: list[dict[str, Any]] = []
    all_lesion_rows: list[dict[str, Any]] = []
    seen_files: set[Path] = set()

    for raw_p in inputs:
        p = Path(raw_p).resolve()
        if p.is_dir():
            for fname, target in (("case_metrics.csv", all_case_rows), ("lesion_scores.csv", all_lesion_rows)):
                csv_path = p / fname
                if csv_path.is_file() and csv_path not in seen_files:
                    seen_files.add(csv_path)
                    _, rows = _read_csv(csv_path)
                    for idx, row in enumerate(rows):
                        row["_source_order"] = f"{p.name}:{idx}"
                    target.extend(rows)
        elif p.is_file():
            if p in seen_files:
                continue
            seen_files.add(p)
            cols, rows = _read_csv(p)
            if "gt_id" in cols:
                all_lesion_rows.extend(rows)
            else:
                all_case_rows.extend(rows)
        else:
            raise FileNotFoundError(f"Merge input does not exist: {raw_p}")

    return all_case_rows, all_lesion_rows


def _assign_group_prefixes(groups: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Choose clean column prefixes for ``(model_name, device)`` groups."""
    by_name: dict[str, int] = {}
    for mname, _ in groups:
        by_name[mname] = by_name.get(mname, 0) + 1

    prefixes: dict[tuple[str, str], str] = {}
    used: set[str] = set()
    for key in groups:
        mname, device = key
        base = mname if by_name[mname] == 1 else (f"{mname}_{device}" if device else mname)
        cand = base
        idx = 2
        while cand in used:
            cand = f"{base}_{idx}"
            idx += 1
        used.add(cand)
        prefixes[key] = cand
    return prefixes


def _lesion_sort_key(k: tuple[str, str, str]) -> tuple[str, str, int, str]:
    ds, cid, gid = k
    try:
        return (ds, cid, int(gid), gid)
    except ValueError:
        return (ds, cid, 0, gid)


def _merge_wide_table(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    shared_fields: list[str],
    model_fields: list[str],
    sort_fn,
) -> tuple[list[str], list[dict[str, Any]]]:
    groups: list[tuple[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    for r in rows:
        grp = (str(r.get("model_name", "") or "model"), str(r.get("device", "")))
        if grp not in seen_groups:
            seen_groups.add(grp)
            groups.append(grp)
    prefixes = _assign_group_prefixes(groups)

    bucket_shared: dict[tuple[str, ...], dict[str, Any]] = {}
    bucket_models: dict[tuple[str, ...], dict[tuple[str, str], dict[str, Any]]] = {}
    for r in rows:
        key = tuple(str(r.get(f, "")) for f in key_fields)
        grp = (str(r.get("model_name", "") or "model"), str(r.get("device", "")))
        sh = bucket_shared.setdefault(key, {})
        for sf in shared_fields:
            if not sh.get(sf) and r.get(sf, "") not in ("", None):
                sh[sf] = r[sf]
        bucket_models.setdefault(key, {})[grp] = r

    columns = list(shared_fields)
    for grp in groups:
        pref = prefixes[grp]
        for mf in model_fields:
            columns.append(f"{pref}_{mf}")

    out_rows: list[dict[str, Any]] = []
    for key in sorted(bucket_shared, key=sort_fn):
        row: dict[str, Any] = {sf: bucket_shared[key].get(sf, "") for sf in shared_fields}
        for grp in groups:
            pref = prefixes[grp]
            src = bucket_models.get(key, {}).get(grp, {})
            for mf in model_fields:
                row[f"{pref}_{mf}"] = src.get(mf, "")
        out_rows.append(row)
    return columns, out_rows


def _build_summary_rows(
    case_rows: list[dict[str, Any]],
    lesion_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in case_rows:
        grp = (str(r.get("model_name", "")), str(r.get("device", "")))
        if grp not in buckets:
            buckets[grp] = []
            groups.append(grp)
        buckets[grp].append(r)

    summary_rows: list[dict[str, Any]] = []
    for model_name, device in sorted(groups):
        rows_group = buckets[(model_name, device)]
        ok_rows = [r for r in rows_group if r.get("status") == "ok"]
        model_lesions = [
            float(lr["dice"])
            for lr in lesion_rows
            if str(lr.get("model_name", "")) == model_name
            and str(lr.get("device", "")) == device
            and lr.get("dice", "") != ""
        ]

        def _vals(col: str) -> list[float]:
            return [float(r[col]) for r in ok_rows if r.get(col, "") not in ("", None)]

        rt_vals = _vals("total_runtime_s")
        load_vals = _vals("model_loading_time_s")
        inf_vals = _vals("model_inference_time_s")
        post_vals = _vals("postprocessing_time_s")
        wall_vals = _vals("end_to_end_wall_time_including_io_s")
        ram_vals = _vals("MaxRAM_MB")
        auc_ram_vals = _vals("AUC_RAM_Time_MB_s")
        cpu_vals = _vals("MaxCPU_Utilization_pct")
        gpu_vals = _vals("MaxGPU_Mem_MB")
        auc_gpu_vals = _vals("AUC_GPU_Time_MB_s")
        rank_gpu_vals = _vals("RankGPU_Mem_MB")

        raw_sanity = [
            r.get("gpu_sanity_ok", "")
            for r in ok_rows
            if r.get("gpu_sanity_ok", "") not in ("", None)
        ]
        norm_sanity = [
            v if isinstance(v, bool) else str(v).strip().lower() == "true"
            for v in raw_sanity
        ]
        if not ok_rows or not norm_sanity:
            all_sanity: Any = ""
        elif any(not v for v in norm_sanity):
            all_sanity = False
        elif len(norm_sanity) < len(ok_rows):
            all_sanity = ""
        else:
            all_sanity = True

        statuses_seen = sorted(
            {str(r["gpu_monitor_status"]) for r in rows_group if r.get("gpu_monitor_status")}
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "device": device,
                "cases_total": len(rows_group),
                "cases_ok": len(ok_rows),
                "cases_failed": len(rows_group) - len(ok_rows),
                "lesions_total": len(
                    [
                        lr
                        for lr in lesion_rows
                        if str(lr.get("model_name", "")) == model_name
                        and str(lr.get("device", "")) == device
                    ]
                ),
                "mean_dice": round(mean(model_lesions), 6) if model_lesions else "",
                "median_dice": round(median(model_lesions), 6) if model_lesions else "",
                "std_dice": (
                    round(pstdev(model_lesions), 6)
                    if len(model_lesions) > 1
                    else ("0.0" if model_lesions else "")
                ),
                "mean_total_runtime_s": round(mean(rt_vals), 4) if rt_vals else "",
                "mean_model_loading_time_s": round(mean(load_vals), 4) if load_vals else "",
                "mean_model_inference_time_s": round(mean(inf_vals), 4) if inf_vals else "",
                "mean_postprocessing_time_s": round(mean(post_vals), 4) if post_vals else "",
                "mean_end_to_end_wall_time_including_io_s": round(mean(wall_vals), 4) if wall_vals else "",
                "mean_MaxRAM_MB": round(mean(ram_vals), 2) if ram_vals else "",
                "max_MaxRAM_MB": round(max(ram_vals), 2) if ram_vals else "",
                "mean_AUC_RAM_Time_MB_s": round(mean(auc_ram_vals), 2) if auc_ram_vals else "",
                "mean_MaxCPU_Utilization_pct": round(mean(cpu_vals), 2) if cpu_vals else "",
                "mean_MaxGPU_Mem_MB": round(mean(gpu_vals), 2) if gpu_vals else "",
                "mean_AUC_GPU_Time_MB_s": round(mean(auc_gpu_vals), 2) if auc_gpu_vals else "",
                "mean_RankGPU_Mem_MB": round(mean(rank_gpu_vals), 2) if rank_gpu_vals else "",
                "gpu_monitor_statuses": ";".join(statuses_seen),
                "all_gpu_sanity_ok": all_sanity,
            }
        )
    return summary_rows


def merge_model_csvs(
    input_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Merge independent per-model CSVs horizontally (with ``<ModelName>_`` prefixes) + summary."""
    out_dir = Path(output_dir).resolve()
    case_rows, lesion_rows = _load_input_csvs([Path(p) for p in input_paths])
    if not case_rows and not lesion_rows:
        raise FileNotFoundError("No case_metrics.csv or lesion_scores.csv rows found in inputs")

    written: dict[str, Path] = {}
    if case_rows:
        case_cols, merged_cases = _merge_wide_table(
            case_rows,
            key_fields=("dataset", "case_id"),
            shared_fields=MERGED_CASE_SHARED,
            model_fields=MERGED_CASE_MODEL_FIELDS,
            sort_fn=lambda k: k,
        )
        merged_case_csv = out_dir / "merged_case_metrics.csv"
        _write_csv(merged_case_csv, case_cols, merged_cases)
        written["merged_case_metrics"] = merged_case_csv

        summary_csv = out_dir / "model_summary.csv"
        _write_csv(summary_csv, SUMMARY_COLUMNS, _build_summary_rows(case_rows, lesion_rows))
        written["model_summary"] = summary_csv

    if lesion_rows:
        lesion_cols, merged_lesions = _merge_wide_table(
            lesion_rows,
            key_fields=("dataset", "case_id", "gt_id"),
            shared_fields=MERGED_LESION_SHARED,
            model_fields=MERGED_LESION_MODEL_FIELDS,
            sort_fn=_lesion_sort_key,
        )
        merged_lesion_csv = out_dir / "merged_lesion_scores.csv"
        _write_csv(merged_lesion_csv, lesion_cols, merged_lesions)
        written["merged_lesion_scores"] = merged_lesion_csv

    return written
