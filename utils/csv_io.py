"""Single-file per-model CSV reporting (``results.csv``) and wide comparison merge (``merged_results.csv``)."""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

RESULT_COLUMNS = [
    "dataset",
    "case_id",
    "lesion_id",
    "model_name",
    "device",
    "dice_score",
    "ground_truth_volume_ml",
    "predicted_volume_ml",
    "total_model_time_load_infer_post_s",
    "model_loading_time_s",
    "model_inference_time_s",
    "postprocessing_time_s",
    "data_loading_time_s",
    "image_preprocessing_time_s",
    "output_saving_time_s",
    "total_end_to_end_time_including_io_s",
    "peak_ram_mb",
    "peak_gpu_memory_mb",
]

MERGED_SHARED_COLUMNS = [
    "dataset",
    "case_id",
    "lesion_id",
    "ground_truth_volume_ml",
]

MERGED_METRIC_COLUMNS = [
    "dice_score",
    "predicted_volume_ml",
    "total_model_time_load_infer_post_s",
    "peak_ram_mb",
    "peak_gpu_memory_mb",
]


def _fmt_float(val: Any, decimals: int) -> Any:
    if val is None or val == "":
        return ""
    return round(float(val), decimals)


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _lookup_lesion_time(
    time_map: dict[str, Any],
    recist_id: Any,
    lesion_id: int,
    case_total_s: float,
    is_first: bool,
) -> float:
    if not time_map:
        return round(case_total_s, 4) if is_first else 0.0
    candidates: list[str] = []
    if recist_id not in (None, ""):
        candidates.append(str(recist_id))
    candidates.append(str(lesion_id))
    for key in candidates:
        if key in time_map and time_map[key] not in ("", None):
            return round(float(time_map[key]), 4)
    return 0.0


def records_to_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten case records into human-readable per-lesion CSV rows using Option A shared timings."""
    out_rows: list[dict[str, Any]] = []

    for record in records:
        metrics = dict(record.get("metrics", {}))
        model_name = record.get("model_name") or record.get("model", "")
        dataset = record.get("dataset", "")
        case_id = record.get("case_id", "")
        device = record.get("device", "")
        spacing = record.get("spacing_xyz", [])
        rows = list(record.get("rows", []))
        lesion_inf_times = dict(record.get("per_lesion_inference_time_s", {}))
        lesion_post_times = dict(record.get("per_lesion_postprocessing_time_s", {}))
        voxel_ml = _voxel_volume_ml(spacing) if spacing else float("nan")

        case_load_s = float(metrics.get("model_loading_time_s") or 0.0)
        case_inf_s = float(metrics.get("model_inference_time_s") or 0.0)
        case_post_s = float(metrics.get("postprocessing_time_s") or 0.0)
        case_data_s = float(metrics.get("data_loading_time_s") or 0.0)
        case_pre_s = float(metrics.get("image_preprocessing_time_s") or 0.0)
        case_save_s = float(metrics.get("output_saving_time_s") or 0.0)
        case_wall_s = float(metrics.get("total_end_to_end_time_including_io_s") or 0.0)

        peak_ram_mb = _fmt_float(metrics.get("MaxRAM_MB"), 1)
        peak_gpu_mb = _fmt_float(metrics.get("MaxGPU_Mem_MB"), 1)

        if not rows:
            c_load = round(case_load_s, 4)
            c_inf = round(case_inf_s, 4)
            c_post = round(case_post_s, 4)
            out_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "lesion_id": "",
                    "model_name": model_name,
                    "device": device,
                    "dice_score": "",
                    "ground_truth_volume_ml": "",
                    "predicted_volume_ml": "",
                    "total_model_time_load_infer_post_s": round(c_load + c_inf + c_post, 4),
                    "model_loading_time_s": c_load,
                    "model_inference_time_s": c_inf,
                    "postprocessing_time_s": c_post,
                    "data_loading_time_s": round(case_data_s, 4),
                    "image_preprocessing_time_s": round(case_pre_s, 4),
                    "output_saving_time_s": round(case_save_s, 4),
                    "total_end_to_end_time_including_io_s": round(case_wall_s, 4),
                    "peak_ram_mb": peak_ram_mb,
                    "peak_gpu_memory_mb": peak_gpu_mb,
                }
            )
            continue

        single_lesion_case = len(rows) == 1
        for row_idx, r in enumerate(rows):
            is_first = row_idx == 0
            gid = int(r["gt_id"])
            rid = r.get("recist_id")
            gt_vox = r.get("gt_voxels")
            pred_vox = r.get("pred_voxels")
            gt_vol_ml = round(gt_vox * voxel_ml, 4) if (gt_vox is not None and np.isfinite(voxel_ml)) else ""
            pred_vol_ml = round(pred_vox * voxel_ml, 4) if (pred_vox is not None and np.isfinite(voxel_ml)) else ""

            row_load_s = round(case_load_s, 4) if is_first else 0.0
            row_data_s = round(case_data_s, 4) if is_first else 0.0
            row_pre_s = round(case_pre_s, 4) if is_first else 0.0
            row_save_s = round(case_save_s, 4) if is_first else 0.0
            row_wall_s = round(case_wall_s, 4) if is_first else 0.0

            if single_lesion_case:
                row_inf_s = round(case_inf_s, 4)
                row_post_s = round(case_post_s, 4)
            else:
                row_inf_s = _lookup_lesion_time(lesion_inf_times, rid, gid, case_inf_s, is_first)
                row_post_s = _lookup_lesion_time(lesion_post_times, rid, gid, case_post_s, is_first)

            row_total_model_s = round(row_load_s + row_inf_s + row_post_s, 4)

            out_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "lesion_id": gid,
                    "model_name": model_name,
                    "device": device,
                    "dice_score": round(float(r["dice"]), 4) if r.get("dice") is not None else "",
                    "ground_truth_volume_ml": gt_vol_ml,
                    "predicted_volume_ml": pred_vol_ml,
                    "total_model_time_load_infer_post_s": row_total_model_s,
                    "model_loading_time_s": row_load_s,
                    "model_inference_time_s": row_inf_s,
                    "postprocessing_time_s": row_post_s,
                    "data_loading_time_s": row_data_s,
                    "image_preprocessing_time_s": row_pre_s,
                    "output_saving_time_s": row_save_s,
                    "total_end_to_end_time_including_io_s": row_wall_s,
                    "peak_ram_mb": peak_ram_mb,
                    "peak_gpu_memory_mb": peak_gpu_mb,
                }
            )

    return out_rows


def write_model_csv(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Write a single human-readable per-model ``results.csv`` (one row per RECIST lesion)."""
    out_dir = Path(output_dir).resolve()
    csv_path = out_dir / "results.csv"
    _write_csv(csv_path, RESULT_COLUMNS, records_to_rows(records))
    return csv_path


def _load_result_csvs(inputs: list[Path]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    seen_files: set[Path] = set()

    for raw_p in inputs:
        p = Path(raw_p).resolve()
        if p.is_dir():
            csv_path = p / "results.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"results.csv not found in model directory: {p}")
        elif p.is_file():
            csv_path = p
        else:
            raise FileNotFoundError(f"Merge input does not exist: {raw_p}")

        if csv_path not in seen_files:
            seen_files.add(csv_path)
            all_rows.extend(_read_csv(csv_path))

    return all_rows


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


def _lesion_sort_key(k: tuple[str, str, str]) -> tuple[str, str, int, int, str]:
    ds, cid, lid = k
    if lid == "":
        return (ds, cid, 1, 0, lid)
    try:
        return (ds, cid, 0, int(lid), lid)
    except ValueError:
        return (ds, cid, 0, 0, lid)


def _summarize_groups(rows: list[dict[str, Any]], groups: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Compute compact per-model console summary stats without writing extra CSV files."""
    prefixes = _assign_group_prefixes(groups)
    summaries: list[dict[str, Any]] = []
    for grp in groups:
        mname, device = grp
        grp_rows = [
            r
            for r in rows
            if (str(r.get("model_name", "") or "model"), str(r.get("device", ""))) == grp
        ]
        case_keys = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in grp_rows}
        dices = [float(r["dice_score"]) for r in grp_rows if r.get("dice_score", "") not in ("", None)]
        timed_rows = [
            r for r in grp_rows if r.get("total_model_time_load_infer_post_s", "") not in ("", None)
        ]
        timed_cases = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in timed_rows}
        tot_times = [float(r["total_model_time_load_infer_post_s"]) for r in timed_rows]
        rams = [float(r["peak_ram_mb"]) for r in grp_rows if r.get("peak_ram_mb", "") not in ("", None)]
        gpus = [float(r["peak_gpu_memory_mb"]) for r in grp_rows if r.get("peak_gpu_memory_mb", "") not in ("", None)]
        summaries.append(
            {
                "column_prefix": prefixes[grp],
                "model_name": mname,
                "device": device,
                "cases": len(case_keys),
                "lesions": sum(1 for r in grp_rows if str(r.get("lesion_id", "")) != ""),
                "mean_dice_score": round(mean(dices), 4) if dices else None,
                "mean_case_total_model_time_load_infer_post_s": (
                    round(sum(tot_times) / len(timed_cases), 4) if tot_times and timed_cases else None
                ),
                "max_peak_ram_mb": round(max(rams), 1) if rams else None,
                "max_peak_gpu_memory_mb": round(max(gpus), 1) if gpus else None,
            }
        )
    return summaries


def merge_model_csvs(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Merge per-model ``results.csv`` files into one metric-grouped wide comparison CSV."""
    rows = _load_result_csvs([Path(p) for p in input_paths])
    if not rows:
        raise FileNotFoundError("No rows found in input results.csv files")

    groups: list[tuple[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    for r in rows:
        grp = (str(r.get("model_name", "") or "model"), str(r.get("device", "")))
        if grp not in seen_groups:
            seen_groups.add(grp)
            groups.append(grp)
    prefixes = _assign_group_prefixes(groups)

    key_fields = ("dataset", "case_id", "lesion_id")
    bucket_shared: dict[tuple[str, str, str], dict[str, Any]] = {}
    bucket_models: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = {}

    for r in rows:
        key = (str(r.get(key_fields[0], "")), str(r.get(key_fields[1], "")), str(r.get(key_fields[2], "")))
        grp = (str(r.get("model_name", "") or "model"), str(r.get("device", "")))
        sh = bucket_shared.setdefault(key, {})
        for sf in MERGED_SHARED_COLUMNS:
            if not sh.get(sf) and r.get(sf, "") not in ("", None):
                sh[sf] = r[sf]
        bucket_models.setdefault(key, {})[grp] = r

    # Group columns by metric across models so models sit side-by-side for direct comparison.
    columns = list(MERGED_SHARED_COLUMNS)
    for metric in MERGED_METRIC_COLUMNS:
        for grp in groups:
            columns.append(f"{prefixes[grp]}_{metric}")

    merged_rows: list[dict[str, Any]] = []
    for key in sorted(bucket_shared, key=_lesion_sort_key):
        row: dict[str, Any] = {sf: bucket_shared[key].get(sf, "") for sf in MERGED_SHARED_COLUMNS}
        for metric in MERGED_METRIC_COLUMNS:
            for grp in groups:
                src = bucket_models.get(key, {}).get(grp, {})
                row[f"{prefixes[grp]}_{metric}"] = src.get(metric, "")
        merged_rows.append(row)

    out = Path(output_path).resolve()
    merged_csv = out if out.suffix.lower() == ".csv" else (out / "merged_results.csv")
    _write_csv(merged_csv, columns, merged_rows)
    return merged_csv, _summarize_groups(rows, groups)
