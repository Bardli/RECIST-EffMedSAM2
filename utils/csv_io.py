"""Single-file per-model CSV reporting (``results.csv``) and plain vertical merge (``merged_results.csv``)."""
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
    "ground_truth_recist_length_mm",
    "predicted_recist_length_mm",
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
    "ram_auc_mb_s",
    "peak_gpu_memory_mb",
    "gpu_memory_auc_mb_s",
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


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _lookup_lesion_float(
    val_map: dict[str, Any],
    recist_id: Any,
    lesion_id: int,
) -> float | None:
    if not val_map:
        return None
    candidates: list[str] = []
    if recist_id not in (None, ""):
        candidates.append(str(recist_id))
    candidates.append(str(lesion_id))
    for key in candidates:
        if key in val_map and val_map[key] not in ("", None):
            return float(val_map[key])
    return None


def _lookup_lesion_time(
    time_map: dict[str, Any],
    recist_id: Any,
    lesion_id: int,
    case_total_s: float,
    is_first: bool,
) -> float:
    if not time_map:
        return round(case_total_s, 4) if is_first else 0.0
    found = _lookup_lesion_float(time_map, recist_id, lesion_id)
    return round(found, 4) if found is not None else 0.0


def _is_prompted_row(row: dict[str, Any]) -> bool:
    if row.get("recist_id") not in (None, ""):
        return True
    status = str(row.get("status", ""))
    return bool(status) and status != "no_recist"


def _allocate_case_aucs(
    rows: list[dict[str, Any]],
    case_auc: Any,
    per_lesion_auc_map: dict[str, Any],
    row_active_durations: list[float],
    case_wall_s: float,
    first_idx: int,
) -> list[Any]:
    """Allocate span-integrated AUC per lesion so each active lesion gets its real AUC without double-counting."""
    if case_auc is None:
        return [""] * len(rows)
    total_rounded = round(float(case_auc), 1)
    if len(rows) == 1:
        return [total_rounded]

    alloc: list[float] = [0.0] * len(rows)
    other_sum = 0.0
    for i, r in enumerate(rows):
        if i == first_idx:
            continue
        rid = r.get("recist_id")
        gid = int(r["gt_id"])
        span_auc = _lookup_lesion_float(per_lesion_auc_map, rid, gid)
        if span_auc is not None:
            val = round(span_auc, 1)
        elif not per_lesion_auc_map and _is_prompted_row(r) and case_wall_s > 0.0 and row_active_durations[i] > 0.0:
            val = round(float(case_auc) * (row_active_durations[i] / case_wall_s), 1)
        else:
            val = 0.0
        alloc[i] = val
        other_sum += val

    alloc[first_idx] = round(max(0.0, total_rounded - other_sum), 1)
    return alloc


def records_to_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten case records into human-readable per-lesion CSV rows without double-counting."""
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
        lesion_ram_aucs = dict(record.get("per_lesion_ram_auc_mb_s", {}))
        lesion_gpu_aucs = dict(record.get("per_lesion_gpu_auc_mb_s", {}))
        voxel_ml = _voxel_volume_ml(spacing) if spacing else float("nan")

        case_load_s = float(metrics.get("model_loading_time_s") or 0.0)
        case_inf_s = float(metrics.get("model_inference_time_s") or 0.0)
        case_post_s = float(metrics.get("postprocessing_time_s") or 0.0)
        case_data_s = float(metrics.get("data_loading_time_s") or 0.0)
        case_pre_s = float(metrics.get("image_preprocessing_time_s") or 0.0)
        case_save_s = float(metrics.get("output_saving_time_s") or 0.0)
        case_wall_s = float(metrics.get("total_end_to_end_time_including_io_s") or 0.0)

        peak_ram_mb = _fmt_float(metrics.get("MaxRAM_MB"), 1)
        case_auc_ram = metrics.get("AUC_RAM_Time_MB_s")
        peak_gpu_mb = _fmt_float(metrics.get("MaxGPU_Mem_MB"), 1)
        case_auc_gpu = metrics.get("AUC_GPU_Time_MB_s")

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
                    "ground_truth_recist_length_mm": "",
                    "predicted_recist_length_mm": "",
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
                    "ram_auc_mb_s": _fmt_float(case_auc_ram, 1),
                    "peak_gpu_memory_mb": peak_gpu_mb,
                    "gpu_memory_auc_mb_s": _fmt_float(case_auc_gpu, 1),
                }
            )
            continue

        first_idx = next((i for i, r in enumerate(rows) if _is_prompted_row(r)), 0)
        single_lesion_case = len(rows) == 1

        row_inf_list: list[float] = []
        row_post_list: list[float] = []
        row_active_durations: list[float] = []
        for row_idx, r in enumerate(rows):
            is_first = row_idx == first_idx
            gid = int(r["gt_id"])
            rid = r.get("recist_id")
            if single_lesion_case:
                r_inf = round(case_inf_s, 4)
                r_post = round(case_post_s, 4)
            else:
                r_inf = _lookup_lesion_time(lesion_inf_times, rid, gid, case_inf_s, is_first)
                r_post = _lookup_lesion_time(lesion_post_times, rid, gid, case_post_s, is_first)
            row_inf_list.append(r_inf)
            row_post_list.append(r_post)
            row_active_durations.append(r_inf + r_post)

        ram_aucs = _allocate_case_aucs(
            rows, case_auc_ram, lesion_ram_aucs, row_active_durations, case_wall_s, first_idx
        )
        gpu_aucs = _allocate_case_aucs(
            rows, case_auc_gpu, lesion_gpu_aucs, row_active_durations, case_wall_s, first_idx
        )

        for row_idx, r in enumerate(rows):
            is_first = row_idx == first_idx
            gid = int(r["gt_id"])
            gt_vox = r.get("gt_voxels")
            pred_vox = r.get("pred_voxels")
            gt_vol_ml = round(gt_vox * voxel_ml, 4) if (gt_vox is not None and np.isfinite(voxel_ml)) else ""
            pred_vol_ml = round(pred_vox * voxel_ml, 4) if (pred_vox is not None and np.isfinite(voxel_ml)) else ""

            row_load_s = round(case_load_s, 4) if is_first else 0.0
            row_data_s = round(case_data_s, 4) if is_first else 0.0
            row_pre_s = round(case_pre_s, 4) if is_first else 0.0
            row_save_s = round(case_save_s, 4) if is_first else 0.0
            row_wall_s = round(case_wall_s, 4) if is_first else 0.0
            row_inf_s = row_inf_list[row_idx]
            row_post_s = row_post_list[row_idx]
            row_total_model_s = round(row_load_s + row_inf_s + row_post_s, 4)

            out_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "lesion_id": gid,
                    "model_name": model_name,
                    "device": device,
                    "dice_score": round(float(r["dice"]), 4) if r.get("dice") is not None else "",
                    "ground_truth_recist_length_mm": _fmt_float(r.get("ground_truth_recist_length_mm"), 2),
                    "predicted_recist_length_mm": _fmt_float(r.get("predicted_recist_length_mm"), 2),
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
                    "ram_auc_mb_s": ram_aucs[row_idx],
                    "peak_gpu_memory_mb": peak_gpu_mb,
                    "gpu_memory_auc_mb_s": gpu_aucs[row_idx],
                }
            )

    return out_rows


def write_model_csv(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Write a single human-readable per-model ``results.csv`` (one row per RECIST lesion)."""
    out_dir = Path(output_dir).resolve()
    csv_path = out_dir / "results.csv"
    _write_csv(csv_path, RESULT_COLUMNS, records_to_rows(records))
    return csv_path


def _load_result_csvs(inputs: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    extra_cols: list[str] = []
    seen_files: set[Path] = set()
    known_cols = set(RESULT_COLUMNS)

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
            hdr, rows = _read_csv(csv_path)
            for col in hdr:
                if col not in known_cols and col not in extra_cols:
                    extra_cols.append(col)
            all_rows.extend(rows)

    columns = list(RESULT_COLUMNS) + extra_cols
    return columns, all_rows


def _summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute compact per-model console summary stats without writing extra CSV files."""
    groups: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        grp = (str(r.get("model_name", "") or "model"), str(r.get("device", "")))
        if grp not in seen:
            seen.add(grp)
            groups.append(grp)

    summaries: list[dict[str, Any]] = []
    for mname, device in groups:
        grp_rows = [
            r
            for r in rows
            if (str(r.get("model_name", "") or "model"), str(r.get("device", ""))) == (mname, device)
        ]
        case_keys = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in grp_rows}
        dices = [float(r["dice_score"]) for r in grp_rows if r.get("dice_score", "") not in ("", None)]
        timed_rows = [
            r for r in grp_rows if r.get("total_model_time_load_infer_post_s", "") not in ("", None)
        ]
        timed_cases = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in timed_rows}
        tot_times = [float(r["total_model_time_load_infer_post_s"]) for r in timed_rows]
        auc_rows = [r for r in grp_rows if r.get("ram_auc_mb_s", "") not in ("", None)]
        auc_cases = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in auc_rows}
        ram_aucs = [float(r["ram_auc_mb_s"]) for r in auc_rows]
        gpu_auc_rows = [r for r in grp_rows if r.get("gpu_memory_auc_mb_s", "") not in ("", None)]
        gpu_auc_cases = {(str(r.get("dataset", "")), str(r.get("case_id", ""))) for r in gpu_auc_rows}
        gpu_aucs = [float(r["gpu_memory_auc_mb_s"]) for r in gpu_auc_rows]
        rams = [float(r["peak_ram_mb"]) for r in grp_rows if r.get("peak_ram_mb", "") not in ("", None)]
        gpus = [float(r["peak_gpu_memory_mb"]) for r in grp_rows if r.get("peak_gpu_memory_mb", "") not in ("", None)]
        summaries.append(
            {
                "model_name": mname,
                "device": device,
                "cases": len(case_keys),
                "lesions": sum(1 for r in grp_rows if str(r.get("lesion_id", "")) != ""),
                "mean_dice_score": round(mean(dices), 4) if dices else None,
                "mean_case_total_model_time_load_infer_post_s": (
                    round(sum(tot_times) / len(timed_cases), 4) if tot_times and timed_cases else None
                ),
                "max_peak_ram_mb": round(max(rams), 1) if rams else None,
                "mean_case_ram_auc_mb_s": (
                    round(sum(ram_aucs) / len(auc_cases), 1) if ram_aucs and auc_cases else None
                ),
                "max_peak_gpu_memory_mb": round(max(gpus), 1) if gpus else None,
                "mean_case_gpu_memory_auc_mb_s": (
                    round(sum(gpu_aucs) / len(gpu_auc_cases), 1) if gpu_aucs and gpu_auc_cases else None
                ),
            }
        )
    return summaries


def merge_model_csvs(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Concatenate per-model ``results.csv`` files vertically into one ``merged_results.csv``."""
    columns, rows = _load_result_csvs([Path(p) for p in input_paths])
    if not rows:
        raise FileNotFoundError("No rows found in input results.csv files")

    out = Path(output_path).resolve()
    merged_csv = out if out.suffix.lower() == ".csv" else (out / "merged_results.csv")
    _write_csv(merged_csv, columns, rows)
    return merged_csv, _summarize_groups(rows)
