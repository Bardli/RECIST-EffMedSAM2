"""Merge per-model results.csv files into one metric-grouped wide comparison table."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from utils.csv_io import merge_model_csvs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="Per-model output directories (or direct results.csv paths) to merge",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Scan a parent runs directory for model folders containing results.csv",
    )
    ap.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("runs/merged"),
        help="Output .csv path or directory for merged_results.csv (default: runs/merged)",
    )
    return ap


def _discover_run_subdirs(runs_dir: Path) -> list[Path]:
    skip = {"merged", "reports", "logs", "scratch"}
    dirs = [
        p
        for p in sorted(runs_dir.iterdir())
        if p.is_dir()
        and p.name not in skip
        and not p.name.startswith("merged")
        and (p / "results.csv").is_file()
    ]
    if not dirs:
        raise FileNotFoundError(f"No model folders containing results.csv found under {runs_dir}")
    return dirs


def main() -> int:
    args = build_parser().parse_args()
    inputs: list[Path] = []
    if args.inputs:
        inputs.extend(args.inputs)
    if args.runs_dir is not None:
        inputs.extend(_discover_run_subdirs(args.runs_dir))
    if not inputs:
        default_runs = Path("runs")
        if default_runs.is_dir():
            inputs = _discover_run_subdirs(default_runs)
        else:
            raise SystemExit("Pass -i/--inputs <model_dir...> or --runs-dir <runs_root>.")

    merged_csv, summaries = merge_model_csvs(inputs, args.output)
    print(f"Wrote: {merged_csv}")
    for s in summaries:
        dice_s = f"{s['mean_dice_score']:.4f}" if s["mean_dice_score"] is not None else "n/a"
        time_s = (
            f"{s['mean_case_total_inference_time_s']:.3f}s"
            if s["mean_case_total_inference_time_s"] is not None
            else "n/a"
        )
        ram_s = f"{s['max_peak_ram_mb']:.1f}MB" if s["max_peak_ram_mb"] is not None else "n/a"
        gpu_s = f"{s['max_peak_gpu_memory_mb']:.1f}MB" if s["max_peak_gpu_memory_mb"] is not None else "n/a"
        print(
            f"  {s['column_prefix']} ({s['device'] or 'n/a'}): "
            f"cases={s['cases']} | lesions={s['lesions']} | "
            f"mean_dice={dice_s} | mean_total_inf={time_s} | peak_ram={ram_s} | peak_gpu={gpu_s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
