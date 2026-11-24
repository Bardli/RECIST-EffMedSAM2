"""Plot and load FLARE-Efficiency style resource traces from case metric JSONs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .monitor import ranking_gpu_memory

logger = logging.getLogger(__name__)

_PHASE_COLORS = {
    "model_load": ("#f59e0b", "model_load"),
    "data_load": ("#6b7280", "data_load"),
    "preprocess": ("#8b5cf6", "preprocess"),
    "inference": ("#10b981", "inference"),
    "postprocess": ("#ec4899", "postprocess"),
    "save": ("#3b82f6", "save"),
}


def load_metric_records(
    metrics_dir: Path,
    gpu_tolerance_mb: float | None = None,
) -> list[dict[str, Any]]:
    """Load all ``*.metrics.json`` or ``*.json`` records in ``metrics_dir`` sorted by filename."""
    if not metrics_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable metrics file %s: %s", path, exc)
            continue
        max_gpu = record.get("metrics", {}).get("MaxGPU_Mem_MB")
        if gpu_tolerance_mb is not None and max_gpu is not None:
            record["metrics"]["RankGPU_Mem_MB"] = ranking_gpu_memory(
                max_gpu,
                gpu_tolerance_mb,
            )
            record["gpu_memory_tolerance_mb"] = gpu_tolerance_mb
        records.append(record)
    return records


def plot_case(record: dict[str, Any], plots_dir: Path) -> list[Path]:
    """Save CPU / RAM / GPU-vs-time curve PNGs with shaded execution phases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = record.get("samples", {})
    seconds = samples.get("seconds", [])
    if not seconds:
        return []

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    case_id = record.get("case_id", "case")
    model = record.get("model", "")
    phase_spans = record.get("phase_spans", {})

    series = [
        ("cpu_percent", "CPU Utilization (%)", f"{case_id}_CPU-Time.png"),
        ("ram_mb", "Process RAM RSS (MiB)", f"{case_id}_RAM-Time.png"),
        ("gpu_mb", "Process GPU Memory (MiB)", f"{case_id}_GPU-Time.png"),
    ]
    written: list[Path] = []
    for key, ylabel, filename in series:
        values = samples.get(key) or []
        pairs = [(t, v) for t, v in zip(seconds, values) if v is not None]
        if not pairs:
            continue
        xs = [t for t, _ in pairs]
        ys = [float(v) for _, v in pairs]

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(xs, ys, color="#1d4ed8", linewidth=1.6)
        labeled_phases: set[str] = set()
        for phase, spans in phase_spans.items():
            color, label = _PHASE_COLORS.get(phase, ("#9ca3af", phase))
            for t0, t1 in spans:
                lbl = label if phase not in labeled_phases else None
                labeled_phases.add(phase)
                ax.axvspan(t0, t1, color=color, alpha=0.12, label=lbl)
        tolerance = record.get("gpu_memory_tolerance_mb")
        max_gpu = record.get("metrics", {}).get("MaxGPU_Mem_MB")
        if key == "gpu_mb" and tolerance and max_gpu is not None and float(max_gpu) > 0:
            ax.axhline(
                float(tolerance),
                color="#dc2626",
                linestyle="--",
                linewidth=1.0,
                label=f"tolerance {float(tolerance):.0f} MiB",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        title_prefix = f"{model} - " if model else ""
        ax.set_title(f"{title_prefix}{case_id} ({record.get('device', '')})", fontsize=9)
        ax.grid(alpha=0.3)
        if labeled_phases or (key == "gpu_mb" and tolerance and max_gpu is not None and float(max_gpu) > 0):
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        out_path = plots_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written
