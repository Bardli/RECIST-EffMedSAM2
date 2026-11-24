"""Sequential per-case inference benchmark runner (CPU and GPU modes).

Runs each (model, case) pair in a fresh Python subprocess so that:
  - Model weights are reloaded per case (clinical single-case protocol),
  - Serial single-threaded execution minimizes peak RAM usage,
  - CPU runs enforce CUDA masking and explicit GPU sanity verification, and
  - An external ProcessTreeMonitor guarantees RAM/GPU/time capture even on OOM/SIGKILL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from ..evaluation.compute_metrics import evaluate_run
from ..profiling.monitor import DEFAULT_GPU_TOLERANCE_MB, ProcessTreeMonitor, write_json_atomic
from ..profiling.report import plot_case

DEFAULT_INPUT_DIR = Path("data/inputs")
DEFAULT_WINDOW_DIR = Path("data/window_twins")
DEFAULT_OUTPUT_DIR = Path("runs/benchmark")
DEFAULT_ASSETS_DIR = Path("assets")


def _repo_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "configs" / "models.json").is_file():
        return candidate
    return None


def default_models_config_path() -> Path:
    candidates = [
        Path(__file__).resolve().parents[3] / "configs" / "models.json",
        Path.cwd() / "configs" / "models.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not locate configs/models.json; pass --models-config explicitly."
    )


def resolve_default_python(
    family: str,
    *,
    cwd: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    """Locate default venv Python for ``sam`` or ``nninteractive``, else ``sys.executable``."""
    base_dirs: list[Path] = []
    cur = (cwd or Path.cwd()).resolve()
    base_dirs.append(cur)
    root = repo_root if repo_root is not None else _repo_root()
    if root is not None and root.resolve() != cur:
        base_dirs.append(root.resolve())

    venv_names = (".venv-sam", ".venv") if family == "sam" else (".venv-nn", ".venv")
    for base in base_dirs:
        for vname in venv_names:
            for py_name in ("python", "python3"):
                cand = base / vname / "bin" / py_name
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)
    return sys.executable


def resolve_default_window_dir(
    input_dir: Path,
    window_dir: Path | None,
    no_window_dir: bool,
    *,
    cwd: Path | None = None,
) -> Path | None:
    if no_window_dir:
        return None
    if window_dir is not None:
        return window_dir
    if input_dir == DEFAULT_INPUT_DIR:
        base = (cwd or Path.cwd()) / DEFAULT_WINDOW_DIR
        if base.is_dir() and any(base.rglob("*.npz")):
            return base
    return None


def load_models(
    config_path: Path,
    *,
    scope: str = "paper",
    selected_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    models = json.loads(config_path.read_text())
    by_name = {m["name"]: m for m in models}
    if selected_names:
        unknown = [n for n in selected_names if n not in by_name]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}; available: {list(by_name)}"
            )
        return [by_name[n] for n in selected_names]
    if scope == "paper":
        return [m for m in models if m.get("paper_default", True)]
    if scope == "all-six":
        return models
    raise ValueError(f"Unknown model scope: {scope!r}")


def discover_cases(
    input_dir: Path,
    *,
    cases: list[str] | None = None,
    limit: int | None = None,
    recursive: bool = False,
) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"input_dir does not exist: {input_dir} "
            "(place NPZ files in data/inputs or pass --input-dir)"
        )
    if cases:
        resolved: list[Path] = []
        for entry in cases:
            p = Path(entry)
            if p.is_file():
                resolved.append(p.resolve())
                continue
            cand = input_dir / entry
            if not cand.is_file() and not entry.endswith(".npz"):
                cand = input_dir / f"{entry}.npz"
            if not cand.is_file():
                raise FileNotFoundError(f"Requested case not found: {entry}")
            resolved.append(cand.resolve())
        selected = resolved
    else:
        pattern = "**/*.npz" if recursive else "*.npz"
        selected = sorted(p.resolve() for p in input_dir.glob(pattern) if p.is_file())
    if not selected:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    stems = [p.stem for p in selected]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate case stems detected across discovered NPZ files")
    return selected[:limit] if limit else selected


def _resolve_python(family: str, sam_python: str | None, nn_python: str | None) -> str:
    if family == "sam":
        return sam_python or resolve_default_python("sam")
    return nn_python or resolve_default_python("nninteractive")


def _read_status(metrics_json: Path) -> str | None:
    try:
        return json.loads(metrics_json.read_text()).get("status")
    except Exception:
        return None


def _build_case_command(
    *,
    model: dict[str, Any],
    case_path: Path,
    output_npz: Path,
    metrics_json: Path,
    plots_dir: Path | None,
    args: argparse.Namespace,
    force_overwrite: bool = False,
) -> tuple[list[str], dict[str, str]]:
    family = model["family"]
    py = _resolve_python(family, args.sam_python, args.nn_python)
    worker_device = "cpu" if args.device == "cpu" else "cuda:0"
    runner_backend = "mock" if args.backend_override == "mock" else family

    cmd: list[str] = [
        py,
        "-m",
        "recist_effmedsam2.runner.run_case",
        "--runner-backend",
        runner_backend,
        "--input-npz",
        str(case_path),
        "--output-npz",
        str(output_npz),
        "--metrics-json",
        str(metrics_json),
        "--model-name",
        model["name"],
        "--dataset",
        args.dataset or args.input_dir.name,
        "--device",
        worker_device,
        "--num-threads",
        str(args.num_threads),
        "--sample-interval",
        str(args.sample_interval),
        "--gpu-tolerance-mb",
        str(args.gpu_tolerance_mb),
    ]
    if args.device == "cuda":
        cmd += ["--gpu-index", str(args.gpu_index)]
    if args.scratch:
        cmd += ["--scratch", str(args.scratch)]
    if plots_dir:
        cmd += ["--plots-dir", str(plots_dir)]
    if args.require_raw_hu:
        cmd.append("--require-raw-hu")
    if args.overwrite or force_overwrite:
        cmd.append("--overwrite")

    if args.window_dir and family == "sam":
        try:
            rel_path = case_path.relative_to(args.input_dir.resolve())
        except ValueError:
            rel_path = Path(case_path.name)
        win_file = args.window_dir / rel_path
        if not win_file.is_file():
            win_file = args.window_dir / case_path.name
        if not win_file.is_file():
            raise FileNotFoundError(f"Window twin NPZ missing: {win_file}")
        cmd += ["--window-npz", str(win_file)]

    if args.postprocess == "moving_centroid" and (
        family == "sam" and model["name"].startswith("eff")
    ):
        cmd += ["--postprocess", "moving_centroid"]

    if runner_backend == "mock":
        cmd += ["--mock-family", family]
        if getattr(args, "mock_fail_mode", "none") != "none":
            cmd += ["--mock-fail-mode", args.mock_fail_mode]
    elif family == "sam":
        assets_dir = args.assets_dir.resolve()
        weights_path = assets_dir / model["checkpoint_relpath"]
        cmd += [
            "--sam-backend",
            model["backend"],
            "--model-size",
            model["model_size"],
            "--weights",
            str(weights_path),
            "--prompt",
            model["prompt"],
        ]
        if model.get("cfg_relpath"):
            cmd += ["--cfg", str(model["cfg_relpath"])]
        if model.get("disable_image_encoder_compile", True):
            cmd.append("--disable-image-encoder-compile")
    else:
        assets_dir = args.assets_dir.resolve()
        ckpt_dir = assets_dir / model["checkpoint_relpath"]
        cmd += [
            "--checkpoint",
            str(ckpt_dir),
            "--prompt-mode",
            model["prompt_mode"],
            "--interaction",
            model.get("interaction", "batch"),
            "--scribble-stroke",
            model.get("scribble_stroke", "board"),
            "--scribble-inset",
            str(model.get("scribble_inset", 0.0)),
        ]

    if args.verify_sha256 and runner_backend != "mock":
        assets_dir = args.assets_dir.resolve()
        for rel, expected_hash in model.get("checkpoint_sha256", {}).items():
            cmd += ["--expected-sha256", f"{assets_dir / rel}={expected_hash}"]

    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[2])
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else src_root
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    tval = str(args.num_threads)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[var] = tval
    env["PYTHONUNBUFFERED"] = "1"
    return cmd, env


def _build_failure_record(
    *,
    metrics_json: Path,
    parent_record: dict[str, Any],
    returncode: int,
    wall_s: float,
    dataset: str,
    case_id: str,
    model_name: str,
    device: str,
    num_threads: int,
    postprocess: str,
    input_npz: Path,
) -> dict[str, Any]:
    """Combine worker partial failure telemetry (if any) with parent-process monitor telemetry."""
    if metrics_json.exists():
        try:
            worker_rec = json.loads(metrics_json.read_text())
        except Exception:
            worker_rec = None
        if isinstance(worker_rec, dict):
            worker_rec["status"] = f"failed({returncode})"
            worker_rec.setdefault("error", f"Worker exited with code {returncode}")
            worker_rec.setdefault("dataset", dataset)
            worker_rec.setdefault("case_id", case_id)
            worker_rec.setdefault("model", model_name)
            worker_rec.setdefault("device", device)
            worker_rec.setdefault("num_threads", num_threads)
            worker_rec.setdefault("postprocess", postprocess)
            worker_rec.setdefault("rows", [])
            w_metrics = worker_rec.setdefault("metrics", {})
            p_metrics = parent_record.get("metrics", {})
            if not worker_rec.get("samples", {}).get("seconds") and parent_record.get("samples", {}).get("seconds"):
                worker_rec["samples"] = parent_record["samples"]
                for k, v in p_metrics.items():
                    if w_metrics.get(k) in (None, "", 0, 0.0) and v is not None:
                        w_metrics[k] = v
                worker_rec["monitor_source"] = "parent_process_monitor"
            w_metrics["wall_total_s"] = round(max(float(w_metrics.get("wall_total_s", 0.0)), wall_s), 6)
            return worker_rec

    p_metrics = dict(parent_record.get("metrics", {}))
    p_metrics["wall_total_s"] = round(max(float(p_metrics.get("wall_total_s", 0.0)), wall_s), 6)
    fail_record = dict(parent_record)
    fail_record.update(
        {
            "dataset": dataset,
            "case_id": case_id,
            "model": model_name,
            "device": device,
            "num_threads": num_threads,
            "postprocess": postprocess,
            "source_npz": str(input_npz),
            "status": f"failed({returncode})",
            "error": f"Worker terminated with exit code {returncode}",
            "monitor_source": "parent_process_monitor",
            "metrics": p_metrics,
            "rows": [],
        }
    )
    return fail_record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder of input *.npz files (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output benchmark run folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"Root directory holding downloaded model checkpoints (default: {DEFAULT_ASSETS_DIR})",
    )
    ap.add_argument(
        "--window-dir",
        type=Path,
        default=None,
        help=f"Optional uint8 window twin dir for SAM (auto-detected at {DEFAULT_WINDOW_DIR} if present)",
    )
    ap.add_argument("--no-window-dir", action="store_true", help="Disable auto-detection of data/window_twins")
    ap.add_argument("--models-config", type=Path, default=None, help="Optional models.json override")
    ap.add_argument("--models", nargs="+", default=None, help="Explicit model names to run")
    ap.add_argument("--model-scope", choices=("paper", "all-six"), default="paper")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gpu-index", type=int, default=0, help="Physical GPU index when --device cuda")
    ap.add_argument("--num-threads", type=int, default=1, help="Threads per case (default: 1)")
    ap.add_argument(
        "--sam-python",
        default=None,
        help="Python executable for SAM models (auto-detects .venv-sam / .venv, else current python)",
    )
    ap.add_argument(
        "--nn-python",
        default=None,
        help="Python executable for nnInteractive models (auto-detects .venv-nn / .venv, else current python)",
    )
    ap.add_argument("--cases", nargs="+", default=None, help="Explicit case filenames or stems")
    ap.add_argument("--limit", type=int, default=None, help="Run at most N cases (e.g. --limit 1)")
    ap.add_argument("--recursive", action="store_true", help="Recursively discover *.npz under --input-dir")
    ap.add_argument("--dataset", default=None, help="Dataset name recorded in CSVs")
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--gpu-tolerance-mb", type=float, default=DEFAULT_GPU_TOLERANCE_MB)
    ap.add_argument("--scratch", type=Path, default=None, help="Temp directory for disk-backed channels")
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--verify-sha256", action="store_true", help="Verify asset SHA256 before each case")
    ap.add_argument("--postprocess", choices=("none", "moving_centroid"), default="none")
    ap.add_argument("--backend-override", choices=("auto", "mock"), default="auto")
    ap.add_argument("--mock-fail-mode", choices=("none", "exception", "sigkill"), default="none")
    ap.add_argument("--plots", action="store_true", help="Write CPU/RAM/GPU-vs-time PNG plots")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be >= 1")

    args.window_dir = resolve_default_window_dir(
        args.input_dir,
        args.window_dir,
        args.no_window_dir,
    )
    config_path = args.models_config or default_models_config_path()
    models = load_models(config_path, scope=args.model_scope, selected_names=args.models)
    cases = discover_cases(
        args.input_dir,
        cases=args.cases,
        limit=args.limit,
        recursive=args.recursive,
    )

    out_root = args.output_dir.resolve()
    dataset_label = args.dataset or args.input_dir.name
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "input_dir": str(args.input_dir.resolve()),
            "window_dir": str(args.window_dir.resolve()) if args.window_dir else None,
            "assets_dir": str(args.assets_dir.resolve()),
            "dataset": dataset_label,
            "device": args.device,
            "gpu_index": args.gpu_index if args.device == "cuda" else None,
            "num_threads": args.num_threads,
            "sam_python": _resolve_python("sam", args.sam_python, args.nn_python),
            "nn_python": _resolve_python("nninteractive", args.sam_python, args.nn_python),
            "models": [m["name"] for m in models],
            "cases": [c.stem for c in cases],
            "backend_override": args.backend_override,
            "postprocess": args.postprocess,
            "started_unix": time.time(),
        }
        write_json_atomic(out_root / "run_manifest.json", run_manifest)

    failures = 0
    for model in models:
        model_dir = out_root / model["name"]
        pred_dir = model_dir / "predictions"
        post_pred_dir = model_dir / "predictions_postprocessed"
        metrics_dir = model_dir / "metrics"
        logs_dir = model_dir / "logs"
        plots_dir = (model_dir / "plots") if args.plots else None
        if not args.dry_run:
            pred_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

        for case_path in cases:
            case_id = case_path.stem
            out_npz = pred_dir / case_path.name
            post_npz = post_pred_dir / case_path.name
            metrics_json = metrics_dir / f"{case_id}.metrics.json"
            log_path = logs_dir / f"{case_id}.log"

            if (
                not args.dry_run
                and not args.overwrite
                and out_npz.exists()
                and metrics_json.exists()
                and _read_status(metrics_json) == "ok"
            ):
                print(f"[skip] {model['name']} / {case_id}")
                continue

            retrying_failed = (not args.dry_run) and metrics_json.exists() and (_read_status(metrics_json) != "ok")
            cmd, env = _build_case_command(
                model=model,
                case_path=case_path,
                output_npz=out_npz,
                metrics_json=metrics_json,
                plots_dir=plots_dir,
                args=args,
                force_overwrite=retrying_failed,
            )
            if args.dry_run:
                prefix = (
                    "CUDA_VISIBLE_DEVICES='' "
                    if args.device == "cpu"
                    else f"CUDA_VISIBLE_DEVICES={args.gpu_index} "
                )
                print("[dry-run]", prefix + shlex.join(cmd))
                continue

            # Clear any stale JSON / partial NPZ before launching so failed runs never leave stale 'ok'
            if metrics_json.exists():
                metrics_json.unlink()
            if (args.overwrite or retrying_failed) and out_npz.exists():
                out_npz.unlink()
            if (args.overwrite or retrying_failed) and post_npz.exists():
                post_npz.unlink()

            t0 = time.perf_counter()
            with log_path.open("w") as log_file:
                log_file.write(f"command: {json.dumps(cmd)}\n\n")
                log_file.flush()
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                with ProcessTreeMonitor(
                    target_pid=proc.pid,
                    device=args.device,
                    physical_gpu_index=args.gpu_index if args.device == "cuda" else None,
                    sample_interval=args.sample_interval,
                    gpu_tolerance_mb=args.gpu_tolerance_mb,
                ) as parent_mon:
                    returncode = proc.wait()
                parent_record = parent_mon.finalize()
            wall_s = time.perf_counter() - t0

            if returncode != 0:
                failures += 1
                fail_record = _build_failure_record(
                    metrics_json=metrics_json,
                    parent_record=parent_record,
                    returncode=returncode,
                    wall_s=wall_s,
                    dataset=dataset_label,
                    case_id=case_id,
                    model_name=model["name"],
                    device=args.device,
                    num_threads=args.num_threads,
                    postprocess=args.postprocess,
                    input_npz=case_path,
                )
                write_json_atomic(metrics_json, fail_record)
                if plots_dir is not None:
                    plot_case(fail_record, plots_dir)
                fm = fail_record.get("metrics", {})
                ram_str = f"{fm['MaxRAM_MB']:.1f} MiB" if fm.get("MaxRAM_MB") is not None else "N/A"
                print(
                    f"[FAIL] {model['name']} / {case_id} (rc={returncode}, wall={wall_s:.2f}s, "
                    f"RAM={ram_str}, src={fail_record.get('monitor_source')}) -> see {log_path}",
                    flush=True,
                )
                continue

            rec = json.loads(metrics_json.read_text())
            m = rec["metrics"]
            gpu_str = f"{m['MaxGPU_Mem_MB']:.1f} MiB" if m.get("MaxGPU_Mem_MB") is not None else "N/A"
            print(
                f"[ok]   {model['name']} / {case_id} | "
                f"runtime={m['runtime_s']:.3f}s (load={m['model_load_s']:.3f}s, "
                f"infer={m['inference_s']:.3f}s, post={m['postprocess_s']:.3f}s, "
                f"wall={m['wall_total_s']:.3f}s) | "
                f"RAM={m['MaxRAM_MB']:.1f} MiB | GPU={gpu_str} ({m['gpu_monitor_status']}) | "
                f"sanity={m['gpu_sanity_ok']}",
                flush=True,
            )

    if args.dry_run:
        return 0

    if not args.skip_eval:
        evaluate_run(
            run_dir=out_root,
            gpu_tolerance_mb=args.gpu_tolerance_mb,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
