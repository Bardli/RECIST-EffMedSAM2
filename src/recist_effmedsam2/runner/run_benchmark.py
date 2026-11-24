"""Sequential per-case inference benchmark runner (CPU and GPU modes).

Runs each (model, case) pair in a fresh Python subprocess so that:
  - Model weights are reloaded per case (clinical single-case protocol),
  - Serial single-threaded execution minimizes peak RAM usage, and
  - CPU runs enforce strict CUDA masking and GPU=0 sanity verification.
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
from ..profiling.monitor import DEFAULT_GPU_TOLERANCE_MB


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
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
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


def _resolve_python(family: str, sam_python: str, nn_python: str) -> str:
    return sam_python if family == "sam" else nn_python


def _build_case_command(
    *,
    model: dict[str, Any],
    case_path: Path,
    output_npz: Path,
    metrics_json: Path,
    plots_dir: Path | None,
    args: argparse.Namespace,
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
    if args.overwrite:
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder of input *.npz files")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output benchmark run folder")
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Root directory holding downloaded model checkpoints",
    )
    ap.add_argument("--window-dir", type=Path, default=None, help="Optional uint8 window twin dir for SAM")
    ap.add_argument("--models-config", type=Path, default=None, help="Optional models.json override")
    ap.add_argument("--models", nargs="+", default=None, help="Explicit model names to run")
    ap.add_argument("--model-scope", choices=("paper", "all-six"), default="paper")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gpu-index", type=int, default=0, help="GPU index when --device cuda")
    ap.add_argument("--num-threads", type=int, default=1, help="Threads per case (default: 1)")
    ap.add_argument("--sam-python", default=sys.executable, help="Python executable for SAM models")
    ap.add_argument("--nn-python", default=sys.executable, help="Python executable for nnInteractive models")
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
    ap.add_argument("--plots", action="store_true", help="Write CPU/RAM/GPU-vs-time PNG plots")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be >= 1")

    config_path = args.models_config or default_models_config_path()
    models = load_models(config_path, scope=args.model_scope, selected_names=args.models)
    cases = discover_cases(
        args.input_dir,
        cases=args.cases,
        limit=args.limit,
        recursive=args.recursive,
    )

    out_root = args.output_dir.resolve()
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "input_dir": str(args.input_dir.resolve()),
            "window_dir": str(args.window_dir.resolve()) if args.window_dir else None,
            "assets_dir": str(args.assets_dir.resolve()),
            "dataset": args.dataset or args.input_dir.name,
            "device": args.device,
            "gpu_index": args.gpu_index if args.device == "cuda" else None,
            "num_threads": args.num_threads,
            "models": [m["name"] for m in models],
            "cases": [c.stem for c in cases],
            "backend_override": args.backend_override,
            "postprocess": args.postprocess,
            "started_unix": time.time(),
        }
        (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    failures = 0
    for model in models:
        model_dir = out_root / model["name"]
        pred_dir = model_dir / "predictions"
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
            metrics_json = metrics_dir / f"{case_id}.metrics.json"
            log_path = logs_dir / f"{case_id}.log"

            if not args.dry_run and not args.overwrite and out_npz.exists() and metrics_json.exists():
                print(f"[skip] {model['name']} / {case_id}")
                continue

            cmd, env = _build_case_command(
                model=model,
                case_path=case_path,
                output_npz=out_npz,
                metrics_json=metrics_json,
                plots_dir=plots_dir,
                args=args,
            )
            if args.dry_run:
                prefix = "CUDA_VISIBLE_DEVICES='' " if args.device == "cpu" else f"CUDA_VISIBLE_DEVICES={args.gpu_index} "
                print("[dry-run]", prefix + shlex.join(cmd))
                continue

            t0 = time.perf_counter()
            with log_path.open("w") as log_file:
                log_file.write(f"command: {json.dumps(cmd)}\n\n")
                log_file.flush()
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            wall_s = time.perf_counter() - t0
            if proc.returncode != 0:
                failures += 1
                print(
                    f"[FAIL] {model['name']} / {case_id} (rc={proc.returncode}, {wall_s:.2f}s) "
                    f"-> see {log_path}",
                    flush=True,
                )
                if not metrics_json.exists():
                    fail_record = {
                        "dataset": args.dataset or args.input_dir.name,
                        "case_id": case_id,
                        "model": model["name"],
                        "device": args.device,
                        "num_threads": args.num_threads,
                        "status": f"failed({proc.returncode})",
                        "metrics": {"wall_total_s": round(wall_s, 6)},
                        "rows": [],
                    }
                    metrics_json.write_text(json.dumps(fail_record, indent=2) + "\n")
                continue

            rec = json.loads(metrics_json.read_text())
            m = rec["metrics"]
            print(
                f"[ok]   {model['name']} / {case_id} | "
                f"runtime={m['runtime_s']:.3f}s (load={m['model_load_s']:.3f}s, infer={m['inference_s']:.3f}s) | "
                f"RAM={m['MaxRAM_MB']:.1f} MiB | GPU={m['MaxGPU_Mem_MB']:.1f} MiB | "
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
