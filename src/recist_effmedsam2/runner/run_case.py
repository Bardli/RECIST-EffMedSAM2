"""Single-case isolated inference worker.

Designed to be launched as a fresh subprocess per (model, case) so that:
  1. The model is reloaded cleanly for every case (clinical single-case protocol),
  2. CPU-only runs enforce CUDA isolation before importing PyTorch, and
  3. Peak RAM and GPU memory measurements belong strictly to one case.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np


def _apply_env(device: str, num_threads: int) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    val = str(num_threads)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = val


def _verify_sha256_pairs(pairs: list[str]) -> None:
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected PATH=SHA256, got {item!r}")
        raw_path, expected = item.split("=", 1)
        p = Path(raw_path)
        if not p.is_file():
            raise FileNotFoundError(f"Expected asset missing: {p}")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {p}: expected {expected}, got {actual}")


def _run_mock_case(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic lightweight mock backend for end-to-end testing without weights."""
    from ..ct_inputs import abdomen_image, nninteractive_image
    from ..postprocess import finalize_and_save_predictions
    from ..profiling.monitor import CaseProfiler, write_json_atomic
    from ..profiling.report import plot_case
    from ..prompts import get_diameter_bbox
    from ..storage import IndependentPredictionWriter, sha256

    input_path = Path(args.input_npz).resolve()
    case_id = input_path.stem
    metrics_path = Path(args.metrics_json)

    with CaseProfiler(
        device=args.device,
        gpu_index=args.gpu_index,
        sample_interval=args.sample_interval,
        gpu_tolerance_mb=args.gpu_tolerance_mb,
    ) as profiler:
        try:
            with profiler.phase("model_load"):
                weights_buf = np.ones((256, 256), dtype=np.float32)
                time.sleep(0.02)

            with profiler.phase("data_load"):
                with np.load(input_path, allow_pickle=False) as data:
                    spacing = np.asarray(data["spacing"])
                    recist = np.asarray(data["recist"])
                    gts = None
                    for k in ("gts", "gts_hidden"):
                        if k in data.files:
                            gts = np.asarray(data[k])
                            break
                    if args.mock_family == "nninteractive":
                        imgs = nninteractive_image(data, require_raw=args.require_raw_hu)
                    else:
                        if args.window_npz:
                            with np.load(Path(args.window_npz).resolve(), allow_pickle=False) as w:
                                imgs = abdomen_image(w, require_raw=False)
                        else:
                            imgs = abdomen_image(data, require_raw=args.require_raw_hu)

            with profiler.phase("preprocess"):
                vol = imgs.astype(np.float32)
                _ = float(vol.mean()) + float(weights_buf[0, 0])

            unique_labs = [int(x) for x in np.unique(recist) if x != 0]
            metadata = {
                "dataset": args.dataset or "",
                "case_id": case_id,
                "model": args.model_name or "mock_model",
                "family": "mock",
                "device": args.device,
                "num_threads": args.num_threads,
                "source_npz": str(input_path),
                "source_sha256": sha256(input_path),
            }

            with IndependentPredictionWriter(
                shape=imgs.shape,
                recist=recist,
                spacing=spacing,
                gt=gts,
                metadata=metadata,
                scratch=args.scratch,
            ) as writer:
                with profiler.phase("inference"):
                    if args.mock_fail_mode == "sigkill":
                        # Allocate a small verifiable RAM buffer then simulate an OOM SIGKILL
                        _oom_buf = np.ones((8 * 1024 * 1024,), dtype=np.uint8)
                        time.sleep(0.08)
                        os.kill(os.getpid(), signal.SIGKILL)
                    if args.mock_fail_mode == "exception":
                        raise RuntimeError("Simulated mock inference exception")

                    for lab in unique_labs:
                        t_lesion = time.perf_counter()
                        writer.mark_attempted(lab)
                        z_indices = np.unique(np.where(recist == lab)[0])
                        z_mid = int(z_indices[0])
                        x0, y0, x1, y1 = get_diameter_bbox((recist[z_mid] == lab).astype(np.uint8))
                        pred_vol = np.zeros(imgs.shape, dtype=np.uint8)
                        z_lo = max(0, z_mid - 1)
                        z_hi = min(imgs.shape[0] - 1, z_mid + 1)
                        pred_vol[z_lo : z_hi + 1, y0 : y1 + 1, x0 : x1 + 1] = 1
                        # Add a distant false-positive voxel to exercise moving-centroid clipping
                        pred_vol[z_mid, 0, 0] = 1
                        writer.write_volume(lab, pred_vol)
                        time.sleep(0.01)
                        profiler.record_lesion_time(lab, time.perf_counter() - t_lesion)

                receipt, post_npz, post_rows = finalize_and_save_predictions(
                    writer,
                    args.output_npz,
                    profiler=profiler,
                    postprocess=args.postprocess,
                    overwrite=args.overwrite,
                    scratch=args.scratch,
                )

            del weights_buf
            record = profiler.finalize(
                extra_metadata={
                    "dataset": args.dataset or "",
                    "case_id": case_id,
                    "model": args.model_name or "mock_model",
                    "family": "mock",
                    "postprocess": args.postprocess,
                    "num_threads": args.num_threads,
                    "n_lesions": len(receipt["instance_ids"]),
                    "shape_zyx": list(imgs.shape),
                    "spacing_xyz": [float(s) for s in spacing],
                    "source_npz": str(input_path),
                    "source_sha256": metadata["source_sha256"],
                    "prediction_npz": str(Path(args.output_npz).resolve()),
                    "prediction_sha256": receipt["output_sha256"],
                    "postprocessed_prediction_npz": str(post_npz.resolve()) if post_npz else None,
                    "rows": receipt["rows"],
                    "postprocessed_rows": post_rows,
                    "status": "ok",
                }
            )
        except BaseException as exc:
            fail_record = profiler.finalize(
                extra_metadata={
                    "dataset": args.dataset or "",
                    "case_id": case_id,
                    "model": args.model_name or "mock_model",
                    "family": "mock",
                    "postprocess": args.postprocess,
                    "num_threads": args.num_threads,
                    "source_npz": str(input_path),
                    "status": "failed(exception)",
                    "error": repr(exc),
                    "rows": [],
                }
            )
            write_json_atomic(metrics_path, fail_record)
            if args.plots_dir:
                plot_case(fail_record, Path(args.plots_dir))
            raise

    write_json_atomic(metrics_path, record)
    if args.plots_dir:
        plot_case(record, Path(args.plots_dir))
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner-backend", choices=("sam", "nninteractive", "mock"), required=True)
    ap.add_argument("--input-npz", required=True, type=Path)
    ap.add_argument("--output-npz", required=True, type=Path)
    ap.add_argument("--metrics-json", required=True, type=Path)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index for NVML monitoring")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--gpu-tolerance-mb", type=float, default=4096.0)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--window-npz", type=Path, default=None)
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--expected-sha256", action="append", default=[])
    ap.add_argument("--postprocess", choices=("none", "moving_centroid"), default="none")

    # SAM-specific options
    ap.add_argument("--sam-backend", choices=("eff", "medsam2"), default="eff")
    ap.add_argument("--model-size", choices=("small", "tiny"), default="tiny")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--cfg", type=str, default=None)
    ap.add_argument("--prompt", default="box")
    ap.add_argument("--disable-image-encoder-compile", action="store_true")

    # nnInteractive-specific options
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--prompt-mode", default="box")
    ap.add_argument("--interaction", default="batch")
    ap.add_argument("--scribble-stroke", default="board")
    ap.add_argument("--scribble-inset", type=float, default=0.0)
    ap.add_argument("--expect-model", default="Dataset225_nnInteractiveV2")

    # Mock-specific options
    ap.add_argument("--mock-family", choices=("sam", "nninteractive"), default="sam")
    ap.add_argument("--mock-fail-mode", choices=("none", "exception", "sigkill"), default="none")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    _apply_env(args.device, args.num_threads)
    if args.expected_sha256:
        _verify_sha256_pairs(args.expected_sha256)

    if args.runner_backend == "mock":
        _run_mock_case(args)
    elif args.runner_backend == "sam":
        from ..backends.sam_infer import run_case as run_sam_case

        run_sam_case(
            input_npz=args.input_npz,
            output_npz=args.output_npz,
            metrics_json=args.metrics_json,
            backend=args.sam_backend,
            model_size=args.model_size,
            device=args.device,
            gpu_index=args.gpu_index,
            weights=args.weights,
            cfg=args.cfg,
            prompt=args.prompt,
            window_npz=args.window_npz,
            require_raw_hu=args.require_raw_hu,
            disable_image_encoder_compile=args.disable_image_encoder_compile,
            num_threads=args.num_threads,
            sample_interval=args.sample_interval,
            gpu_tolerance_mb=args.gpu_tolerance_mb,
            scratch=args.scratch,
            plots_dir=args.plots_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            postprocess=args.postprocess,
            overwrite=args.overwrite,
        )
    else:
        from ..backends.nninteractive_infer import run_case as run_nn_case

        run_nn_case(
            input_npz=args.input_npz,
            output_npz=args.output_npz,
            metrics_json=args.metrics_json,
            prompt_mode=args.prompt_mode,
            checkpoint=args.checkpoint,
            expect_model=args.expect_model,
            device=args.device,
            gpu_index=args.gpu_index,
            interaction=args.interaction,
            scribble_stroke=args.scribble_stroke,
            scribble_inset=args.scribble_inset,
            require_raw_hu=args.require_raw_hu,
            num_threads=args.num_threads,
            sample_interval=args.sample_interval,
            gpu_tolerance_mb=args.gpu_tolerance_mb,
            scratch=args.scratch,
            plots_dir=args.plots_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            postprocess=args.postprocess,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
