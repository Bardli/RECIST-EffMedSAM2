"""nnInteractive inference backend for RECIST NPZ CT volumes (CPU & GPU).

Supports box, center-out point prompts (``3pt_25``), and skeletonized inset
scribble prompts (``line --scribble-stroke skeleton --scribble-inset 3``) with
per-case model reloading, independent-instance NPZ output, and embedded
FLARE-Efficiency resource profiling.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..ct_inputs import nninteractive_image
from ..postprocess import finalize_and_save_predictions
from ..profiling.monitor import DEFAULT_GPU_TOLERANCE_MB, CaseProfiler, write_json_atomic
from .device_memory import reset_cuda_peaks
from ..profiling.report import plot_case
from ..prompts import (
    MASK_MODES,
    RECIST_NEG_MODES,
    SCRIBBLE_STROKES,
    get_diameter_bbox,
    mask_points_for_mode,
    points_for_mode,
    recist_points_for_mode,
    scribble_plane,
    seed_all,
)
from ..storage import IndependentPredictionWriter, sha256

RECIST_PROMPT_MODES = (
    "box",
    "line",
    "1_center_point",
    "3_points",
    "5_points",
    "3pt_25",
    "5pt_25_42",
    "5pt_10_90",
)
PROMPT_MODES = RECIST_PROMPT_MODES + RECIST_NEG_MODES + MASK_MODES
RECIST_POS_ONLY_POINT_MODES = (
    "1_center_point",
    "3_points",
    "5_points",
    "3pt_25",
    "5pt_25_42",
    "5pt_10_90",
)
OFFICIAL_MODEL_DATASET_NAME = "Dataset225_nnInteractiveV2"


def apply_thread_env(num_threads: int | None) -> None:
    if num_threads is None:
        return
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


def configure_device_env(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"


def resolve_checkpoint_dir(checkpoint: str | None) -> str:
    if checkpoint:
        return str(Path(checkpoint).expanduser().resolve())
    from nnInteractive.model_management import ensure_model_available, get_default_model_id

    return str(ensure_model_available(get_default_model_id()))


def detect_fold(checkpoint_dir: str) -> int | str:
    folds = sorted(d for d in os.listdir(checkpoint_dir) if d.startswith("fold_"))
    if not folds:
        raise FileNotFoundError(f"No fold_* directory under {checkpoint_dir}")
    if "fold_all" in folds:
        return "all"
    return int(folds[0].split("_")[1])


def checkpoint_fingerprint(checkpoint_dir: str) -> dict[str, str]:
    fold = detect_fold(checkpoint_dir)
    weights = Path(checkpoint_dir) / f"fold_{fold}" / "checkpoint_final.pth"
    h = hashlib.md5()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    with open(Path(checkpoint_dir) / "plans.json") as f:
        dataset_name = json.load(f).get("dataset_name", "?")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "weights_path": str(weights),
        "weights_sha256": sha256(weights),
        "fold": str(fold),
        "dataset_name": dataset_name,
        "md5": h.hexdigest(),
    }


def build_session(
    checkpoint_dir: str,
    device: str,
    num_threads: int | None = None,
):
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    has_pinned_memory = "use_pinned_memory" in inspect.signature(
        nnInteractiveInferenceSession.__init__
    ).parameters
    torch_device = torch.device(
        device if device.startswith("cuda") or device == "cpu" else f"cuda:{device}"
    )
    session_kwargs: dict[str, Any] = dict(
        device=torch_device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=num_threads if num_threads is not None else os.cpu_count(),
        do_autozoom=True,
    )
    if has_pinned_memory:
        session_kwargs["use_pinned_memory"] = torch_device.type == "cuda"
    session = nnInteractiveInferenceSession(**session_kwargs)
    session.initialize_from_trained_model_folder(
        model_training_output_dir=checkpoint_dir,
        use_fold=detect_fold(checkpoint_dir),
    )
    return session


def run_case(
    *,
    input_npz: str | Path,
    output_npz: str | Path,
    metrics_json: str | Path,
    prompt_mode: str,
    checkpoint: str | None = None,
    expect_model: str = OFFICIAL_MODEL_DATASET_NAME,
    device: str = "cpu",
    gpu_index: int | None = None,
    interaction: str = "batch",
    scribble_stroke: str = "board",
    scribble_inset: float = 0.0,
    require_raw_hu: bool = False,
    num_threads: int = 1,
    sample_interval: float = 0.1,
    gpu_tolerance_mb: float = DEFAULT_GPU_TOLERANCE_MB,
    scratch: str | Path | None = None,
    plots_dir: str | Path | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    postprocess: str = "none",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run fresh-session nnInteractive inference and efficiency profiling for one case."""
    if not np.isfinite(scribble_inset) or scribble_inset < 0:
        raise ValueError("--scribble-inset must be finite and nonnegative")

    configure_device_env(device)
    apply_thread_env(num_threads)

    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnunetv2.utilities.helpers import empty_cache

    seed_all()
    if num_threads is not None and num_threads >= 1:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if device == "cpu" and torch.cuda.is_available():
        raise RuntimeError(
            "CPU mode requested, but torch.cuda.is_available() is True. "
            "Set CUDA_VISIBLE_DEVICES='' before starting the worker."
        )

    has_interaction_bbox = "interaction_bbox" in inspect.signature(
        nnInteractiveInferenceSession.add_scribble_interaction
    ).parameters

    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    fp = checkpoint_fingerprint(checkpoint_dir)
    if expect_model and fp["dataset_name"] != expect_model:
        raise RuntimeError(
            f"Checkpoint dataset_name is {fp['dataset_name']!r}, expected {expect_model!r}"
        )

    torch_device = torch.device(
        device if device.startswith("cuda") or device == "cpu" else f"cuda:{device}"
    )
    reset_cuda_peaks(torch_device)

    input_path = Path(input_npz).resolve()
    case_id = input_path.stem
    eff_model_name = model_name or f"nninteractive_{prompt_mode}"
    metrics_path = Path(metrics_json)

    with CaseProfiler(
        device=device,
        gpu_index=gpu_index,
        sample_interval=sample_interval,
        gpu_tolerance_mb=gpu_tolerance_mb,
    ) as profiler:
        session = None
        try:
            # 1. Model load (counted in runtime_s)
            with profiler.phase("model_load"):
                session = build_session(checkpoint_dir, device=device, num_threads=num_threads)

            # 2. Data load & decompression (excluded from runtime_s)
            with profiler.phase("data_load"):
                with np.load(input_path, allow_pickle=False) as data:
                    imgs = nninteractive_image(data, require_raw=require_raw_hu)
                    recist = np.asarray(data["recist"])
                    spacing = np.asarray(data["spacing"])
                    gts = None
                    for k in ("gts", "gts_hidden"):
                        if k in data.files:
                            gts = np.asarray(data[k])
                            break

            # 3. Asynchronous set_image + target buffer setup (excluded from runtime_s)
            with profiler.phase("preprocess"):
                session.set_image(imgs[None].astype(np.float32))
                target_buffer = torch.zeros(imgs.shape, dtype=torch.uint8, device="cpu")
                session.set_target_buffer(target_buffer)
                if hasattr(session, "_finish_preprocessing_and_initialize_interactions"):
                    session._finish_preprocessing_and_initialize_interactions()

            unique_labs = [int(x) for x in np.unique(recist) if x != 0]
            is_mask_mode = prompt_mode in MASK_MODES
            is_recist_neg_mode = prompt_mode in RECIST_NEG_MODES
            if is_mask_mode and gts is None:
                raise ValueError(f"Mask prompt mode {prompt_mode!r} requires GT volume")

            metadata = {
                "dataset": dataset_name or "",
                "case_id": case_id,
                "model": eff_model_name,
                "prompt_mode": prompt_mode,
                "interaction": interaction,
                "scribble_stroke": scribble_stroke,
                "scribble_inset": scribble_inset,
                "device": str(session.device),
                "num_threads": int(torch.get_num_threads()),
                "source_npz": str(input_path),
                "source_sha256": sha256(input_path),
                "checkpoint": fp,
            }

            with IndependentPredictionWriter(
                shape=imgs.shape,
                recist=recist,
                spacing=spacing,
                gt=gts,
                metadata=metadata,
                scratch=scratch,
            ) as writer:
                # 4. Pure model inference (counted in runtime_s)
                with profiler.phase("inference"):
                    for lab in unique_labs:
                        recist_mask = recist == lab
                        z_indices = np.unique(np.where(recist_mask)[0])
                        if len(z_indices) == 0:
                            continue
                        z_mid = int(z_indices[0])
                        slice_2d = recist_mask[z_mid].astype(np.uint8)

                        if session.device.type == "cuda":
                            torch.cuda.synchronize(session.device)
                        t_lesion = time.perf_counter()

                        session.reset_interactions()
                        if prompt_mode == "box":
                            bbox_2d = get_diameter_bbox(slice_2d, shift=0)
                            x_min, y_min, x_max, y_max = [int(v) for v in bbox_2d]
                            bbox_3d = [
                                [z_mid, z_mid + 1],
                                [y_min, y_max + 1],
                                [x_min, x_max + 1],
                            ]
                            session.add_bbox_interaction(bbox_3d, include_interaction=True)
                        elif prompt_mode == "line":
                            plane = scribble_plane(
                                recist_mask[z_mid],
                                session.preferred_scribble_thickness,
                                stroke=scribble_stroke,
                                inset=scribble_inset,
                            )
                            if has_interaction_bbox:
                                ys, xs = np.nonzero(plane)
                                y0, y1 = int(ys.min()), int(ys.max()) + 1
                                x0, x1 = int(xs.min()), int(xs.max()) + 1
                                crop = np.ascontiguousarray(plane[None, y0:y1, x0:x1].astype(np.uint8))
                                session.add_scribble_interaction(
                                    crop,
                                    include_interaction=True,
                                    interaction_bbox=[[z_mid, z_mid + 1], [y0, y1], [x0, x1]],
                                )
                            else:
                                scribble_vol = np.zeros(recist.shape, dtype=np.uint8)
                                scribble_vol[z_mid] = plane
                                session.add_scribble_interaction(scribble_vol, include_interaction=True)
                        elif prompt_mode in RECIST_POS_ONLY_POINT_MODES:
                            pts_xy = points_for_mode(slice_2d, prompt_mode)
                            if interaction == "progressive":
                                for px, py in pts_xy:
                                    session.add_point_interaction(
                                        (z_mid, int(py), int(px)),
                                        include_interaction=True,
                                    )
                            else:
                                order = list(range(len(pts_xy)))
                                order = order[1:] + order[:1]
                                for i in order:
                                    px, py = pts_xy[i]
                                    session.add_point_interaction(
                                        (z_mid, int(py), int(px)),
                                        include_interaction=True,
                                        run_prediction=(i == 0),
                                    )
                        elif is_mask_mode or is_recist_neg_mode:
                            if is_mask_mode:
                                mask_slice = (gts[z_mid] == lab).astype(np.int32)
                                fg_xy, bg_xy = mask_points_for_mode(mask_slice, prompt_mode)
                            else:
                                fg_xy, bg_xy = recist_points_for_mode(slice_2d, prompt_mode)
                            prediction_center = None
                            if len(fg_xy):
                                cx, cy = np.mean(fg_xy, axis=0)
                                prediction_center = (z_mid, int(round(cy)), int(round(cx)))
                            for px, py in fg_xy:
                                session.add_point_interaction(
                                    (z_mid, int(py), int(px)),
                                    include_interaction=True,
                                    run_prediction=False,
                                )
                            for px, py in bg_xy:
                                session.add_point_interaction(
                                    (z_mid, int(py), int(px)),
                                    include_interaction=False,
                                    run_prediction=False,
                                )
                            if prediction_center is not None:
                                session.new_interaction_centers = [prediction_center]
                            else:
                                session.new_interaction_centers = [session.new_interaction_centers[-1]]
                            session.new_interaction_zoom_out_factors = [
                                session.new_interaction_zoom_out_factors[-1]
                            ]
                            session._predict()
                        else:
                            raise ValueError(f"Unhandled prompt_mode: {prompt_mode!r}")

                        mask = session.target_buffer.cpu().numpy() > 0
                        writer.mark_attempted(lab)
                        writer.write_volume(lab, mask)

                        if session.device.type == "cuda":
                            torch.cuda.synchronize(session.device)
                        profiler.record_lesion_time(lab, time.perf_counter() - t_lesion)

                # 5. Optional postprocess (counted in runtime_s) + compressed NPZ save
                receipt, post_npz, post_rows = finalize_and_save_predictions(
                    writer,
                    output_npz,
                    profiler=profiler,
                    postprocess=postprocess,
                    overwrite=overwrite,
                    scratch=scratch,
                )

            del session
            session = None
            gc.collect()
            if torch_device.type == "cuda":
                empty_cache(torch_device)

            record = profiler.finalize(
                extra_metadata={
                    "dataset": dataset_name or "",
                    "case_id": case_id,
                    "model": eff_model_name,
                    "family": "nninteractive",
                    "prompt_mode": prompt_mode,
                    "interaction": interaction,
                    "postprocess": postprocess,
                    "num_threads": int(torch.get_num_threads()),
                    "n_lesions": len(receipt["instance_ids"]),
                    "shape_zyx": list(imgs.shape),
                    "spacing_xyz": [float(s) for s in spacing],
                    "source_npz": str(input_path),
                    "source_sha256": metadata["source_sha256"],
                    "prediction_npz": str(Path(output_npz).resolve()),
                    "prediction_sha256": receipt["output_sha256"],
                    "postprocessed_prediction_npz": str(post_npz.resolve()) if post_npz else None,
                    "checkpoint_path": fp["weights_path"],
                    "checkpoint_sha256": fp["weights_sha256"],
                    "rows": receipt["rows"],
                    "postprocessed_rows": post_rows,
                    "status": "ok",
                }
            )
        except BaseException as exc:
            if session is not None:
                del session
            gc.collect()
            fail_record = profiler.finalize(
                extra_metadata={
                    "dataset": dataset_name or "",
                    "case_id": case_id,
                    "model": eff_model_name,
                    "family": "nninteractive",
                    "prompt_mode": prompt_mode,
                    "interaction": interaction,
                    "postprocess": postprocess,
                    "num_threads": int(torch.get_num_threads()),
                    "source_npz": str(input_path),
                    "status": "failed(exception)",
                    "error": repr(exc),
                    "rows": [],
                }
            )
            write_json_atomic(metrics_path, fail_record)
            if plots_dir is not None:
                plot_case(fail_record, Path(plots_dir))
            raise

    write_json_atomic(metrics_path, record)
    if plots_dir is not None:
        plot_case(record, Path(plots_dir))
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/inputs"),
        help="Input NPZ file or directory (default: data/inputs)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("runs/nninteractive"),
        help="Output directory (default: runs/nninteractive)",
    )
    ap.add_argument("--checkpoint", default=None, help="nnInteractive_v1.0 folder")
    ap.add_argument(
        "--expect-model",
        default=OFFICIAL_MODEL_DATASET_NAME,
        help=f"Assert plans.json:dataset_name matches (default {OFFICIAL_MODEL_DATASET_NAME!r}; '' to skip)",
    )
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index for NVML monitoring")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--prompt_mode", "--prompt-mode", dest="prompt_mode", default="box", choices=PROMPT_MODES)
    ap.add_argument("--interaction", choices=("batch", "progressive"), default="batch")
    ap.add_argument("--scribble-stroke", choices=SCRIBBLE_STROKES, default="board")
    ap.add_argument("--scribble-inset", type=float, default=0.0)
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--gpu-tolerance-mb", type=float, default=DEFAULT_GPU_TOLERANCE_MB)
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--postprocess", choices=("none", "moving_centroid"), default="none")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    configure_device_env(args.device)
    apply_thread_env(args.num_threads)
    inputs = [args.input] if args.input.is_file() else sorted(args.input.glob("*.npz"))
    if not inputs:
        raise SystemExit(f"No NPZ files found at {args.input}")

    pred_dir = args.output_dir / "predictions"
    metrics_dir = args.output_dir / "metrics"
    plots_dir = (args.output_dir / "plots") if args.plots else None
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in inputs:
        out_npz = pred_dir / npz_path.name
        out_json = metrics_dir / f"{npz_path.stem}.metrics.json"
        if out_npz.exists() and out_json.exists() and not args.overwrite:
            print(f"[skip] {npz_path.name} already exists")
            continue
        rec = run_case(
            input_npz=npz_path,
            output_npz=out_npz,
            metrics_json=out_json,
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
            plots_dir=plots_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            postprocess=args.postprocess,
            overwrite=args.overwrite,
        )
        m = rec["metrics"]
        gpu_str = f"{m['MaxGPU_Mem_MB']:.1f} MiB" if m["MaxGPU_Mem_MB"] is not None else "N/A"
        print(
            f"[done] {npz_path.name} | runtime={m['runtime_s']:.3f}s "
            f"(load={m['model_load_s']:.3f}s, infer={m['inference_s']:.3f}s, post={m['postprocess_s']:.3f}s) | "
            f"RAM={m['MaxRAM_MB']:.1f} MiB | GPU={gpu_str} ({m['gpu_monitor_status']}) | "
            f"sanity={m['gpu_sanity_ok']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
