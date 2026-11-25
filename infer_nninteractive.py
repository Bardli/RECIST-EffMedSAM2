"""Standalone nnInteractive inference (box / 3-point / scribble) with efficiency profiling and CSV export."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time

# Configure device visibility and thread pools BEFORE importing PyTorch.
from utils.monitor import apply_thread_env, configure_device_env

DEFAULT_CHECKPOINT = Path("assets/nninteractive_hf_models/models/nnInteractive_v1.0")
OFFICIAL_DATASET_NAME = "Dataset225_nnInteractiveV2"
DEFAULT_MODEL_NAMES = {
    "box": "nnInteractive_box",
    "3pt_25": "nnInteractive_point",
    "line": "nnInteractive_scribble",
}


def detect_fold(checkpoint_dir: Path) -> int | str:
    folds = sorted(d for d in os.listdir(checkpoint_dir) if d.startswith("fold_"))
    if not folds:
        raise FileNotFoundError(f"No fold_* directory under {checkpoint_dir}")
    return "all" if "fold_all" in folds else int(folds[0].split("_")[1])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input-dir", type=Path, default=Path("data/inputs"), help="Input NPZ file or folder")
    ap.add_argument("-o", "--output-dir", type=Path, default=None, help="Output folder (default: runs/<model_name>)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="nnInteractive_v1.0 directory")
    ap.add_argument("--expect-model", type=str, default=OFFICIAL_DATASET_NAME, help="Expected dataset_name in plans.json")
    ap.add_argument("--prompt-mode", choices=("box", "3pt_25", "line"), default="box")
    ap.add_argument("--model-name", type=str, default=None, help="Model name written into CSVs")
    ap.add_argument("--interaction", choices=("batch", "progressive"), default="batch")
    ap.add_argument("--scribble-stroke", choices=("board", "raw", "skeleton", "skeleton_thick"), default="skeleton")
    ap.add_argument("--scribble-inset", type=float, default=3.0)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index when --device cuda")
    ap.add_argument("--num-threads", type=int, default=1, help="Threads per run (default: 1)")
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset name recorded in CSVs")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N cases")
    ap.add_argument("--recursive", action="store_true", help="Search --input-dir recursively")
    ap.add_argument("--sample-interval", type=float, default=0.1, help="Resource polling interval (s)")
    ap.add_argument("--gpu-tolerance-mb", type=float, default=4096.0)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--plots", action="store_true", help="Save CPU/RAM/GPU-vs-time plots")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be >= 1")

    configure_device_env(args.device, args.gpu_index)
    apply_thread_env(args.num_threads)

    import numpy as np
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnunetv2.utilities.helpers import empty_cache

    from utils.csv_io import write_model_csvs
    from utils.ct_inputs import discover_cases, load_gt, nninteractive_image
    from utils.device_memory import reset_cuda_peaks
    from utils.monitor import CaseProfiler, plot_case, write_json_atomic
    from utils.postprocess import finalize_and_save_predictions
    from utils.prompts import get_diameter_bbox, points_for_mode, scribble_plane, seed_all
    from utils.storage import IndependentPredictionWriter, sha256

    if args.device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("CPU mode requested, but torch.cuda.is_available() is True.")

    seed_all()
    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ckpt_dir = args.checkpoint.resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"nnInteractive checkpoint dir not found: {ckpt_dir}. Run scripts/download_weights.sh or pass --checkpoint."
        )
    if args.expect_model:
        plans_ds = json.loads((ckpt_dir / "plans.json").read_text()).get("dataset_name")
        if plans_ds != args.expect_model:
            raise RuntimeError(f"Checkpoint dataset_name is {plans_ds!r}, expected {args.expect_model!r}")

    fold = detect_fold(ckpt_dir)
    ckpt_weights = ckpt_dir / f"fold_{fold}" / "checkpoint_final.pth"
    ckpt_sha = sha256(ckpt_weights)

    has_pinned_memory = "use_pinned_memory" in inspect.signature(nnInteractiveInferenceSession.__init__).parameters
    has_interaction_bbox = "interaction_bbox" in inspect.signature(
        nnInteractiveInferenceSession.add_scribble_interaction
    ).parameters

    model_name = args.model_name or DEFAULT_MODEL_NAMES[args.prompt_mode]
    out_dir = (args.output_dir or Path("runs") / model_name).resolve()
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    cases = discover_cases(args.input_dir, limit=args.limit, recursive=args.recursive)
    dataset_label = args.dataset or (args.input_dir.parent.name if args.input_dir.is_file() else args.input_dir.name)

    records: list[dict] = []
    failures = 0

    for idx, case_path in enumerate(cases):
        case_id = case_path.stem
        out_npz = pred_dir / case_path.name
        metrics_json = metrics_dir / f"{case_id}.metrics.json"
        if out_npz.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists (pass --overwrite): {out_npz}")
        if metrics_json.exists():
            metrics_json.unlink()

        reset_cuda_peaks(torch_device)
        session = None
        record: dict | None = None

        with CaseProfiler(
            device=args.device,
            gpu_index=args.gpu_index,
            sample_interval=args.sample_interval,
            gpu_tolerance_mb=args.gpu_tolerance_mb,
        ) as profiler:
            try:
                # 1. Model loading (counted in total_runtime_s)
                with profiler.phase("model_loading_time"):
                    session_kwargs = dict(
                        device=torch_device,
                        use_torch_compile=False,
                        verbose=False,
                        torch_n_threads=args.num_threads,
                        do_autozoom=True,
                    )
                    if has_pinned_memory:
                        session_kwargs["use_pinned_memory"] = torch_device.type == "cuda"
                    session = nnInteractiveInferenceSession(**session_kwargs)
                    session.initialize_from_trained_model_folder(model_training_output_dir=str(ckpt_dir), use_fold=fold)

                # 2. Data load & decompression (excluded from total_runtime_s)
                with profiler.phase("data_loading_time"):
                    with np.load(case_path, allow_pickle=False) as data:
                        imgs = nninteractive_image(data, require_raw=args.require_raw_hu)
                        recist = np.asarray(data["recist"])
                        spacing = np.asarray(data["spacing"])
                        gts, _ = load_gt(data, "auto", case_path.name)

                # 3. Preprocessing + session image setup (excluded from total_runtime_s)
                with profiler.phase("image_preprocessing_time"):
                    session.set_image(imgs[None].astype(np.float32))
                    session.set_target_buffer(torch.zeros(imgs.shape, dtype=torch.uint8, device="cpu"))
                    if hasattr(session, "_finish_preprocessing_and_initialize_interactions"):
                        session._finish_preprocessing_and_initialize_interactions()

                unique_labs = [int(x) for x in np.unique(recist) if x != 0]
                metadata = {
                    "dataset": dataset_label,
                    "case_id": case_id,
                    "model_name": model_name,
                    "prompt_mode": args.prompt_mode,
                    "device": args.device,
                    "postprocess": "none",
                    "source_npz": str(case_path),
                    "source_sha256": sha256(case_path),
                    "checkpoint_path": str(ckpt_weights),
                    "checkpoint_sha256": ckpt_sha,
                }

                with IndependentPredictionWriter(
                    shape=imgs.shape,
                    recist=recist,
                    spacing=spacing,
                    gt=gts,
                    metadata=metadata,
                    scratch=args.scratch,
                ) as writer:
                    # 4. Pure model inference (counted in total_runtime_s)
                    with profiler.phase("model_inference_time"):
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
                            if args.prompt_mode == "box":
                                x_min, y_min, x_max, y_max = [int(v) for v in get_diameter_bbox(slice_2d)]
                                session.add_bbox_interaction(
                                    [[z_mid, z_mid + 1], [y_min, y_max + 1], [x_min, x_max + 1]],
                                    include_interaction=True,
                                )
                            elif args.prompt_mode == "line":
                                plane = scribble_plane(
                                    recist_mask[z_mid],
                                    session.preferred_scribble_thickness,
                                    stroke=args.scribble_stroke,
                                    inset=args.scribble_inset,
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
                                    svol = np.zeros(recist.shape, dtype=np.uint8)
                                    svol[z_mid] = plane
                                    session.add_scribble_interaction(svol, include_interaction=True)
                            else:
                                pts_xy = points_for_mode(slice_2d, args.prompt_mode)
                                if args.interaction == "progressive":
                                    for px, py in pts_xy:
                                        session.add_point_interaction((z_mid, int(py), int(px)), include_interaction=True)
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

                            writer.mark_attempted(lab)
                            writer.write_volume(lab, session.target_buffer.cpu().numpy() > 0)

                            if session.device.type == "cuda":
                                torch.cuda.synchronize(session.device)
                            profiler.record_lesion_time(lab, time.perf_counter() - t_lesion)

                    # 5. Save final NPZ
                    receipt = finalize_and_save_predictions(
                        writer,
                        out_npz,
                        profiler=profiler,
                        postprocess="none",
                        overwrite=args.overwrite,
                        scratch=args.scratch,
                    )

                del session
                session = None
                gc.collect()
                if torch_device.type == "cuda":
                    empty_cache(torch_device)

                record = profiler.finalize(
                    use_ru_maxrss=(idx == 0),
                    extra_metadata={
                        "dataset": dataset_label,
                        "case_id": case_id,
                        "model_name": model_name,
                        "device": args.device,
                        "postprocess": "none",
                        "num_threads": int(torch.get_num_threads()),
                        "n_lesions": len(receipt["instance_ids"]),
                        "shape_zyx": list(imgs.shape),
                        "spacing_xyz": [float(s) for s in spacing],
                        "rows": receipt["rows"],
                        "status": "ok",
                        "error": "",
                    },
                )
            except BaseException as exc:
                failures += 1
                if session is not None:
                    del session
                gc.collect()
                record = profiler.finalize(
                    use_ru_maxrss=(idx == 0),
                    extra_metadata={
                        "dataset": dataset_label,
                        "case_id": case_id,
                        "model_name": model_name,
                        "device": args.device,
                        "postprocess": "none",
                        "num_threads": int(torch.get_num_threads()),
                        "rows": [],
                        "status": "failed(exception)",
                        "error": repr(exc),
                    },
                )

        assert record is not None
        write_json_atomic(metrics_json, record)
        if args.plots:
            plot_case(record, plots_dir)
        records.append(record)
        m = record["metrics"]
        print(
            f"[{idx + 1}/{len(cases)}] {model_name} | {case_id} | {record['status']} | "
            f"total_runtime={m['total_runtime_s']:.3f}s | wall={m['end_to_end_wall_time_including_io_s']:.3f}s | "
            f"MaxRAM={m['MaxRAM_MB']:.1f}MB"
        )

    csv_paths = write_model_csvs(records, out_dir)
    print(f"Saved: {csv_paths['case_metrics']}")
    print(f"Saved: {csv_paths['lesion_scores']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
