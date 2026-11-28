"""Standalone RECIST-EffMedSAM2 inference (Tiny/Small) with efficiency profiling and CSV export."""
from __future__ import annotations

import argparse
import gc
import importlib
import os
from pathlib import Path
import random
import sys

# Configure device visibility and thread pools BEFORE importing PyTorch.
from utils.exemptions import load_policy
from utils.monitor import apply_thread_env, configure_device_env

SEED = 2024
IMAGE_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BOX_SCALE = 1.0
Z_SLAB_SCALE = 1.6

DEFAULT_CHECKPOINT_BY_SIZE = {
    "tiny": Path("assets/RECIST-EffTiny-m8/checkpoints/checkpoint_40.pt"),
    "small": Path("assets/RECIST-EffSmall-m8/checkpoints/checkpoint_40.pt"),
}
DEFAULT_CFG_BY_SIZE = {
    "tiny": "configs/efficienttam_ti_512x512.yaml",
    "small": "configs/efficienttam_s_512x512.yaml",
}


def resolve_cfg(cfg: str | None, model_size: str) -> str:
    pkg = importlib.import_module("efficient_track_anything")
    pkg_dir = Path(pkg.__file__).resolve().parent
    rel = DEFAULT_CFG_BY_SIZE[model_size]
    if cfg is None:
        cand = pkg_dir / rel
    else:
        p = Path(cfg).expanduser()
        cand = p.resolve() if p.is_file() else (pkg_dir / cfg).resolve()
    if not cand.is_file():
        raise FileNotFoundError(
            f"Config YAML not found: {cand} (install MedSAM2 editable via -e .venv-sam/src/MedSAM2)"
        )
    return "/" + str(cand)


def build_parser(*, nodal: bool = False) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input-dir", type=Path, required=nodal, default=Path("data/inputs"), help="Input NPZ file or folder")
    ap.add_argument("-o", "--output-dir", type=Path, required=nodal, default=Path("runs/EffMedSAM2"), help="Output folder")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint (.pt)")
    ap.add_argument("--model-size", choices=("tiny", "small"), default="tiny")
    ap.add_argument("--cfg", type=str, default=None, help="Optional YAML config override")
    ap.add_argument("--model-name", type=str, default=None, help="Model name written into CSVs (default: EffMedSAM2)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index when --device cuda")
    ap.add_argument("--num-threads", type=int, default=1, help="Threads per run (default: 1)")
    ap.add_argument("--window-dir", type=Path, default=None, help="Optional pre-windowed/normalized twin dir")
    ap.add_argument(
        "--postprocess",
        choices=("moving_centroid", "none"),
        default="moving_centroid",
        help="Postprocessing applied before saving final prediction (default: moving_centroid)",
    )
    policy = ap.add_mutually_exclusive_group()
    policy.add_argument("--exempt-ids", type=int, nargs="+", metavar="ID",
                        help="Tiny: skip moving-centroid clipping for these original lesion IDs (single input file)")
    policy.add_argument("--exempt-manifest", type=Path, metavar="JSON",
                        help="Tiny: per-input SHA256 to exempt ID lists (recist-exempt-v1)")
    ap.add_argument("--shift", type=int, default=0, help="Box padding in pixels")
    ap.add_argument("--gt-key", choices=("auto", "gts", "gts_hidden"), default="auto")
    ap.add_argument("--gts-path", type=Path, default=None)
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--compile-image-encoder", action="store_true", help="Enable torch.compile on encoder (GPU)")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset name recorded in CSVs")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N cases")
    ap.add_argument("--recursive", action="store_true", help="Search --input-dir recursively")
    ap.add_argument("--sample-interval", type=float, default=0.1, help="Resource polling interval (s)")
    ap.add_argument("--gpu-tolerance-mb", type=float, default=4096.0)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--plots", action="store_true", help="Save CPU/RAM/GPU-vs-time plots")
    ap.add_argument("--overwrite", action="store_true")
    if nodal:
        ap.add_argument("--nodal-manifest", required=True, type=Path, help="Hash-bound confirmed short-axis IDs for this input")
    return ap


def main(argv=None, *, nodal: bool = False) -> int:
    args = build_parser(nodal=nodal).parse_args(argv)
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be >= 1")

    try:
        policy = load_policy(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid exemption policy: {exc}") from exc
    configure_device_env(args.device, args.gpu_index)
    apply_thread_env(args.num_threads)

    import numpy as np
    from utils.nodal import bind_request, derived_geometry
    nodal_ids, nodal_meta = bind_request(args) if nodal else (None, {})
    import torch
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor_npz

    from utils.csv_io import write_model_csv
    from utils.ct_inputs import (
        abdomen_image,
        auto_window_dir,
        discover_cases,
        load_gt,
        resize_grayscale_to_rgb_and_resize,
        resolve_window_npz,
    )
    from utils.device_memory import reset_cuda_peaks
    from utils.monitor import CaseProfiler, plot_case, write_json_atomic
    from utils.postprocess import finalize_and_save_predictions
    from utils.prompts import find_recist_slices, recist_bbox, recist_diameter
    from utils.storage import IndependentPredictionWriter, sha256

    if args.device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("CPU mode requested, but torch.cuda.is_available() is True.")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ckpt_path = (args.checkpoint or DEFAULT_CHECKPOINT_BY_SIZE[args.model_size]).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run scripts/download_weights.sh or pass --checkpoint."
        )
    cfg_arg = resolve_cfg(args.cfg, args.model_size)
    model_name = args.model_name or ("EffMedSAM2" if args.model_size == "tiny" else "EffMedSAM2_small")
    window_dir = auto_window_dir(args.input_dir, args.window_dir)
    cases = discover_cases(args.input_dir, limit=args.limit, recursive=args.recursive)

    if policy is not None and len({p.stem for p in cases}) != len(cases):
        raise SystemExit("Duplicate input basenames: use separate run directories for these cases")

    out_dir = args.output_dir.resolve()
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    dataset_label = args.dataset or (args.input_dir.parent.name if args.input_dir.is_file() else args.input_dir.name)
    ckpt_sha = sha256(ckpt_path)

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
        predictor = None
        record: dict | None = None

        with CaseProfiler(
            device=args.device,
            gpu_index=args.gpu_index,
            sample_interval=args.sample_interval,
            gpu_tolerance_mb=args.gpu_tolerance_mb,
        ) as profiler:
            try:
                # 1. Model loading (counted in total_model_time_load_infer_post_s)
                with profiler.phase("model_loading_time"):
                    overrides = (
                        []
                        if (args.compile_image_encoder and args.device == "cuda")
                        else ["++model.compile_image_encoder=false"]
                    )
                    predictor = build_efficienttam_video_predictor_npz(
                        cfg_arg,
                        str(ckpt_path),
                        device=str(torch_device),
                        hydra_overrides_extra=overrides,
                    )

                # 2. Data load & decompression (excluded from total_model_time_load_infer_post_s)
                with profiler.phase("data_loading_time"):
                    win_npz = resolve_window_npz(case_path, args.input_dir, window_dir)
                    with np.load(case_path, allow_pickle=False) as npz_data:
                        spacing = np.asarray(npz_data["spacing"])
                        recist = np.asarray(npz_data["recist"])
                        gts_3d, _ = load_gt(npz_data, args.gt_key, case_path.name, args.gts_path)
                        if win_npz is not None:
                            with np.load(win_npz, allow_pickle=False) as w_data:
                                img_3d = abdomen_image(w_data, require_raw=False)
                        else:
                            img_3d = abdomen_image(npz_data, require_raw=args.require_raw_hu)
                    if img_3d.shape != recist.shape:
                        raise ValueError(f"Image shape {img_3d.shape} != RECIST shape {recist.shape}")

                # 3. Resize + RGB + ImageNet norm + RECIST prompt extraction (excluded from total_model_time_load_infer_post_s)
                with profiler.phase("image_preprocessing_time"):
                    video_height, video_width = img_3d.shape[1], img_3d.shape[2]
                    if (video_height, video_width) != (IMAGE_SIZE, IMAGE_SIZE):
                        img_resized = resize_grayscale_to_rgb_and_resize(img_3d, IMAGE_SIZE)
                    else:
                        img_resized = np.repeat(img_3d[:, None], 3, axis=1).astype(np.float32, copy=False)
                    img_tensor = torch.from_numpy(img_resized)
                    img_tensor -= torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
                    img_tensor /= torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

                    slices_by_label = find_recist_slices(recist)
                    axes = {}
                    if nodal_ids is not None:
                        if not nodal_ids <= set(slices_by_label):
                            raise ValueError("Unknown or unprompted confirmed lesion IDs")
                        slices_by_label = {i: slices_by_label[i] for i in sorted(nodal_ids)}
                    prepared_prompts: list[tuple[int, int, int, int, int, np.ndarray]] = []
                    for ulab, z_indices in slices_by_label.items():
                        if len(z_indices) != 1:
                            raise ValueError(f"Expected 1 RECIST slice for ID {ulab}, got {z_indices}")
                        z_mid_orig = z_indices[0]
                        recist_2d = (recist[z_mid_orig] == ulab).astype(np.uint8)
                        if nodal:
                            axes[ulab] = derived_geometry(recist_2d, spacing, z_mid_orig)
                            diameter = axes[ulab]["diameter_px"]
                        else:
                            diameter = recist_diameter(recist_2d)
                        z_span = diameter * (((spacing[0] + spacing[1]) / 2.0) / spacing[2])
                        half_z = int(z_span * Z_SLAB_SCALE / 2.0)
                        z_lo = max(0, z_mid_orig - half_z)
                        z_hi = min(img_3d.shape[0] - 1, z_mid_orig + half_z)
                        z_mid = z_mid_orig - z_lo
                        if nodal:
                            axes[ulab]["z_bounds_inclusive"] = [z_lo, z_hi]
                            box_2d = np.array(axes[ulab]["box_xyxy"])
                        else:
                            box_2d = recist_bbox(recist_2d, scale=BOX_SCALE, shift=args.shift)
                        prepared_prompts.append((ulab, z_mid_orig, z_lo, z_hi, z_mid, box_2d))

                source_sha = sha256(case_path)
                exempt_ids = policy.ids_for(source_sha, set(slices_by_label)) if policy is not None else None
                policy_info = {} if policy is None else {
                    "exemption_source": "manifest" if policy.by_input is not None else "cli",
                    "exemption_manifest_sha256": policy.manifest_sha256,
                }
                metadata = {
                    "dataset": dataset_label,
                    "case_id": case_id,
                    "model_name": model_name,
                    "device": args.device,
                    "postprocess": args.postprocess,
                    **({**nodal_meta, "nodal_geometry": axes} if nodal else {}),
                    "source_npz": str(case_path),
                    "source_sha256": source_sha,
                    **policy_info,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": ckpt_sha,
                }

                with IndependentPredictionWriter(
                    shape=img_3d.shape,
                    recist=recist,
                    spacing=spacing,
                    gt=gts_3d,
                    metadata=metadata,
                    only_ids=nodal_ids,
                    scratch=args.scratch,
                ) as writer:
                    # 4. Pure model inference (counted in total_model_time_load_infer_post_s)
                    with profiler.phase("model_inference_time"):
                        with torch.inference_mode():
                            for ulab, z_mid_orig, z_lo, z_hi, z_mid, box_2d in prepared_prompts:
                                with profiler.lesion_inference(ulab):
                                    img_crop = img_tensor[z_lo : z_hi + 1]
                                    writer.mark_attempted(ulab)
                                    state = predictor.init_state(img_crop, video_height, video_width)
                                    _, _, logits = predictor.add_new_points_or_box(
                                        inference_state=state, frame_idx=z_mid, obj_id=1, box=box_2d
                                    )
                                    mask_prompt = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                                    _, _, masks = predictor.add_new_mask(state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
                                    writer.write_slice(ulab, z_mid_orig, (masks[0] > 0.0).cpu().numpy()[0])
                                    for fi, _, lg in predictor.propagate_in_video(state, start_frame_idx=z_mid, reverse=False):
                                        writer.write_slice(ulab, z_lo + int(fi), (lg[0] > 0.0).cpu().numpy()[0])

                                    predictor.reset_state(state)
                                    state = predictor.init_state(img_crop, video_height, video_width)
                                    predictor.add_new_mask(state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
                                    for fi, _, lg in predictor.propagate_in_video(state, start_frame_idx=z_mid, reverse=True):
                                        writer.write_slice(ulab, z_lo + int(fi), (lg[0] > 0.0).cpu().numpy()[0])
                                    predictor.reset_state(state)

                    # 5. Final postprocess (counted in total_model_time_load_infer_post_s) + save final NPZ only
                    receipt = finalize_and_save_predictions(
                        writer,
                        out_npz,
                        profiler=profiler,
                        postprocess=args.postprocess,
                        exempt_ids=exempt_ids,
                        nodal_lengths={lab: a["nominal_mm"] for lab, a in axes.items()} if nodal else None,
                        overwrite=args.overwrite,
                        scratch=args.scratch,
                    )

                del predictor
                predictor = None
                gc.collect()
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()

                record = profiler.finalize(
                    use_ru_maxrss=(idx == 0),
                    extra_metadata={
                        "dataset": dataset_label,
                        "case_id": case_id,
                        "model_name": model_name,
                        "device": args.device,
                        "postprocess": args.postprocess,
                        "num_threads": int(torch.get_num_threads()),
                        "n_lesions": len(receipt["instance_ids"]),
                        "shape_zyx": list(img_3d.shape),
                        "spacing_xyz": [float(s) for s in spacing],
                        "rows": receipt["rows"],
                        **({**nodal_meta, "nodal_geometry": axes, "source_sha256": metadata["source_sha256"],
                            "checkpoint_sha256": ckpt_sha} if nodal else {}),
                        **({"postprocess_actions": receipt["postprocess_actions"]} if nodal else {}),
                        **({
                            **policy_info,
                            "source_sha256": source_sha,
                            "postprocess_exempt_ids": receipt["postprocess_exempt_ids"],
                            "postprocess_actions": receipt["postprocess_actions"],
                        } if policy is not None else {}),
                        "status": "ok",
                        "error": "",
                    },
                )
            except BaseException as exc:
                failures += 1
                if predictor is not None:
                    del predictor
                gc.collect()
                record = profiler.finalize(
                    use_ru_maxrss=(idx == 0),
                    extra_metadata={
                        "dataset": dataset_label,
                        "case_id": case_id,
                        "model_name": model_name,
                        "device": args.device,
                        "postprocess": args.postprocess,
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
            f"model_time={m['total_model_time_load_infer_post_s']:.3f}s | end_to_end={m['total_end_to_end_time_including_io_s']:.3f}s | "
            f"MaxRAM={m['MaxRAM_MB']:.1f}MB"
        )

    csv_path = write_model_csv(records, out_dir)
    print(f"Saved: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
