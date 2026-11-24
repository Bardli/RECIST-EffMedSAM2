"""Unified MedSAM2 / EfficientMedSAM2 inference for RECIST NPZ CT volumes (CPU & GPU).

Every invocation processes one or more NPZ cases sequentially, reloading the
model fresh per case (matching clinical single-case usage and isolating per-case
peak RAM/VRAM measurements), recording time breakdown + FLARE-Efficiency resource
metrics, and saving ``independent_instances_v2`` channel predictions.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

from ..ct_inputs import abdomen_image
from ..postprocess import finalize_and_save_predictions
from ..profiling.monitor import DEFAULT_GPU_TOLERANCE_MB, CaseProfiler, write_json_atomic
from ..profiling.report import plot_case
from ..storage import IndependentPredictionWriter, sha256

SEED = 2024
IMAGE_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BOX_SCALE = 1.0
Z_SLAB_SCALE = 1.6
SLAB_FORMULA = "correct"
SUBTRACT_THICKNESS = 2.0
N_POINTS = 5

PROMPT_CHOICES = (
    "box",
    "1point",
    "3points",
    "5points",
    "5points_10_90",
    "npoints",
    "grid9",
)
PROMPT_PERCENTILES = {
    "1point": (50.0,),
    "3points": (25.0, 50.0, 75.0),
    "5points": (7.5, 25.0, 50.0, 75.0, 92.5),
    "5points_10_90": (10.0, 30.0, 50.0, 70.0, 90.0),
}
HF_REPO_ID = "wanglab/MedSAM2"
HF_WEIGHTS = {
    ("medsam2", "small"): "medsam2_FLARE25_RECIST_baseline.pt",
    ("medsam2", "tiny"): "medsam2_FLARE25_RECIST_baseline.pt",
    ("eff", "small"): "eff_medsam2_small_FLARE25_RECIST_baseline.pt",
    ("eff", "tiny"): "eff_medsam2_tiny_FLARE25_RECIST_baseline.pt",
}
DEFAULT_CFG_BY_TARGET = {
    ("medsam2", "small"): ("sam2", "configs/sam2.1_hiera_t512.yaml"),
    ("medsam2", "tiny"): ("sam2", "configs/sam2.1_hiera_t512.yaml"),
    ("eff", "small"): ("efficient_track_anything", "configs/efficienttam_s_512x512.yaml"),
    ("eff", "tiny"): ("efficient_track_anything", "configs/efficienttam_ti_512x512.yaml"),
}


def apply_thread_env(num_threads: int | None) -> None:
    """Apply BLAS/OpenMP thread environment limits before importing PyTorch."""
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
    """Enforce CPU isolation before importing PyTorch when ``device == 'cpu'``."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"


def recist_endpoints(
    recist_per_lab: np.ndarray,
    subtract_thickness: float = SUBTRACT_THICKNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal-axis (SVD) endpoints of the rasterized RECIST line as float32 [x, y]."""
    ys, xs = np.where(recist_per_lab > 0)
    if len(xs) < 2:
        raise ValueError("RECIST line must contain at least two pixels")
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
    else:
        centroid = coords.mean(0)
        c0 = coords - centroid
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        u = vt[0].astype(np.float32)
        proj = c0 @ u
        lo, hi = float(proj.min()), float(proj.max())
        mid = centroid + ((hi + lo) / 2.0) * u
        half = (hi - lo) / 2.0
        p1, p2 = mid - half * u, mid + half * u

    if subtract_thickness > 0:
        v = p2 - p1
        length = float(np.linalg.norm(v))
        if length > subtract_thickness:
            mid = (p1 + p2) / 2.0
            u = v / length
            half = (length - subtract_thickness) / 2.0
            p1, p2 = mid - half * u, mid + half * u
    return p1.astype(np.float32), p2.astype(np.float32)


def recist_diameter(recist_per_lab: np.ndarray) -> float:
    p1, p2 = recist_endpoints(recist_per_lab)
    return float(np.linalg.norm(p1 - p2))


def recist_bbox(
    recist_per_lab: np.ndarray,
    scale: float = BOX_SCALE,
    shift: int = 0,
) -> np.ndarray:
    """Square box centered on the RECIST midpoint with side = ``scale * diameter``."""
    h, w = recist_per_lab.shape
    p1, p2 = recist_endpoints(recist_per_lab)
    center = (p1 + p2) / 2.0
    half = (float(np.linalg.norm(p1 - p2)) * scale) / 2.0
    x_min = int(round(center[0] - half)) - shift
    x_max = int(round(center[0] + half)) + shift
    y_min = int(round(center[1] - half)) - shift
    y_max = int(round(center[1] + half)) + shift
    return np.array([
        max(0, x_min),
        max(0, y_min),
        min(w - 1, x_max),
        min(h - 1, y_max),
    ])


def recist_points_at(recist_per_lab: np.ndarray, percentiles) -> np.ndarray:
    """Sample [x, y] points along the RECIST line at given percentiles (center-first)."""
    p1, p2 = recist_endpoints(recist_per_lab)
    mid = (p1 + p2) / 2.0
    pts = []
    for p in sorted(percentiles, key=lambda q: abs(q - 50.0)):
        f = (p - 50.0) / 50.0
        end = p2 if f >= 0 else p1
        pts.append(mid + abs(f) * (end - mid))
    return np.stack(pts).astype(np.float32)


def sample_points_in_bbox_grid(bbox: np.ndarray, n: int) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    g = int(np.ceil(np.sqrt(n)))
    xv, yv = np.meshgrid(
        np.linspace(x_min, x_max, g, dtype=int),
        np.linspace(y_min, y_max, g, dtype=int),
    )
    return np.stack([xv.ravel(), yv.ravel()], axis=1)[:n]


def prompt_points(recist_per_lab: np.ndarray, prompt: str) -> np.ndarray:
    if prompt in PROMPT_PERCENTILES:
        return recist_points_at(recist_per_lab, PROMPT_PERCENTILES[prompt])
    if prompt == "npoints":
        return recist_points_at(recist_per_lab, np.linspace(0.0, 100.0, N_POINTS + 2)[1:-1])
    if prompt == "grid9":
        return sample_points_in_bbox_grid(recist_bbox(recist_per_lab, scale=BOX_SCALE), n=9)
    raise ValueError(f"Unsupported point prompt: {prompt!r}")


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    """Convert ``(D, H, W)`` uint8 grayscale volume to ``(D, 3, image_size, image_size)``."""
    from PIL import Image

    d = array.shape[0]
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        out[i] = np.array(img.resize((image_size, image_size))).transpose(2, 0, 1)
    return out


def load_gt(npz_data, gt_key: str, npz_name: str, gts_path: str | None) -> tuple[np.ndarray | None, str | None]:
    src = npz_data
    if gts_path is not None:
        src = np.load(os.path.join(gts_path, npz_name), "r", allow_pickle=False)
    keys = getattr(src, "files", list(src.keys()))
    if gt_key == "auto":
        for k in ("gts", "gts_hidden"):
            if k in keys:
                return src[k], k
        return None, None
    if gt_key not in keys:
        raise KeyError(f"{npz_name}: missing GT key {gt_key!r}; found {keys}")
    return src[gt_key], gt_key


def resolve_cfg(cfg: str | None, backend: str, model_size: str) -> str:
    """Resolve Hydra config YAML to a leading-slash absolute path."""
    default_pkg, default_rel = DEFAULT_CFG_BY_TARGET[(backend, model_size)]
    pkg_name = default_pkg
    pkg = importlib.import_module(pkg_name)
    pkg_dir = Path(pkg.__file__).resolve().parent

    if cfg is None:
        candidate = pkg_dir / default_rel
    else:
        p = Path(cfg).expanduser()
        if p.is_file():
            candidate = p.resolve()
        elif (pkg_dir / cfg).is_file():
            candidate = (pkg_dir / cfg).resolve()
        else:
            raise FileNotFoundError(
                f"Config YAML not found: {cfg!r} (checked filesystem and {pkg_dir})"
            )

    if candidate.suffix not in (".yaml", ".yml"):
        raise ValueError(f"--cfg must be a YAML config, got {candidate}")
    return "/" + str(candidate)


def build_predictor(
    *,
    backend: str,
    model_size: str,
    device: str,
    weights: str | None = None,
    cfg: str | None = None,
    disable_image_encoder_compile: bool = True,
    weights_cache_dir: str | None = None,
) -> tuple[object, str]:
    """Build SAM2 or EfficientTAM predictor and load weights onto ``device``."""
    cfg_arg = resolve_cfg(cfg, backend, model_size)
    if weights:
        ckpt = os.path.abspath(weights)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"--weights not found: {ckpt}")
    else:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_WEIGHTS[(backend, model_size)],
            cache_dir=weights_cache_dir,
        )

    if backend == "medsam2":
        from sam2.build_sam import build_sam2_video_predictor_npz

        predictor = build_sam2_video_predictor_npz(cfg_arg, ckpt, device=device)
        return predictor, ckpt

    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor_npz

    disable_compile = disable_image_encoder_compile or device == "cpu"
    overrides = ["++model.compile_image_encoder=false"] if disable_compile else []
    predictor = build_efficienttam_video_predictor_npz(
        cfg_arg,
        ckpt,
        device=device,
        hydra_overrides_extra=overrides,
    )
    return predictor, ckpt


def run_case(
    *,
    input_npz: str | Path,
    output_npz: str | Path,
    metrics_json: str | Path,
    backend: str,
    model_size: str,
    device: str = "cpu",
    gpu_index: int | None = None,
    weights: str | None = None,
    cfg: str | None = None,
    prompt: str = "box",
    shift: int = 0,
    gt_key: str = "auto",
    gts_path: str | None = None,
    window_npz: str | Path | None = None,
    require_raw_hu: bool = False,
    disable_image_encoder_compile: bool = True,
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
    """Run fresh-model-reload inference and efficiency profiling for one case."""
    configure_device_env(device)
    apply_thread_env(num_threads)

    import torch

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
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

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    input_path = Path(input_npz).resolve()
    case_id = input_path.stem
    eff_model_name = model_name or f"{backend}_{model_size}_{prompt}"
    metrics_path = Path(metrics_json)

    with CaseProfiler(
        device=device,
        gpu_index=gpu_index,
        sample_interval=sample_interval,
        gpu_tolerance_mb=gpu_tolerance_mb,
    ) as profiler:
        predictor = None
        try:
            # 1. Model load (counted in runtime_s)
            with profiler.phase("model_load"):
                predictor, ckpt_path = build_predictor(
                    backend=backend,
                    model_size=model_size,
                    device=device,
                    weights=weights,
                    cfg=cfg,
                    disable_image_encoder_compile=disable_image_encoder_compile,
                )

            # 2. Data load & decompression (excluded from runtime_s)
            with profiler.phase("data_load"):
                with np.load(input_path, allow_pickle=False) as npz_data:
                    spacing = np.asarray(npz_data["spacing"])
                    recist = np.asarray(npz_data["recist"])
                    gts_3d, resolved_gt_key = load_gt(npz_data, gt_key, input_path.name, gts_path)
                    if window_npz is not None:
                        with np.load(Path(window_npz).resolve(), allow_pickle=False) as w_data:
                            img_3d_ori = abdomen_image(w_data, require_raw=False)
                    else:
                        img_3d_ori = abdomen_image(npz_data, require_raw=require_raw_hu)
                if img_3d_ori.shape != recist.shape:
                    raise ValueError(f"Image shape {img_3d_ori.shape} != RECIST shape {recist.shape}")
                if np.max(img_3d_ori) >= 256:
                    raise ValueError("SAM input image intensities must lie in [0, 255]")

            # 3. Resize + RGB + ImageNet normalization (excluded from runtime_s)
            with profiler.phase("preprocess"):
                video_height, video_width = img_3d_ori.shape[1], img_3d_ori.shape[2]
                if (video_height, video_width) != (IMAGE_SIZE, IMAGE_SIZE):
                    img_resized = resize_grayscale_to_rgb_and_resize(img_3d_ori, IMAGE_SIZE)
                else:
                    img_resized = img_3d_ori[:, None].repeat(3, axis=1).astype(np.float32)
                img_tensor = torch.from_numpy(img_resized / 255.0)
                img_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
                img_std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
                img_tensor -= img_mean
                img_tensor /= img_std

            unique_labs = [int(x) for x in np.unique(recist) if x != 0]
            metadata = {
                "dataset": dataset_name or "",
                "case_id": case_id,
                "model": eff_model_name,
                "backend": backend,
                "model_size": model_size,
                "prompt": prompt,
                "device": device,
                "num_threads": int(torch.get_num_threads()),
                "source_npz": str(input_path),
                "source_sha256": sha256(input_path),
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": sha256(ckpt_path),
                "gt_key": resolved_gt_key,
            }

            with IndependentPredictionWriter(
                shape=img_3d_ori.shape,
                recist=recist,
                spacing=spacing,
                gt=gts_3d,
                metadata=metadata,
                scratch=scratch,
            ) as writer:
                # 4. Pure model inference (counted in runtime_s)
                with profiler.phase("inference"):
                    with torch.inference_mode():
                        for ulab in unique_labs:
                            recist_per_lab = (recist == ulab) * ulab
                            z_indices = np.unique(np.where(recist == ulab)[0])
                            if len(z_indices) != 1:
                                raise ValueError(
                                    f"Expected one z slice for recist=={ulab}, got {z_indices.tolist()}"
                                )
                            z_mid_orig = int(z_indices[0])
                            diameter = recist_diameter(recist_per_lab[z_mid_orig])
                            z_span = diameter * (((spacing[0] + spacing[1]) / 2.0) / spacing[2])
                            half_z = int(z_span * Z_SLAB_SCALE / 2.0)
                            z_lo = max(0, z_mid_orig - half_z)
                            z_hi = min(img_3d_ori.shape[0] - 1, z_mid_orig + half_z)
                            if backend == "medsam2":
                                z_lo, z_hi = 0, img_3d_ori.shape[0] - 1

                            recist_crop = recist_per_lab[z_lo : z_hi + 1]
                            img_crop = img_tensor[z_lo : z_hi + 1]
                            z_mid = z_mid_orig - z_lo
                            box_2d = recist_bbox(recist_crop[z_mid], scale=BOX_SCALE, shift=shift)
                            pts = None if prompt == "box" else prompt_points(recist_crop[z_mid], prompt)

                            if torch_device.type == "cuda":
                                torch.cuda.synchronize(torch_device)
                            t_lesion = time.perf_counter()

                            writer.mark_attempted(ulab)
                            inference_state = predictor.init_state(img_crop, video_height, video_width)
                            if prompt == "box":
                                _, _, logits = predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=z_mid,
                                    obj_id=1,
                                    box=box_2d,
                                )
                            else:
                                _, _, logits = predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=z_mid,
                                    obj_id=1,
                                    points=pts,
                                    labels=np.ones(len(pts)),
                                )
                            mask_prompt = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

                            # Forward propagation
                            _, _, masks = predictor.add_new_mask(
                                inference_state,
                                frame_idx=z_mid,
                                obj_id=1,
                                mask=mask_prompt,
                            )
                            writer.write_slice(ulab, z_mid_orig, (masks[0] > 0.0).cpu().numpy()[0])
                            for fi, _, lg in predictor.propagate_in_video(
                                inference_state,
                                start_frame_idx=z_mid,
                                reverse=False,
                            ):
                                writer.write_slice(ulab, z_lo + int(fi), (lg[0] > 0.0).cpu().numpy()[0])

                            # Reset and reverse propagation
                            predictor.reset_state(inference_state)
                            inference_state = predictor.init_state(img_crop, video_height, video_width)
                            predictor.add_new_mask(
                                inference_state,
                                frame_idx=z_mid,
                                obj_id=1,
                                mask=mask_prompt,
                            )
                            for fi, _, lg in predictor.propagate_in_video(
                                inference_state,
                                start_frame_idx=z_mid,
                                reverse=True,
                            ):
                                writer.write_slice(ulab, z_lo + int(fi), (lg[0] > 0.0).cpu().numpy()[0])
                            predictor.reset_state(inference_state)

                            if torch_device.type == "cuda":
                                torch.cuda.synchronize(torch_device)
                            profiler.record_lesion_time(ulab, time.perf_counter() - t_lesion)

                # 5. Optional postprocess (counted in runtime_s) + compressed NPZ save
                receipt, post_npz, post_rows = finalize_and_save_predictions(
                    writer,
                    output_npz,
                    profiler=profiler,
                    postprocess=postprocess,
                    overwrite=overwrite,
                    scratch=scratch,
                )

            del predictor
            predictor = None
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

            record = profiler.finalize(
                extra_metadata={
                    "dataset": dataset_name or "",
                    "case_id": case_id,
                    "model": eff_model_name,
                    "family": "sam",
                    "backend": backend,
                    "model_size": model_size,
                    "prompt": prompt,
                    "postprocess": postprocess,
                    "num_threads": int(torch.get_num_threads()),
                    "n_lesions": len(receipt["instance_ids"]),
                    "shape_zyx": list(img_3d_ori.shape),
                    "spacing_xyz": [float(s) for s in spacing],
                    "source_npz": str(input_path),
                    "source_sha256": metadata["source_sha256"],
                    "prediction_npz": str(Path(output_npz).resolve()),
                    "prediction_sha256": receipt["output_sha256"],
                    "postprocessed_prediction_npz": str(post_npz.resolve()) if post_npz else None,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "rows": receipt["rows"],
                    "postprocessed_rows": post_rows,
                    "status": "ok",
                }
            )
        except BaseException as exc:
            if predictor is not None:
                del predictor
            gc.collect()
            fail_record = profiler.finalize(
                extra_metadata={
                    "dataset": dataset_name or "",
                    "case_id": case_id,
                    "model": eff_model_name,
                    "family": "sam",
                    "backend": backend,
                    "model_size": model_size,
                    "prompt": prompt,
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
        default=Path("runs/sam"),
        help="Output directory (default: runs/sam)",
    )
    ap.add_argument("--backend", choices=("eff", "medsam2"), default="eff")
    ap.add_argument("--model-size", choices=("small", "tiny"), default="tiny")
    ap.add_argument("--weights", type=str, default=None, help="Path to local checkpoint (.pt)")
    ap.add_argument("--cfg", type=str, default=None, help="YAML config path (optional)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index for NVML monitoring")
    ap.add_argument("--prompt", choices=PROMPT_CHOICES, default="box")
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--gt-key", choices=("auto", "gts", "gts_hidden"), default="auto")
    ap.add_argument("--gts-path", type=str, default=None)
    ap.add_argument("--window-dir", type=Path, default=None, help="Optional uint8 window twin directory")
    ap.add_argument("--require-raw-hu", action="store_true")
    ap.add_argument("--disable-image-encoder-compile", action="store_true")
    ap.add_argument("--num-threads", type=int, default=1)
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
        window_path = (args.window_dir / npz_path.name) if args.window_dir else None
        out_npz = pred_dir / npz_path.name
        out_json = metrics_dir / f"{npz_path.stem}.metrics.json"
        if out_npz.exists() and out_json.exists() and not args.overwrite:
            print(f"[skip] {npz_path.name} already exists")
            continue
        rec = run_case(
            input_npz=npz_path,
            output_npz=out_npz,
            metrics_json=out_json,
            backend=args.backend,
            model_size=args.model_size,
            device=args.device,
            gpu_index=args.gpu_index,
            weights=args.weights,
            cfg=args.cfg,
            prompt=args.prompt,
            shift=args.shift,
            gt_key=args.gt_key,
            gts_path=args.gts_path,
            window_npz=window_path,
            require_raw_hu=args.require_raw_hu,
            disable_image_encoder_compile=args.disable_image_encoder_compile,
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
