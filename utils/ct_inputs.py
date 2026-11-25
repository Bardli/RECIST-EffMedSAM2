"""CT NPZ input loading, HU decoding, full float32 windowing, and slice resizing."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ENCODING = "nifti_stored_scaled_v1"
_UNIT_TOL = 1e-5


def raw_hu(data) -> np.ndarray:
    """Reconstruct float64 Hounsfield units from a ``nifti_stored_scaled_v1`` NPZ."""
    if "intensity_encoding" not in data or str(data["intensity_encoding"].item()) != ENCODING:
        raise ValueError("Raw HU requires nifti_stored_scaled_v1 NPZ; uint8 inputs cannot restore HU")
    stored = data["imgs"]
    slope, intercept = float(data["hu_slope"]), float(data["hu_intercept"])
    if stored.ndim != 3 or not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("Invalid stored CT or NIfTI scaling")
    hu = stored.astype(np.float64) * slope + intercept
    if not np.isfinite(hu).all():
        raise ValueError("Non-finite CT intensities")
    return hu


def abdomen_image(data, require_raw: bool = False) -> np.ndarray:
    """Return a float32 ``(D, H, W)`` volume in ``[0.0, 1.0]`` (L40/W400 ``[-160, 240]`` if raw HU)."""
    if "intensity_encoding" in data:
        hu = raw_hu(data)
        return ((np.clip(hu, -160.0, 240.0) + 160.0) / 400.0).astype(np.float32)
    if require_raw:
        raise ValueError("Strict raw-HU mode requires preserved nifti_stored_scaled_v1 intensities")
    image = np.asarray(data["imgs"])
    if image.ndim != 3 or image.size == 0:
        raise ValueError(f"Windowed/normalized SAM input must be a non-empty 3D volume, got shape {image.shape}")
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Non-finite SAM input intensities")
        vmin = float(image.min())
        vmax = float(image.max())
        if vmin < -_UNIT_TOL:
            raise ValueError(
                f"Floating SAM input must be normalized in [0, 1] or [0, 255]; got min={vmin}"
            )
        if vmax <= 1.0 + _UNIT_TOL:
            return np.clip(image.astype(np.float32, copy=False), 0.0, 1.0).astype(np.float32, copy=False)
        if vmax <= 255.0 + _UNIT_TOL:
            return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
        raise ValueError(
            f"Floating SAM input must be normalized in [0, 1] or [0, 255]; got max={vmax}"
        )
    raise ValueError(
        f"Windowed/normalized SAM input must be a 3D float ([0, 1] or [0, 255]) or uint8 volume, "
        f"got dtype={image.dtype}"
    )


def nninteractive_image(data, require_raw: bool = False) -> np.ndarray:
    """Return raw HU (or pass-through array when ``require_raw=False``) for nnInteractive."""
    if "intensity_encoding" in data:
        return raw_hu(data)
    if require_raw:
        raise ValueError("nnInteractive raw-HU mode rejects pre-windowed uint8 inputs")
    return data["imgs"]


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Resize ``(D, H, W)`` float32 grayscale volume to ``(D, 3, image_size, image_size)`` without uint8 truncation."""
    import cv2

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp and omp.isdigit():
        n_threads = int(omp)
        cv2.setNumThreads(0 if n_threads <= 1 else n_threads)

    arr = np.asarray(array, dtype=np.float32)
    d = arr.shape[0]
    out = np.empty((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        resized = cv2.resize(arr[i], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out[i] = np.clip(resized, 0.0, 1.0)
    return out


def load_gt(
    npz_data,
    gt_key: str = "auto",
    npz_name: str = "",
    gts_path: str | Path | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Load optional GT mask from ``npz_data`` or an external ``gts_path`` directory."""
    src = npz_data
    if gts_path is not None:
        with np.load(Path(gts_path) / npz_name, allow_pickle=False) as ext:
            return _extract_gt(ext, gt_key, npz_name)
    return _extract_gt(src, gt_key, npz_name)


def _extract_gt(src, gt_key: str, npz_name: str) -> tuple[np.ndarray | None, str | None]:
    keys = getattr(src, "files", list(src.keys()))
    if gt_key == "auto":
        for k in ("gts", "gts_hidden"):
            if k in keys:
                return np.asarray(src[k]), k
        return None, None
    if gt_key not in keys:
        raise KeyError(f"{npz_name}: missing GT key {gt_key!r}; found {keys}")
    return np.asarray(src[gt_key]), gt_key


def discover_cases(
    input_path: str | Path,
    limit: int | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Return sorted ``*.npz`` paths from a single file or a directory."""
    p = Path(input_path)
    if p.is_file():
        return [p.resolve()]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    pattern = "**/*.npz" if recursive else "*.npz"
    cases = sorted(c.resolve() for c in p.glob(pattern) if c.is_file())
    if not cases:
        raise FileNotFoundError(f"No *.npz files found under {p} (recursive={recursive})")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        cases = cases[:limit]
    return cases


def auto_window_dir(input_path: Path, window_dir: Path | None) -> Path | None:
    """Use ``data/window_twins`` automatically when reading from default ``data/inputs``."""
    if window_dir is not None:
        return Path(window_dir)
    if input_path.resolve() == (Path.cwd() / "data" / "inputs").resolve():
        cand = Path.cwd() / "data" / "window_twins"
        if cand.is_dir() and any(cand.rglob("*.npz")):
            return cand
    return None


def resolve_window_npz(
    case_path: Path,
    input_root: Path,
    window_dir: Path | None,
) -> Path | None:
    """Resolve the matching pre-windowed companion NPZ for ``case_path``."""
    if window_dir is None:
        return None
    if Path(window_dir).is_file():
        return Path(window_dir).resolve()
    try:
        rel = case_path.resolve().relative_to(input_root.resolve())
        cand = Path(window_dir) / rel
        if cand.is_file():
            return cand.resolve()
    except ValueError:
        pass
    cand = Path(window_dir) / case_path.name
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Window twin NPZ missing for {case_path.name} under {window_dir}")
