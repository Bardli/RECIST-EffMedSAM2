"""Decode preserved NIfTI intensities; window continuously in float32 at the SAM boundary."""
from __future__ import annotations

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
    """Return a float32 (D, H, W) volume in [0.0, 1.0], applying L40/W400 [-160, 240] when raw HU is present."""
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
    """Return raw HU (or legacy array when ``require_raw=False``) for nnInteractive."""
    if "intensity_encoding" in data:
        return raw_hu(data)
    if require_raw:
        raise ValueError("nnInteractive raw-HU mode rejects pre-windowed uint8 inputs")
    return data["imgs"]
