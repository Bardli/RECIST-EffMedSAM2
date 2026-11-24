"""Decode preserved NIfTI intensities; window only at the SAM model boundary."""
from __future__ import annotations

import numpy as np

ENCODING = "nifti_stored_scaled_v1"


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
    """Return a uint8 (D, H, W) volume in [0, 255] using L40/W400 [-160, 240] when raw HU is present."""
    if "intensity_encoding" in data:
        hu = raw_hu(data)
        return ((np.clip(hu, -160.0, 240.0) + 160.0) / 400.0 * 255.0).astype(np.uint8)
    if require_raw:
        raise ValueError("Strict raw-HU mode requires preserved nifti_stored_scaled_v1 intensities")
    image = data["imgs"]
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Windowed SAM input must be a 3D uint8 volume")
    return image


def nninteractive_image(data, require_raw: bool = False) -> np.ndarray:
    """Return raw HU (or legacy array when ``require_raw=False``) for nnInteractive."""
    if "intensity_encoding" in data:
        return raw_hu(data)
    if require_raw:
        raise ValueError("nnInteractive raw-HU mode rejects pre-windowed uint8 inputs")
    return data["imgs"]
