"""Load direct raw-HU CT volumes (no reconstruction); window only at the SAM boundary."""
from __future__ import annotations

import numpy as np

ENCODING = "nifti_stored_scaled_v1"


def _is_direct_raw_hu_array(arr: np.ndarray) -> bool:
    """Return True if ``arr`` is a 3D signed-integer or floating-point CT volume."""
    return arr.ndim == 3 and (
        np.issubdtype(arr.dtype, np.signedinteger) or np.issubdtype(arr.dtype, np.floating)
    )


def _decode_legacy_scaled_hu(data) -> np.ndarray:
    """Reconstruct float64 HU from an optional legacy ``nifti_stored_scaled_v1`` NPZ."""
    encoding = str(np.asarray(data["intensity_encoding"]).item())
    if encoding != ENCODING:
        raise ValueError(f"Unsupported intensity_encoding {encoding!r}; expected raw HU or {ENCODING!r}")
    stored = np.asarray(data["imgs"])
    slope, intercept = float(data["hu_slope"]), float(data["hu_intercept"])
    if stored.ndim != 3 or not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("Invalid stored CT or NIfTI scaling")
    hu = stored.astype(np.float64) * slope + intercept
    if not np.isfinite(hu).all():
        raise ValueError("Non-finite CT intensities")
    return hu


def _window_abdomen_hu(hu: np.ndarray) -> np.ndarray:
    """Apply standard abdomen L40/W400 windowing ([-160, 240] HU -> uint8 [0, 255])."""
    return ((np.clip(hu, -160.0, 240.0) + 160.0) / 400.0 * 255.0).astype(np.uint8)


def raw_hu(data) -> np.ndarray:
    """Return raw Hounsfield units directly from ``imgs`` (no slope/intercept reconstruction).

    Direct raw-HU NPZs store raw CT HU values directly in ``imgs`` as a 3D
    signed-integer (e.g. ``int16``) or floating-point (e.g. ``float32``) array
    without scaling metadata. Legacy ``nifti_stored_scaled_v1`` NPZs carrying
    ``intensity_encoding`` remain optionally supported for backward compatibility.
    """
    if "intensity_encoding" in data:
        return _decode_legacy_scaled_hu(data)
    imgs = np.asarray(data["imgs"])
    if imgs.ndim != 3:
        raise ValueError(f"Raw HU volume must be 3D, got shape {imgs.shape}")
    if imgs.dtype == np.uint8:
        raise ValueError("Input imgs is uint8 (pre-windowed) and cannot be treated as raw HU")
    if not _is_direct_raw_hu_array(imgs):
        raise ValueError(
            f"Direct raw HU requires a 3D signed-integer or floating-point array, "
            f"got dtype={imgs.dtype}, ndim={imgs.ndim}"
        )
    if np.issubdtype(imgs.dtype, np.floating) and not np.isfinite(imgs).all():
        raise ValueError("Non-finite CT intensities")
    return imgs


def abdomen_image(data, require_raw: bool = False) -> np.ndarray:
    """Return a uint8 (D, H, W) volume in [0, 255], applying L40/W400 [-160, 240] to raw HU."""
    if "intensity_encoding" in data:
        return _window_abdomen_hu(raw_hu(data))
    image = np.asarray(data["imgs"])
    if image.ndim != 3:
        raise ValueError(f"SAM input volume must be 3D, got shape {image.shape}")
    if image.dtype == np.uint8:
        if require_raw:
            raise ValueError("Strict raw-HU mode rejects pre-windowed uint8 inputs")
        return image
    return _window_abdomen_hu(raw_hu(data))


def nninteractive_image(data, require_raw: bool = False) -> np.ndarray:
    """Return raw HU directly for nnInteractive (or pass through uint8 if ``require_raw=False``)."""
    if "intensity_encoding" in data:
        return raw_hu(data)
    image = np.asarray(data["imgs"])
    if image.ndim != 3:
        raise ValueError(f"nnInteractive input volume must be 3D, got shape {image.shape}")
    if image.dtype == np.uint8:
        if require_raw:
            raise ValueError("nnInteractive raw-HU mode rejects pre-windowed uint8 inputs")
        return image
    return raw_hu(data)
