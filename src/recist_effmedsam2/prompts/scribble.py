"""RECIST raster to scribble transform (pixels, not mm)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _dilate_plane_to_thickness(plane: np.ndarray, thickness) -> np.ndarray:
    """Dilate a 2D scribble mask in-plane to match preferred scribble thickness."""
    out = plane > 0
    t = [int(round(v)) for v in thickness] if thickness is not None else [2, 2, 2]
    in_plane = max(1, max(t[1:]))
    if in_plane <= 1 or not out.any():
        return out
    return binary_dilation(out, iterations=in_plane - 1)


def _inset_line_endpoints(plane: np.ndarray, inset: float) -> np.ndarray:
    """Trim a 1px RECIST line by ``inset`` px (Euclidean) at both endpoints."""
    line = plane > 0
    if inset <= 0 or not line.any():
        return line
    coords = np.argwhere(line).astype(np.float64)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    near_end = (np.linalg.norm(coords - coords[i], axis=1) <= inset) | (
        np.linalg.norm(coords - coords[j], axis=1) <= inset
    )
    keep = coords[~near_end]
    if len(keep) == 0:
        mid = (coords[i] + coords[j]) / 2
        keep = coords[[np.argmin(np.linalg.norm(coords - mid, axis=1))]]
    out = np.zeros_like(line)
    out[tuple(keep.astype(int).T)] = True
    return out


SCRIBBLE_STROKES = ("board", "raw", "skeleton", "skeleton_thick")


def scribble_plane(plane: np.ndarray, thickness, stroke: str = "board", inset: float = 0) -> np.ndarray:
    """Convert a 2D RECIST line slice into a boolean 2D scribble stroke."""
    line = plane > 0
    if stroke in ("skeleton", "skeleton_thick"):
        from skimage.morphology import skeletonize

        line = skeletonize(line)
    line = _inset_line_endpoints(line, inset)
    if stroke in ("board", "skeleton_thick"):
        return _dilate_plane_to_thickness(line, thickness)
    if stroke in ("raw", "skeleton"):
        return line
    raise ValueError(f"unknown scribble stroke {stroke!r}")
