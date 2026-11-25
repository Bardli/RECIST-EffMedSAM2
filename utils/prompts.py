"""RECIST prompt conversion helpers shared by EffMedSAM2, MedSAM2, and nnInteractive."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

SEED = 2024
SUBTRACT_THICKNESS = 2.0
SCRIBBLE_STROKES = ("board", "raw", "skeleton", "skeleton_thick")


def seed_all() -> None:
    np.random.seed(SEED)


def recist_endpoints(
    recist_per_lab: np.ndarray,
    subtract_thickness: float = SUBTRACT_THICKNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal-axis (SVD) endpoints of the rasterized RECIST line as float32 ``[x, y]``."""
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
    """Euclidean length (pixels) of the RECIST major-axis segment."""
    p1, p2 = recist_endpoints(recist_per_lab)
    return float(np.linalg.norm(p1 - p2))


def recist_bbox(
    recist_per_lab: np.ndarray,
    scale: float = 1.0,
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


def get_diameter_bbox(recist_per_lab: np.ndarray, shift: int = 0) -> np.ndarray:
    """Return ``[x_min, y_min, x_max, y_max]`` square box inscribed around the RECIST endpoints."""
    h, w = recist_per_lab.shape
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)
    p1 = coords[0]
    p2 = coords[-1]
    center = ((p1 + p2) / 2).astype(int)
    half_side = int(np.linalg.norm(p1 - p2) / 2)
    return np.array([
        max(0, center[0] - half_side - shift),
        max(0, center[1] - half_side - shift),
        min(w - 1, center[0] + half_side + shift),
        min(h - 1, center[1] + half_side + shift),
    ])


def _recist_midpoint_and_halfvecs(recist_per_lab: np.ndarray):
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(coords) < 2:
        raise ValueError("RECIST line must contain at least two points")
    c0 = coords - coords.mean(0)
    _, _, vt = np.linalg.svd(c0, full_matrices=False)
    proj = c0 @ vt[0]
    e1 = coords[proj.argmin()]
    e2 = coords[proj.argmax()]
    mid = (e1 + e2) / 2.0
    return mid, e1 - mid, e2 - mid


def points_for_mode(recist_per_lab: np.ndarray, mode: str) -> np.ndarray:
    """Return ``(N, 2)`` ``[x, y]`` points along the RECIST line (midpoint first)."""
    if mode in ("1_center_point", "1center"):
        ys, xs = np.where(recist_per_lab > 0)
        coords = np.stack([xs, ys], axis=1)
        return np.array([((coords[0] + coords[-1]) / 2)], dtype=np.float32)
    if mode in ("3_points", "3points"):
        ys, xs = np.where(recist_per_lab > 0)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        if len(coords) < 2:
            raise ValueError("RECIST line must contain at least two points")
        return np.array([(coords[0] + coords[-1]) / 2, coords[0], coords[-1]], dtype=np.float32)
    if mode in ("5_points", "5points"):
        ys, xs = np.where(recist_per_lab > 0)
        coords = np.stack([xs, ys], axis=1)
        if len(coords) < 5:
            raise ValueError(f"Cannot sample 5 points; RECIST line only has {len(coords)} pixels.")
        return coords[np.random.choice(len(coords), size=5, replace=False)]

    mid, h1, h2 = _recist_midpoint_and_halfvecs(recist_per_lab)
    if mode == "3pt_25":
        return np.asarray([mid, mid + 0.50 * h1, mid + 0.50 * h2], dtype=np.float32)
    if mode == "5pt_25_42":
        return np.asarray(
            [mid, mid + 0.50 * h1, mid + 0.50 * h2, mid + 0.85 * h1, mid + 0.85 * h2],
            dtype=np.float32,
        )
    if mode == "5pt_10_90":
        pts = []
        for p in sorted([10, 30, 50, 70, 90], key=lambda q: abs(q - 50)):
            f = (p - 50) / 50.0
            pts.append(mid + abs(f) * (h2 if f >= 0 else h1))
        return np.asarray(pts, dtype=np.float32)
    raise ValueError(f"Unknown point prompt mode: {mode}")


def _inset_line_endpoints(plane: np.ndarray, inset: float) -> np.ndarray:
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


def scribble_plane(
    plane: np.ndarray,
    thickness,
    stroke: str = "skeleton",
    inset: float = 3.0,
) -> np.ndarray:
    """Convert a 2D RECIST line slice into a boolean 2D scribble stroke."""
    line = plane > 0
    if stroke in ("skeleton", "skeleton_thick"):
        from skimage.morphology import skeletonize

        line = skeletonize(line)
    line = _inset_line_endpoints(line, inset)
    if stroke in ("board", "skeleton_thick"):
        t = [int(round(v)) for v in thickness] if thickness is not None else [2, 2, 2]
        in_plane = max(1, max(t[1:]))
        if in_plane > 1 and line.any():
            return binary_dilation(line, iterations=in_plane - 1)
        return line
    if stroke in ("raw", "skeleton"):
        return line
    raise ValueError(f"unknown scribble stroke {stroke!r}")
