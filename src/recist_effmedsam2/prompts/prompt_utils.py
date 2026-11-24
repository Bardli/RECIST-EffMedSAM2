"""Canonical RECIST -> prompt helpers shared across SAM2/EffMedSAM2 and nnInteractive.

All helpers operate on a single 2D RECIST slice (H, W) where nonzero pixels
trace the RECIST diameter line for one lesion. Returned coordinates are (x, y).
"""
from __future__ import annotations

import re as _re

import numpy as np

from . import prompts_from_3d_mask as _p3d

SEED = 2024


def seed_all() -> None:
    np.random.seed(SEED)


def get_diameter_bbox(recist_per_lab: np.ndarray, shift: int = 0) -> np.ndarray:
    """Return [x_min, y_min, x_max, y_max] box inscribed around the RECIST line."""
    H, W = recist_per_lab.shape
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    p1 = coords[0]
    p2 = coords[-1]

    center = ((p1 + p2) / 2).astype(int)
    diameter = np.linalg.norm(p1 - p2)
    half_side = int(diameter / 2)

    x_min = center[0] - half_side
    x_max = center[0] + half_side
    y_min = center[1] - half_side
    y_max = center[1] + half_side

    x_min = max(0, x_min - shift)
    y_min = max(0, y_min - shift)
    x_max = min(W - 1, x_max + shift)
    y_max = min(H - 1, y_max + shift)

    return np.array([x_min, y_min, x_max, y_max])


def get_center_from_recist(recist_per_lab: np.ndarray) -> np.ndarray:
    """Return (1, 2) array containing the midpoint of the RECIST line as [[x, y]]."""
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)
    p1 = coords[0]
    p2 = coords[-1]
    center = ((p1 + p2) / 2).astype(np.float32)
    return np.array([[center[0], center[1]]])


def get_n_points_from_recist(recist_per_lab: np.ndarray, n: int = 5) -> np.ndarray:
    """Return (n, 2) array of [x, y] points sampled from RECIST line pixels."""
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)
    if len(coords) < n:
        raise ValueError(
            f"Cannot sample {n} points; RECIST line only has {len(coords)} pixels."
        )
    sampled_indices = np.random.choice(len(coords), size=n, replace=False)
    return coords[sampled_indices]


def get_center_and_endpoints_from_recist(recist_per_lab: np.ndarray) -> np.ndarray:
    """Return (3, 2) array: [center, endpoint1, endpoint2] as [x, y]."""
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)
    if len(coords) < 2:
        raise ValueError("RECIST line must contain at least two points")
    p1 = coords[0].astype(np.float32)
    p2 = coords[-1].astype(np.float32)
    center = ((p1 + p2) / 2).astype(np.float32)
    return np.array([center, p1, p2])


def _recist_midpoint_and_halfvecs(recist_per_lab: np.ndarray):
    """Return (mid, h1, h2) in [x, y] along the line's principal axis (SVD)."""
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


def get_centerout_points_from_recist(recist_per_lab: np.ndarray, n: int) -> np.ndarray:
    """Return (n, 2) [x, y] points sampled outward from the RECIST midpoint."""
    mid, h1, h2 = _recist_midpoint_and_halfvecs(recist_per_lab)
    if n == 3:
        pts = [mid, mid + 0.50 * h1, mid + 0.50 * h2]
    elif n == 5:
        pts = [
            mid,
            mid + 0.50 * h1,
            mid + 0.50 * h2,
            mid + 0.85 * h1,
            mid + 0.85 * h2,
        ]
    else:
        raise ValueError(f"center-out sampling supports n in (3, 5), got {n}")
    return np.asarray(pts, dtype=np.float32)


def points_along_recist_percentiles(recist_per_lab: np.ndarray, percentiles) -> np.ndarray:
    """Return points at the given along-line percentiles (0..100), center-first."""
    mid, h1, h2 = _recist_midpoint_and_halfvecs(recist_per_lab)
    order = sorted(percentiles, key=lambda p: abs(p - 50))
    pts = []
    for p in order:
        f = (p - 50) / 50.0
        h = h2 if f >= 0 else h1
        pts.append(mid + abs(f) * h)
    return np.asarray(pts, dtype=np.float32)


def points_for_mode(recist_per_lab: np.ndarray, mode: str) -> np.ndarray:
    """Dispatch to the right helper; returns (N, 2) array of [x, y] points."""
    if mode == "5pt_10_90":
        return points_along_recist_percentiles(recist_per_lab, [10, 30, 50, 70, 90])
    if mode in ("1_center_point", "1center"):
        return get_center_from_recist(recist_per_lab)
    if mode in ("3_points", "3points"):
        return get_center_and_endpoints_from_recist(recist_per_lab)
    if mode in ("5_points", "5points"):
        return get_n_points_from_recist(recist_per_lab, n=5)
    if mode == "3pt_25":
        return get_centerout_points_from_recist(recist_per_lab, n=3)
    if mode == "5pt_25_42":
        return get_centerout_points_from_recist(recist_per_lab, n=5)
    raise ValueError(f"Unknown point prompt mode: {mode}")


RECIST_NEG_MODES: tuple[str, ...] = (
    "recist_3pos_2neg",
    "recist_3pos_4neg",
    "recist_3pos_6neg",
)


def _recist_endpoints_and_center(recist_per_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(recist_per_lab > 0)
    if len(xs) < 2:
        raise ValueError("RECIST line must contain at least two points")
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    p1 = coords[0]
    p2 = coords[-1]
    center = ((p1 + p2) / 2).astype(np.float32)
    return p1, p2, center


def _filter_bg_points(
    bg: np.ndarray,
    fg: np.ndarray,
    shape: tuple[int, int],
    forbidden_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Drop unsafe background points instead of clipping them onto the target."""
    h, w = shape
    fg_rounded = {(int(round(x)), int(round(y))) for x, y in np.asarray(fg)}
    kept: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in np.asarray(bg, dtype=np.float32):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        if (xi, yi) in fg_rounded or (xi, yi) in seen:
            continue
        if forbidden_mask is not None and bool(forbidden_mask[yi, xi]):
            continue
        kept.append([float(x), float(y)])
        seen.add((xi, yi))
    return np.asarray(kept, dtype=np.float32).reshape(-1, 2)


def recist_points_for_mode(
    recist_per_lab: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (fg_xy, bg_xy) from a RECIST line only."""
    if mode not in RECIST_NEG_MODES:
        raise ValueError(f"Unknown RECIST negative mode: {mode!r}")

    p1, p2, center = _recist_endpoints_and_center(recist_per_lab)
    fg = np.array([center, p1, p2], dtype=np.float32)

    vec = p2 - p1
    diameter = float(np.linalg.norm(vec))
    if diameter <= 0:
        return fg, np.empty((0, 2), dtype=np.float32)
    unit = vec / diameter
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)

    half_side = diameter / 2.0
    buffered = half_side * 1.2
    corners = np.array(
        [
            p1 + perp * half_side,
            p1 - perp * half_side,
            p2 + perp * half_side,
            p2 - perp * half_side,
        ],
        dtype=np.float32,
    )
    minor = np.array(
        [
            center + perp * buffered,
            center - perp * buffered,
        ],
        dtype=np.float32,
    )

    if mode.endswith("2neg"):
        bg = minor
    elif mode.endswith("4neg"):
        bg = corners
    else:
        bg = np.concatenate([corners, minor], axis=0)

    return fg, _filter_bg_points(bg, fg, recist_per_lab.shape, forbidden_mask=recist_per_lab > 0)


MASK_MODES: tuple[str, ...] = tuple(
    [f"mask_{n}" for n in range(1, 6)]
    + [f"mask_{n}pos_4neg" for n in range(1, 6)]
    + [f"mask_{n}pos_6neg" for n in range(1, 6)]
)


def parse_mask_mode(mode: str) -> tuple[int, int]:
    """Return (fg_count, bg_count) for a ``mask_*`` mode name."""
    m = _re.fullmatch(r"mask_(\d)(?:pos_(\d)neg)?", mode)
    if not m:
        raise ValueError(f"Unknown mask mode: {mode!r}")
    fg_n = int(m.group(1))
    bg_n = int(m.group(2)) if m.group(2) is not None else 0
    if fg_n not in (1, 2, 3, 4, 5):
        raise ValueError(f"fg count must be 1..5, got {fg_n} (mode={mode!r})")
    if bg_n not in (0, 4, 6):
        raise ValueError(f"bg count must be 0/4/6, got {bg_n} (mode={mode!r})")
    return fg_n, bg_n


def _fallback_centroid(mask_slice: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_slice > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)


def _pair(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1, 2)


def mask_points_for_mode(
    mask_slice_2d: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (fg_xy, bg_xy) point arrays from a single GT mask slice."""
    fg_n, bg_n = parse_mask_mode(mode)
    slice_int = (mask_slice_2d > 0).astype(np.int32)
    H, W = slice_int.shape
    neg_spacing = (0, W, H)

    try:
        center = _p3d.get_center_pt(slice_int)
        ends = _pair(_p3d.get_recist_pts(slice_int))
        quart = _pair(_p3d.get_recist_25_75_pts(slice_int))
    except Exception:
        return _fallback_centroid(slice_int), np.empty((0, 2), dtype=np.float32)

    if fg_n == 1:
        fg = center.reshape(1, 2)
    elif fg_n == 2:
        fg = ends
    elif fg_n == 3:
        fg = np.concatenate([center.reshape(1, 2), ends], axis=0)
    elif fg_n == 4:
        fg = np.concatenate([ends, quart], axis=0)
    else:
        fg = np.concatenate([center.reshape(1, 2), ends, quart], axis=0)
    fg = fg.astype(np.float32)

    if bg_n == 0:
        bg = np.empty((0, 2), dtype=np.float32)
    else:
        try:
            recist_pts = _p3d.get_recist_pts(slice_int)
            neg_flat = _p3d.get_negative_pts(
                slice_int, spacing=neg_spacing, recist_pts=recist_pts,
            )
            neg_all = _pair(neg_flat)
            bg = neg_all[:4] if bg_n == 4 else neg_all
            bg = _filter_bg_points(bg, fg, slice_int.shape, forbidden_mask=slice_int > 0)
        except Exception:
            bg = np.empty((0, 2), dtype=np.float32)

    return fg, bg
