"""Generate prompts from a 2D lesion mask slice using scikit-image regionprops."""
from __future__ import annotations

import numpy as np
from skimage.measure import regionprops


def get_slice_properties(mask_slice: np.ndarray) -> tuple[float, float, float, float]:
    """Return (x_cent, y_cent, orientation, semi_major_axis_len) for a binary/label 2D mask."""
    try:
        props = regionprops(mask_slice)[0]
        y_cent, x_cent = props.centroid
        orientation = props.orientation
        semi_maj_axis_len = props.axis_major_length / 2
    except Exception as exc:
        raise ValueError(f"regionprops failed (sum={int(mask_slice.sum())}): {exc}") from exc
    return x_cent, y_cent, orientation, semi_maj_axis_len


def get_center_pt(mask_slice: np.ndarray) -> np.ndarray:
    """Return [x_cent, y_cent] centroid of the mask slice."""
    x_cent, y_cent, _orientation, _semi_maj = get_slice_properties(mask_slice)
    return np.array([x_cent, y_cent])


def get_recist_pts(mask_slice: np.ndarray) -> np.ndarray:
    """Return [x_r1, y_r1, x_r2, y_r2] major-axis endpoints of the mask slice."""
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)
    x_r1 = x_cent - np.sin(orientation) * semi_maj_axis_len
    y_r1 = y_cent - np.cos(orientation) * semi_maj_axis_len
    x_r2 = x_cent + np.sin(orientation) * semi_maj_axis_len
    y_r2 = y_cent + np.cos(orientation) * semi_maj_axis_len
    return np.array([x_r1, y_r1, x_r2, y_r2])


def get_negative_pts(
    mask_slice: np.ndarray,
    spacing: np.ndarray | tuple[int, int, int],
    recist_pts: np.ndarray | None = None,
) -> np.ndarray:
    """Return 4 rotated-bbox corners + 2 buffered minor-axis points."""
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)
    if recist_pts is None:
        recist_pts = get_recist_pts(mask_slice)
    x_r1, y_r1, x_r2, y_r2 = recist_pts

    x_min1 = min(spacing[1] - 1, x_cent + np.cos(orientation) * semi_maj_axis_len * 1.2)
    y_min1 = min(spacing[2] - 1, y_cent - np.sin(orientation) * semi_maj_axis_len * 1.2)
    x_min2 = min(spacing[1] - 1, x_cent - np.cos(orientation) * semi_maj_axis_len * 1.2)
    y_min2 = min(spacing[2] - 1, y_cent + np.sin(orientation) * semi_maj_axis_len * 1.2)

    x1 = min(spacing[1] - 1, x_r1 + np.cos(orientation) * semi_maj_axis_len)
    y1 = min(spacing[2] - 1, y_r1 - np.sin(orientation) * semi_maj_axis_len)
    x2 = min(spacing[1] - 1, x_r1 - np.cos(orientation) * semi_maj_axis_len)
    y2 = min(spacing[2] - 1, y_r1 + np.sin(orientation) * semi_maj_axis_len)
    x3 = min(spacing[1] - 1, x_r2 + np.cos(orientation) * semi_maj_axis_len)
    y3 = min(spacing[2] - 1, y_r2 - np.sin(orientation) * semi_maj_axis_len)
    x4 = min(spacing[1] - 1, x_r2 - np.cos(orientation) * semi_maj_axis_len)
    y4 = min(spacing[2] - 1, y_r2 + np.sin(orientation) * semi_maj_axis_len)

    return np.array([x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2])


def get_recist_25_75_pts(mask_slice: np.ndarray) -> np.ndarray:
    """Return [x_r25, y_r25, x_r75, y_r75] points along the major axis."""
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)
    x_r25 = x_cent - np.sin(orientation) * semi_maj_axis_len / 2
    y_r25 = y_cent - np.cos(orientation) * semi_maj_axis_len / 2
    x_r75 = x_cent + np.sin(orientation) * semi_maj_axis_len / 2
    y_r75 = y_cent + np.cos(orientation) * semi_maj_axis_len / 2
    return np.array([x_r25, y_r25, x_r75, y_r75])
