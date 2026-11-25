"""Tests for CT intensity decoding, RECIST prompts, storage v2, and moving-centroid postprocess."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scipy.spatial.distance import pdist

from utils.ct_inputs import (
    abdomen_image,
    nninteractive_image,
    raw_hu,
    resize_grayscale_to_rgb_and_resize,
)
from utils.postprocess import apply_moving_centroid_to_channels
from utils.prompts import (
    axial_diameter_mm,
    find_recist_slices,
    get_diameter_bbox,
    max_axial_diameter_mm,
    points_for_mode,
    recist_bbox,
    recist_endpoints,
    recist_length_mm,
    scribble_plane,
)
from utils.storage import IndependentPredictionWriter, load_independent_prediction


def test_ct_inputs_raw_hu_and_windowing():
    stored = np.array([[[-1000, 40], [240, 500]]], dtype=np.int16)
    stored_3d = np.repeat(stored, 2, axis=0)
    data = {
        "intensity_encoding": np.array("nifti_stored_scaled_v1"),
        "imgs": stored_3d,
        "hu_slope": np.array(1.0),
        "hu_intercept": np.array(0.0),
    }

    hu = raw_hu(data)
    assert np.allclose(hu[0], [[-1000.0, 40.0], [240.0, 500.0]])
    win = abdomen_image(data, require_raw=True)
    assert win.dtype == np.float32
    assert win[0, 0, 0] == pytest.approx(0.0)
    assert win[0, 0, 1] == pytest.approx(0.5)
    assert win[0, 1, 0] == pytest.approx(1.0)
    assert win[0, 1, 1] == pytest.approx(1.0)

    # Sub-uint8 precision (0.5 HU difference -> 0.00125 < 1/255) is preserved end-to-end
    sub_data = {
        "intensity_encoding": np.array("nifti_stored_scaled_v1"),
        "imgs": np.full((2, 32, 32), 40.5, dtype=np.float32),
        "hu_slope": np.array(1.0),
        "hu_intercept": np.array(0.0),
    }
    sub_win = abdomen_image(sub_data, require_raw=True)
    assert sub_win.dtype == np.float32
    assert np.allclose(sub_win, 0.50125, atol=1e-6)
    assert not np.isclose(sub_win[0, 0, 0], 128.0 / 255.0, atol=1e-4)

    resized = resize_grayscale_to_rgb_and_resize(sub_win, 64)
    assert resized.shape == (2, 3, 64, 64)
    assert resized.dtype == np.float32
    assert np.allclose(resized, 0.50125, atol=1e-6)

    nn_img = nninteractive_image(data, require_raw=True)
    assert np.allclose(nn_img, hu)

    # Pre-windowed/normalized float32 in [0, 1] passes through losslessly
    f32_unit = {"imgs": sub_win.copy()}
    out_unit = abdomen_image(f32_unit, require_raw=False)
    assert out_unit.dtype == np.float32
    assert np.allclose(out_unit, 0.50125, atol=1e-6)

    # Pre-windowed float32 in [0, 255] scales smoothly to [0, 1] without uint8 quantization
    f32_255 = {"imgs": np.full((2, 4, 4), 127.5, dtype=np.float32)}
    out_255 = abdomen_image(f32_255, require_raw=False)
    assert out_255.dtype == np.float32
    assert np.allclose(out_255, 0.5, atol=1e-6)

    # Legacy uint8 [0, 255] remains compatible when require_raw=False and rejected when require_raw=True
    legacy_u8 = np.array([[[0, 128], [255, 255]]] * 2, dtype=np.uint8)
    uint8_data = {"imgs": legacy_u8}
    out_u8 = abdomen_image(uint8_data, require_raw=False)
    assert out_u8.dtype == np.float32
    assert np.allclose(out_u8, legacy_u8.astype(np.float32) / 255.0)
    with pytest.raises(ValueError):
        abdomen_image(uint8_data, require_raw=True)
    with pytest.raises(ValueError):
        abdomen_image(f32_unit, require_raw=True)
    with pytest.raises(ValueError):
        nninteractive_image(uint8_data, require_raw=True)

    # Out-of-range, non-finite, or unencoded int16 SAM inputs are rejected
    for bad in (
        np.full((2, 2, 2), -0.1, dtype=np.float32),
        np.full((2, 2, 2), 256.0, dtype=np.float32),
        np.full((2, 2, 2), np.nan, dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.int16),
    ):
        with pytest.raises(ValueError):
            abdomen_image({"imgs": bad}, require_raw=False)


def test_recist_prompts_and_scribble():
    plane = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(plane, (16, 32), (46, 32), 1, thickness=2)

    p1, p2 = recist_endpoints(plane)
    diameter = float(np.linalg.norm(p2 - p1))
    assert 28.5 <= diameter <= 31.5  # true 30px line after 2px cap removal

    # Horizontal line of ~30 px scaled by anisotropic XY spacing [0.5, 1.0, 2.0] -> ~15 mm
    len_mm = recist_length_mm(plane, [0.5, 1.0, 2.0])
    assert 14.2 <= len_mm <= 15.8

    box = recist_bbox(plane)
    nn_box = get_diameter_bbox(plane)
    assert box[0] < box[2] and box[1] < box[3]
    assert nn_box[0] < nn_box[2] and nn_box[1] < nn_box[3]

    pts_3 = points_for_mode(plane, "3pt_25")
    pts_5 = points_for_mode(plane, "5pt_10_90")
    assert pts_3.shape == (3, 2)
    assert pts_5.shape == (5, 2)
    assert abs(float(pts_3[0, 0]) - 31.0) <= 1.0
    assert abs(float(pts_5[0, 0]) - 31.0) <= 1.0

    scrib = scribble_plane(plane, thickness=[1, 2, 2], stroke="skeleton", inset=3)
    assert scrib.dtype == bool
    assert 0 < int(scrib.sum()) < int((plane > 0).sum())


def test_axial_feret_diameter_equivalence_and_edge_cases():
    sx, sy = 0.7, 1.3

    def brute_force_diameter(mask: np.ndarray) -> float:
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            return 0.0
        pts = np.column_stack((xs * sx, ys * sy)).astype(np.float64)
        return float(pdist(pts).max())

    # Empty & single pixel
    assert axial_diameter_mm(np.zeros((16, 16), dtype=bool), sx, sy) == 0.0
    single = np.zeros((16, 16), dtype=bool)
    single[4, 7] = True
    assert axial_diameter_mm(single, sx, sy) == 0.0

    # Collinear 1-pixel horizontal, vertical, and diagonal (>3 points -> exercises QhullError fallback)
    horiz = np.zeros((16, 16), dtype=bool)
    horiz[5, 2:9] = True
    assert np.isclose(axial_diameter_mm(horiz, sx, sy), 6 * sx)

    vert = np.zeros((16, 16), dtype=bool)
    vert[1:8, 6] = True
    assert np.isclose(axial_diameter_mm(vert, sx, sy), 6 * sy)

    diag = np.zeros((16, 16), dtype=bool)
    for k in range(6):
        diag[2 + k, 3 + 2 * k] = True
    assert np.isclose(axial_diameter_mm(diag, sx, sy), brute_force_diameter(diag))

    # Arbitrary ellipse + disjoint island on anisotropic XY grid
    yy, xx = np.ogrid[:48, :48]
    blob = ((yy - 18) ** 2 / 6.0**2 + (xx - 20) ** 2 / 9.0**2) <= 1.0
    blob[35:39, 36:40] = True
    assert np.isclose(axial_diameter_mm(blob, sx, sy), brute_force_diameter(blob))

    vol = np.zeros((5, 48, 48), dtype=bool)
    vol[1, :16, :16] = horiz
    vol[3] = blob
    assert np.isclose(max_axial_diameter_mm(vol, [sx, sy, 2.5]), brute_force_diameter(blob))


def test_find_recist_slices_multilabel():
    recist_3d = np.zeros((10, 32, 32), dtype=np.uint16)
    cv2.line(recist_3d[7], (4, 8), (18, 8), 5, thickness=2)
    cv2.line(recist_3d[2], (10, 20), (24, 20), 2, thickness=2)
    cv2.line(recist_3d[7], (20, 20), (28, 20), 2, thickness=2)

    slices_map = find_recist_slices(recist_3d)
    assert list(slices_map.keys()) == [2, 5]
    assert slices_map == {2: [2, 7], 5: [7]}


def test_independent_storage_overlap_empty_and_postprocess(tmp_path: Path):
    shape = (6, 32, 32)
    gt = np.zeros(shape, dtype=np.uint16)
    recist = np.zeros(shape, dtype=np.uint16)
    spacing = np.array([1.0, 1.0, 2.0], dtype=np.float32)

    # Label 2: overlapping candidate ID with 5
    gt[2:4, 6:14, 6:14] = 2
    cv2.line(recist[3], (6, 10), (13, 10), 2, thickness=2)

    # Label 5: predicted empty
    gt[2:4, 18:24, 18:24] = 5
    cv2.line(recist[3], (18, 21), (23, 21), 5, thickness=2)

    # Label 9: no RECIST
    gt[4:5, 2:6, 2:6] = 9

    out_npz = tmp_path / "pred.npz"
    with IndependentPredictionWriter(
        shape=shape,
        recist=recist,
        spacing=spacing,
        gt=gt,
        metadata={"case_id": "c1"},
        scratch=tmp_path,
    ) as writer:
        writer.mark_attempted(2)
        pred_2 = np.zeros(shape, dtype=np.uint8)
        pred_2[2:4, 6:14, 6:14] = 1
        pred_2[2, 28:31, 28:31] = 1  # far false positive to test moving-centroid clip
        writer.write_volume(2, pred_2)

        writer.mark_attempted(5)  # intentionally empty
        receipt = writer.save(out_npz)

    assert receipt["instance_ids"] == [2, 5, 9]
    assert receipt["statuses"] == ["predicted", "predicted_empty", "no_recist"]
    # Predicted lesion 2 has non-zero GT and predicted axial Feret diameter; empty 5 & no_recist 9 have 0.0 predicted
    assert receipt["rows"][0]["ground_truth_recist_length_mm"] == pytest.approx(float(np.hypot(7.0, 7.0)))
    assert receipt["rows"][0]["predicted_recist_length_mm"] > receipt["rows"][0]["ground_truth_recist_length_mm"]
    assert receipt["rows"][1]["ground_truth_recist_length_mm"] == pytest.approx(float(np.hypot(5.0, 5.0)))
    assert receipt["rows"][1]["predicted_recist_length_mm"] == 0.0
    assert receipt["rows"][2]["ground_truth_recist_length_mm"] == pytest.approx(float(np.hypot(3.0, 3.0)))
    assert receipt["rows"][2]["predicted_recist_length_mm"] == 0.0
    assert receipt["rows"][0]["recist_prompt_length_mm"] is not None
    assert receipt["rows"][2]["recist_prompt_length_mm"] is None

    # When GT is omitted (gt=None), ground_truth_recist_length_mm is None and predicted_recist_length_mm is still computed
    out_nogt = tmp_path / "pred_nogt.npz"
    with IndependentPredictionWriter(shape=shape, recist=recist, spacing=spacing, gt=None, scratch=tmp_path) as w2:
        w2.mark_attempted(2)
        w2.write_volume(2, pred_2)
        w2.mark_attempted(5)
        rec_nogt = w2.save(out_nogt)
    assert rec_nogt["rows"][0]["ground_truth_recist_length_mm"] is None
    assert rec_nogt["rows"][0]["predicted_recist_length_mm"] > 0.0
    loaded = load_independent_prediction(out_npz)
    assert loaded["segs"].shape == (3, *shape)
    assert loaded["instance_ids"] == [2, 5, 9]
    assert loaded["prediction_status"] == ["predicted", "predicted_empty", "no_recist"]

    clipped, actions = apply_moving_centroid_to_channels(
        loaded["segs"],
        loaded["instance_ids"],
        recist,
        spacing,
        exempt_ids={9},
    )
    assert actions[0]["action"] == "xy_centroid_clip"
    assert actions[0]["removed_voxels"] > 0
    assert clipped[0, 2, 29, 29] == 0
    assert actions[2]["action"] == "exempt_lymph_node"
