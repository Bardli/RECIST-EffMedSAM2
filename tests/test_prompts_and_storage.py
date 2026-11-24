"""Tests for CT intensity decoding, RECIST prompts, storage v2, and moving-centroid postprocess."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recist_effmedsam2.backends.sam_infer import (
    recist_bbox,
    recist_endpoints,
    recist_points_at,
    resize_grayscale_to_rgb_and_resize,
)
from recist_effmedsam2.ct_inputs import abdomen_image, nninteractive_image, raw_hu
from recist_effmedsam2.postprocess import apply_moving_centroid_to_channels
from recist_effmedsam2.prompts import points_for_mode, scribble_plane
from recist_effmedsam2.storage import IndependentPredictionWriter, load_independent_prediction


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

    box = recist_bbox(plane)
    assert box[0] < box[2] and box[1] < box[3]

    sam_pts = recist_points_at(plane, (25.0, 50.0, 75.0))
    nn_pts = points_for_mode(plane, "3pt_25")
    assert sam_pts.shape == (3, 2)
    assert nn_pts.shape == (3, 2)
    # Midpoint (index 0) near (31, 32)
    assert abs(float(sam_pts[0, 0]) - 31.0) <= 1.0
    assert abs(float(nn_pts[0, 0]) - 31.0) <= 1.0

    scrib = scribble_plane(plane, thickness=[1, 2, 2], stroke="skeleton", inset=3)
    assert scrib.dtype == bool
    assert 0 < int(scrib.sum()) < int((plane > 0).sum())


def test_independent_storage_overlap_empty_and_postprocess(tmp_path: Path):
    shape = (6, 32, 32)
    gt = np.zeros(shape, dtype=np.uint16)
    recist = np.zeros(shape, dtype=np.uint16)
    spacing = np.array([1.0, 1.0, 2.0], dtype=np.float32)

    # Label 2: predicted overlapping with 5
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
    loaded = load_independent_prediction(out_npz)
    assert loaded["segs"].shape == (3, *shape)
    assert loaded["instance_ids"] == [2, 5, 9]
    assert loaded["prediction_status"] == ["predicted", "predicted_empty", "no_recist"]

    # Moving-centroid clip should remove the distant false positive on slice 2 for label 2
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
