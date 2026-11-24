"""Tests for CT intensity decoding, RECIST prompts, storage v2, and moving-centroid postprocess."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recist_effmedsam2.backends.sam_infer import recist_bbox, recist_endpoints, recist_points_at
from recist_effmedsam2.ct_inputs import abdomen_image, nninteractive_image, raw_hu
from recist_effmedsam2.postprocess import apply_moving_centroid_to_channels
from recist_effmedsam2.prompts import points_for_mode, scribble_plane
from recist_effmedsam2.storage import IndependentPredictionWriter, load_independent_prediction


def test_ct_inputs_raw_hu_and_windowing():
    base_hu = np.array([[[-1000, 40], [240, 500]]], dtype=np.int16)
    direct_int16_3d = np.repeat(base_hu, 2, axis=0)

    # 1. Primary format: direct int16 raw HU without intensity_encoding / slope / intercept
    direct_data = {"imgs": direct_int16_3d}
    hu = raw_hu(direct_data)
    assert hu.dtype == np.int16
    assert np.array_equal(hu, direct_int16_3d)

    for strict in (False, True):
        win = abdomen_image(direct_data, require_raw=strict)
        assert win.dtype == np.uint8
        assert win[0, 0, 0] == 0
        assert win[0, 1, 0] == 255
        assert win[0, 1, 1] == 255
        assert 125 <= int(win[0, 0, 1]) <= 130  # 40 HU -> mid-range of [-160, 240]

        nn_img = nninteractive_image(direct_data, require_raw=strict)
        assert nn_img.dtype == np.int16
        assert np.array_equal(nn_img, direct_int16_3d)

    # 2. Direct float32 raw HU without reconstruction metadata
    direct_f32 = {"imgs": direct_int16_3d.astype(np.float32)}
    hu_f32 = raw_hu(direct_f32)
    assert hu_f32.dtype == np.float32
    assert np.allclose(hu_f32, direct_int16_3d)
    assert np.array_equal(abdomen_image(direct_f32, require_raw=True), win)
    assert np.allclose(nninteractive_image(direct_f32, require_raw=True), direct_int16_3d)

    # Non-finite float32 and unscaled uint16 without intensity_encoding must be rejected
    bad_nan = direct_int16_3d.astype(np.float32)
    bad_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        raw_hu({"imgs": bad_nan})
    with pytest.raises(ValueError):
        raw_hu({"imgs": np.zeros((2, 2, 2), dtype=np.uint16)})

    # 3. Pre-windowed uint8 behavior under require_raw=False vs require_raw=True
    uint8_data = {"imgs": win}
    assert np.array_equal(abdomen_image(uint8_data, require_raw=False), win)
    assert np.array_equal(nninteractive_image(uint8_data, require_raw=False), win)
    with pytest.raises(ValueError):
        raw_hu(uint8_data)
    with pytest.raises(ValueError):
        abdomen_image(uint8_data, require_raw=True)
    with pytest.raises(ValueError):
        nninteractive_image(uint8_data, require_raw=True)

    # 4. Optional legacy nifti_stored_scaled_v1 backward compatibility
    legacy_data = {
        "intensity_encoding": np.array("nifti_stored_scaled_v1"),
        "imgs": np.full((2, 2, 2), 520, dtype=np.int16),
        "hu_slope": np.array(2.0),
        "hu_intercept": np.array(-1000.0),
    }
    legacy_hu = raw_hu(legacy_data)
    assert legacy_hu.dtype == np.float64
    assert np.allclose(legacy_hu, 40.0)
    legacy_win = abdomen_image(legacy_data, require_raw=True)
    assert np.all((125 <= legacy_win) & (legacy_win <= 130))
    assert np.allclose(nninteractive_image(legacy_data, require_raw=True), 40.0)


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
