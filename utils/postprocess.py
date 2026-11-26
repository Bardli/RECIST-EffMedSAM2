"""Moving-centroid XY disk clipping postprocessing (saves final predictions only)."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

from .storage import IndependentPredictionWriter


def _svd_recist_endpoints_xy(
    recist_2d: np.ndarray,
    subtract_thickness: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (p1, p2) [x, y] endpoints from a 2D RECIST raster via SVD."""
    ys, xs = np.where(recist_2d > 0)
    if len(xs) < 2:
        raise ValueError("recist_too_short")
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
    else:
        centroid = coords.mean(0)
        c0 = coords - centroid
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        u = vt[0]
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
    return p1, p2


def prompt_length(recist: np.ndarray, lab: int, z0: int, spacing: np.ndarray, endpoint: dict) -> float:
    """Verify saved centerline endpoints reproduce the prompt raster; return physical length in mm."""
    points = np.asarray(endpoint["points"], dtype=float)
    if points.shape != (2, 3) or not np.all(np.isfinite(points)):
        raise ValueError("invalid_endpoints")
    if not np.all(points[:, 2] == z0):
        raise ValueError("endpoint_slice_mismatch")
    if np.any(points < 0) or np.any(points > np.array(recist.shape[::-1]) - 1):
        raise ValueError("endpoints_outside_image")
    xy = np.rint(points[:, :2]).astype(int)
    expected = np.zeros(recist.shape[1:], dtype=np.uint8)
    cv2.line(expected, tuple(xy[0]), tuple(xy[1]), 1, thickness=2)
    if not np.array_equal(expected > 0, recist[z0] == lab):
        raise ValueError("endpoint_raster_mismatch")
    length = float(np.linalg.norm((points[1] - points[0]) * spacing))
    if length <= 0:
        raise ValueError("zero_prompt_length")
    return length


def xy_centroid_clip_channel(
    pred_mask: np.ndarray,
    recist_3d: np.ndarray,
    label_id: int,
    spacing: np.ndarray,
    endpoint: dict | None = None,
    exempt: bool = False,
) -> tuple[np.ndarray, dict]:
    """Clip one independent binary 3D mask ``[Z, Y, X]`` using its RECIST prompt."""
    sp = np.asarray(spacing, dtype=np.float64).ravel()
    if len(sp) != 3 or not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("positive XYZ spacing required")
    if pred_mask.ndim != 3 or pred_mask.shape != recist_3d.shape:
        raise ValueError("original ZYX grids must match")

    out = (pred_mask > 0).astype(np.uint8)
    row = dict(
        label=int(label_id),
        seed_z="",
        radius_mm="",
        prompt_length_mm="",
        status="ok",
        action="preserved",
        removed_voxels=0,
        affected_slices=0,
    )
    if exempt:
        row["action"] = "exempt_lymph_node"
        return out, row

    seeds = [z for z, s in enumerate(recist_3d) if np.any(s == label_id)]
    if len(seeds) != 1:
        row["status"] = "missing_recist" if not seeds else "multiple_recist_slices"
        return out, row
    z0 = seeds[0]
    row["seed_z"] = z0

    if endpoint is not None:
        try:
            length_mm = prompt_length(recist_3d, label_id, z0, sp, endpoint)
        except ValueError as exc:
            row["status"] = str(exc)
            return out, row
    else:
        p1, p2 = _svd_recist_endpoints_xy(recist_3d[z0] == label_id, subtract_thickness=2.0)
        length_mm = float(np.linalg.norm((p2 - p1) * sp[:2]))
        if length_mm <= 0:
            row["status"] = "zero_prompt_length"
            return out, row

    radius = length_mm / 2.0
    row.update(radius_mm=radius, prompt_length_mm=length_mm, action="xy_centroid_clip")

    yy, xx = np.ogrid[: out.shape[1], : out.shape[2]]
    r2_tol = radius**2 + 1e-9 * max(1.0, radius**2)
    for z in range(len(out)):
        original = out[z] > 0
        before = int(original.sum())
        if before == 0:
            continue
        ys, xs = np.nonzero(original)
        cx, cy = float(xs.mean()), float(ys.mean())
        distance2 = ((xx - cx) * sp[0]) ** 2 + ((yy - cy) * sp[1]) ** 2
        allowed = distance2 <= r2_tol
        out[z][original & ~allowed] = 0
        after = int(np.count_nonzero(out[z]))
        row["removed_voxels"] += before - after
        row["affected_slices"] += int(after < before)
    return out, row


def apply_moving_centroid_to_channels(
    segs_nzyx: np.ndarray,
    instance_ids: list[int],
    recist: np.ndarray,
    spacing: np.ndarray,
    endpoints: dict[int, dict] | None = None,
    exempt_ids: set[int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Apply moving-centroid clipping independently to each channel of an ``[N, Z, Y, X]`` array."""
    if segs_nzyx.ndim != 4 or segs_nzyx.shape[0] != len(instance_ids):
        raise ValueError("segs_nzyx must have shape (len(instance_ids), Z, Y, X)")
    clipped = np.empty_like(segs_nzyx, dtype=np.uint8)
    rows: list[dict] = []
    endpoints = endpoints or {}
    exempt_ids = exempt_ids or set()
    for idx, lab in enumerate(instance_ids):
        ch, row = xy_centroid_clip_channel(
            segs_nzyx[idx],
            recist,
            int(lab),
            spacing,
            endpoint=endpoints.get(int(lab)),
            exempt=int(lab) in exempt_ids,
        )
        clipped[idx] = ch
        rows.append(row)
    return clipped, rows


def apply_moving_centroid_to_writer(
    writer: IndependentPredictionWriter,
    *,
    profiler=None,
    scratch: str | Path | None = None,
    endpoints: dict[int, dict] | None = None,
    exempt_ids: set[int] | None = None,
) -> tuple[IndependentPredictionWriter, list[dict]]:
    """Apply moving-centroid clipping directly from disk-backed channels without NPZ reload."""
    endpoints = endpoints or {}
    exempt_ids = exempt_ids or set()
    post_meta = {**writer.metadata, "postprocess": "moving_centroid"}
    post_writer = IndependentPredictionWriter(
        shape=writer.shape,
        recist=writer.recist,
        spacing=writer.spacing,
        gt=writer.gt,
        metadata=post_meta,
        scratch=scratch,
    )
    actions: list[dict] = []
    try:
        for lab in writer.ids:
            if lab not in writer.prompt_ids:
                actions.append(
                    dict(
                        label=int(lab),
                        seed_z="",
                        radius_mm="",
                        prompt_length_mm="",
                        status="no_recist",
                        action="preserved",
                        removed_voxels=0,
                        affected_slices=0,
                    )
                )
                continue
            ctx = profiler.lesion_postprocess(lab) if profiler is not None else nullcontext()
            with ctx:
                post_writer.mark_attempted(lab)
                clipped, row = xy_centroid_clip_channel(
                    writer.channels[lab],
                    writer.recist,
                    int(lab),
                    writer.spacing,
                    endpoint=endpoints.get(int(lab)),
                    exempt=int(lab) in exempt_ids,
                )
                post_writer.write_volume(lab, clipped)
            actions.append(row)
    except BaseException:
        post_writer.close()
        raise
    return post_writer, actions


def finalize_and_save_predictions(
    writer: IndependentPredictionWriter,
    output_npz: str | Path,
    *,
    profiler,
    postprocess: str = "none",
    exempt_ids: set[int] | None = None,
    overwrite: bool = False,
    scratch: str | Path | None = None,
) -> dict:
    """Optionally apply postprocessing inside ``profiler`` and save ONLY the final prediction NPZ."""
    post_writer: IndependentPredictionWriter | None = None
    if exempt_ids is not None:
        if postprocess != "moving_centroid":
            raise ValueError("Exemptions require moving_centroid")
        if not exempt_ids <= writer.prompt_ids:
            raise ValueError(f"Unknown or unprompted exempt lesion IDs: {sorted(exempt_ids - writer.prompt_ids)}")
    try:
        if postprocess == "moving_centroid":
            with profiler.phase("postprocessing_time"):
                post_writer, actions = apply_moving_centroid_to_writer(
                    writer, profiler=profiler, scratch=scratch, exempt_ids=exempt_ids
                )
                if exempt_ids is not None:
                    # Explicit IDs do not imply an anatomical diagnosis.
                    for action in actions:
                        if action["action"] == "exempt_lymph_node":
                            action["action"] = "exempt_by_id"
                    post_writer.metadata.update(postprocess_exempt_ids=sorted(exempt_ids),
                                                postprocess_actions=actions)
            final_writer = post_writer
        elif postprocess == "none":
            final_writer = writer
        else:
            raise ValueError(f"Unsupported postprocess mode: {postprocess!r}")

        with profiler.phase("output_saving_time"):
            receipt = final_writer.save(output_npz, overwrite=overwrite)
        return receipt
    finally:
        if post_writer is not None:
            post_writer.close()
