"""Opt-in segmentation prompts for explicitly confirmed nodal short-axis measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .exemptions import _unique_keys

PROTOCOL = "confirmed-nodal-short-to-long-1.25-v1"
SCALE = 1.25


def bind_request(args):
    """One physical primary input per invocation. No anatomical inference or defaults."""
    from .storage import sha256
    from .ct_inputs import resolve_window_npz
    if not args.input_dir.is_file() or args.input_dir.suffix.lower() != ".npz":
        raise ValueError("Nodal mode requires one explicit primary NPZ")
    if (args.shift != 0 or args.overwrite or getattr(args, "model_size", "tiny") != "tiny"
            or getattr(args, "postprocess", "moving_centroid") != "moving_centroid"
            or getattr(args, "exempt_ids", None) is not None
            or getattr(args, "exempt_manifest", None) is not None):
        raise ValueError("Fixed nodal protocol: no padding, Small, exemptions, disabled clipping or overwrite")
    if args.gts_path is not None:
        raise ValueError("Use the primary NPZ for GT or run without GT; external GT is not hash-bound")
    raw = args.nodal_manifest.read_bytes()
    m = json.loads(raw, object_pairs_hook=_unique_keys)
    if not isinstance(m, dict) or set(m) != {
        "protocol", "input_sha256", "window_sha256", "confirmed_short_axis_ids"
    } or m["protocol"] != PROTOCOL:
        raise ValueError("Expected a confirmed-nodal-short-to-long-1.25-v1 case manifest")
    ids = m["confirmed_short_axis_ids"]
    if (not isinstance(ids, list) or not ids or any(type(i) is not int or i <= 0 for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Provide a nonempty, unique list of confirmed positive case-local IDs")
    win = resolve_window_npz(args.input_dir, args.input_dir, args.window_dir)
    if sha256(args.input_dir) != m["input_sha256"] or (sha256(win) if win else None) != m["window_sha256"]:
        raise ValueError("Nodal manifest mismatch: primary input or window SHA256")
    # Refuse reuse, even when only a prior results.csv (no predictions) remains.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return set(ids), dict(nodal_protocol=PROTOCOL, nodal_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                          window_sha256=m["window_sha256"], confirmed_short_axis_ids=sorted(ids))


def derived_geometry(raster, spacing, z):
    """Study's physical perpendicular, float32 clamped endpoints and nominal clip length."""
    import numpy as np
    from .prompts import recist_endpoints
    sp = np.asarray(spacing, dtype=np.float64)
    if sp.shape != (3,) or not np.isfinite(sp).all() or (sp <= 0).any():
        raise ValueError("Nodal prompt requires positive finite XYZ spacing")
    h, w = raster.shape
    p = np.array(recist_endpoints(raster), dtype=np.float64)
    mm = p * sp[:2]
    delta = mm[1] - mm[0]
    short = float(np.linalg.norm(delta))
    if not np.isfinite(short) or short <= 0:
        raise ValueError("Invalid short-axis length")
    mid = mm.mean(axis=0)
    length = SCALE * short
    arm = (0.5 * length) * (np.array([-delta[1], delta[0]]) / short)
    nominal = np.stack((mid - arm, mid + arm)) / sp[:2]
    q = np.clip(nominal, [0, 0], [w - 1, h - 1]).astype(np.float32)
    diameter = float(np.linalg.norm(q[1] - q[0]))
    if diameter <= 0:
        raise ValueError("Derived prompt collapsed at the image edge")
    center = q.mean(axis=0)
    half = diameter / 2
    box = [max(0, round(float(center[0] - half))), max(0, round(float(center[1] - half))),
           min(w - 1, round(float(center[0] + half))), min(h - 1, round(float(center[1] + half)))]
    return dict(source_endpoints_xy=p.tolist(), seed_z=int(z), short_mm=short,
                nominal_endpoints_xy=nominal.tolist(), endpoints_xy=q.tolist(),
                nominal_mm=length, chord_mm=float(np.linalg.norm((q[1]-q[0])*sp[:2])),
                endpoint_clamped=bool(((nominal < 0) | (nominal > [w-1, h-1])).any()),
                diameter_px=diameter, box_xyxy=box)
