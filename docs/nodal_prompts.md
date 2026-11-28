# Run the lymph-node short-axis protocol (1.25×)

Use [`infer_nodal.py`](../infer_nodal.py) **only for lesion IDs whose original annotation was verified to be a nodal short axis**. It processes one physical CT input per invocation and writes a **subset** prediction and native `results.csv`: no other ID in that CT is run, relabeled, or emitted as a zero. The ordinary model scripts still use the original supplied prompt.

| Dedicated command | Model input | Final prediction |
|---|---|---|
| `infer_nodal.py tiny` | Square box and 1.6× Z slab from the derived diameter on the original RECIST slice | Per-slice moving-centroid disk, radius = half the nominal derived physical length |
| `infer_nodal.py medsam2` | Same square box; full-Z propagation | No postprocessing |

There is no nodal command for Small or nnInteractive. For a historical Tiny protocol requiring clipping exemptions, use the [ordinary exemption options](../README.md#exempt-specific-tiny-lesions-from-moving-centroid-clipping). An exemption does not scale the box. Its IDs are **not** a roster of verified short-axis measurements.

## Run one case

1. Use the positive **case-local ID in the original RECIST raster**, not a channel index. Verify the input and its exact window twin against your dataset ledger. Do not select lesions by filename, anatomical site, a previous exemption, or predicted shape.
2. Create a private JSON manifest for that case; example below. Replace the digest strings with actual SHA256 values (use `sha256sum` on the original files). Use JSON `null` for `window_sha256` **only** when the intended protocol uses the primary input itself for intensities. Do not substitute that mode for a protocol requiring a particular window.
3. After installing the [SAM runtime and weights](../README.md#installation), run one or both methods with **new, nonexistent** output directories:

```json
{
  "protocol": "confirmed-nodal-short-to-long-1.25-v1",
  "input_sha256": "<sha256 of the actual primary NPZ>",
  "window_sha256": "<sha256 of the actual window twin>",
  "confirmed_short_axis_ids": [3]
}
```

```bash
PY=.venv-sam/bin/python
CT=data/inputs/case.npz
WIN=data/window_twins/case.npz
"$PY" infer_nodal.py tiny -i "$CT" --window-dir "$WIN" --nodal-manifest case.nodal.json \
  -o runs/nodal125/case/tiny --device cpu --num-threads 1
"$PY" infer_nodal.py medsam2 -i "$CT" --window-dir "$WIN" --nodal-manifest case.nodal.json \
  -o runs/nodal125/case/medsam2 --device cpu --num-threads 1
```

For a GPU run use `--device cuda --gpu-index 0` and **different output directories**. Omit `--window-dir` only for the manifest's null-window case. For multiple confirmed IDs in the same NPZ, list them together so the model loads once per CT. Keep cohort manifests, identifiers, images and predictions out of this public repository.

## Geometry and audit

From the supplied raster, fit the source endpoints once using the same SVD / raster-thickness convention as the ordinary scripts. Keep the original axial slice, lesion ID and midpoint. Rotate the direction by 90° **in physical XY millimetres**, and set the nominal diameter to **1.25 × measured short diameter**. This is an inference prompt, not a measured anatomical long axis. No GT is used to derive it. If supplied, GT is used for same-ID scoring; it is not a substitute for the prompt.

At an image edge, clip the derived endpoints to the original pixel grid. As in the reference study, use that in-bounds chord and its midpoint for the axis-aligned square box (and Tiny slab); keep the **nominal pre-clamp** length for Tiny’s clip radius. The run records both lengths, both endpoint sets, seed slice and box. On square pixels away from the edge, increasing diameter changes the box and radius; rotation by itself does not.

Check `metrics/*.metrics.json` or the prediction NPZ provenance for the protocol, input/window/manifest and checkpoint hashes, the selected IDs, and per-ID geometry. Tiny additionally records Z bounds, actual clipping radius and removed voxel count. The original RECIST raster and native CSV definitions are unchanged. Join a subset result back by case identity and original lesion ID, **not by row number**; never interpret unselected lesions as empty predictions.
