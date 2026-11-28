# Lymph-node RECIST prompts: which axis does inference use?

**The released CLI uses the RECIST supplied in the NPZ.** If that RECIST is a lymph-node short axis, the scripts do not recognize the anatomy and do not convert it into a long axis. There is no nodal scaling option in this release.

| Configuration | Input to the model today | After inference |
|---|---|---|
| EffMedSAM2 Tiny | Axis-aligned square box and a 1.6× Z slab from the supplied diameter | By default, a disk centered at the prediction centroid on each slice clips that lesion using half the supplied physical diameter. An explicit exemption skips **only** this clip. |
| MedSAM2 | Square box from the supplied diameter, propagated in the full Z volume | No moving-centroid clip. |
| nnInteractive (box / three points / scribble) | The configured prompt is built from the supplied RECIST on its original slice | No SAM nodal conversion or moving-centroid clip. |

For a protocol that calls for Tiny clipping exemptions, follow the [single-input and hash-bound manifest examples](../README.md#exempt-specific-tiny-lesions-from-moving-centroid-clipping). Use the positive lesion ID **within that input**. An exemption neither selects a lesion for inference nor changes its box or slab. Removing an exemption just applies the existing clip to the supplied diameter.

## What the proposed long-axis route would change (not a CLI option)

For a **verified nodal short-axis prompt**, use the original lesion ID, slice and midpoint. In physical XY coordinates (mm), rotate the short-axis direction by 90° and set the new diameter to a protocol-specified multiple of the original length. This constructs a segmentation prompt, **not** a measured anatomical long axis. A ground-truth segmentation or an oracle long axis must not be needed to generate it.

- **Tiny:** build the square box and Z slab from the derived diameter, then use its physical length for that same lesion’s moving-centroid clip.
- **MedSAM2:** use the derived diameter for its square box; retain full-Z propagation and no clip.
- **nnInteractive, non-nodal lesions and unverified prompt types:** keep the original prompt. An anatomical site or an old clipping-exemption list alone does not establish that the supplied measurement is a short axis.

There is **no released production scale**. On square in-plane pixels, away from image boundaries, the SAM square box, slab and disk radius depend on midpoint and length, not the direction of the line. Before implementing this route, define how an out-of-bounds endpoint affects the box, slab and radius; record the source and derived endpoints/lengths separately. Do not rewrite a shared RECIST raster to approximate this protocol: that would also change the inputs for methods meant to remain unchanged.

Implementation references: [Tiny inference](../infer_effmedsam2.py), [MedSAM2 inference](../infer_medsam2.py), [nnInteractive inference](../infer_nninteractive.py), [prompt geometry](../utils/prompts.py) and [moving-centroid clipping](../utils/postprocess.py).
