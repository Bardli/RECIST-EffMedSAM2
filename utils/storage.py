"""Independent-instance prediction schema v2 storage (disk-backed streamed NPZ)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any

import numpy as np

from .prompts import axial_diameter_mm, find_recist_slices, recist_length_mm

SCHEMA = "independent_instances_v2"


def sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file using 8 MiB streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """Compute SHA-256 over array dtype, shape, and C-contiguous slice bytes."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    for plane in arr:
        h.update(np.ascontiguousarray(plane).tobytes())
    return h.hexdigest()


def write_member(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with archive.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.asarray(array), allow_pickle=False)


class IndependentPredictionWriter:
    """Disk-backed independent-instance binary channel writer.

    Stores one uint8 ``[Z, Y, X]`` memmap channel per original lesion ID so
    overlapping predictions never overwrite each other and large 3D volumes do
    not require holding an ``[N, Z, Y, X]`` tensor in RAM.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        recist: np.ndarray,
        spacing: np.ndarray,
        gt: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        scratch: str | Path | None = None,
    ) -> None:
        if len(shape) != 3 or recist.shape != tuple(shape):
            raise ValueError(f"Expected 3D shape matching recist {recist.shape}, got {shape}")
        if gt is not None and gt.shape != tuple(shape):
            raise ValueError(f"GT shape {gt.shape} does not match volume shape {shape}")

        self.shape = tuple(int(x) for x in shape)
        self.gt = gt
        self.recist = recist
        self.spacing = np.asarray(spacing)
        self.metadata = dict(metadata or {})

        self.prompt_ids: set[int] = {int(x) for x in np.unique(recist) if x != 0}
        if gt is not None:
            self.ids: list[int] = sorted(int(x) for x in np.unique(gt) if x != 0)
            if not self.prompt_ids <= set(self.ids):
                raise ValueError("RECIST contains label ID absent from GT")
        else:
            self.ids = sorted(self.prompt_ids)

        self.seen: set[int] = set()
        self.saved = False
        self.tmp = Path(tempfile.mkdtemp(prefix="instance-channels-", dir=str(scratch) if scratch else None))
        self.channels: dict[int, np.memmap] = {}
        for lab in self.ids:
            self.channels[lab] = np.lib.format.open_memmap(
                self.tmp / f"{lab}.npy",
                mode="w+",
                dtype=np.uint8,
                shape=self.shape,
            )

    def write_slice(self, label: int, z: int, mask_2d: np.ndarray) -> None:
        """Record predictions for ``label`` on slice ``z`` (unioned if called multiple times)."""
        lab = int(label)
        if lab not in self.prompt_ids:
            raise ValueError(f"Cannot write unprompted or unknown label {lab}")
        self.seen.add(lab)
        m = np.asarray(mask_2d, dtype=bool)
        if m.any():
            self.channels[lab][int(z)][m] = 1

    def write_volume(self, label: int, mask_3d: np.ndarray) -> None:
        """Record a 3D binary prediction volume for ``label`` slice-by-slice."""
        lab = int(label)
        if lab not in self.prompt_ids:
            raise ValueError(f"Cannot write unprompted or unknown label {lab}")
        if mask_3d.shape != self.shape:
            raise ValueError(f"Expected mask shape {self.shape}, got {mask_3d.shape}")
        self.seen.add(lab)
        chan = self.channels[lab]
        for zi in range(self.shape[0]):
            sl = np.asarray(mask_3d[zi], dtype=bool)
            if sl.any():
                chan[zi][sl] = 1

    def mark_attempted(self, label: int) -> None:
        """Explicitly mark a prompted label as attempted even if predicted mask is empty."""
        lab = int(label)
        if lab not in self.prompt_ids:
            raise ValueError(f"Unknown prompted label {lab}")
        self.seen.add(lab)

    def save(self, path: str | Path, overwrite: bool = False) -> dict[str, Any]:
        """Stream all channels into a Zip64 compressed NPZ and return receipt metadata."""
        if self.saved:
            raise RuntimeError("IndependentPredictionWriter.save() already called")
        if self.seen != self.prompt_ids:
            missing = sorted(self.prompt_ids - self.seen)
            raise ValueError(f"Prompted IDs never emitted a prediction: {missing}")

        pred_counts = dict.fromkeys(self.ids, 0)
        overlap_voxels = 0
        for zi in range(self.shape[0]):
            occupancy = np.zeros(self.shape[1:], dtype=np.uint32)
            for lab in self.ids:
                mask = self.channels[lab][zi] != 0
                c = int(mask.sum())
                if c:
                    pred_counts[lab] += c
                    occupancy += mask
            overlap_voxels += int((occupancy > 1).sum())

        statuses = [
            "no_recist"
            if i not in self.prompt_ids
            else ("predicted" if pred_counts[i] > 0 else "predicted_empty")
            for i in self.ids
        ]

        recist_slices = find_recist_slices(self.recist)
        sp = np.asarray(self.spacing, dtype=np.float64).ravel()
        has_valid_xy = len(sp) >= 2 and bool(np.all(np.isfinite(sp[:2]))) and bool(np.all(sp[:2] > 0))
        sx, sy = (float(sp[0]), float(sp[1])) if has_valid_xy else (0.0, 0.0)

        rows: list[dict[str, Any]] = []
        for k, lab in enumerate(self.ids):
            pred_voxels = pred_counts[lab]
            rlen_mm: float | None = None
            zs = recist_slices.get(lab, [])
            if lab in self.prompt_ids and len(zs) == 1:
                try:
                    rlen_mm = recist_length_mm(self.recist[zs[0]] == lab, self.spacing)
                except ValueError:
                    rlen_mm = None

            pred_recist_mm: float | None = 0.0 if has_valid_xy else None
            gt_recist_mm: float | None = (0.0 if has_valid_xy else None) if self.gt is not None else None
            ng = 0 if self.gt is not None else None
            ni = 0 if self.gt is not None else None
            chan = self.channels[lab]

            for zi in range(self.shape[0]):
                p = chan[zi] != 0
                p_any = bool(p.any())
                if p_any and has_valid_xy:
                    d_p = axial_diameter_mm(p, sx, sy)
                    if pred_recist_mm is not None and d_p > pred_recist_mm:
                        pred_recist_mm = d_p
                if self.gt is not None:
                    g = self.gt[zi] == lab
                    g_cnt = int(g.sum())
                    if g_cnt:
                        assert ng is not None and ni is not None
                        ng += g_cnt
                        if p_any:
                            ni += int((g & p).sum())
                        if has_valid_xy:
                            d_g = axial_diameter_mm(g, sx, sy)
                            if gt_recist_mm is not None and d_g > gt_recist_mm:
                                gt_recist_mm = d_g

            if self.gt is not None:
                assert ng is not None and ni is not None
                denom = ng + pred_voxels
                dice: float | None = (2.0 * ni / denom) if denom > 0 else 1.0
            else:
                dice = None

            rows.append(
                dict(
                    channel_index=k,
                    gt_id=lab,
                    recist_id=lab if lab in self.prompt_ids else None,
                    ground_truth_recist_length_mm=gt_recist_mm,
                    predicted_recist_length_mm=pred_recist_mm,
                    recist_prompt_length_mm=rlen_mm,
                    prediction_id=lab,
                    status=statuses[k],
                    gt_voxels=ng,
                    pred_voxels=pred_voxels,
                    intersection=ni,
                    dice=dice,
                    score_protocol="independent_same_id_raw_v2",
                )
            )

        receipt: dict[str, Any] = dict(
            schema_version=SCHEMA,
            **self.metadata,
            shape_nzyx=[len(self.ids), *self.shape],
            spacing=[float(s) for s in self.spacing],
            instance_ids=self.ids,
            statuses=statuses,
            rows=rows,
            overlap_voxels=overlap_voxels,
            has_gt=self.gt is not None,
        )

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {dest}")
        part = dest.with_name(dest.name + ".part")
        try:
            with zipfile.ZipFile(
                part,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                with archive.open("segs.npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array_header_2_0(
                        stream,
                        dict(
                            descr=np.dtype("uint8").str,
                            fortran_order=False,
                            shape=(len(self.ids), *self.shape),
                        ),
                    )
                    for lab in self.ids:
                        self.channels[lab].flush()
                        for plane in self.channels[lab]:
                            stream.write(np.ascontiguousarray(plane).tobytes())
                write_member(archive, "instance_ids", np.asarray(self.ids, dtype=np.int64))
                write_member(archive, "prediction_status", np.asarray(statuses, dtype="U24"))
                write_member(archive, "schema_version", np.array(SCHEMA))
                write_member(archive, "spacing", self.spacing)
                write_member(archive, "provenance_json", np.array(json.dumps(receipt)))

            with zipfile.ZipFile(part) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError(f"NPZ CRC check failed on member {bad}")
            os.replace(part, dest)
            receipt["output_sha256"] = sha256(dest)
            self.saved = True
            return receipt
        finally:
            if part.exists():
                part.unlink()

    def close(self) -> None:
        self.channels.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self) -> "IndependentPredictionWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_independent_prediction(path: str | Path) -> dict[str, Any]:
    """Load an ``independent_instances_v2`` NPZ prediction safely."""
    with np.load(path, allow_pickle=False) as f:
        schema = str(f["schema_version"].item()) if "schema_version" in f.files else ""
        if schema != SCHEMA:
            raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {schema!r}")
        return {
            "segs": f["segs"],
            "instance_ids": [int(x) for x in f["instance_ids"]],
            "prediction_status": [str(x) for x in f["prediction_status"]],
            "spacing": f["spacing"],
            "provenance": json.loads(str(f["provenance_json"].item())),
        }
