"""Explicit case-local Tiny moving-centroid exemptions; no anatomy inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate exemption manifest key: {key}")
        result[key] = value
    return result


def _ids(values) -> frozenset[int]:
    if not isinstance(values, list) or any(type(i) is not int or i <= 0 for i in values):
        raise ValueError("Exempt IDs must be a list of positive integer lesion IDs")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate exempt lesion ID")
    return frozenset(values)


@dataclass(frozen=True)
class ExemptionPolicy:
    single: frozenset[int] | None = None
    by_input: dict[str, frozenset[int]] | None = None
    manifest_sha256: str | None = None

    def ids_for(self, input_sha256: str, prompt_ids: set[int]) -> set[int]:
        if self.by_input is not None:
            if input_sha256 not in self.by_input:
                raise ValueError(f"Input SHA256 absent from exemption manifest: {input_sha256}")
            ids = self.by_input[input_sha256]
        else:
            assert self.single is not None
            ids = self.single
        if ids - prompt_ids:
            raise ValueError(f"Unknown or unprompted exempt lesion IDs: {sorted(ids - prompt_ids)}")
        return set(ids)


def load_policy(args) -> ExemptionPolicy | None:
    if args.exempt_ids is None and args.exempt_manifest is None:
        return None
    if args.model_size != "tiny" or args.postprocess != "moving_centroid":
        raise ValueError("Exemptions require --model-size tiny --postprocess moving_centroid")
    if args.exempt_ids is not None:
        if not args.input_dir.is_file():
            raise ValueError("--exempt-ids requires one explicit input NPZ file, not a folder")
        return ExemptionPolicy(single=_ids(args.exempt_ids))
    raw = args.exempt_manifest.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_keys)
    if (not isinstance(data, dict) or set(data) != {"schema", "inputs"}
            or data["schema"] != "recist-exempt-v1" or not isinstance(data["inputs"], dict)
            or not data["inputs"]):
        raise ValueError("Expected recist-exempt-v1 schema and nonempty inputs mapping")
    by_input = {}
    for digest, values in data["inputs"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid lowercase input SHA256: {digest}")
        by_input[digest] = _ids(values)
    return ExemptionPolicy(by_input=by_input, manifest_sha256=hashlib.sha256(raw).hexdigest())
