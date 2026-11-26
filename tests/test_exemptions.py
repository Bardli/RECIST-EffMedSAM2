"""The CLI policy selects original case-local IDs, never prediction channel indices."""
from contextlib import nullcontext
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.postprocess import finalize_and_save_predictions
from utils.storage import IndependentPredictionWriter


class Profiler:
    def phase(self, name):
        return nullcontext()

    def lesion_postprocess(self, label):
        return nullcontext()


def test_exemption_preserves_entire_raw_channel_and_scores_both_ids(tmp_path):
    shape = (2, 32, 32)
    recist = np.zeros(shape, dtype=np.uint16)
    cv2.line(recist[0], (5, 8), (13, 8), 2, 2)
    cv2.line(recist[1], (5, 15), (13, 15), 19, 2)
    gt = np.zeros_like(recist)
    gt[0, 8, 9], gt[1, 15, 9] = 2, 19
    raw = np.zeros(shape, dtype=np.uint8)
    raw[:, 3:22, 3:22] = 1
    path = tmp_path / 'pred.npz'
    with IndependentPredictionWriter(shape=shape, recist=recist, spacing=np.ones(3), gt=gt) as writer:
        for lab in (2, 19):
            writer.mark_attempted(lab)
            writer.write_volume(lab, raw)
        got = finalize_and_save_predictions(writer, path, profiler=Profiler(),
                                            postprocess='moving_centroid', exempt_ids={19})
    with np.load(path, allow_pickle=False) as pred:
        assert pred['instance_ids'].tolist() == [2, 19]
        assert 0 < pred['segs'][0].sum() < raw.sum()
        np.testing.assert_array_equal(pred['segs'][1], raw)
        prov = json.loads(pred['provenance_json'].item())
    assert got['postprocess_actions'] == prov['postprocess_actions']
    assert prov['postprocess_exempt_ids'] == [19]
    assert [a['action'] for a in got['postprocess_actions']] == ['xy_centroid_clip', 'exempt_by_id']
    assert got['postprocess_actions'][1]['removed_voxels'] == 0
    assert got['rows'][1]['gt_id'] == 19
    assert got['rows'][1]['dice'] == pytest.approx(2 / (1 + int(raw.sum())))


def test_manifest_resolves_per_physical_input_and_fails_closed(tmp_path):
    from infer_effmedsam2 import build_parser
    from utils.exemptions import load_policy
    manifest = tmp_path / 'policy.json'
    a, b = '1' * 64, '2' * 64
    manifest.write_text(json.dumps({'schema': 'recist-exempt-v1', 'inputs': {a: [19], b: []}}))
    args = build_parser().parse_args(['-i', str(tmp_path), '--exempt-manifest', str(manifest)])
    policy = load_policy(args)
    assert policy.ids_for(a, {2, 19}) == {19}
    assert policy.ids_for(b, {2, 19}) == set()
    with pytest.raises(ValueError, match='absent from exemption manifest'):
        policy.ids_for('f' * 64, {2, 19})
    with pytest.raises(ValueError, match='unprompted'):
        policy.ids_for(a, {2})


def test_direct_ids_are_single_file_only_and_cannot_disable_all_postprocessing(tmp_path):
    from infer_effmedsam2 import build_parser
    from utils.exemptions import load_policy
    parser = build_parser()
    f = tmp_path / 'case.npz'
    f.touch()
    assert load_policy(parser.parse_args([])) is None
    single = ['-i', str(f), '--exempt-ids', '2', '19']
    assert load_policy(parser.parse_args(single)).ids_for('any', {2, 19, 80}) == {2, 19}
    with pytest.raises(ValueError, match='unprompted'):
        load_policy(parser.parse_args(single)).ids_for('any', {1, 2})
    for argv, error in [(['-i', str(tmp_path), '--exempt-ids', '1'], 'one explicit input'),
                        (single + ['--model-size', 'small'], 'tiny'),
                        (single + ['--postprocess', 'none'], 'moving_centroid'),
                        (['-i', str(f), '--exempt-ids', '1', '1'], 'Duplicate'),
                        (['-i', str(f), '--exempt-ids', '0'], 'positive')]:
        with pytest.raises(ValueError, match=error):
            load_policy(parser.parse_args(argv))


@pytest.mark.parametrize("inputs", [
    '{"bad": []}',
    '{"%s": [true]}' % ('a' * 64),
    '{"%s": [0]}' % ('a' * 64),
    '{"%s": [1,1]}' % ('a' * 64),
    '{"%s": [], "%s": [1]}' % ('a' * 64, 'a' * 64),
])
def test_bad_manifest_is_rejected(tmp_path, inputs):
    from infer_effmedsam2 import build_parser
    from utils.exemptions import load_policy
    f = tmp_path / 'policy.json'
    f.write_text('{"schema":"recist-exempt-v1","inputs":' + inputs + '}')
    with pytest.raises(ValueError):
        load_policy(build_parser().parse_args(['--exempt-manifest', str(f)]))


def test_tiny_entrypoint_keeps_csv_and_npz_with_a_hash_bound_manifest(tmp_path, monkeypatch):
    """Exercise the CLI with real Torch/NPZ/CSV; only the prediction backend is a test double."""
    import csv
    import hashlib
    import sys
    from types import ModuleType
    torch = pytest.importorskip("torch")
    import infer_effmedsam2 as cli

    class Predictor:
        def init_state(self, image, height, width):
            return (len(image), height, width)

        def reset_state(self, state):
            pass

        def add_new_points_or_box(self, *, inference_state, **kwargs):
            _, h, w = inference_state
            return 0, [1], torch.ones((1, 1, h, w))

        def add_new_mask(self, state, **kwargs):
            _, h, w = state
            return 0, [1], torch.ones((1, 1, h, w))

        def propagate_in_video(self, state, start_frame_idx, reverse):
            d, h, w = state
            for i in (range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, d)):
                yield i, [1], torch.ones((1, 1, h, w))

    pkg = ModuleType("efficient_track_anything")
    pkg.__path__ = []
    pkg.__file__ = str(tmp_path / "__init__.py")
    backend = ModuleType("efficient_track_anything.build_efficienttam")
    backend.build_efficienttam_video_predictor_npz = lambda *a, **kw: Predictor()
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ckpt, cfg = tmp_path / "test.pt", tmp_path / "test.yaml"
    ckpt.touch()
    cfg.touch()
    prompt = np.zeros((2, 32, 32), dtype=np.uint16)
    cv2.line(prompt[0], (6, 8), (17, 8), 2, thickness=2)
    cv2.line(prompt[1], (6, 17), (17, 17), 19, thickness=2)
    src = tmp_path / "sample.npz"
    np.savez(src, imgs=np.zeros(prompt.shape, dtype=np.uint8), gts=prompt,
             recist=prompt, spacing=np.ones(3))
    manifest = tmp_path / "policy.json"
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema": "recist-exempt-v1", "inputs": {h: [19]}}))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["infer_effmedsam2.py", "-i", str(src), "-o", str(out),
                        "--checkpoint", str(ckpt), "--cfg", str(cfg), "--exempt-manifest", str(manifest)])
    assert cli.main() == 0
    record = json.loads((out / "metrics/sample.metrics.json").read_text())
    assert record["source_sha256"] == h
    assert record["exemption_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert record["postprocess_exempt_ids"] == [19]
    assert [r["action"] for r in record["postprocess_actions"]] == ["xy_centroid_clip", "exempt_by_id"]
    with np.load(out / "predictions/sample.npz", allow_pickle=False) as pred:
        assert pred["instance_ids"].tolist() == [2, 19]
        assert 0 < pred["segs"][0].sum() < prompt.size
        assert pred["segs"][1].all()
    with (out / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["lesion_id"] for r in rows] == ["2", "19"]
    n = int((prompt == 19).sum())
    assert float(rows[1]["dice_score"]) == pytest.approx(2*n/(n + prompt.size), abs=0.000051)
