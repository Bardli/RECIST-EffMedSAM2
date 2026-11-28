import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.nodal import PROTOCOL, bind_request, derived_geometry
from utils.storage import IndependentPredictionWriter, load_independent_prediction
from utils.postprocess import finalize_and_save_predictions
from contextlib import nullcontext


def test_geometry_from_prompt_not_gt():
    r = np.zeros((60, 60), np.uint8)
    r[18, 19] = r[23, 25] = 1
    sp = np.array([2., 1., 3.])
    g = derived_geometry(r, sp, 9)
    p = np.array(g["source_endpoints_xy"])
    q = np.array(g["endpoints_xy"])
    assert g["seed_z"] == 9 and not g["endpoint_clamped"]
    np.testing.assert_allclose(q.mean(0), p.mean(0))
    ds, dl = (p[1]-p[0])*sp[:2], (q[1]-q[0])*sp[:2]
    assert ds @ dl == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(dl) == pytest.approx(1.25*np.linalg.norm(ds))


def test_edge_matches_study_length_convention():
    r = np.zeros((32, 32), np.uint8)
    r[1, 2:21] = 1
    g = derived_geometry(r, [1,1,1], 0)
    assert g["endpoint_clamped"] and g["nominal_mm"] == 20 and g["chord_mm"] == 11
    assert g["box_xyxy"] == [6,0,16,11]
    for bad in ([0,1,1], [1,np.nan,1], [1,1], [1,1,-1]):
        with pytest.raises(ValueError): derived_geometry(r,bad,0)


def test_case_bound_manifest(tmp_path):
    source, win, policy = (tmp_path/n for n in ['ct.npz','window.npz','manifest.json'])
    source.write_bytes(b'primary');win.write_bytes(b'twin')
    m = dict(protocol=PROTOCOL, input_sha256=hashlib.sha256(b'primary').hexdigest(),
             window_sha256=hashlib.sha256(b'twin').hexdigest(), confirmed_short_axis_ids=[3])
    args=SimpleNamespace(input_dir=source, window_dir=win, output_dir=tmp_path/'new',
                         nodal_manifest=policy, shift=0, overwrite=False, gts_path=None)
    for patch in [dict(protocol='exempt'),dict(input_sha256=m['window_sha256']),
                  dict(window_sha256=None),dict(confirmed_short_axis_ids=[]),
                  dict(confirmed_short_axis_ids=[3,3]),dict(confirmed_short_axis_ids=[False])]:
        policy.write_text(json.dumps({**m, **patch}))
        with pytest.raises(ValueError):bind_request(args)
    policy.write_text(json.dumps(m));assert bind_request(args)[0]=={3}
    with pytest.raises(FileExistsError): bind_request(args)


def test_same_id_subset_output_and_nominal_clip(tmp_path):
    r=np.zeros((2,31,31),np.uint8);r[0,8,10:20]=2;r[1,12,5:21]=19
    sp=[1.,1.,2.]; g=derived_geometry(r[1]==19,sp,1)
    with pytest.raises(ValueError): IndependentPredictionWriter(r.shape,r,sp,only_ids={5})
    prof=SimpleNamespace(phase=lambda _:nullcontext(),lesion_postprocess=lambda _:nullcontext())
    with IndependentPredictionWriter(r.shape,r,sp,gt=r.copy(),only_ids={19},scratch=tmp_path) as w:
        w.write_volume(19,np.ones(r.shape,np.uint8))
        target=tmp_path/'pred.npz'
        rec=finalize_and_save_predictions(w,target,profiler=prof,postprocess='moving_centroid',
                                          nodal_lengths={19:g['nominal_mm']},scratch=tmp_path)
    assert rec['instance_ids']==[19] and rec['postprocess_actions'][0]['radius_mm']==g['nominal_mm']/2
    assert rec['rows'][0]['recist_prompt_length_mm']==pytest.approx(g['short_mm'])
    with np.load(target) as z:
        assert z['instance_ids'].tolist()==[19] and len(z['segs'])==1
        yy,xx=np.ogrid[:31,:31]
        expected=(xx-15)**2+(yy-15)**2<=(g['nominal_mm']/2)**2
        np.testing.assert_array_equal(z['segs'][0,0], expected)
