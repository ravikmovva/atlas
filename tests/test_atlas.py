"""Fast checks that the instrument measures what it claims to. Run: pytest -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skimage import data  # noqa: E402

from atlas import degrade as D  # noqa: E402
from atlas.flow import endpoint_error, estimate_flow  # noqa: E402
from atlas.metrics import reference_motion, warp_error_curve, psnr  # noqa: E402
from atlas.synth import make_sequence  # noqa: E402
from atlas.warp import backward_warp, fb_consistency_mask  # noqa: E402


@pytest.fixture(scope="module")
def seq():
    return make_sequence(data.astronaut(), n_frames=6, size=(128, 160))


def test_backward_warp_with_gt_flow_reconstructs_frame(seq):
    ref, gt, fg = seq
    w, valid = backward_warp(ref[1], gt[0])
    keep = valid & ~fg[0] & ~fg[1]          # skip the moving patch's occlusion edges
    assert np.abs(w - ref[0]).mean(axis=-1)[keep].mean() < 0.02


def test_flow_estimator_is_accurate_on_ground_truth(seq):
    ref, gt, _ = seq
    assert endpoint_error(estimate_flow(ref[0], ref[1], "dis"), gt[0]) < 0.5


def test_fb_mask_accepts_consistent_flow():
    f = np.full((32, 32, 2), 1.5, np.float32)
    m = fb_consistency_mask(f, -f)
    assert m[2:-3, 2:-3].all()


def test_fb_mask_rejects_inconsistent_flow():
    f = np.full((32, 32, 2), 3.0, np.float32)
    assert not fb_consistency_mask(f, f)[4:-4, 4:-4].any()


def test_match_psnr_hits_target(seq):
    ref = seq[0]
    target = D.mean_psnr(ref, D.blur(ref, 1.0))
    _, rec, db = D.match_psnr(D.noise_flicker, ref, target)
    assert abs(db - target) < 0.05


def test_flicker_exceeds_blur_at_matched_psnr(seq):
    ref = seq[0]
    mot = reference_motion(ref)
    b = D.blur(ref, 1.0)
    _, f, _ = D.match_psnr(D.noise_flicker, ref, D.mean_psnr(ref, b))
    assert np.nanmean(warp_error_curve(f, mot)) > 3 * np.nanmean(warp_error_curve(b, mot))


def test_psnr_identity():
    x = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert psnr(x, x) == float("inf")
